import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
SEQ_LEN    = 48      # 12 hours past input (24/48)
HORIZON    = 12      # 6 hours ahead prediction
BATCH_SIZE = 256
EPOCHS     = 50
LR         = 1e-3
D_MODEL    = 128      # attention dimension (64/128)
N_HEADS    = 8       # attention heads (4/8)
DROPOUT    = 0.3
PATIENCE   = 10
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── FEATURES & TARGETS ─────────────────────────────────
FEATURE_COLS = [
    'wind_dir_sin','wind_dir_cos','hour_sin','hour_cos',
    'month_sin','month_cos','u_wind','v_wind',
    'pressure_tendency','temp_trend','dewpoint_depression',
    'cooling_rate','monsoon_flag','sea_breeze_phase',
    'wind_speed','gust','pressure','visibility','temp','dewpoint'
]

REG_TARGETS = ['temp','wind_speed','gust','pressure','visibility','u_wind','v_wind']
LOW_VIS_THR = 1000

# ── LOAD DATA ──────────────────────────────────────────
from pathlib import Path
BASE_DIR = Path(__file__).parent
csv_path = BASE_DIR.parent / "Vabb_Metar_Data" / "vabb_metar_features_updated.csv"
df = pd.read_csv(csv_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"Loaded: {len(df)} rows")

# ── SCALE ──────────────────────────────────────────────
split_idx = int(len(df) * 0.8)

X_all = df[FEATURE_COLS].values
y_all = df[REG_TARGETS].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
scaler_X.fit(X_all[:split_idx])
scaler_y.fit(y_all[:split_idx])

X_scaled = scaler_X.transform(X_all)
y_scaled  = scaler_y.transform(y_all)

vis_idx  = REG_TARGETS.index('visibility')
vis_mean = scaler_y.mean_[vis_idx]
vis_std  = scaler_y.scale_[vis_idx]

# ── DATASET ────────────────────────────────────────────
class METARDataset(Dataset):
    def __init__(self, X, y, gap_flags, seq_len, horizon):
        self.X         = torch.tensor(X, dtype=torch.float32)
        self.y         = torch.tensor(y, dtype=torch.float32)
        self.gap_flags = gap_flags
        self.seq_len   = seq_len
        self.horizon   = horizon
        self.indices   = self._build_indices()

    def _build_indices(self):
        valid = []
        for i in range(self.seq_len, len(self.X) - self.horizon):
            if self.gap_flags[i - self.seq_len : i + self.horizon + 1].max() == 0:
                valid.append(i)
        return valid

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i - self.seq_len : i]
        y = self.y[i + self.horizon - 1]
        return x, y

gap_flags = df['gap_flag'].values
train_ds  = METARDataset(X_scaled[:split_idx], y_scaled[:split_idx],
                          gap_flags[:split_idx], SEQ_LEN, HORIZON)
test_ds   = METARDataset(X_scaled[split_idx:], y_scaled[split_idx:],
                          gap_flags[split_idx:], SEQ_LEN, HORIZON)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

# ── TFT MODEL ──────────────────────────────────────────
class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT:
    1. Input projection
    2. Multi-head self-attention (finds which past hours matter most)
    3. Gating (decides how much attention output to use)
    4. Regression head (outputs 7 weather variables)
    5. Classification head (fog yes/no)

    Key difference from LSTM:
    LSTM reads sequence left to right, one step at a time.
    TFT looks at ALL 24 past timesteps simultaneously and
    learns which specific ones to pay attention to.
    """
    def __init__(self, input_size, d_model, n_heads, n_targets, dropout):
        super().__init__()

        # Project input features to d_model dimensions
        self.input_proj = nn.Linear(input_size, d_model)

        # Multi-head attention — core of TFT
        # Finds which past timesteps matter most for prediction
        self.attention = nn.MultiheadAttention(
            embed_dim    = d_model,
            num_heads    = n_heads,
            dropout      = dropout,
            batch_first  = True
        )

        # Gating — learned switch between attention and direct input
        # Prevents attention from hurting when sequence is simple
        self.gate     = nn.Linear(d_model * 2, d_model)
        self.gate_act = nn.Sigmoid()

        # Layer normalisation — stabilises training
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), # 2 can be 4 if needed for better
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model) # 2 can be 4 if needed for better
        )

        self.dropout = nn.Dropout(dropout)

        # Regression head — 7 weather variables
        self.fc_reg = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, n_targets)
        )

        # Classification head — fog yes/no
        self.fc_cls = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        proj = self.input_proj(x)          # (batch, seq, d_model)

        # Self-attention: each timestep attends to all others
        attn_out, attn_weights = self.attention(proj, proj, proj)

        # Gating mechanism
        gate_input  = torch.cat([proj, attn_out], dim=-1)
        gate_values = self.gate_act(self.gate(gate_input))
        gated       = gate_values * attn_out + (1 - gate_values) * proj

        # Residual + norm
        x1 = self.norm1(proj + self.dropout(gated))

        # Feed-forward + residual + norm
        x2 = self.norm2(x1 + self.dropout(self.ffn(x1)))

        # Use last timestep for prediction
        last = x2[:, -1, :]

        reg = self.fc_reg(last)
        cls = self.fc_cls(last).squeeze(-1)

        return reg, cls, attn_weights

model = TemporalFusionTransformer(
    input_size = len(FEATURE_COLS),
    d_model    = D_MODEL,
    n_heads    = N_HEADS,
    n_targets  = len(REG_TARGETS),
    dropout    = DROPOUT
).to(DEVICE)

print(f"TFT parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── LOSS & OPTIMISER ───────────────────────────────────
pos_weight     = torch.tensor([50.0]).to(DEVICE)
criterion_reg  = nn.MSELoss()
criterion_cls  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer      = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode='min', factor=0.5, patience=5)

# ── TRAINING ───────────────────────────────────────────
best_val_loss    = float('inf')
patience_counter = 0
history          = {'train_loss': [], 'val_loss': []}

print("\nTraining TFT...")
for epoch in range(EPOCHS):

    # Train
    model.train()
    train_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        vis_metres = y_batch[:, vis_idx] * vis_std + vis_mean
        cls_label  = (vis_metres < LOW_VIS_THR).float()

        optimizer.zero_grad()
        pred_reg, pred_cls, _ = model(x_batch)

        loss = criterion_reg(pred_reg, y_batch) + \
               0.1 * criterion_cls(pred_cls, cls_label)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    # Validate
    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            vis_metres = y_batch[:, vis_idx] * vis_std + vis_mean
            cls_label  = (vis_metres < LOW_VIS_THR).float()

            pred_reg, pred_cls, _ = model(x_batch)
            loss = criterion_reg(pred_reg, y_batch) + \
                   0.1 * criterion_cls(pred_cls, cls_label)
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss   = np.mean(val_losses)
    scheduler.step(val_loss)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'tft_best.pt')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")

print(f"\nBest val loss: {best_val_loss:.4f}")

# ── EVALUATION ─────────────────────────────────────────
print("\nEvaluating best TFT model...")
model.load_state_dict(torch.load('tft_best.pt'))
model.eval()

all_preds, all_true   = [], []
all_cls_probs, all_cls_true = [], []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        pred_reg, pred_cls, _ = model(x_batch)
        all_preds.append(pred_reg.cpu().numpy())
        all_true.append(y_batch.numpy())

        vis_metres = y_batch[:, vis_idx].numpy() * vis_std + vis_mean
        all_cls_true.append((vis_metres < LOW_VIS_THR).astype(int))
        all_cls_probs.append(torch.sigmoid(pred_cls).cpu().numpy())

preds_orig = scaler_y.inverse_transform(np.concatenate(all_preds))
true_orig  = scaler_y.inverse_transform(np.concatenate(all_true))

# Regression results
print(f"\n── TFT Regression Results (6hr horizon) ──")
print(f"{'Variable':12}  {'MAE':>8}  {'RMSE':>8}")
tft_results = []
for i, var in enumerate(REG_TARGETS):
    mae  = mean_absolute_error(true_orig[:, i], preds_orig[:, i])
    rmse = np.sqrt(mean_squared_error(true_orig[:, i], preds_orig[:, i]))
    tft_results.append({'variable': var, 'MAE': round(mae,3), 'RMSE': round(rmse,3)})
    print(f"{var:12}  {mae:>8.3f}  {rmse:>8.3f}")

#Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix, recall_score

cls_probs = np.concatenate(all_cls_probs)
cls_true  = np.concatenate(all_cls_true)

print(f"\n── TFT Low-vis Classification ──")
print(f"{'Threshold':>10}  {'Recall':>8}  {'Precision':>10}  {'F1':>6}")
best_f1, best_thresh = 0, 0.3
for thresh in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1]:
    pred_t = (cls_probs >= thresh).astype(int)
    rec  = recall_score(cls_true, pred_t, zero_division=0)
    from sklearn.metrics import precision_score
    prec = precision_score(cls_true, pred_t, zero_division=0)
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
    flag = " ← best F1" if f1 > best_f1 else ""
    print(f"{thresh:>10.2f}  {rec:>8.3f}  {prec:>10.3f}  {f1:>6.3f}{flag}")
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

best_preds = (cls_probs >= best_thresh).astype(int)
print(f"\nBest threshold: {best_thresh}")
print(classification_report(cls_true, best_preds,
      target_names=['Normal vis', 'Low vis (<1000m)']))
print("Confusion matrix:")
print(confusion_matrix(cls_true, best_preds))

# Comparison table
print(f"\n── Model Comparison — RMSE @6hr ──")
lstm_rmse = {'temp':1.133,'wind_speed':2.267,'gust':3.127,
             'pressure':0.990,'visibility':509.526,'u_wind':3.039,'v_wind':2.668}
xgb_rmse  = {'temp':1.137,'wind_speed':2.351,'gust':3.223,
             'pressure':0.739,'visibility':550.126,'u_wind':3.156,'v_wind':2.758}

print(f"{'Variable':12}  {'XGB':>8}  {'LSTM':>8}  {'TFT':>8}  {'Best':>6}")
for r in tft_results:
    var  = r['variable']
    xgb  = xgb_rmse[var]
    lstm = lstm_rmse[var]
    tft  = r['RMSE']
    best = min(xgb, lstm, tft)
    who  = 'XGB' if best==xgb else ('LSTM' if best==lstm else 'TFT')
    print(f"{var:12}  {xgb:>8.3f}  {lstm:>8.3f}  {tft:>8.3f}  {who:>6}")

# Save results
pd.DataFrame(tft_results).to_csv('tft_results.csv', index=False)

# Save timestamps + predictions for ensemble
test_timestamps = [
    df['timestamp'].iloc[split_idx + idx + HORIZON - 1]
    for idx in test_ds.indices
]
pred_df = pd.DataFrame({'timestamp': test_timestamps})
for i, target in enumerate(REG_TARGETS):
    pred_df[f'actual_{target}']    = true_orig[:, i]
    pred_df[f'predicted_{target}'] = preds_orig[:, i]
pred_df.to_csv('tft_predictions_with_timestamps.csv', index=False)

pd.DataFrame(history).to_csv('tft_training_history.csv', index=False)
print("\nSaved: tft_results.csv, tft_predictions_with_timestamps.csv")
print("Done.")