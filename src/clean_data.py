import pandas as pd
import numpy as np

df = pd.read_csv('vabb_metar_clean.csv')

print(f"Starting rows: {len(df)}")

# Step 1: Convert timestamp to datetime, so that python and my model understands it as time, not just a string. Also set dayfirst=True since the format is YYYYMMDDHHMM.
df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)

# Step 2: Sort by time
df = df.sort_values('timestamp').reset_index(drop=True)

# Step 3: Drop duplicate timestamps, keeping only the first occurrence. This is important because we want a unique timestamp for each row, and duplicates can cause issues in time series analysis and modeling. We keep the first occurrence because it is likely the most complete record, while duplicates may have missing values or inconsistencies.
df = df.drop_duplicates(subset=['timestamp'], keep='first')
print(f"After dropping duplicates: {len(df)}")

# Step 4: Fix wind_dir=None when wind_speed=0 (calm wind)
df.loc[df['wind_speed'] == 0, 'wind_dir'] = 0

# Step 5: Drop physical impossibles
df = df[df['wind_dir'] <= 360]
df = df[df['wind_speed'] <= 80]
df = df[df['gust'] <= 100]
df = df[df['temp'].between(10, 42)]
df = df[df['dewpoint'] <= 35]
print(f"After dropping impossibles: {len(df)}")

# Step 6: Forward fill small gaps (pressure, visibility, remaining missing)
# Only fill gaps of 3 hours or less (6 x 30min intervals), if done more, than the data can be inaccurate. thus, what we do is, tell python to just fill for the next 6 rows maximum with the value previous it, and if there are more than 6 rows of missing values, then it will not fill those and will leave them as NaN, which we will drop in the next step.
df = df.set_index('timestamp')
#df = df.fillna(method='ffill', limit=6) -- old way because of this FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.
# The updated, correct way
df = df.ffill(limit=6)
df = df.reset_index()

# Step 7: Drop any remaining rows with missing values
df = df.dropna()
print(f"After dropping remaining nulls: {len(df)}")

# Step 8: Save cleaned file
df.to_csv('vabb_metar_cleaned_v2.csv', index=False)
print(f"Saved. Final rows: {len(df)}")
print(df.describe())