# CPU-only Python base — no CUDA needed for inference
FROM python:3.12-slim

# System deps for torch CPU build
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install torch CPU first separately — avoids pulling CUDA (~3GB)
RUN pip install torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

# Install remaining requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# Copy project files
COPY src/       ./src/
COPY api/       ./api/
COPY models/    ./models/
COPY config.yaml .

# Create directories
RUN mkdir -p data/features data/raw data/processed logs

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check — Docker pings this every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
