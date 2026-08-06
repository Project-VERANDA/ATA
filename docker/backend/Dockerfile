# ============================================================================
# Backend Application Dockerfile (Python 3.12 from slim image)
# ============================================================================

# Start with Python 3.12 slim image (includes CUDA libraries)
FROM python:3.12-slim-bookworm AS base

# Install CUDA runtime libraries separately (we don't need the full devel image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ca-certificates \
    ffmpeg \
    build-essential \
    libnvidia-opencl1 \
    libcuda1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy requirements files
COPY docker/backend/requirements.txt ./requirements.txt
COPY ../../requirements-lock.txt ./requirements-lock.txt

# Install dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-lock.txt

# Copy application code
COPY --chown=appuser:appgroup interactive_app/ ./interactive_app/
COPY --chown=appuser:appgroup pipeline/ ./pipeline/
COPY --chown=appuser:appgroup interactive_app/audio_utils.py ./audio_utils.py

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
