# ============================================================================
# Backend Application Dockerfile (CUDA 13.3.1)
# ============================================================================

FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy requirements files (CORRECTED PATHS)
COPY ../../docker/backend/requirements.txt ./requirements.txt
COPY ../../requirements-lock.txt ./requirements-lock.txt

# Create virtual environment and install dependencies
RUN python3.12 -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install --no-cache-dir -r requirements-lock.txt

# Copy application code (CORRECTED PATHS)
COPY --chown=appuser:appgroup interactive_app/ ./interactive_app/
COPY --chown=appuser:appgroup pipeline/ ./pipeline/
COPY --chown=appuser:appgroup interactive_app/audio_utils.py ./audio_utils.py

# Set environment variables
ENV PATH="/app/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
