# ============================================================================
# Backend Application Dockerfile (CUDA 13.3.1 + Python 3.12)
# ============================================================================

FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies AND add Python 3.12 PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
    ca-certificates \
    ffmpeg \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add DeadSnakes PPA for Python 3.12 (not in Ubuntu 22.04 default repos)
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        libpython3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy requirements files (docker/ exclusion now fixed in .dockerignore)
COPY docker/backend/requirements.txt ./requirements.txt
COPY ../../requirements-lock.txt ./requirements-lock.txt

# Create virtual environment and install dependencies
RUN python3.12 -m venv /app/venv && \
    /app/venv/bin/pip install --upgrade pip setuptools wheel && \
    /app/venv/bin/pip install --no-cache-dir -r requirements-lock.txt

# Copy application code
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
