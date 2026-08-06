# =============================================================================
# BACKEND DOCKERFILE - Medical Data Processing
# CUDA 13.0 Compatible
# =============================================================================

FROM nvidia/cuda:13.0-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="VERANDA Project"
LABEL description="Speech Anonymization Backend - PHI/PII Processing"
LABEL version="1.0.0"
LABEL security="Medical Data Safe"

# ============================================================================
# SYSTEM USER SETUP (No root execution)
# ============================================================================

RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core utilities
    curl \
    wget \
    git \
    ca-certificates \
    gnupg \
    \
    # FFmpeg (audio processing)
    ffmpeg \
    \
    # Python
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3-venv \
    \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    \
    # Security
    openssl \
    libssl3 \
    \
    # Cleaning
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# ============================================================================
# PYTHON VIRTUAL ENVIRONMENT
# ============================================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python3.12 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# ============================================================================
# COPY REQUIREMENTS FIRST (Layer Caching)
# ============================================================================

COPY backend/requirements.lock.txt ./requirements.lock.txt
COPY backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.lock.txt \
    && pip check

# ============================================================================
# COPY APPLICATION CODE (As Non-Root User)
# ============================================================================

COPY --chown=appuser:appgroup interactive_app/ ./interactive_app/
COPY --chown=appuser:appgroup pipeline/ ./pipeline/
COPY --chown=appuser:appgroup audio_utils.py ./
COPY --chown=appuser:appgroup process.py ./pipeline/process.py
COPY --chown=appuser:appgroup tts_engine.py ./pipeline/tts/tts_engine.py

# ============================================================================
# DIRECTORY STRUCTURE WITH RESTRICTIVE PERMISSIONS
# ============================================================================

RUN mkdir -p \
    /app/pipeline/{videos,audios,transcripts,model,anonym,LLM-Anonym} \
    /app/pipeline/tts/{bin,voices} \
    /app/uploads \
    /app/logs \
    /app/tmp \
    && chown -R appuser:appgroup /app \
    && chmod -R 750 /app \
    && chmod 1770 /app/tmp

# Set restrictive permissions on model folder (read-only for inference)
RUN chmod 550 /app/pipeline/model \
    && find /app/pipeline/model -type f -exec chmod 440 {} \; \
    && find /app/pipeline/model -type d -exec chmod 550 {} \;

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================

ENV ATA_HOME=/app
ENV MODEL_DIR=/app/pipeline/model
ENV PYTHONPATH=/app:$PYTHONPATH
ENV FLASK_ENV=production
ENV LOG_LEVEL=INFO

# GPU configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV CUDA_VISIBLE_DEVICES=0

# Security - Disable any network calls from container except necessary
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0

# ============================================================================
# HEALTH CHECK
# ============================================================================

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -kf https://localhost:5001/health || exit 1

# ============================================================================
# PORT EXPOSURE
# ============================================================================

EXPOSE 5001

# ============================================================================
# RUN AS NON-ROOT USER
# ============================================================================

USER appuser

# ============================================================================
# ENTRYPOINT
# ============================================================================

CMD ["python", "interactive_app/app.py"]
