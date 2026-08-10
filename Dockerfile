# ============================================================================
# Dialogue Anonymizer Dockerfile
# Supports optional features via build arguments
# ============================================================================

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

# Build arguments for optional features
ARG BUILD_WEB=true
ARG DOWNLOAD_MODELS=false
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm

# Proxy configuration
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTPS_PROXY}
ENV NO_PROXY=${NO_PROXY}
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git ca-certificates ffmpeg \
    libgomp1 libsm6 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy requirements files
COPY requirements.txt ./requirements.txt
COPY requirements-web.txt ./requirements-web.txt

# Install core ML dependencies (always required)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Conditionally install web dependencies
ARG BUILD_WEB
RUN if [ "$BUILD_WEB" = "true" ]; then \
      pip install --no-cache-dir -r requirements-web.txt; \
    fi

# Copy application code
COPY --chown=appuser:appgroup interactive_app/ ./interactive_app/
COPY --chown=appuser:appgroup pipeline/ ./pipeline/
COPY --chown=appuser:appgroup interactive_app/audio_utils.py ./audio_utils.py

# Create model directory with proper permissions for optional model downloads at runtime
RUN mkdir -p /app/pipeline/model && chown -R appuser:appgroup /app/pipeline/model

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

# Health check endpoint (adjust if your Flask app uses different path)
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Entry point script for optional model initialization
# COPY entrypoint.sh ./entrypoint.sh
# RUN chmod +x ./entrypoint.sh
# ENTRYPOINT ["./entrypoint.sh"]

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]