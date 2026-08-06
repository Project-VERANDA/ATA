# ============================================================================
# Backend Application Dockerfile (NVIDIA PyTorch Base)
# ============================================================================

# NVIDIA PyTorch container has Python 3.12+, CUDA, PyTorch, ffmpeg pre-installed
FROM nvcr.io/nvidia/pytorch:24.02-py3 AS base

WORKDIR /app

# Copy requirements files
COPY docker/backend/requirements.txt ./requirements.txt
COPY ../../requirements-lock.txt ./requirements-lock.txt

# Install ONLY Python dependencies (no apt-get!)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-lock.txt

# Copy application code
COPY --chown=1000:100 interactive_app/ ./interactive_app/
COPY --chown=1000:100 pipeline/ ./pipeline/
COPY --chown=1000:100 interactive_app/audio_utils.py ./audio_utils.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
