# ============================================================================
# Backend Application Dockerfile (Pre-built Dependencies - No apt-get)
# ============================================================================

# Jupyter datascience notebook has Python 3.12, ffmpeg, and system libs pre-installed
FROM jupyter/datascience-notebook:python-5a8a51e7387d AS base

# Already includes: Python 3.12, ffmpeg, curl, git, wget, libgomp1, libsm6, libxext6
# No apt-get needed!

WORKDIR /app

# Copy requirements files
COPY docker/backend/requirements.txt ./requirements.txt
COPY ../../requirements-lock.txt ./requirements-lock.txt

# Install ONLY Python dependencies (no system packages!)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-lock.txt

# Copy application code
COPY --chown=1000:100 interactive_app/ ./interactive_app/
COPY --chown=1000:100 pipeline/ ./pipeline/
COPY --chown=1000:100 interactive_app/audio_utils.py ./audio_utils.py

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5000

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
