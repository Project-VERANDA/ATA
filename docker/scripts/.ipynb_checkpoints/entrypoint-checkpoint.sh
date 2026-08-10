#!/bin/bash

# ============================================================================
# Docker Entrypoint Script
# Handles model initialization and container startup
# ============================================================================

set -e

APP_DIR="/app"
MODEL_DIR="$APP_DIR/pipeline/model"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-false}"
HF_TOKEN="${HUGGINGFACE_TOKEN:-}"
WHISPER_MODEL="${WHISPER_MODEL:-large}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[ENTRYPOINT]${NC} $*"; }
log_success() { echo -e "${GREEN}[ENTRYPOINT]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[ENTRYPOINT]${NC} $*"; }
log_error()   { echo -e "${RED}[ENTRYPOINT]${NC} $*" >&2; }

# Ensure model directory exists and has proper permissions
mkdir -p "$MODEL_DIR"
chown -R appuser:appgroup "$MODEL_DIR" 2>/dev/null || true

echo ""
log_info "=========================================="
log_info "  Dialogue Anonymizer Container Startup  "
log_info "=========================================="
echo ""

# Check if models exist
count_whisper() {
    ls -1 "$MODEL_DIR"/Systran* 2>/dev/null | wc -l || echo 0
}

count_pyannote() {
    [ -d "$MODEL_DIR/models--pyannote--speaker-diarization-community-1" ] && echo 1 || echo 0
}

count_pii() {
    [ -d "$MODEL_DIR/multilingual_DialogPII_NER" ] && echo 1 || echo 0
}

WHISPER_COUNT=$(count_whisper)
PYANNOTE_COUNT=$(count_pyannote)
PII_COUNT=$(count_pii)

log_info "Current model state:"
log_info "  WhisperX:   $WHISPER_COUNT models"
log_info "  Pyannote:   $PYANNOTE_COUNT model"
log_info "  PII NER:    $PII_COUNT model"
echo ""

# Download models if requested and none exist
if [ "$DOWNLOAD_MODELS" = "true" ]; then
    log_info "DOWNLOAD_MODELS=true - downloading models if missing..."
    
    if [ "$WHISPER_COUNT" -eq 0 ]; then
        log_info "Downloading WhisperX model ($WHISPER_MODEL)..."
        python "$APP_DIR/scripts/download_models.py" \
            --whisper "$WHISPER_MODEL" \
            --base-path "$APP_DIR" \
            ${HF_TOKEN:+--hf-token "$HF_TOKEN"}
    else
        log_info "WhisperX models already present"
    fi
    
    if [ "$PYANNOTE_COUNT" -eq 0 ]; then
        log_info "Downloading Pyannote model..."
        python "$APP_DIR/scripts/download_models.py" \
            --pyannote \
            --base-path "$APP_DIR" \
            ${HF_TOKEN:+--hf-token "$HF_TOKEN"}
    else
        log_info "Pyannote model already present"
    fi
    
    if [ "$PII_COUNT" -eq 0 ]; then
        log_info "Downloading PII NER model..."
        python "$APP_DIR/scripts/download_models.py" \
            --pii \
            --base-path "$APP_DIR" \
            ${HF_TOKEN:+--hf-token "$HF_TOKEN"}
    else
        log_info "PII NER model already present"
    fi
    
    log_success "Model download complete"
    echo ""
else
    log_info "DOWNLOAD_MODELS=false - skipping automatic model download"
    log_info "Models will be mounted from volume or downloaded on-demand by application"
    echo ""
fi

# Verify .env exists
if [ ! -f "$APP_DIR/.env" ]; then
    log_warn ".env not found - application may use defaults or fail"
    log_info "  Mount .env file via docker-compose volumes"
fi

# Check required directories
for dir in pipeline interactive_app; do
    if [ ! -d "$APP_DIR/$dir" ]; then
        log_warn "Directory missing: $APP_DIR/$dir"
    fi
done

echo ""
log_info "Starting Flask application..."
echo ""

# Execute the main command (passed as arguments to entrypoint)
# This allows docker compose CMD to be respected
exec "$@"