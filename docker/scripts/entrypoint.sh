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

# Ensure model directory exists
mkdir -p "$MODEL_DIR"

echo ""
log_info "=========================================="
log_info "  Dialogue Anonymizer Container Startup  "
log_info "=========================================="
echo ""

# Check if models exist
count_whisper() {
    find "$MODEL_DIR" -maxdepth 1 -name "Systran*" -type d 2>/dev/null | wc -l
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

# Download models if requested and missing
if [ "$DOWNLOAD_MODELS" = "true" ]; then
    log_info "DOWNLOAD_MODELS=true - downloading models if missing..."
    
    # Use the existing download_models.py script
    DL_SCRIPT="$APP_DIR/scripts/download_models.py"
    
    if [ ! -f "$DL_SCRIPT" ]; then
        log_warn "Download script not found at $DL_SCRIPT"
        log_info "Available scripts:"
        ls -la "$APP_DIR/scripts/" 2>/dev/null || echo "  No scripts directory"
    else
        if [ "$WHISPER_COUNT" -eq 0 ]; then
            log_info "Downloading WhisperX model ($WHISPER_MODEL)..."
            python "$DL_SCRIPT" \
                --whisper "$WHISPER_MODEL" \
                --base-path "$APP_DIR" \
                ${HF_TOKEN:+--hf-token "$HF_TOKEN"} || \
                log_warn "WhisperX download failed - continuing anyway"
        else
            log_info "WhisperX models already present"
        fi
        
        if [ "$PYANNOTE_COUNT" -eq 0 ]; then
            log_info "Downloading Pyannote model..."
            python "$DL_SCRIPT" \
                --pyannote \
                --base-path "$APP_DIR" \
                ${HF_TOKEN:+--hf-token "$HF_TOKEN"} || \
                log_warn "Pyannote download failed - continuing anyway"
        else
            log_info "Pyannote model already present"
        fi
        
        if [ "$PII_COUNT" -eq 0 ]; then
            log_info "Downloading PII NER model..."
            python "$DL_SCRIPT" \
                --pii \
                --base-path "$APP_DIR" \
                ${HF_TOKEN:+--hf-token "$HF_TOKEN"} || \
                log_warn "PII model download failed - continuing anyway"
        else
            log_info "PII NER model already present"
        fi
    fi
    
    log_success "Model download phase complete"
    echo ""
else
    log_info "DOWNLOAD_MODELS=false - models expected from volume mount"
    log_info "  If models are missing, set DOWNLOAD_MODELS=true in .env"
    echo ""
fi

# Verify .env exists
if [ ! -f "$APP_DIR/.env" ]; then
    log_warn ".env not found - application may use defaults or fail"
fi

echo ""
log_info "Starting application: $@"
echo ""

# Execute the CMD passed by Docker
exec "$@"