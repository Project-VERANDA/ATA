#!/usr/bin/bash

# ============================================================================
# ATA Speech Anonymizer Installer (v3.1 - Latest Dependencies)
# NumPy 2.x Compatible | Latest PyTorch 2.8+ | Streamlined TTS
# ============================================================================

set -e

SHOW_HELP=false
SKIP_WEB=false
NO_MODELS=false
SKIP_WHISPER=false
SKIP_PYANNOTE=false
AUTO_LOGIN=false
ANSWER_YES=false
QUIET_MODE=false
WHISPER_SELECTION=""
FORCE_REFRESH=false
HUGGINGFACE_TOKEN=""

usage() {
    cat << 'EOF'
Usage: ./ATA_SelfInstall.sh [OPTIONS]

Options:
  -h, --help              Show this help message
  -y, --yes               Auto-answer 'yes' to all prompts
  --skip-web              Skip web interface installation (Flask, TTS)
  --no-models             Skip ALL ML model downloads
  --skip-whisper          Skip WhisperX model download
  --skip-pyannote         Skip Pyannote diarization model download
  --whisper-models LIST   Comma-separated Whisper models (default: large-v3)
  --auto-login            Auto-authenticate with HUGGINGFACE_TOKEN env var
  -q, --quiet             Minimal output mode
  --force-refresh         Unconditionally reinstall all packages
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        -y|--yes) ANSWER_YES=true; shift ;;
        --skip-web) SKIP_WEB=true; shift ;;
        --no-models) NO_MODELS=true; shift ;;
        --skip-whisper) SKIP_WHISPER=true; shift ;;
        --skip-pyannote) SKIP_PYANNOTE=true; shift ;;
        --whisper-models) WHISPER_SELECTION="$2"; shift 2 ;;
        --auto-login) AUTO_LOGIN=true; shift ;;
        -q|--quiet) QUIET_MODE=true; shift ;;
        --force-refresh) FORCE_REFRESH=true; shift ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    usage
    exit 0
fi

if [ "$QUIET_MODE" = true ]; then
    exec >/dev/null 2>&1
fi

# --- 1. MAIN FOLDER LOGIC ---
CURRENT_DIR="$(pwd)"
SCRIPT_NAME="$(basename "$0")"
MAIN_DIR_NAME="ATA"

shopt -s nullglob

if [ "$(basename "$CURRENT_DIR")" == "$MAIN_DIR_NAME" ]; then
    echo "✅ Already inside '$MAIN_DIR_NAME' folder. Proceeding..."
else
    echo "📂 Not inside '$MAIN_DIR_NAME'. Checking for project files..."
    if [ -d "pipeline" ] || [ -f "README.md" ]; then
        echo "✅ Detected project root. Creating '$MAIN_DIR_NAME' and moving files..."
        mkdir -p "$MAIN_DIR_NAME"
        for item in *; do
            if [ "$item" != "$MAIN_DIR_NAME" ]; then
                mv "$item" "$MAIN_DIR_NAME/"
            fi
        done
        cd "$MAIN_DIR_NAME"
        CURRENT_DIR="$(pwd)"
        echo "✅ Moved all files into '$MAIN_DIR_NAME'. New location: $CURRENT_DIR"
    else
        echo "❌ Error: Not inside '$MAIN_DIR_NAME' and no project files found."
        exit 1
    fi
fi

cd "$CURRENT_DIR"

echo "=== ATA Speech Anonymizer Installer (v3.1 - Latest Dependencies) ==="
echo "Working Directory: $(pwd)"
echo ""

# 2. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
else
    echo "Conda already installed."
    eval "$(conda shell.bash hook)"
fi

# 3. Create the environment
ENV_NAME="whisperx"
TARGET_PYTHON="3.13"

echo "Creating conda environment '$ENV_NAME' with Python $TARGET_PYTHON..."

if conda env list | grep -q "^$ENV_NAME "; then
    if [ "$FORCE_REFRESH" = true ]; then
        echo "🔄 Environment exists (--force-refresh). Removing and recreating..."
        conda env remove -n $ENV_NAME -y
        rm -rf ~/miniconda3/envs/$ENV_NAME 2>/dev/null || true
        rm -rf ~/miniforge3/envs/$ENV_NAME 2>/dev/null || true
    else
        echo "⚠️  Environment '$ENV_NAME' already exists."
        echo "   Using existing environment (use --force-refresh to recreate)"
    fi
else
    echo "Creating fresh environment with Python $TARGET_PYTHON..."
fi

conda create -n $ENV_NAME python=$TARGET_PYTHON -c conda-forge -y

# 4. Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME
MODEL_DIR="$CURRENT_DIR/pipeline/model"
mkdir -p "$MODEL_DIR"

ACTUAL_PYTHON=$(conda run -n $ENV_NAME python --version | awk '{print $2}')
EXPECTED_PYTHON="3.12"
if [[ ! "$ACTUAL_PYTHON" =~ ^3\.12 ]]; then
    echo "❌ ERROR: Expected Python 3.12.x, got $ACTUAL_PYTHON"
    exit 1
fi
echo "✅ Python version verified: $ACTUAL_PYTHON"

# ============================================================================
# 🔧 SYSTEM DEPENDENCIES
# ============================================================================

echo ""
echo "============================================================"
echo "Installing System Dependencies"
echo "============================================================"

# FFmpeg (required for video/audio processing)
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    sudo apt-get update -q
    sudo apt-get install -y ffmpeg || echo "⚠️  FFmpeg installation failed (may work without it)"
else
    echo "✅ FFmpeg already installed"
fi

# ============================================================================
# 🔧 NUMPY 2.x + LATEST PYTORCH INSTALLATION
# ============================================================================

echo ""
echo "============================================================"
echo "Installing ML Stack (NumPy 2.x + Latest PyTorch 2.8+)"
echo "============================================================"

pip install --upgrade pip -q

# Core ML stack with NumPy 2.x compatibility
echo "  → numpy>=2.0.0 (NumPy 2.x series)..."
pip install "numpy>=2.0.0,<3.0.0" --no-cache-dir

echo "  → scipy..."
pip install "scipy>=1.18.0" --no-cache-dir

# LATEST PyTorch 2.8+ (updated from 2.0.0)
echo "  → torch>=2.8.0 (LATEST STABLE)..."
pip install "torch>=2.8.0,<3.0.0" --no-cache-dir || \
pip install torch --no-cache-dir

echo "  → torchaudio..."
pip install "torchaudio>=2.8.0" --no-cache-dir || \
pip install torchaudio --no-cache-dir

echo "  → torchvision..."
pip install "torchvision>=0.23.0" --no-cache-dir || \
pip install torchvision --no-cache-dir

# LATEST torchcodec 0.14.0 (updated from 0.7.0)
echo "  → torchcodec>=0.14.0 (LATEST WITH NUMPY 2.x SUPPORT)..."
pip install "torchcodec>=0.14.0" --no-cache-dir || \
pip install "torchcodec>=0.10.0" --no-cache-dir || \
echo "⚠️  torchcodec installation failed (optional)"

# Transformers & dependencies (NumPy 2.x compatible)
echo "  → transformers..."
pip install "transformers>=4.50.0" --no-cache-dir

echo "  → accelerate..."
pip install "accelerate>=0.30.0" --no-cache-dir

echo "  → sentencepiece..."
pip install "sentencepiece>=0.1.99" --no-cache-dir

echo "  → huggingface-hub..."
pip install "huggingface-hub>=0.34.0" --no-cache-dir

# Other utilities
echo "  → pandas..."
pip install "pandas>=2.2.0" --no-cache-dir

echo "  → pydub..."
pip install pydub --no-cache-dir

echo "  → ffmpeg-python..."
pip install ffmpeg-python --no-cache-dir

echo "  → openai..."
pip install "openai>=1.0.0" --no-cache-dir

echo "  → python-dotenv..."
pip install python-dotenv --no-cache-dir

echo "  → torchcrf..."
pip install pytorch-crf --no-cache-dir

# CRITICAL: spaCy 3.8+ + thinc 8.3+ (NumPy 2.x compatible)
echo "  → thinc>=8.3.0 (NumPy 2.x compatible)..."
pip install "thinc>=8.3.0" --no-cache-dir

echo "  → spacy>=3.8.0 (NumPy 2.x compatible)..."
pip install "spacy>=3.8.0" --no-cache-dir

echo "  → blis..."
pip install "blis>=0.7.0" --no-cache-dir

echo "  → click..."
pip install "click>=8.1.7" --no-cache-dir

echo "  → typer..."
pip install "typer>=0.9.0" --no-cache-dir

# Speaker diarization (NumPy 2.x compatible)
echo "  → pyannote.audio (speaker diarization)..."
pip install "pyannote.audio>=4.0.0" --no-cache-dir || \
pip install "pyannote.audio>=4.0.0,<5.0.0" --no-cache-dir || \
echo "⚠️  Warning installing pyannote.audio"

# whisperx
echo "  → whisperx..."
pip uninstall whisperx -y 2>/dev/null || true
pip install git+https://github.com/m-bain/whisperx.git --no-cache-dir || \
pip install whisperx --no-cache-dir || \
echo "⚠️  Warning installing whisperx"

echo "✅ Base ML stack installation complete."

# ============================================================================
# 🔧 VERSION VERIFICATION
# ============================================================================

echo ""
echo "=== Version Verification ==="
NUMPY_VER=$(pip show numpy | grep Version | awk '{print $2}')
Torch_VER=$(pip show torch | grep Version | awk '{print $2}')
THINC_VER=$(pip show thinc | grep Version | awk '{print $2}')
SPACY_VER=$(pip show spacy | grep Version | awk '{print $2}')
PYANNOTE_VER=$(pip show pyannote.audio | grep Version | awk '{print $2}')
TORCHCODEC_VER=$(pip show torchcodec | grep Version | awk '{print $2}')

echo "  numpy: $NUMPY_VER"
echo "  torch: $Torch_VER"
echo "  thinc: $THINC_VER"
echo "  spacy: $SPACY_VER"
echo "  pyannote.audio: ${PYANNOTE_VER:-installed}"
echo "  torchcodec: ${TORCHCODEC_VER:-installed}"

if [[ ! "$NUMPY_VER" =~ ^2\. ]]; then
    echo "⚠️  WARNING: NumPy $NUMPY_VER (expected 2.x)"
else
    echo "✅ NumPy $NUMPY_VER confirmed (2.x series)"
fi

if [[ ! "$Torch_VER" =~ ^2\.[89] ]] && [[ ! "$Torch_VER" =~ ^3\. ]]; then
    echo "⚠️  NOTE: PyTorch $Torch_VER (2.8+ recommended)"
else
    echo "✅ PyTorch $Torch_VER confirmed (2.8+ latest)"
fi

if python -c "import spacy; import thinc; import pyannote.audio; import whisperx; import torchcodec" 2>/dev/null; then
    echo "✅ All core packages imported successfully"
else
    echo "❌ CRITICAL: Import test failed"
    exit 1
fi

# ============================================================================
# 🔧 SPACY MODEL INSTALLATION
# ============================================================================

echo ""
echo "============================================================"
echo "Downloading spaCy Models (for PII Detection)"
echo "============================================================"

if python -m spacy check en_core_web_sm &> /dev/null; then
    echo "✅ en_core_web_sm already installed."
else
    echo "Installing spaCy English model (en_core_web_sm)..."
    python -m spacy download en_core_web_sm
fi

if python -m spacy check xx_ent_wiki_sm &> /dev/null; then
    echo "✅ xx_ent_wiki_sm already installed."
else
    echo "Installing spaCy multilingual model (xx_ent_wiki_sm)..."
    python -m spacy download xx_ent_wiki_sm
fi

# ============================================================================
# 🖥️ WEB INTERFACE & TTS BACKENDS
# ============================================================================

echo ""
echo "============================================================"
echo "Web Interface & TTS Backend Installation"
echo "============================================================"

if [ "$SKIP_WEB" = true ]; then
    echo "Skipping Web Interface (--skip-web)"
    INSTALL_WEB="n"
elif [ "$ANSWER_YES" = true ]; then
    echo "Web Interface: YES (auto-answered --yes)"
    INSTALL_WEB="y"
else
    read -p "Install Web Interface (Flask + TTS)? (y/n): " INSTALL_WEB
    INSTALL_WEB=${INSTALL_WEB:-y}
fi

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    pip install flask requests cryptography --no-cache-dir
    
    # TTS BACKEND SELECTION
    echo ""
    echo "Select TTS Backend:"
    echo "  1. piper       - Fast offline neural TTS (recommended)"
    echo "  2. coqui_xtts  - High-quality voice cloning (17 langs)"
    
    if [ "$ANSWER_YES" = true ]; then
        TTS_BACKEND_CHOICE="piper"
    else
        read -p "Enter choice [1]: " TTS_CHOICE
        TTS_BACKEND_CHOICE=${TTS_CHOICE:-piper}
        [ "$TTS_CHOICE" = "2" ] && TTS_BACKEND_CHOICE="coqui_xtts"
    fi
    
    MODEL_DIR="$CURRENT_DIR/pipeline/models"
    mkdir -p "$MODEL_DIR"
    
    echo "Installing TTS Backend: $TTS_BACKEND_CHOICE..."
    
    if [ "$TTS_BACKEND_CHOICE" = "piper" ]; then
        # PIPER TTS
        pip install "piper-tts>=1.4.2" --no-cache-dir
        
        PIPER_VOICE_DIR="$MODEL_DIR/piper-voices"
        mkdir -p "$PIPER_VOICE_DIR"
        
        echo "Downloading default voice (en_US-lessac-medium)..."
        curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" \
               -o "$PIPER_VOICE_DIR/en_US-lessac-medium.onnx"
        curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" \
               -o "$PIPER_VOICE_DIR/en_US-lessac-medium.onnx.json"
        
        if [ -s "$PIPER_VOICE_DIR/en_US-lessac-medium.onnx" ]; then
            echo "✅ Piper TTS installed with voice model"
            TTS_CONFIG="piper"
            TTS_VOICE_PATH="$PIPER_VOICE_DIR/en_US-lessac-medium.onnx"
            TTS_CONFIG_PATH="$PIPER_VOICE_DIR/en_US-lessac-medium.onnx.json"
        else
            echo "❌ Voice download failed"
            exit 1
        fi
        
    elif [ "$TTS_BACKEND_CHOICE" = "coqui_xtts" ]; then
        # COQUI XTTS v2
        echo "⚠️  CPML License: Non-commercial use only"
        export COQUI_TOS_AGREED=1
        
        pip install "TTS>=0.27.0" --no-cache-dir
        
        COQUI_MODEL_DIR="$MODEL_DIR/coqui-xtts"
        mkdir -p "$COQUI_MODEL_DIR"
        
        echo "Downloading XTTS v2 model (~2GB)..."
        python <<EOF
import os
os.environ['COQUI_TOS_AGREED'] = '1'
from TTS.api import TTS
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)
import shutil
for root, dirs, files in os.walk(tts.model_path):
    for f in files:
        shutil.copy2(os.path.join(root, f), "$COQUI_MODEL_DIR")
print("Done")
EOF
        
        if [ "$(ls -A "$COQUI_MODEL_DIR" 2>/dev/null)" ]; then
            echo "✅ Coqui XTTS v2 installed"
            TTS_CONFIG="coqui_xtts"
            TTS_VOICE_PATH="$COQUI_MODEL_DIR"
            TTS_CONFIG_PATH="$COQUI_MODEL_DIR"
        else
            echo "❌ Model download failed"
            exit 1
        fi
    fi
else
    echo "Skipping TTS installation"
    TTS_CONFIG="none"
fi

# ============================================================================
# 🧩 MODEL DOWNLOADS
# ============================================================================

echo ""
echo "============================================================"
echo "Model Downloads (WhisperX, Pyannote, DFKI-SLT PII)"
echo "============================================================"

if [ "$NO_MODELS" = true ]; then
    echo "Skipping all model downloads (--no-models)"
    SKIP_WHISPER=true
    SKIP_PYANNOTE=true
else
    # Helper function for model downloads
    download_model_hf() {
        local repo_id=$1
        local target_dir=$2
        local repo_type=${3:-model}
        
        echo "Downloading $repo_id..."
        mkdir -p "$target_dir"
        
        python -c "
from huggingface_hub import snapshot_download
import sys
try:
    snapshot_download(
        '$repo_id',
        local_dir='$target_dir',
        local_dir_use_symlinks=False,
        repo_type='$repo_type'
    )
    print('✅ Download complete')
except Exception as e:
    print(f'❌ Download failed: {e}', file=sys.stderr)
    sys.exit(1)
" || {
        echo "❌ Failed to download $repo_id"
        return 1
    }
    }
    
    # WhisperX Models
    declare -A MODEL_MAP
    MODEL_MAP["tiny"]="Systran/faster-whisper-tiny"
    MODEL_MAP["base"]="Systran/faster-whisper-base"
    MODEL_MAP["small"]="Systran/faster-whisper-small"
    MODEL_MAP["medium"]="Systran/faster-whisper-medium"
    MODEL_MAP["large"]="Systran/faster-whisper-large-v3"
    
    if [ "$SKIP_WHISPER" = true ]; then
        echo "Skipping WhisperX models (--skip-whisper)"
    elif [ -n "$WHISPER_SELECTION" ]; then
        WHISPER_MODELS_INPUT="$WHISPER_SELECTION"
    elif [ "$ANSWER_YES" = true ]; then
        WHISPER_MODELS_INPUT="large"
    else
        read -p "Enter WhisperX models to download (tiny, base, small, medium, large) or press Enter to skip: " WHISPER_MODELS_INPUT
    fi
    
    if [ -n "$WHISPER_MODELS_INPUT" ]; then
        for model_name in $WHISPER_MODELS_INPUT; do
            model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]')
            if [ -z "${MODEL_MAP[$model_name]}" ]; then
                echo "⚠️  Invalid: '$model_name'. Skipping."
                continue
            fi
            
            hf_repo="${MODEL_MAP[$model_name]}"
            target_dir="$MODEL_DIR/$(echo "$hf_repo" | sed 's/\//\--/g')"
            
            if [ -d "$target_dir" ] && [ "$(ls -A "$target_dir")" ]; then
                echo "✅ Model '$model_name' already exists."
            else
                download_model_hf "$hf_repo" "$target_dir"
            fi
        done
    elif [ "$ANSWER_YES" = true ]; then
        echo "Downloading 'large-v3' by default (auto-answered --yes)..."
        download_model_hf "Systran/faster-whisper-large-v3" "$MODEL_DIR/models--Systran--faster-whisper-large-v3"
    fi
    
    # Pyannote Diarization Model
    TARGET="$MODEL_DIR/models--pyannote--speaker-diarization-community-1"

    if [ "$SKIP_PYANNOTE" = true ]; then
        echo "Skipping Pyannote (--skip-pyannote)"
    elif [ -d "$TARGET" ] && [ "$(ls -A "$TARGET" 2>/dev/null)" ]; then
        echo "✅ Pyannote model already exists."
    else
        echo "Downloading Pyannote to $TARGET..."
        
        # Authenticate if token provided
        if [ "$AUTO_LOGIN" = true ] && [ -n "$HUGGINGFACE_TOKEN" ]; then
            echo "Authenticating with HUGGINGFACE_TOKEN..."
            hf auth login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
        fi
        
        # Check auth status using hf whoami
        if hf whoami &>/dev/null; then
            echo "✅ Hugging Face authenticated"
            
            python -c "
    from huggingface_hub import snapshot_download
    import sys
    snapshot_download(
        repo_id='pyannote/speaker-diarization-community-1',
        local_dir='$TARGET',
        local_dir_use_symlinks=False
    )
    print('✅ Download complete')
    "
        else
            echo "⚠️  Not logged in to Hugging Face."
            echo "   Run: hf auth login"
            echo "   Or: export HUGGINGFACE_TOKEN=your_token && ./ATA_SelfInstall.sh --auto-login"
            echo "   Verify: hf whoami"
        fi
    fi
    
    # mmBERT Base & PII Models (11 Languages for PII Detection)
    echo ""
    echo "=========================================================="
    echo "Downloading mBert Base & DFKI-SLT PII NER Model"
    echo "(Supports 11 languages for PII detection)"
    echo "=========================================================="
    echo ""
    echo "Languages: AR, DE, EN, FI, FR, HI, IT, PL, PT, SP/TR"
    echo ""
    echo "  AR - Arabic     FI - Finnish    PL - Polish"
    echo "  DE - German     FR - French     PT - Portuguese"
    echo "  EN - English    HI - Hindi      SP - Spanish"
    echo "  IT - Italian    PL - Polish     TR - Turkish"
    echo ""
    
    BASE_TARGET="$MODEL_DIR/jhu-clsp/mmBERT-base"
    if [ -d "$BASE_TARGET" ] && [ "$(ls -A "$BASE_TARGET")" ]; then
        echo "✅ mmBERT-base already present."
    else
        download_model_hf "jhu-clsp/mmBERT-base" "$BASE_TARGET" "model"
    fi
    
    PII_TARGET="$MODEL_DIR/multilingual_DialogPII_NER"
    echo ""
    echo "📦 Downloading DFKI-SLT Multilingual DialogPII NER Model"
    
    if [ -d "$PII_TARGET" ] && [ "$(ls -A "$PII_TARGET")" ]; then
        echo "✅ DFKI-SLT PII model already exists."
    else
        download_model_hf "DFKI-SLT/multilingual_DialogPII_NER" "$PII_TARGET" "model"
        
        if [ "$(ls -A "$PII_TARGET" 2>/dev/null)" ]; then
            echo "✅ DFKI-SLT PII model downloaded successfully."
        else
            echo "❌ Download failed. Manual download required:"
            echo "   https://huggingface.co/DFKI-SLT/multilingual_DialogPII_NER"
        fi
    fi
fi

# ============================================================================
# 📋 ENV FILE CONFIGURATION
# ============================================================================

if [ -f ".env" ]; then
    echo "⚠️  Backing up existing .env..."
    cp .env ".env.backup.$(date +%Y%m%d%H%M%S)"
fi

cat > .env <<EOF
# ATA Speech Anonymizer Configuration
CHAT_AI_API_KEY=your_api_key_here
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1

# TTS CONFIGURATION
TTS_BACKEND=$TTS_CONFIG
EOF

# Add backend-specific paths
case "$TTS_CONFIG" in
    piper)
        cat >> .env <<EOF
Piper_Voice_Path=$TTS_VOICE_PATH
Piper_Config_Path=$TTS_CONFIG_PATH
EOF
        ;;
    coqui_xtts)
        cat >> .env <<EOF
XTTS_Model_Path=$TTS_CONFIG_PATH
XTTS_Reference_Audio_Path=./reference_audio.wav
EOF
        ;;
esac

# Common config
cat >> .env <<EOF

# COMPLIANCE & LOGGING
COMPLIANCE_MODE=standard
COMPLIANCE_ENCRYPTION=false
COMPLIANCE_AUDIT_LOG=false
LOG_LEVEL=INFO
LOG_FILE=./logs/ata.log

# PII MODEL (DFKI-SLT - 11 languages)
PII_Model_Path=pipeline/model/multilingual_DialogPII_NER
PII_Languages=AR,DE,EN,FI,FR,H,I,IT,PL,PT,SP,TR

# WHISPERX
Whisper_Model_Path=pipeline/model/models--Systran--faster-whisper-large-v3
Whisper_Device=cuda
Whisper_Compute_Type=float16

# PYANNOTE
Pyannote_Model_Path=pipeline/model/models--pyannote--speaker-diarization-community-1
EOF

echo "✅ .env configured (TTS_BACKEND=$TTS_CONFIG)"

# ============================================================================
# 📋 FINAL INSTRUCTIONS
# ============================================================================

echo ""
echo "=========================================================================="
echo "✅ SETUP COMPLETE"
echo "=========================================================================="
echo ""
echo "Activate environment:"
echo "  conda activate whisperx"
echo ""
echo "TTS Backend: $TTS_CONFIG"
echo "  Piper:    Offline, 30+ langs, GPL v3"
echo "  Coqui:    Voice cloning, 17 langs, CPML (non-commercial)"
echo ""
echo "Model Location: pipeline/models/"
echo ""
echo "Next steps:"
echo "  1. Edit .env with API key"
echo "  2. Place videos in pipeline/videos/"
echo "  3. Run: python pipeline/process.py"
echo "=========================================================================="