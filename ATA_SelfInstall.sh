#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# ============================================================================
# 🔧 COMMAND-LINE ARGUMENT PARSING (AUTOMATION)
# ============================================================================
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

Environment Variables:
  HUGGINGFACE_TOKEN       Used with --auto-login for Hugging Face auth
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

echo "=== ATA Speech Anonymizer Installer (v2.1 - NumPy 2.x Compatible) ==="
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
TARGET_PYTHON="3.12"

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
# 🔧 PACKAGE INSTALLATION (NUMPY 2.x COMPATIBLE STACK)
# ============================================================================

echo "Upgrading pip..."
pip install --upgrade pip -q

echo "Installing base ML stack..."

# CORE: NumPy 2.x (required by whisperx/pyannote/thinc 8.3+)
echo "  → numpy>=2.0.0 (NumPy 2.x compatible stack)..."
pip install "numpy>=2.0.0,<3.0.0" --no-cache-dir

echo "  → scipy..."
pip install scipy --no-cache-dir

echo "  → pandas..."
pip install "pandas>=2.2.0" --no-cache-dir

echo "  → torch..."
pip install torch --no-cache-dir || pip install torch torchvision torchaudio --no-cache-dir

echo "  → torchaudio..."
pip install torchaudio --no-cache-dir

echo "  → torchvision..."
pip install torchvision --no-cache-dir

echo "  → transformers..."
pip install "transformers>=4.48.0" --no-cache-dir

echo "  → accelerate..."
pip install "accelerate>=0.20.0" --no-cache-dir

echo "  → sentencepiece..."
pip install sentencepiece --no-cache-dir

echo "  → huggingface-hub..."
pip install "huggingface-hub>=1.5.0" --no-cache-dir

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

# KEY FIX: thinc 8.3+ and spacy 3.8+ support NumPy 2.x
echo "  → thinc>=8.3.0 (NumPy 2.x compatible)..."
pip install "thinc>=8.3.0" --no-cache-dir

echo "  → spacy>=3.8.0 (NumPy 2.x compatible)..."
pip install "spacy>=3.8.0" --no-cache-dir

echo "  → pyannote.audio..."
pip install "pyannote.audio>=3.0.0" --no-cache-dir || echo "⚠️  Warning installing pyannote.audio"

echo "✅ Base ML stack installation complete."

# ============================================================================
# 🔧 CLICK & TYPER VERSION ENFORCEMENT
# ============================================================================
echo "Ensuring click and typer are configured..."
pip install "click>=8.1.7" --no-cache-dir
pip install "typer>=0.9.0" --no-cache-dir

CLICK_VER=$(pip show click | grep Version | awk '{print $2}')
TYPER_VER=$(pip show typer | grep Version | awk '{print $2}')
echo "  click: $CLICK_VER | typer: $TYPER_VER"

# ============================================================================
# 🔧 WHISPERX INSTALLATION
# ============================================================================
echo "Installing whisperx from source..."
pip uninstall whisperx -y 2>/dev/null || true
pip install git+https://github.com/m-bain/whisperx.git --no-cache-dir

# ============================================================================
# 🔧 SPACY MODEL INSTALLATION
# ============================================================================
echo ""
echo "Checking spaCy models..."

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
# 🔧 TORCHCODEC & OTHER CHECKS
# ============================================================================
echo "Checking torchcodec..."
if python -c "import torchcodec" 2>/dev/null; then
    echo "✅ torchcodec already available."
else
    echo "Installing torchcodec..."
    pip install "torchcodec>=0.7.0" --no-cache-dir || echo "⚠️  torchcodec not available (optional)"
fi

echo "Checking CRF library..."
if python -c "from torchcrf import CRF" 2>/dev/null; then
    echo "✅ CRF library working."
else
    echo "Installing CRF library..."
    pip install pytorch-crf --no-cache-dir
fi

# ============================================================================
# 🔧 POST-INSTALLATION VERIFICATION
# ============================================================================
echo ""
echo "=== Verifying Critical Dependencies ==="

if python -c "import click; from click import Choice; c = Choice(['a','b'])" 2>/dev/null; then
    echo "✅ click is compatible"
else
    echo "❌ WARNING: click may not be compatible."
fi

if python -c "import typer" 2>/dev/null; then
    echo "✅ typer imported successfully"
else
    echo "❌ WARNING: typer import failed"
fi

if python -c "import spacy; import thinc; print('✅ OK')" 2>/dev/null; then
    echo "✅ spacy and thinc work with NumPy 2.x"
else
    echo "❌ WARNING: spacy/thinc import failed"
fi

if python -c "import whisperx; print('✅ whisperx OK')" 2>/dev/null; then
    echo "✅ whisperx imported successfully"
else
    echo "⚠️  whisperx may have issues"
fi

if python -c "from pyannote.audio import Pipeline; print('✅ pyannote OK')" 2>/dev/null; then
    echo "✅ pyannote imported successfully"
else
    echo "⚠️  pyannote may have issues (may need HuggingFace auth)"
fi

# ============================================================================
# 🖥️ WEB INTERFACE OPTION (FLASK ONLY - NO COQUI)
# ============================================================================
echo ""
echo "-------------------------------------------------"
if [ "$SKIP_WEB" = true ]; then
    echo "Skipping Web Interface (--skip-web)"
    INSTALL_WEB="n"
elif [ "$ANSWER_YES" = true ]; then
    echo "Web Interface: YES (auto-answered --yes)"
    INSTALL_WEB="y"
else
    read -p "Do you want to install the Web Interface (Flask)? (y/n): " INSTALL_WEB
    INSTALL_WEB=${INSTALL_WEB:-y}
fi

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    echo "Installing Web Interface dependencies..."
    pip install flask requests cryptography --no-cache-dir
    
    echo ""
    echo "============================================================"
    echo "TTS Backend Installation (Piper or MeloTTS)"
    echo "============================================================"
    
    TTS_BACKEND="${TTS_BACKEND:-piper}"
    
    case "$TTS_BACKEND" in
        piper)
            echo "Installing Piper TTS..."
            PIPER_VERSION="1.4.2"
            ARCH=$(uname -m)
            
            case $ARCH in
                x86_64) PIPER_PLATFORM="linux-x86_64" ;;
                aarch64) PIPER_PLATFORM="linux-aarch64" ;;
                armv7l) PIPER_PLATFORM="linux-armv7l" ;;
                *)
                    echo "❌ ERROR: Unsupported architecture: $ARCH"
                    exit 1
                    ;;
            esac
            
            wget -q "https://github.com/rhasspy/piper/releases/download/v${PIPER_VERSION}/piper_${PIPER_VERSION}_${PIPER_PLATFORM}.tar.xz" -O piper.tar.xz
            tar xf piper.tar.xz
            mkdir -p "$CURRENT_DIR/pipeline/tts/bin"
            mv "piper_${PIPER_VERSION}_${PIPER_PLATFORM}/piper" "$CURRENT_DIR/pipeline/tts/bin/"
            chmod +x "$CURRENT_DIR/pipeline/tts/bin/piper"
            rm -rf "piper_${PIPER_VERSION}_${PIPER_PLATFORM}" piper.tar.xz
            echo "✅ Piper binary installed."
            
            pip install dimits --no-cache-dir || echo "⚠️  dimits optional"
            
            DEFAULT_VOICE="en_US-ryan-high"
            VOICE_MODEL="${TTS_VOICE:-$DEFAULT_VOICE}"
            VOICE_DIR="$MODEL_DIR/piper-voices"
            mkdir -p "$VOICE_DIR"
            
            if [ ! -f "$VOICE_DIR/${VOICE_MODEL}.onnx" ]; then
                echo "Downloading Piper voice: $VOICE_MODEL..."
                curl -sL "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/${VOICE_MODEL}/medium/${VOICE_MODEL}.onnx" \
                    -o "$VOICE_DIR/${VOICE_MODEL}.onnx"
                
                if [ -f "$VOICE_DIR/${VOICE_MODEL}.onnx" ] && [ "$(stat -c%s "$VOICE_DIR/${VOICE_MODEL}.onnx" 2>/dev/null || echo 0)" -gt 100000 ]; then
                    echo "✅ Piper voice downloaded: $VOICE_MODEL"
                else
                    echo "⚠️  Piper voice download incomplete."
                fi
            else
                echo "✅ Piper voice already exists: $VOICE_MODEL"
            fi
            ;;
        
        melotts)
            echo "Installing MeloTTS..."
            pip install git+https://github.com/myshell-ai/MeloTTS.git --no-cache-dir || \
            pip install melotts --no-cache-dir || echo "⚠️  MeloTTS may require manual setup"
            ;;
        
        *)
            echo "❌ Unknown TTS backend: $TTS_BACKEND"
            echo "Valid options: piper, melotts"
            exit 1
            ;;
    esac
    
    echo "✅ Web Interface dependencies installed."
else
    echo "Skipping Web Interface installation."
fi

# ============================================================================
# 🧩 MODEL DOWNLOADS
# ============================================================================
echo ""
echo "-------------------------------------------------"
echo "WhisperX & NLP Model Download"
echo "Target Directory: $MODEL_DIR"

if [ "$NO_MODELS" = true ]; then
    echo "Skipping all model downloads (--no-models)"
    SKIP_WHISPER=true
    SKIP_PYANNOTE=true
else
    declare -A MODEL_MAP
    MODEL_MAP["tiny"]="Systran/faster-whisper-tiny"
    MODEL_MAP["base"]="Systran/faster-whisper-base"
    MODEL_MAP["small"]="Systran/faster-whisper-small"
    MODEL_MAP["medium"]="Systran/faster-whisper-medium"
    MODEL_MAP["large"]="Systran/faster-whisper-large-v3"
    
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
    
    if [ "$SKIP_WHISPER" = true ]; then
        echo "Skipping WhisperX models (--skip-whisper)"
    elif [ -n "$WHISPER_SELECTION" ]; then
        WHISPER_MODELS_INPUT="$WHISPER_SELECTION"
    elif [ "$ANSWER_YES" = true ]; then
        WHISPER_MODELS_INPUT="large"
    else
        read -p "Enter WhisperX models (tiny/base/small/medium/large) or press Enter to skip: " WHISPER_MODELS_INPUT
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
        echo "Downloading 'large-v3' by default..."
        download_model_hf "Systran/faster-whisper-large-v3" "$MODEL_DIR/models--Systran--faster-whisper-large-v3"
    fi
    
    TARGET="$MODEL_DIR/models--pyannote--speaker-diarization-community-1"
    
    if [ "$SKIP_PYANNOTE" = true ]; then
        echo "Skipping Pyannote (--skip-pyannote)"
    elif [ -d "$TARGET" ] && [ "$(ls -A "$TARGET")" ]; then
        echo "✅ Pyannote model already exists."
    else
        echo "Downloading Pyannote to $TARGET..."
        
        if [ "$AUTO_LOGIN" = true ] && [ -n "$HUGGINGFACE_TOKEN" ]; then
            echo "Auto-authenticating with HUGGINGFACE_TOKEN..."
            hf auth login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential 2>/dev/null || true
            AUTHENTICATED=true
        elif hf auth status &> /dev/null 2>&1; then
            AUTHENTICATED=true
        else
            AUTHENTICATED=false
        fi
        
        if [ "$AUTHENTICATED" = true ]; then
            download_model_hf "pyannote/speaker-diarization-community-1" "$TARGET" "model"
        else
            echo "⚠️  Not logged in to Hugging Face. Pyannote requires authentication."
            echo "   Options: hf auth login, export HUGGINGFACE_TOKEN, or --auto-login"
        fi
    fi
    
    echo ""
    echo "=========================================================="
    echo "Downloading mBert Base & DFKI-SLT PII NER Model"
    echo "=========================================================="
    
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
# 📋 ENV FILE CREATION (DO NOT OVERWRITE EXISTING)
# ============================================================================
echo ""
echo "-------------------------------------------------"
echo "Environment File Configuration"

if [ -f ".env" ]; then
    echo "⚠️  Existing .env file found. Preserving it."
    echo "   Created backup: .env.backup.$(date +%Y%m%d%H%M%S)"
    cp .env ".env.backup.$(date +%Y%m%d%H%M%S)"
else
    echo "Creating new .env file..."
    
    cat > .env <<'EOF'
# ATA Speech Anonymizer Configuration
# Generated by ATA_SelfInstall.sh

CHAT_AI_API_KEY=your_api_key_here
CHAT_AI_ENDPOINT=https://your-endpoint.com/v1

# ============================================================================
# TTS BACKEND CONFIGURATION
# ============================================================================
TTS_BACKEND=piper

TTS_MODEL_NAME=en_US-ryan-high

TTS_BIN_PATH=./pipeline/tts/bin/piper
TTS_VOICE_PATH=./pipeline/tts/voices/en_US-ryan-high.onnx

# Compliance Settings
COMPLIANCE_MODE=standard
COMPLIANCE_ENCRYPTION=false
COMPLIANCE_AUDIT_LOG=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/ata.log
EOF
    
    echo "✅ .env file created."
fi

# ============================================================================
# 📋 FINAL INSTRUCTIONS
# ============================================================================
echo ""
echo "=========================================================================="
echo "SETUP COMPLETE!"
echo "=========================================================================="
echo ""
echo "To use the environment:"
echo "  source \$HOME/miniforge3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "Hugging Face Authentication:"
echo "  • Run: hf auth login"
echo "  • Or: export HUGGINGFACE_TOKEN=your_token && ./ATA_SelfInstall.sh --auto-login"
echo ""
echo "Automation Flags:"
echo "  --yes              Auto-accept all prompts"
echo "  --skip-web         Skip web UI dependencies"
echo "  --no-models        Skip all model downloads"
echo "  --whisper-models LIST  Specify Whisper models"
echo "  --force-refresh    Reinstall all packages"
echo ""
echo "Project Structure:"
echo "  $(pwd)/"
echo "    ├── .env"
echo "    ├── ATA_SelfInstall.sh"
echo "    └── pipeline/"
echo "        ├── model/"
echo "        ├── videos/"
echo "        ├── audios/"
echo "        ├── transcripts/"
echo "        ├── anonym/"
echo "        └── LLM-Anon/"
echo ""
echo "Next Steps:"
echo "  1. Edit '.env' with your API Key"
echo "  2. Place videos in 'pipeline/videos'"
echo "  3. Run: python pipeline/process.py"
echo ""
echo "✨ FEATURES:"
echo "  • NumPy 2.x compatible stack (thinc>=8.3, spacy>=3.8)"
echo "  • DFKI-SLT Multilingual DialogPII NER (11 languages)"
echo "  • FLERT-style context windowing"
echo "  • Piper/MeloTTS for voice synthesis"
echo "  • WhisperX for transcription"
echo "  • Pyannote for speaker diarization"
echo "=========================================================================="
exit 0