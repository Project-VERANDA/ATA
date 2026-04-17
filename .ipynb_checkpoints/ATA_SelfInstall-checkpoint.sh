#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== ATA Speech Anonymizer Installer ==="
echo "This script sets up the 'whisperx' environment and installs all necessary dependencies."
echo "It excludes the ModelTraining folder (assumes you have your own fine-tuned models)."
echo ""

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download Miniconda
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda silently (-b for batch, -p for path)
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Clean up installer
    rm miniconda.sh
    
    # Initialize Conda for the current shell session
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
else
    echo "Conda already installed."
    # Ensure conda is initialized in the current session
    eval "$(conda shell.bash hook)"
fi

# 2. Create the environment
ENV_NAME="whisperx"
echo "Creating conda environment '$ENV_NAME' with Python 3.12..."
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists. Removing and recreating..."
    conda env remove -n $ENV_NAME -y
fi
conda create -n $ENV_NAME python=3.12 -y

# 3. Activate the environment
echo "Activating environment..."
conda activate $ENV_NAME

# 4. Install core system dependencies
echo "Installing ffmpeg via conda-forge..."
conda install ffmpeg -c conda-forge -y

# 5. Install Python packages
echo "Installing core Python packages..."
pip install --upgrade pip

# WhisperX and dependencies
echo "Installing whisperx..."
pip install whisperx

# Audio processing
echo "Installing pydub and python-ffmpeg..."
pip install pydub python-ffmpeg

# Pyannote (for diarization) - Requires HF Token later
echo "Installing pyannote-audio and related dependencies..."
pip install pyannote.audio pyannote.pipeline pyannote.metrics

# Anonymization Libraries (Local BERT/spaCy)
echo "Installing anonymization libraries..."
pip install transformers accelerate sentencepiece
pip install spacy
python -m spacy download de_core_news_sm

# Note: We do NOT install 'bert-anonymizer' as a package because it's not on PyPI.
# The user will use the custom code in ModelTraining/ensemble_anonymizer.
# However, we ensure the base libraries are there.

# Torchcodec specific versions (if needed for your setup)
echo "Handling torchcodec..."
pip uninstall torchcodec -y || true
pip install torchcodec==0.7.0 || echo "Warning: torchcodec installation failed, skipping."

# 6. Web Interface Option
echo ""
echo "-------------------------------------------------"
read -p "Do you want to install the Web Interface (Flask, gTTS, etc.)? (y/n): " INSTALL_WEB
INSTALL_WEB=${INSTALL_WEB:-y} # Default to 'y' if empty

if [[ "$INSTALL_WEB" =~ ^[Yy]$ ]]; then
    echo "Installing Web Interface dependencies..."
    pip install flask python-dotenv requests cryptography
    echo "Web Interface dependencies installed."
else
    echo "Skipping Web Interface installation. Command-line only mode selected."
fi

# 7. Hugging Face Token for Gated Models
echo ""
echo "-------------------------------------------------"
echo "The Pyannote Diarization model is gated on Hugging Face."
echo "You need a token to download it automatically."
echo "Get your token here: https://huggingface.co/settings/tokens"
echo "(Read access is sufficient)"
echo ""
read -sp "Enter your Hugging Face Token (will not be stored): " HF_TOKEN
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "No token provided. You may need to manually download the Pyannote model later."
    echo "Set the token manually later with: huggingface-cli login --token YOUR_TOKEN"
else
    echo "Logging in to Hugging Face..."
    pip install huggingface_hub
    huggingface-cli login --token "$HF_TOKEN"
    
    # Trigger a dummy download to verify access and cache the model
    echo "Verifying access and caching Pyannote model..."
    python -c "from pyannote.audio import Pipeline; print('Downloading model...'); Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')"
    echo "Pyannote model cached successfully!"
fi

# 8. Final Instructions
echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To use the environment:"
echo "  source $HOME/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_NAME"
echo ""
echo "Project Structure Expectations:"
echo "  - Place your 'pipeline' folder (with process.py) in the project root."
echo "  - Place your 'ModelTraining' folder (with custom anonymizers) in the project root."
echo "  - If Web Interface installed: Run 'python interactive_app/app.py'."
echo "  - If CLI only: Run 'python pipeline/process.py'."
echo ""
echo "IMPORTANT: Ensure your .env file (if using Web Interface) contains:"
echo "  CHAT_AI_API_KEY=your_api_key_here"
echo "  CHAT_AI_ENDPOINT=https://your-endpoint.com/v1"
echo ""