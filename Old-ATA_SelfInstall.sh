#!/usr/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Starting WhisperX Environment Setup ==="

# 1. Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    
    # Download Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    
    # Install Miniconda silently (-b for batch, -p for path)
    bash miniconda.sh -b -p $HOME/miniconda3
    
    # Clean up installer
    rm miniconda.sh
    
    # Initialize Conda for the current shell session
    # This is CRITICAL for scripts to recognize 'conda' commands
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
else
    echo "Conda already installed."
    # Ensure conda is initialized in the current session just in case
    eval "$(conda shell.bash hook)"
fi

# 2. Create the environment
echo "Creating conda environment 'whisperx' with Python 3.12..."
conda create -n whisperx python=3.12 -y

# 3. Activate the environment
# Since we used 'eval' above, 'conda activate' should work in this script context
echo "Activating environment..."
conda activate whisperx

# 4. Install ffmpeg via conda (recommended for stability)
echo "Installing ffmpeg via conda-forge..."
conda install ffmpeg -c conda-forge -y

# 5. Install Python packages via pip
echo "Installing whisperx..."
pip install whisperx

echo "Installing python-ffmpeg..."
pip install python-ffmpeg

echo "Installing pydub..."
pip install pydub

# 6. Handle torchcodec specific versions
echo "Attempting to uninstall existing torchcodec (ignoring if not present)..."
# The '|| true' ensures the script continues even if the package isn't installed
pip uninstall torchcodec -y || true

echo "Installing torchcodec==0.7.0..."
pip install torchcodec==0.7.0

echo "=== Setup Complete! ==="
echo "To use the environment later, run:"
echo "  source $HOME/miniconda3/etc/profile.d/conda.sh"
echo "  conda activate whisperx"
