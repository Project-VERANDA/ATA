#!/usr/bin/env bash
# ============================================================================
# Environment Reporting Script - Capture Current Package Versions
# ============================================================================
# Run: bash report_env.sh > env_report_$(date +%Y%m%d_%H%M%S).txt

OUTPUT_FILE="env_report_$(date +%Y%m%d_%H%M%S).txt"

echo "==========================================" | tee "$OUTPUT_FILE"
echo "ENVIRONMENT REPORT GENERATOR"
echo "Generated: $(date)"
echo "Hostname: $(hostname)"
echo "==========================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 1. SYSTEM INFORMATION
# ============================================================================
echo "=== SYSTEM INFORMATION ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- OS Release ---" | tee -a "$OUTPUT_FILE"
cat /etc/os-release 2>/dev/null || echo "Not available" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- Kernel Version ---" | tee -a "$OUTPUT_FILE"
uname -a | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- GPU Information ---" | tee -a "$OUTPUT_FILE"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | tee -a "$OUTPUT_FILE"
else
    echo "No NVIDIA GPU detected" | tee -a "$OUTPUT_FILE"
fi
echo "" | tee -a "$OUTPUT_FILE"

echo "--- CPU Information ---" | tee -a "$OUTPUT_FILE"
nproc | tee -a "$OUTPUT_FILE"
lscpu | grep -E "Model name|CPU\(s\)|Thread" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- RAM ---" | tee -a "$OUTPUT_FILE"
free -h | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- Disk Space ---" | tee -a "$OUTPUT_FILE"
df -h . | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 2. PYTHON ENVIRONMENT
# ============================================================================
echo "=== PYTHON ENVIRONMENT ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- Active Environment ---" | tee -a "$OUTPUT_FILE"
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda env: $CONDA_DEFAULT_ENV" | tee -a "$OUTPUT_FILE"
    echo "Conda prefix: $CONDA_PREFIX" | tee -a "$OUTPUT_FILE"
else
    echo "Not in conda environment" | tee -a "$OUTPUT_FILE"
fi
echo "" | tee -a "$OUTPUT_FILE"

echo "--- Python Version ---" | tee -a "$OUTPUT_FILE"
python --version | tee -a "$OUTPUT_FILE"
which python | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- pip Version ---" | tee -a "$OUTPUT_FILE"
pip --version | tee -a "$OUTPUT_FILE"
which pip | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 3. PIP PACKAGES (Full list)
# ============================================================================
echo "=== PIP INSTALLED PACKAGES ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

pip freeze --all 2>/dev/null | tee -a "$OUTPUT_FILE"

# Also save as requirements.txt format
pip freeze --all 2>/dev/null > /tmp/pip_packages.txt
echo "Saved to: /tmp/pip_packages.txt" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 4. KEY PACKAGE VERSIONS (Formatted Table)
# ============================================================================
echo "=== KEY PACKAGE VERSIONS ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

python << 'PYTHON_EOF'
import subprocess
import sys
import json

packages_to_check = [
    # ML Core
    'numpy', 'scipy', 'pandas',
    'torch', 'torchaudio', 'torchvision', 'torchcodec',
    
    # Transformers
    'transformers', 'tokenizers', 'accelerate', 'huggingface-hub',
    
    # NLP
    'spacy', 'thinc', 'blis', 'catalogue',
    
    # Audio/Speech
    'pyannote.audio', 'whisperx', 'ffmpeg-python',
    
    # Web & Utils
    'flask', 'requests', 'python-dotenv', 'openai',
    
    # TTS
    'piper-tts',
    
    # Other
    'pytorch-crf', 'werkzeug', 'cryptography', 'pydub',
]

print(f"{'Package':<25} {'Version':<15} {'Status':<10}")
print("-" * 50)

for pkg in packages_to_check:
    try:
        # Try importing the package
        module = __import__(pkg.replace('-', '_'))
        
        # Get version from various attributes
        version = None
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                version = str(getattr(module, attr))
                break
        
        if version is None:
            # Fallback: check pip show
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', pkg],
                capture_output=True, text=True
            )
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    break
        
        status = "OK" if version else "UNKNOWN"
        print(f"{pkg:<25} {version:<15} {status:<10}")
        
    except ImportError:
        print(f"{pkg:<25} {'NOT INSTALLED':<15} {'MISSING':<10}")
    except Exception as e:
        print(f"{pkg:<25} {'ERROR':<15} {str(e)[:10]:<10}")

PYTHON_EOF

echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 5. CONDA PACKAGES (if applicable)
# ============================================================================
echo "=== CONDA PACKAGES ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

if command -v conda &> /dev/null; then
    conda env export --name "$CONDA_DEFAULT_ENV" --no-builds > /tmp/conda_env.yml 2>/dev/null
    echo "Full conda export saved to: /tmp/conda_env.yml" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    echo "--- Conda Env Summary ---" | tee -a "$OUTPUT_FILE"
    conda list --explicit | head -50 | tee -a "$OUTPUT_FILE"
    echo "..." | tee -a "$OUTPUT_FILE"
else
    echo "Conda not installed" | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 6. MODEL FILES CHECK
# ============================================================================
echo "=== MODEL FILES CHECK ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# Find pipeline/model directory
MODEL_DIR="pipeline/model"
if [ -d "$MODEL_DIR" ]; then
    echo "Directory: $MODEL_DIR" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    find "$MODEL_DIR" -maxdepth 2 -type d | sort | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    for model in $(find "$MODEL_DIR" -maxdepth 1 -mindepth 1 -type d); do
        echo "--- $(basename $model) ---" | tee -a "$OUTPUT_FILE"
        du -sh "$model" | tee -a "$OUTPUT_FILE"
        ls "$model" | head -10 | tee -a "$OUTPUT_FILE"
        if [ $(ls "$model" | wc -l) -gt 10 ]; then
            echo "... and more files" | tee -a "$OUTPUT_FILE"
        fi
        echo "" | tee -a "$OUTPUT_FILE"
    done
else
    echo "Model directory not found at: $MODEL_DIR" | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 7. TTS VOICES CHECK
# ============================================================================
echo "=== TTS VOICES CHECK ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

TTS_VOICE_DIR="pipeline/tts/voices"
if [ -d "$TTS_VOICE_DIR" ]; then
    echo "Directory: $TTS_VOICE_DIR" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    ls -lh "$TTS_VOICE_DIR"/*.onnx 2>/dev/null | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    echo "Voice count: $(ls "$TTS_VOICE_DIR"/*.onnx 2>/dev/null | wc -l)" | tee -a "$OUTPUT_FILE"
else
    echo "TTS voice directory not found at: $TTS_VOICE_DIR" | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 8. SYSTEM PACKAGES
# ============================================================================
echo "=== SYSTEM PACKAGES ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- FFmpeg ---" | tee -a "$OUTPUT_FILE"
if command -v ffmpeg &> /dev/null; then
    ffmpeg -version | head -3 | tee -a "$OUTPUT_FILE"
else
    echo "FFmpeg not installed" | tee -a "$OUTPUT_FILE"
fi
echo "" | tee -a "$OUTPUT_FILE"

echo "--- Git ---" | tee -a "$OUTPUT_FILE"
if command -v git &> /dev/null; then
    git --version | tee -a "$OUTPUT_FILE"
else
    echo "Git not installed" | tee -a "$OUTPUT_FILE"
fi
echo "" | tee -a "$OUTPUT_FILE"

echo "--- GCC ---" | tee -a "$OUTPUT_FILE"
if command -v gcc &> /dev/null; then
    gcc --version | head -1 | tee -a "$OUTPUT_FILE"
else
    echo "GCC not installed" | tee -a "$OUTPUT_FILE"
fi
echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 9. CUDA/CuDNN (if GPU available)
# ============================================================================
echo "=== CUDA INFORMATION ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

python << 'PYTHON_CUDA_EOF'
try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory (total): {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available - CPU-only mode")
except Exception as e:
    print(f"Error checking CUDA: {e}")
PYTHON_CUDA_EOF

echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 10. .ENV FILE (Redacted for security)
# ============================================================================
echo "=== .ENV CONFIGURATION (REDACTED) ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

if [ -f ".env" ]; then
    echo "WARNING: API keys redacted for security" | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
    
    while IFS= read -r line; do
        # Mask API keys
        if [[ "$line" == *"API_KEY"* ]] || [[ "$line" == *"TOKEN"* ]]; then
            key=$(echo "$line" | cut -d'=' -f1)
            echo "${key}=***REDACTED***" | tee -a "$OUTPUT_FILE"
        elif [[ "$line" != \#* ]] && [[ -n "$line" ]]; then
            echo "$line" | tee -a "$OUTPUT_FILE"
        fi
    done < .env
else
    echo ".env file not found" | tee -a "$OUTPUT_FILE"
fi

echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 11. IMPORTANT DIRECTORIES
# ============================================================================
echo "=== DIRECTORY STRUCTURE ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- Working Directory ---" | tee -a "$OUTPUT_FILE"
pwd | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

echo "--- Top-level Files/Directories ---" | tee -a "$OUTPUT_FILE"
ls -la | head -30 | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# 12. IMPORT TEST
# ============================================================================
echo "=== CRITICAL IMPORT TEST ===" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

python << 'IMPORT_TEST_EOF'
import sys
import traceback

critical_imports = [
    'numpy', 'torch', 'transformers', 'spacy', 
    'pyannote.audio', 'whisperx', 'flask', 'openai'
]

print(f"{'Module':<20} {'Result':<10} {'Details':<40}")
print("-" * 70)

for module in critical_imports:
    try:
        __import__(module)
        print(f"{module:<20} {'SUCCESS':<10} {'OK':<40}")
    except Exception as e:
        print(f"{module:<20} {'FAILED':<10} {str(e)[:40]:<40}")
        traceback.print_exc()

IMPORT_TEST_EOF

echo "" | tee -a "$OUTPUT_FILE"

# ============================================================================
# SUMMARY
# ============================================================================
echo "==========================================" | tee -a "$OUTPUT_FILE"
echo "REPORT COMPLETE" | tee -a "$OUTPUT_FILE"
echo "Output saved to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "Temporary files: /tmp/pip_packages.txt, /tmp/conda_env.yml" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"
echo "Next steps:" | tee -a "$OUTPUT_FILE"
echo "1. Review $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "2. Copy contents and share with installer updater" | tee -a "$OUTPUT_FILE"
echo "3. Delete /tmp/pip_packages.txt and /tmp/conda_env.yml if sensitive" | tee -a "$OUTPUT_FILE"
echo "==========================================" | tee -a "$OUTPUT_FILE"

echo ""
echo "Report saved to: $OUTPUT_FILE"
echo "View it with: cat $OUTPUT_FILE | less"