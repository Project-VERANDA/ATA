#!/usr/bin/env python3

# ============================================================================
# Model Download Utility
# Mirrors Installer.sh model download logic for Docker/runtime usage
# ============================================================================

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("  Install with: pip install huggingface-hub")
    sys.exit(1)


def setup_model_dirs(base_path: Path):
    """Ensure model directory structure exists"""
    model_dir = base_path / "pipeline" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def download_whisper_model(model_name: str, model_dir: Path, hf_token: str | None = None) -> bool:
    """Download a specific WhisperX model"""
    
    model_map = {
        "tiny": "Systran/faster-whisper-tiny",
        "base": "Systran/faster-whisper-base",
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large": "Systran/faster-whisper-large-v3",
        "large-turbo": "Systran/faster-whisper-large-v3-turbo",
    }
    
    if model_name not in model_map:
        print(f"❌ Unknown model: {model_name}")
        print(f"  Valid options: {', '.join(model_map.keys())}")
        return False
    
    hf_repo = model_map[model_name]
    safe_name = hf_repo.replace("/", "--")
    target_dir = model_dir / safe_name
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"✅ {model_name} already exists at {target_dir}")
        return True
    
    print(f"⬇️  Downloading {hf_repo}...")
    try:
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(target_dir),
            token=hf_token,
            local_dir_use_symlinks=False,
        )
        print(f"✅ {model_name} downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False


def download_pyannote_model(model_dir: Path, hf_token: str | None = None) -> bool:
    """Download Pyannote speaker diarization model"""
    
    target_dir = model_dir / "models--pyannote--speaker-diarization-community-1"
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"✅ Pyannote model already exists")
        return True
    
    print(f"⬇️  Downloading pyannote/speaker-diarization-community-1...")
    try:
        snapshot_download(
            repo_id="pyannote/speaker-diarization-community-1",
            local_dir=str(target_dir),
            token=hf_token,
            local_dir_use_symlinks=False,
        )
        print(f"✅ Pyannote model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download Pyannote model: {e}")
        return False


def download_pii_model(model_dir: Path, hf_token: str | None = None) -> bool:
    """Download DFKI-SLT PII NER model"""
    
    target_dir = model_dir / "multilingual_DialogPII_NER"
    
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"✅ PII model already exists")
        return True
    
    print(f"⬇️  Downloading DFKI-SLT/multilingual_DialogPII_NER...")
    try:
        snapshot_download(
            repo_id="DFKI-SLT/multilingual_DialogPII_NER",
            local_dir=str(target_dir),
            token=hf_token,
            local_dir_use_symlinks=False,
        )
        print(f"✅ PII model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download PII model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download ML models for Dialogue Anonymizer")
    
    parser.add_argument(
        "--whisper",
        nargs="+",
        choices=["tiny", "base", "small", "medium", "large", "large-turbo"],
        help="WhisperX models to download (space-separated)",
    )
    parser.add_argument(
        "--pyannote",
        action="store_true",
        help="Download Pyannote speaker diarization model",
    )
    parser.add_argument(
        "--pii",
        action="store_true",
        help="Download PII NER model",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all models",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=Path(os.environ.get("CURRENT_DIR", "/app")),
        help="Base path for model directories (default: /app or $CURRENT_DIR)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HUGGINGFACE_TOKEN"),
        help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)",
    )
    
    args = parser.parse_args()
    
    # Determine which models to download
    models_to_download = []
    
    if args.all:
        models_to_download = [
            ("whisper", ["large"]),
            ("pyannote", True),
            ("pii", True),
        ]
    else:
        if args.whisper:
            models_to_download.append(("whisper", args.whisper))
        if args.pyannote:
            models_to_download.append(("pyannote", True))
        if args.pii:
            models_to_download.append(("pii", True))
    
    if not models_to_download:
        print("No models specified to download.")
        print("Use --whisper, --pyannote, --pii, or --all")
        sys.exit(0)
    
    # Setup directories
    model_dir = setup_model_dirs(args.base_path)
    print(f"📁 Model directory: {model_dir}")
    print("")
    
    # Track success
    success_count = 0
    total_count = 0
    
    # Download Whisper models
    for model_type, items in models_to_download:
        if model_type == "whisper":
            for whisper_model in items:
                total_count += 1
                if download_whisper_model(whisper_model, model_dir, args.hf_token):
                    success_count += 1
    
    # Download Pyannote
    for model_type, _ in models_to_download:
        if model_type == "pyannote":
            total_count += 1
            if download_pyannote_model(model_dir, args.hf_token):
                success_count += 1
    
    # Download PII
    for model_type, _ in models_to_download:
        if model_type == "pii":
            total_count += 1
            if download_pii_model(model_dir, args.hf_token):
                success_count += 1
    
    # Summary
    print("")
    print("=" * 60)
    print(f"Download complete: {success_count}/{total_count} successful")
    
    if success_count < total_count:
        print("Some downloads failed - check errors above")
        sys.exit(1)
    else:
        print("All models ready!")
        sys.exit(0)


if __name__ == "__main__":
    main()