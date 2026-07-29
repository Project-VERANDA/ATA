# ============================================================================
# ENVIRONMENT CONFIGURATION (must precede all ML library imports)
# ============================================================================

import os
import sys

# Pin HF Hub and Transformers to offline mode
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'warning')

# ============================================================================
# STANDARD IMPORTS
# ============================================================================

import subprocess
import gc
import logging
import re
import time
import argparse
import json
import traceback
import numpy as np
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, List
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, AutoTokenizer
from torchcrf import CRF

# Import torch FIRST, configure TF32 BEFORE importing pyannote/whisperx
import torch

# Enable TF32 for CUDA matmul and cuDNN convolution kernels
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ML libraries imported after torch and TF32 are configured
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False

import ffmpeg

load_dotenv()

# ============================================================================
# GLOBAL VARIABLES (must be declared before functions that use them)
# ============================================================================

_loaded_whisper_model = None
_loaded_diarize_model = None
args = None  # For CLI argument parsing
logger = logging.getLogger(__name__)

# ============================================================================
# GPU DETECTION & SESSION LOGGER
# ============================================================================

def check_gpu_resources():
    """Checks for GPU availability and logs the status."""
    logger.info("="*40)
    logger.info("GPU DETECTION CHECK")
    logger.info("="*40)
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()
    
    if cuda_available:
        logger.info(f"✅ CUDA is AVAILABLE!")
        logger.info(f"   Number of GPUs detected: {device_count}")
        for i in range(device_count):
            logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"   Current Device: cuda:{torch.cuda.current_device()}")
    else:
        logger.warning("❌ CUDA is NOT available. Falling back to CPU.")
        logger.warning("   This will cause significantly slower processing speeds.")
        logger.warning("   Check if NVIDIA drivers are installed or if a GPU instance is attached.")
    logger.info("="*40)

class SessionLogger:
    """Manages the generation of a summary log file for every script execution."""
    def __init__(self, base_path):
        self.base_path = base_path
        self.log_dir = base_path / "logs"
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = self.log_dir / f"session_{self.session_id}.txt"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        self.settings = {}
        self.stats = {
            "videos_processed": 0,
            "audios_processed": 0,
            "transcripts_generated": 0,
            "anonymizations_success": 0,
            "anonymizations_failed": 0,
            "llm_rewrites_success": 0,
            "llm_rewrites_failed": 0,
            "errors": []
        }
        
        self._write_header()

    def _write_header(self):
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("AUDIO ANONYMIZATION PIPELINE - SESSION LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Script Path: {Path(__file__).resolve()}\n")
            f.write("-" * 60 + "\n")
            f.write("SETTINGS:\n")
            f.write("-" * 60 + "\n")

    def log_settings(self, settings_dict):
        self.settings = settings_dict
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            for key, value in settings_dict.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    def log_stats_update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value

    def log_error(self, error_msg):
        self.stats["errors"].append(error_msg)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] {error_msg}\n")

    def finish(self):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write("-" * 60 + "\n")
            f.write("EXECUTION SUMMARY:\n")
            f.write("-" * 60 + "\n")
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {duration}\n")
            f.write(f"Duration (Seconds): {duration.total_seconds():.2f}\n")
            f.write("\n")
            f.write("FILES PROCESSED:\n")
            f.write(f"  Videos Extracted: {self.stats['videos_processed']}\n")
            f.write(f"  Audios Transcribed: {self.stats['audios_processed']}\n")
            f.write(f"  Transcripts Generated: {self.stats['transcripts_generated']}\n")
            f.write(f"  Anonymizations Success: {self.stats['anonymizations_success']}\n")
            f.write(f"  Anonymizations Failed: {self.stats['anonymizations_failed']}\n")
            f.write(f"  LLM Rewrites Success: {self.stats['llm_rewrites_success']}\n")
            f.write(f"  LLM Rewrites Failed: {self.stats['llm_rewrites_failed']}\n")
            
            if self.stats["errors"]:
                f.write("\nERRORS ENCOUNTERED:\n")
                for err in self.stats["errors"]:
                    f.write(f"  - {err}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("SESSION COMPLETE\n")
            f.write("=" * 60 + "\n")

        logger.info(f"Session log saved to: {self.log_file_path}")
        return self.log_file_path

# ============================================================================
# CONFIGURATION & SECURITY
# ============================================================================

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"DEBUG: API Key loaded? {'YES' if os.getenv('CHAT_AI_API_KEY') else 'NO'}")

# Path Definitions
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_PATH = SCRIPT_DIR

# Traverse up until we find a folder named 'ATA' or hit the root
while BASE_PATH.name != "ATA" and BASE_PATH != BASE_PATH.parent:
    BASE_PATH = BASE_PATH.parent

if BASE_PATH.name != "ATA":
    logger.critical(f"Could not locate 'ATA' folder. Script expects to be run from within .../ATA/")
    logger.critical(f"Current script location: {SCRIPT_DIR}")
    logger.critical(f"Detected base path: {BASE_PATH}")
    sys.exit(1)

logger.info(f"Base path detected: {BASE_PATH}")

pipeline_dir = BASE_PATH / "pipeline"

# Derived paths
VIDEOS_FOLDER = pipeline_dir / "videos"
AUDIOS_FOLDER = pipeline_dir / "audios"
TRANSCRIPTS_FOLDER = pipeline_dir / "transcripts"
MODEL_FOLDER = pipeline_dir / "model"
ANONYM_FOLDER = pipeline_dir / "anonym"
LLM_ANONYM_FOLDER = pipeline_dir / "LLM-Anonymized"

# Create directories if they don't exist
for folder in [TRANSCRIPTS_FOLDER, ANONYM_FOLDER, MODEL_FOLDER, LLM_ANONYM_FOLDER]:
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {folder}")

# Configuration
SUPPORTED_EXTENSIONS = ('.mp4', '.mp3', '.mkv', '.m4a', '.m4p')
DEVICE = "cuda"
BATCH_SIZE = 32
COMPUTE_TYPE = "float16"
MIN_SPEAKERS = 2
MAX_SPEAKERS = 4

# Local Model Paths - CORRECTED to match official DFKI-SLT model name
WHISPERX_MODEL_PATH = MODEL_FOLDER / "Systran--faster-whisper-large-v3"
# The BERT anonymization model (official DFKI-SLT name)
BLANK_NAME = "multilingual_DialogPII_NER"

# System prompts per language
PARAPHRASE_PROMPTS = {
    'DE': (
        "Du bist ein linguistischer Anonymisierungsassistent. Deine Aufgabe ist es, Dialoge "
        "leicht umzuformulieren, um individuelle Sprachmuster zu entfernen und eine "
        "Sprechererkennung zu verhindern. Behalte die exakte ursprüngliche Bedeutung bei "
        "und verwende standardisiertes Hochdeutsch. Weise jedem Sprecher einen leicht "
        "anderen, aber vollständig neutralen Vokabelstil zu. "
        "Falls Platzhalter in eckigen Klammern vorkommen (z. B. [NAME], [HOUSENUMBER], "
        "[CITY], [DATE]), ersetze sie durch passende, realistisch wirkende, aber "
        "vollständig fiktive Werte, die zum Kontext passen. Verwende niemals echte "
        "Personen- oder Kontaktdaten. "
        "Behalte die ursprünglichen Sprecherbezeichnungen unverändert bei. "
        "Gib NUR den finalen Dialog aus. Keine Einleitungen, keine Erklärungen."
    ),
    'EN': (
        "You are a linguistic anonymization assistant. Your task is to slightly "
        "rephrase and modify dialogue to remove unique individual speech patterns, "
        "preventing speaker identification. Maintain the exact original meaning "
        "and use standard neutral English. Assign a slightly different but "
        "entirely neutral vocabulary style to each speaker. "
        "If placeholders enclosed in square brackets appear (e.g., [NAME], "
        "[HOUSENUMBER], [CITY], [DATE]), replace them with suitable, realistic-looking "
        "but entirely fictitious values that fit the context. Never use real personal "
        "or contact information. "
        "Keep the original speaker labels unchanged. "
        "Output ONLY the final dialogue. No introductions, no explanations."
    ),
}

USER_PROMPTS = {
    'DE': "Bitte anonymisiere das folgende Gespräch:",
    'EN': "Please anonymize the following conversation:",
}

# Single consolidated model fallback check
if not WHISPERX_MODEL_PATH.exists():
    logger.warning("❌ large-v3 model NOT found. Searching for alternatives...")
    
    fallback_candidates = [
        "models--Systran--faster-whisper-tiny",
        "models--Systran--faster-whisper-base",
        "models--Systran--faster-whisper-small",
        "models--Systran--faster-whisper-medium"
    ]
    
    found_fallback = False
    for candidate_name in fallback_candidates:
        candidate_path = MODEL_FOLDER / candidate_name
        if candidate_path.exists():
            WHISPERX_MODEL_PATH = candidate_path
            logger.info(f"✅ SUCCESS: Switching to fallback model: {candidate_name}")
            found_fallback = True
            break
    
    if not found_fallback:
        whisper_folders = [f for f in MODEL_FOLDER.iterdir() if f.is_dir() and f.name.startswith("models--Systran--faster-whisper-")]
        if whisper_folders:
            WHISPERX_MODEL_PATH = whisper_folders[0]
            logger.warning(f"⚠️  Using arbitrary fallback: {WHISPERX_MODEL_PATH.name}")
        else:
            logger.critical(f"CRITICAL: No WhisperX model found in {MODEL_FOLDER}.")
            logger.critical(f"Available folders: {list(MODEL_FOLDER.iterdir())}")
            sys.exit(1)
else:
    logger.info(f"✅ Using default WhisperX model: large-v3")

# Final verification
if not WHISPERX_MODEL_PATH.exists():
    logger.critical(f"CRITICAL: Final model path {WHISPERX_MODEL_PATH} does not exist.")
    sys.exit(1)

logger.info(f"✅ WhisperX Model Path Set: {WHISPERX_MODEL_PATH.name}")
DIARIZATION_MODEL_PATH = MODEL_FOLDER / "models--pyannote--speaker-diarization-community-1"

if not MODEL_FOLDER.exists():
    logger.critical(f"CRITICAL: Model folder not found at {MODEL_FOLDER}.")
    sys.exit(1)

if not DIARIZATION_MODEL_PATH.exists():
    logger.warning(f"WARNING: Diarization model not found at {DIARIZATION_MODEL_PATH}.")
    logger.warning("Speaker diarization will be disabled. Using generic speaker labels.")

logger.info(f"✅ Paths verified successfully.")
logger.info(f"   Model Folder: {MODEL_FOLDER}")
logger.info(f"   WhisperX Model: {WHISPERX_MODEL_PATH.name}")
logger.info(f"   Diarization Model: {DIARIZATION_MODEL_PATH.name if DIARIZATION_MODEL_PATH.exists() else 'MISSING'}")

# Supported Languages
SUPPORTED_LANGUAGES = {
    'AR': 'Arabic', 'DE': 'German', 'EN': 'English', 'FI': 'Finnish', 'FR': 'French',
    'HI': 'Hindi', 'IT': 'Italian', 'PL': 'Polish', 'PT': 'Portuguese',
    'SP': 'Spanish', 'ES': 'Spanish', 'TR': 'Turkish'
}

WHISPER_LANG_MAP = {
    'AR': 'ar', 'DE': 'de', 'EN': 'en', 'FI': 'fi', 'FR': 'fr',
    'HI': 'hi', 'IT': 'it', 'PL': 'pl', 'PT': 'pt', 
    'SP': 'es', 'ES': 'es', 'TR': 'tr'
}

# Anonymization Configuration
ANONYMIZATION_ENABLED = True
ANONYMIZATION_LEVEL = "standard"
ANONYMIZATION_METHOD = "local_mmbert"

AVAILABLE_TAGS = [
    "PERSON", "PERSON_EMAIL", "PERSON_SOCIAL_RELATION", "ORG", 
    "LOC_CITY", "LOC_COUNTRY", "LOC_STREET", "LOC_ZIP", "LOC_HOUSENUMBER", "LOC_OTHER",
    "DATETIME", "DATETIME_AGE", "CODE", "CODE_PHONE", "CODE_URL",
    "PROFESSION", "PRODUCT", "QUANTITY", "MISC"
]

# Remote Chat AI API Configuration
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY', '')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')
DEFAULT_CHAT_AI_MODEL = os.getenv('CHAT_AI_MODEL', 'gpt-oss-120b')
LLM_REWRITE_ENABLED = True

LLM_REWRITE_SYSTEM_PROMPT = (
    "You are an expert anonymizer. Your goal is to protect privacy by removing or generalizing identifiers.\n\n"
    "TWO STRATEGIES:\n"
    "1. **REPLACE** specific values with placeholders (e.g., 'John Smith' → '[NAME_OTHER]', '123 Main St' → '[ADDRESS]').\n"
    "2. **GENERALIZE** descriptions when the combination of details could identify someone.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. PRESERVE SPEAKER TAGS: Lines starting with 'SPEAKER_XX:' are structural metadata. NEVER modify them.\n"
    "2. PRESERVE EXISTING TAGS: Do not touch [NAME_OTHER], [DATE], [ID], [LOCATION_CITY], etc.\n"
    "3. MINIMAL CHANGE: Only generalize when necessary for privacy.\n"
    "4. PRESERVE STRUCTURE: Keep all newlines, line breaks, and grammar exactly as they are.\n"
    "5. LANGUAGE: Do not translate. Keep the original language.\n\n"
    "Return ONLY the anonymized text. No explanations, no reasoning, no commentary."
)

AVAILABLE_LLM_MODELS = {
    'medgemma': 'medgemma', 'medgemma27b': 'medgemma27b', 'gpt-oss-120b': 'gpt-oss-120b',
    'Qwen3.6-27B': 'Qwen3.6-27B', 'Qwen3.5-27B': 'Qwen3.5-27B', 'qwen3-asr-1.7b': 'qwen3-asr-1.7b',
    'cle-Kimi-K2.6': 'cle-Kimi-K2.6', 'cle-Qwen3-Coder-Next-FP8': 'cle-Qwen3-Coder-Next-FP8',
    'cle-Qwen3.5-397B-A17B-FP8': 'cle-Qwen3.5-397B-A17B-FP8'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sanitize_filename(filename):
    """Removes potentially dangerous characters from filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

def validate_path(path, base):
    """Ensures the resolved path is within the base directory."""
    resolved = path.resolve()
    base_resolved = base.resolve()
    try:
        resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False

def convert_numpy(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

def cleanup_gpu_resources(*objects_to_delete):
    """Aggressively clears GPU memory and runs garbage collection."""
    for obj in objects_to_delete:
        try:
            del obj
        except NameError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("GPU and System RAM resources cleaned up.")

def merge_consecutive_speaker_segments(segments, max_gap_seconds=2.0):
    """Merges consecutive segments spoken by the same speaker."""
    if not segments:
        return []

    merged_segments = []
    current_segment = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "")
        start = segment.get("start", 0)
        end = segment.get("end", 0)

        if current_segment is None:
            current_segment = {"speaker": speaker, "text": text, "start": start, "end": end}
        else:
            if current_segment["speaker"] == speaker:
                gap = start - current_segment["end"]
                if gap <= max_gap_seconds:
                    current_segment["text"] += " " + text
                    current_segment["end"] = end
                else:
                    merged_segments.append(current_segment)
                    current_segment = {"speaker": speaker, "text": text, "start": start, "end": end}
            else:
                merged_segments.append(current_segment)
                current_segment = {"speaker": speaker, "text": text, "start": start, "end": end}

    if current_segment:
        merged_segments.append(current_segment)

    return merged_segments

def verify_whisperx_model(model_path):
    """Verifies WhisperX model (faster-whisper format)."""
    path = Path(model_path)
    if not path.exists():
        logger.error(f"WhisperX model directory not found at: {model_path}")
        return False
    config_files = list(path.glob("config.*"))
    if not config_files:
        logger.error(f"WhisperX model missing config file")
        return False
    logger.info(f"WhisperX model verified at: {model_path}")
    return True

def verify_diarization_model(model_path):
    """Verifies Pyannote Diarization model."""
    path = Path(model_path)
    if not path.exists():
        logger.error(f"Diarization model directory not found at: {model_path}")
        return False
    logger.info(f"Diarization model verified at: {model_path}")
    return True

# ============================================================================
# PROCESSING PIPELINE FUNCTIONS
# ============================================================================

def process_videos(file_list=None):
    """Extract audio from video files."""
    if not VIDEOS_FOLDER.exists():
        logger.warning(f"No 'videos' folder found at {VIDEOS_FOLDER}. Skipping audio extraction.")
        return 0

    logger.info(f"Found 'videos' folder at {VIDEOS_FOLDER}. Processing supported files...")

    if not AUDIOS_FOLDER.exists():
        AUDIOS_FOLDER.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    
    if file_list:
        files = [Path(f) for f in file_list]
    else:
        files = [f for f in VIDEOS_FOLDER.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]

    for file in files:
        if not validate_path(file, VIDEOS_FOLDER):
            logger.error(f"Security Alert: Attempted path traversal detected for {file.name}. Skipping.")
            continue

        base_name = sanitize_filename(file.stem)
        audio_filename = f"{base_name}.wav"
        audio_path = AUDIOS_FOLDER / audio_filename

        if audio_path.exists():
            logger.info(f"Audio already exists: {audio_path}")
            continue

        logger.info(f"Processing video: {file.name} -> {audio_filename}")

        try:
            subprocess.run([
                'ffmpeg', '-i', str(file),
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y',
                str(audio_path)
            ], check=True, capture_output=True, text=True)

            processed_count += 1
            logger.info(f"Successfully saved: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error processing {file.name}: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file.name}: {e}")

    logger.info(f"Audio extraction complete. Processed {processed_count} files.")
    return processed_count

def process_audios(enable_diarization=True, lang_code=None, file_list=None):
    """Process audio files with optional diarization control."""
    check_gpu_resources()

    force_language = lang_code if lang_code is not None else getattr(args, 'lang', None)

    if file_list:
        files = [Path(f) for f in file_list]
    elif getattr(args, 'file', None):
        files = [Path(args.file)]
    elif getattr(args, 'files', None):
        files = [Path(f) for f in args.files]
    else:
        files = [f for f in AUDIOS_FOLDER.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]

    # Normalize language code
    whisper_code = None
    if force_language:
        whisper_code = WHISPER_LANG_MAP.get(force_language)
        if not whisper_code:
            logger.error(f"Invalid language code: {force_language}")
            return
        logger.info(f"Language forced to: {SUPPORTED_LANGUAGES[force_language]} ({whisper_code})")
    else:
        logger.info("Language set to Auto-Detect.")

    if not enable_diarization:
        logger.info("⚠️  Speaker diarization DISABLED. Using generic speaker labels.")
        diarize_model = None
    else:
        logger.info("✅ Speaker diarization ENABLED.")

    # Load WhisperX Model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"

        model = whisperx.load_model(str(WHISPERX_MODEL_PATH), device, compute_type=compute_type, local_files_only=True)
        logger.info("WhisperX model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return

    # Load Diarization Pipeline
    if enable_diarization and DIARIZATION_MODEL_PATH.exists():
        try:
            from pyannote.audio import Pipeline
            diarize_pipeline = Pipeline.from_pretrained(str(DIARIZATION_MODEL_PATH))
            diarize_model = diarize_pipeline
            logger.info("Diarization Pipeline loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Diarization Pipeline: {e}")
            diarize_model = None
    else:
        diarize_model = None

    total_files = len(files)
    if total_files == 0:
        logger.info("No .wav files found.")
        return

    logger.info(f"Found {total_files} files to process.")

    for idx, file in enumerate(files, 1):
        if not validate_path(file, AUDIOS_FOLDER):
            logger.error(f"Security Alert: Skipping {file.name}.")
            continue

        logger.info(f"[{idx}/{total_files}] Processing: {file.name}")

        try:
            audio = whisperx.load_audio(str(file))

            transcribe_kwargs = {
                "audio": audio, "batch_size": BATCH_SIZE, "verbose": False, "print_progress": False, "task": "transcribe"
            }
            if whisper_code:
                transcribe_kwargs["language"] = whisper_code

            result = model.transcribe(**transcribe_kwargs)

            # Align
            if result.get("language"):
                try:
                    align_device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=align_device)
                    result = whisperx.align(result["segments"], model_a, metadata, audio, align_device, return_char_alignments=False)
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}")

            # Diarize
            if enable_diarization and diarize_model:
                try:
                    diarize_output = diarize_model(str(file))
                    speaker_diarization = diarize_output.speaker_diarization

                    import pandas as pd
                    segments_list = []
                    for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                        duration = turn.end - turn.start
                        if duration < 0.5: continue
                        segments_list.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})

                    if segments_list:
                        diarize_df = pd.DataFrame(segments_list)
                        result = whisperx.assign_word_speakers(diarize_df, result)
                    else:
                        for i, seg in enumerate(result["segments"]):
                            seg["speaker"] = f"SPEAKER_{i%2:02d}"
                except Exception as e:
                    logger.error(f"Diarization failed: {e}")
                    for i, seg in enumerate(result["segments"]):
                        seg["speaker"] = f"SPEAKER_{i%2:02d}"
            else:
                for i, seg in enumerate(result["segments"]):
                    seg["speaker"] = f"SPEAKER_{i%2:02d}"

            # Merge & Save
            result["segments"] = merge_consecutive_speaker_segments(result["segments"])

            base_name = sanitize_filename(file.stem)
            transcript_file = TRANSCRIPTS_FOLDER / f"{base_name}.txt"

            with open(transcript_file, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        f.write(f"{segment.get('speaker', 'Unknown')}: {text}\n")

            logger.info(f"✅ COMPLETED: {file.name} -> {transcript_file.name}")

        except Exception as e:
            logger.error(f"❌ FAILED: {file.name} - {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if 'audio' in locals(): del audio
            if 'result' in locals(): del result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if 'model' in locals(): del model
    if 'diarize_model' in locals(): del diarize_model
    cleanup_gpu_resources()
    logger.info("Stream processing finished.")
    return total_files

def load_models():
    """Loads models locally on GPU if available."""
    global _loaded_whisper_model, _loaded_diarize_model

    check_gpu_resources()

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if _loaded_whisper_model and _loaded_diarize_model:
        return _loaded_whisper_model, _loaded_diarize_model

    # Load WhisperX Model
    try:
        logger.info(f"Loading WhisperX model directly from: {WHISPERX_MODEL_PATH}")
        from faster_whisper import WhisperModel
        compute_type = "float16" if device_str == "cuda" else "float32"

        _loaded_whisper_model = WhisperModel(
            str(WHISPERX_MODEL_PATH), 
            device=device_str, 
            compute_type=compute_type,
            local_files_only=True
        )
        logger.info("✅ WhisperX model loaded successfully (direct path).")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        logger.critical(f"Folder contents: {list(WHISPERX_MODEL_PATH.iterdir()) if WHISPERX_MODEL_PATH.exists() else 'Folder missing'}")
        return None, None

    # Load Diarization Pipeline
    if DIARIZATION_MODEL_PATH and DIARIZATION_MODEL_PATH.exists():
        try:
            logger.info(f"Loading Diarization Pipeline from: {DIARIZATION_MODEL_PATH}")
            from pyannote.audio import Pipeline
            _loaded_diarize_model = Pipeline.from_pretrained(str(DIARIZATION_MODEL_PATH))
            if device_str == "cuda":
                _loaded_diarize_model.to(device)
                logger.info("✅ Diarization Pipeline moved to GPU.")
            else:
                logger.warning("⚠️ Diarization Pipeline loaded on CPU.")
        except Exception as e:
            logger.error(f"Failed to load Diarization Pipeline: {e}")
            _loaded_diarize_model = None
    else:
        logger.warning("Diarization model path not found or invalid. Skipping.")
        _loaded_diarize_model = None

    logger.info("Models loaded successfully.")
    return _loaded_whisper_model, _loaded_diarize_model

def transcribe_audio_locally(audio_path, language='auto'):
    """Transcribes a single audio file using local models."""
    global _loaded_whisper_model, _loaded_diarize_model

    logger.info(f"--- Starting transcription for: {os.path.basename(audio_path)} ---")

    if not _loaded_whisper_model or not _loaded_diarize_model:
        logger.info("Loading models (if not already loaded)...")
        _loaded_whisper_model, _loaded_diarize_model = load_models()

    if not _loaded_whisper_model:
        error_msg = "Error: WhisperX model not loaded."
        logger.error(error_msg)
        return error_msg, []

    try:
        audio = whisperx.load_audio(audio_path)
        logger.info(f"Audio loaded. Duration: {len(audio)/16000:.2f} seconds.")

        # Determine language code (app.py already converts to Whisper-compatible code)
        lang_code = language
        if language and language != 'auto' and language in WHISPER_LANG_MAP.values():
            logger.info(f"Transcribing with forced language: {lang_code}")
        else:
            logger.info("Transcribing with auto-detect")

        segments, info = _loaded_whisper_model.transcribe(
            audio, beam_size=BATCH_SIZE, language=lang_code, vad_filter=True
        )

        raw_segments = list(segments)

        segments_list = []
        for seg in raw_segments:
            seg_dict = {
                "start": seg.start, "end": seg.end, "text": seg.text,
                "words": getattr(seg, 'words', None)
            }
            segments_list.append(seg_dict)

        detected_language = info.language if info else lang_code
        logger.info(f"Transcription completed. Detected language: {detected_language}")

        result = {"segments": segments_list, "language": detected_language}

        # Align
        if result.get("language"):
            try:
                logger.info("Aligning transcription segments...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=device
                )
                result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
                logger.info("Alignment completed.")
            except Exception as e:
                logger.warning(f"Alignment failed: {e}. Proceeding without alignment.")

        # Diarize
        if _loaded_diarize_model:
            logger.info("Running speaker diarization...")
            try:
                _loaded_diarize_model.min_duration_on = 4.0
                _loaded_diarize_model.min_duration_off = 2

                diarize_output = _loaded_diarize_model(audio_path, min_speakers=2, max_speakers=10)
                speaker_diarization = diarize_output.speaker_diarization

                import pandas as pd
                segments_list = []
                for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                    segments_list.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})

                segments_list = sorted(segments_list, key=lambda x: x.get('start', 0))
                diarize_df = pd.DataFrame(segments_list)
                result = whisperx.assign_word_speakers(diarize_df, result)
                logger.info("Speakers assigned to segments.")
            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                logger.warning("Falling back to generic speaker labels.")
                for i, segment in enumerate(result["segments"]):
                    segment["speaker"] = f"SPEAKER_{i%2:02d}"
        else:
            logger.warning("No diarization model loaded. Using generic speaker labels.")
            for i, segment in enumerate(result["segments"]):
                segment["speaker"] = f"SPEAKER_{i%2:02d}"

        if 'segments' in result:
            result["segments"] = sorted(result["segments"], key=lambda x: x.get('start', 0))

        # Merge segments
        logger.info("Merging consecutive speaker segments...")
        merged_segments = []
        prev_segment = None

        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()

            if len(text) < 4 and text.lower() not in ["i", "a", "ok", "no", "yes", "hi"]:
                logger.debug(f"Discarding short segment: '{text}'")
                continue

            if prev_segment and prev_segment["speaker"] == speaker:
                prev_segment["text"] += " " + text
            else:
                if prev_segment:
                    merged_segments.append(prev_segment)
                prev_segment = {"speaker": speaker, "text": text}

        if prev_segment:
            merged_segments.append(prev_segment)

        logger.info(f"Merged into {len(merged_segments)} final segments.")

        result_lines = []
        for seg in merged_segments:
            if seg['text'].strip():
                line = f"{seg['speaker']}: {seg['text'].strip()}"
                result_lines.append(line)

        result_text = "\n".join(result_lines) if result_lines else "No speech detected."
        logger.info(f"--- Transcription Complete. Total lines: {len(result_lines)} ---")

        result_offset = convert_numpy(result["segments"])
        return result_text, result_offset

    except Exception as e:
        error_msg = f"Transcription failed: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg, []

# ============================================================================
# TTS SERVICE LAYER (Piper + Coqui XTTS compatible)
# ============================================================================

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("Pydub not available — using ffmpeg fallback for beep generation.")

TTS_BACKEND = os.getenv('TTS_BACKEND', 'piper').lower()
TTS_ENABLED = os.getenv('TTS_ENABLED', 'true').lower() == 'true'
TTS_DEFAULT_LANG = os.getenv('TTS_DEFAULT_LANG', 'en')

if TTS_BACKEND not in ['piper', 'coqui_xtts']:
    logger.warning(f"Invalid TTS_BACKEND '{TTS_BACKEND}'. Defaulting to 'piper'")
    TTS_BACKEND = 'piper'

TTS_BIN_PATH = Path(os.getenv('TTS_BIN_PATH', ''))
TTS_VOICE_DIR = Path(os.getenv('TTS_VOICE_DIR', str(pipeline_dir / 'model' / 'piper-voices')))
TTS_VOICE_PATH = Path(os.getenv('TTS_VOICE_PATH', ''))
TTS_SAMPLE_RATE = int(os.getenv('TTS_SAMPLE_RATE', '22050'))

TTS_MODEL_NAME = os.getenv('TTS_MODEL_NAME', 'tts_models/multilingual/multi-dataset/xtts_v2')
XTTS_MODEL_PATH = os.getenv('XTTS_Model_Path', str(pipeline_dir / 'model' / 'coqui-xtts'))
XTTS_REFERENCE_AUDIO = os.getenv('XTTS_Reference_Audio_Path', '')

_tts_engine = None
_tts_engine_lang = None

def generate_beep(duration_ms=400, freq=1000):
    """Generate a beep sound. Uses pydub if available, ffmpeg fallback otherwise."""
    if PYDUB_AVAILABLE:
        try:
            return Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-12)
        except Exception as e:
            logger.error(f"Pydub beep generation failed: {e}")
    
    try:
        import tempfile
        temp_path = tempfile.mktemp(suffix='.wav')
        duration_sec = float(duration_ms) / 1000.0
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i',
            f'sine=frequency={freq}:duration={duration_ms/1000}',
            '-ar', '16000', '-ac', '1', temp_path
        ], capture_output=True, check=True)
        return temp_path
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg beep generation failed: {e.stderr}")
        return None

def _get_tts_engine():
    """Lazy-load TTS backend singleton."""
    global _tts_engine, _tts_engine_lang

    if _tts_engine is not None:
        return _tts_engine

    if not TTS_ENABLED:
        logger.warning("TTS is disabled in configuration (TTS_ENABLED=false)")
        return None

    try:
        tts_module_dir = pipeline_dir / 'tts'
        if str(tts_module_dir) not in sys.path:
            sys.path.insert(0, str(tts_module_dir))
        
        from backend import get_tts_backend, TTSError

        config = {
            'TTS_BACKEND': TTS_BACKEND,
            'TTS_BIN_PATH': TTS_BIN_PATH,
            'TTS_VOICE_DIR': TTS_VOICE_DIR,
            'TTS_VOICE_PATH': TTS_VOICE_PATH,
            'TTS_MODEL_NAME': TTS_MODEL_NAME,
            'XTTS_Model_Path': XTTS_MODEL_PATH,
            'XTTS_Reference_Audio_Path': XTTS_REFERENCE_AUDIO,
            'TTS_SAMPLE_RATE': TTS_SAMPLE_RATE,
            'TTS_DEFAULT_LANG': TTS_DEFAULT_LANG,
        }

        _tts_engine = get_tts_backend(config=config)
        logger.info(f"TTS engine initialized: {_tts_engine.backend_name}")
        return _tts_engine
    except ImportError as e:
        logger.error(f"TTS module import failed: {e}")
        return None
    except TTSError as e:
        logger.error(f"TTS backend initialization failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def get_tts_status():
    """Return TTS configuration status."""
    return {
        'backend': TTS_BACKEND,
        'enabled': TTS_ENABLED,
        'default_language': TTS_DEFAULT_LANG,
        'backend_details': f'{TTS_BACKEND.title()} TTS' if TTS_BACKEND == 'piper' else 'Coqui XTTS v2',
        'supported_languages': ['en', 'de', 'fr', 'es', 'it', 'pl', 'pt', 'fi', 'ar', 'hi', 'tr'] if TTS_BACKEND == 'piper' else ['en', 'de', 'fr', 'es', 'it', 'pl', 'pt', 'zh', 'ja', 'ko', 'ar', 'hi', 'tr', 'fi'],
    }

def get_available_tts_voices(language='en'):
    """Return available voice models for the current backend."""
    if not _get_tts_engine():
        return []
    
    try:
        return _get_tts_engine().get_available_voices(language)
    except Exception:
        return [{'id': 'default', 'name': f'{language.upper()} Default', 'language': [language], 'engine': TTS_BACKEND}]

def generate_speech(text, language='en', output_dir=None, speaker_id=0, return_bytes=False):
    """Main speech generation wrapper - delegates to backend."""
    engine = _get_tts_engine()
    
    if not engine:
        logger.error("TTS engine not available")
        return None
    
    if return_bytes:
        import io
        import tempfile
        import os
        
        fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='piper_tts_', dir='/tmp')
        os.close(fd)
        
        try:
            engine.synthesize(
                text=text,
                output_path=temp_path,
                speaker_id=int(speaker_id),
                language=str(language)
            )
            
            buffer = io.BytesIO()
            with open(temp_path, 'rb') as f:
                buffer.write(f.read())
            buffer.seek(0)
            return buffer
            
        except Exception as e:
            logger.error(f"TTS synthesis to buffer failed: {e}")
            return None
        finally:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
    else:
        if output_dir is None:
            output_dir = BASE_PATH / "pipeline" / "audios" / "tts_output"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"speech_{timestamp}.wav"
        
        try:
            engine.synthesize(
                text=text,
                output_path=str(output_path),
                speaker_id=int(speaker_id),
                language=str(language)
            )
            return str(output_path)
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

def synthesize_segment(text, language='en'):
    """Synthesize a single text segment."""
    return generate_speech(text, language=language)

# ============================================================================
# ANONYMIZATION ENGINE (ModernBertCRF with OFFICIAL FLERT context windowing)
# ============================================================================

_nlp_multilingual = None

def get_nlp_pipeline(lang_code=None):
    """Loads or retrieves the SpaCy pipeline for sentence splitting."""
    global _nlp_multilingual
    
    spacy_lang = 'xx'
    
    if spacy_lang == 'xx':
        if _nlp_multilingual is None:
            try:
                import spacy
                _nlp_multilingual = spacy.blank("xx")
                _nlp_multilingual.add_pipe("sentencizer")
                logger.info("Loaded SpaCy multilingual sentencizer.")
            except Exception as e:
                logger.error(f"Failed to load SpaCy 'xx' model: {e}. Falling back to regex splitting.")
                return None
        return _nlp_multilingual
    else:
        return None

def split_dialogue_into_sentences(text, nlp_pipeline=None):
    """
    Splits dialogue text into a list of sentences.
    PER OFFICIAL DFKI-SLT MODEL CARD SPECIFICATIONS.
    """
    if nlp_pipeline is None:
        nlp_pipeline = get_nlp_pipeline()

    sentences_with_speakers = []

    if nlp_pipeline is None:
        logger.warning("SpaCy not available. Using naive sentence splitting.")
        lines = text.strip().split('\n')
        for line in lines:
            match = re.match(r"^(SPEAKER_\d+)\s*:\s*(.*)", line.strip())
            if match:
                speaker = match.group(1)
                content = match.group(2)
                raw_sents = re.split(r'(?<=[.!?])\s+', content)
                for sent in raw_sents:
                    tokens = sent.split()
                    if tokens:
                        sentences_with_speakers.append((speaker, tokens))
        return sentences_with_speakers

    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.match(r"^(SPEAKER_\d+)\s*:\s*(.*)", line)
        if not match:
            continue

        speaker = match.group(1)
        content = match.group(2)

        if not content:
            continue

        doc = nlp_pipeline(content)

        for sent in doc.sents:
            tokens = [tok.text for tok in sent if not tok.is_space]
            if tokens:
                sentences_with_speakers.append((speaker, tokens))

    return sentences_with_speakers

class ModernBertCRF(torch.nn.Module):
    """Custom ModernBERT + CRF model as per DFKI-SLT specification."""
    
    def __init__(self, base_model_name, num_labels, id2label, label2id):
        super().__init__()
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        
        self.transformer = AutoModel.from_pretrained(base_model_name, local_files_only=True)
        hidden_size = self.transformer.config.hidden_size
        
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(0.1)
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        kwargs.pop("token_type_ids", None)
        
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return {"loss": loss, "logits": emissions}
        else:
            return {"logits": emissions}
    
    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask=mask)

def predict_dialogue_flert(sentences_tokens, model, tokenizer, id_to_tag_map,
                           device="cpu", context_window=2, use_sep_marker=True):
    """
    Predicts NER labels using FLERT-style context windowing.
    PER OFFICIAL DFKI-SLT MULTILINGUAL_DIALOGPII_NER MODEL CARD.
    
    Args:
        sentences_tokens: List of token lists, one per sentence
        model: Loaded ModernBertCRF model
        tokenizer: Associated HuggingFace tokenizer
        id_to_tag_map: Mapping from prediction ID to anonymization tag
        device: 'cuda' or 'cpu'
        context_window: Number of surrounding sentences (from flert_config.json, default: 2)
        use_sep_marker: Whether to insert SEP tokens between sentences
    
    Returns:
        list of lists: [[tag, tag, ...], ...] one label list per sentence
    """
    sep_token = getattr(tokenizer, 'sep_token', '[SEP]')
    all_predictions = []
    
    for i, target_tokens in enumerate(sentences_tokens):
        # Build context window
        left_ctx = sentences_tokens[max(0, i - context_window):i]
        right_ctx = sentences_tokens[i + 1:i + 1 + context_window]
        
        # Flatten tokens with SEP markers
        flat_tokens = []
        
        # Add left context
        for ctx_sent in left_ctx:
            flat_tokens.extend(ctx_sent)
            if use_sep_marker:
                flat_tokens.append(sep_token)
        
        # Record target boundaries (indices in flat_tokens)
        tgt_start_idx = len(flat_tokens)
        flat_tokens.extend(target_tokens)
        tgt_end_idx = len(flat_tokens)
        
        # Add right context
        if right_ctx and use_sep_marker:
            flat_tokens.append(sep_token)
        for ctx_sent in right_ctx:
            flat_tokens.extend(ctx_sent)
        
        # Tokenize
        enc = tokenizer(
            flat_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=False
        ).to(device)
        
        word_ids = enc.word_ids(batch_index=0)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**enc)
            emissions = outputs["logits"]
            mask = enc["attention_mask"].bool()
            preds = model.decode(emissions, mask)[0]
        
        # Extract predictions ONLY for target sentence words
        word_predictions = defaultdict(list)
        for idx, wid in enumerate(word_ids):
            if wid is None or wid < tgt_start_idx or wid >= tgt_end_idx:
                continue
            
            pred_id = preds[idx]
            tag = id_to_tag_map.get(str(pred_id), "O")
            target_position = wid - tgt_start_idx
            
            if 0 <= target_position < len(target_tokens):
                word_predictions[target_position].append(tag)
        
        # Aggregate: majority vote for subword fragments
        target_labels = ["O"] * len(target_tokens)
        for pos in range(len(target_tokens)):
            if pos in word_predictions and word_predictions[pos]:
                votes = word_predictions[pos]
                non_o_votes = [v for v in votes if v != "O"]
                if non_o_votes:
                    target_labels[pos] = Counter(non_o_votes).most_common(1)[0][0]
                else:
                    target_labels[pos] = "O"
            else:
                target_labels[pos] = "O"
        
        all_predictions.append(target_labels)
    
    return all_predictions

def reconstruct_text_from_predictions(original_sentences, predictions, speaker_map):
    """Reconstructs the text from predictions. Merges consecutive sentences from same speaker."""
    reconstructed_blocks = []
    current_speaker = None
    current_text_parts = []

    for i, (speaker, tokens) in enumerate(original_sentences):
        labels = predictions[i] if i < len(predictions) else []
        reconstructed_words = []

        for w_idx, word in enumerate(tokens):
            tag = labels[w_idx] if w_idx < len(labels) else "O"
            if not tag or tag == "O":
                reconstructed_words.append(word)
            else:
                reconstructed_words.append(tag)

        sentence_text = " ".join(reconstructed_words)

        if speaker == current_speaker:
            current_text_parts.append(sentence_text)
        else:
            if current_speaker and current_text_parts:
                full_text = " ".join(current_text_parts)
                reconstructed_blocks.append(f"{current_speaker}: {full_text}")
            current_speaker = speaker
            current_text_parts = [sentence_text]

    if current_speaker and current_text_parts:
        full_text = " ".join(current_text_parts)
        reconstructed_blocks.append(f"{current_speaker}: {full_text}")

    return "\n".join(reconstructed_blocks)

def merge_adjacent_tags(text):
    """Merges duplicate adjacent tags like [PERSON][PERSON]."""
    parts = re.split(r"(\s+|\[[A-Z_]+\])", text)
    merged = []
    last_tag = None
    only_whitespace_since_tag = False

    for part in parts:
        if not part:
            continue

        if re.fullmatch(r"\[[A-Z_]+\]", part):
            if part == last_tag and only_whitespace_since_tag:
                continue
            merged.append(part)
            last_tag = part
            only_whitespace_since_tag = True
        elif part.isspace():
            merged.append(part)
        else:
            merged.append(part)
            last_tag = None
            only_whitespace_since_tag = False

    out_str = "".join(merged)
    out_str = re.sub(r"  *", " ", out_str)
    return out_str

def normalize_punctuation(text):
    """Fixes punctuation spacing issues."""
    if not text:
        return text

    lines = text.split('\n')
    normalized_lines = []

    for line in lines:
        match = re.match(r'^(SPEAKER_\d+):\s*(.*)$', line)
        if not match:
            normalized_lines.append(line)
            continue

        speaker = match.group(1)
        content = match.group(2)

        content = re.sub(r"\s+'(\w)", r"'\1", content)
        content = re.sub(r'\s+-\s+', '-', content)
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        content = re.sub(r'([.!?;:])([A-Za-z])', r'\1 \2', content)

        if content:
            content = content[0].upper() + content[1:]

        normalized_lines.append(f"{speaker}: {content}")

    return "\n".join(normalized_lines)

def parse_transcript_into_blocks(transcript_text):
    """Parses a transcript string into a list of (speaker, text_block) tuples."""
    lines = transcript_text.strip().split('\n')
    blocks = []
    current_speaker = None
    current_text = []

    for line in lines:
        match = re.match(r'^(SPEAKER_\d+):\s*(.*)$', line)
        if match:
            speaker = match.group(1)
            text = match.group(2)
            if speaker == current_speaker:
                current_text.append(text)
            else:
                if current_speaker and current_text:
                    blocks.append((current_speaker, " ".join(current_text)))
                current_speaker = speaker
                current_text = [text]
        else:
            if current_speaker:
                current_text.append(line)

    if current_speaker and current_text:
        blocks.append((current_speaker, " ".join(current_text)))

    return blocks

class AnonymizationEngine:
    """Anonymization Engine using ModernBERT + CRF with OFFICIAL FLERT implementation."""
    
    def __init__(self, method="local_mmbert", level="standard", model_path=None,
                 include_tags=None, exclude_tags=None):
        self.method = method
        self.level = level
        # CORRECTED: Use official DFKI-SLT model name
        official_path = MODEL_FOLDER / "multilingual_DialogPII_NER"
        existing_path = MODEL_FOLDER / "mmbert_multilingual_pii_ner"
        self.model_path = model_path or (official_path if official_path.exists() else existing_path)
        self.include_tags = include_tags
        self.exclude_tags = exclude_tags
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = None
        self.crf_config = None
        self.flert_config = None
        self.label_mapping = {}
        self.original_id2label = {}
        self.original_label2id = {}

        if ANONYMIZATION_ENABLED:
            logger.info(f"AnonymizationEngine initialized: Method={self.method}, Level={self.level}, Model={self.model_path.name}")
            self._load_model()
    
    def _build_safe_label_mapping(self):
        """Constructs mapping from model labels to anonymization tags."""
        all_target_tags = {
            'PERSON': '[PERSON]', 'PERSON_EMAIL': '[EMAIL]', 'PERSON_SOCIAL_RELATION': '[NAME_RELATIVE]',
            'ORG': '[ORGANISATION]', 'LOC_CITY': '[CITY]', 'LOC_COUNTRY': '[COUNTRY]',
            'LOC_STREET': '[STREET]', 'LOC_ZIP': '[ZIP]', 'LOC_HOUSENUMBER': '[HOUSENUMBER]',
            'LOC_OTHER': '[LOCATION]', 'DATETIME': '[DATETIME]', 'DATETIME_AGE': '[AGE]',
            'CODE': '[CODE]', 'CODE_PHONE': '[PHONE]', 'CODE_URL': '[URL]',
            'PROFESSION': '[PROFESSION]', 'PRODUCT': '[PRODUCT]', 'QUANTITY': '[QUANTITY]',
            'MISC': '[MISC]', 'O': ''
        }

        active_tags = set(all_target_tags.keys())

        if self.include_tags:
            include_set = set(t.upper() for t in self.include_tags)
            active_tags = active_tags.intersection(include_set)
            logger.info(f"Restricting to tags: {active_tags}")

        if self.exclude_tags:
            exclude_set = set(t.upper() for t in self.exclude_tags)
            active_tags = active_tags.difference(exclude_set)
            logger.info(f"Excluding tags: {exclude_set}")

        mapping = {}
        for label_id, label_name in self.original_id2label.items():
            clean_label = label_name.replace("B-", "").replace("I-", "").replace("S-", "").replace("E-", "")
            clean_label_upper = clean_label.upper()

            if clean_label_upper in active_tags:
                mapping[str(label_id)] = all_target_tags[clean_label_upper]
            else:
                mapping[str(label_id)] = ""

        logger.info(f"Built label mapping with {len(mapping)} entries.")
        return mapping
    
    def _load_model(self):
        """Loads the mmbert model following DFKI-SLT official pattern."""
        if not self.model_path.exists():
            logger.error(f"Model path not found: {self.model_path}")
            self.method = None
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            import torch.nn as nn
            import json

            # 1. Load CRF config (MUST USE crf_config.json)
            crf_config_path = self.model_path / "crf_config.json"
            if not crf_config_path.exists():
                logger.error(f"crf_config.json not found at {crf_config_path}")
                self.method = None
                return

            with open(crf_config_path, "r") as f:
                self.crf_config = json.load(f)

            required_keys = ["base_model_name", "num_labels", "id2label", "label2id"]
            if not all(k in self.crf_config for k in required_keys):
                logger.error(f"Missing required keys in crf_config.json: {required_keys}")
                self.method = None
                return

            self.original_id2label = self.crf_config.get("id2label", {})
            self.original_label2id = self.crf_config.get("label2id", {})
            logger.info(f"Loaded {len(self.original_id2label)} original model labels from crf_config.json")

            # 2. Load FLERT config (MUST USE flert_config.json)
            flert_config_path = self.model_path / "flert_config.json"
            if flert_config_path.exists():
                with open(flert_config_path, "r") as f:
                    self.flert_config = json.load(f)
                logger.info(f"Loaded FLERT config: context_window={self.flert_config.get('context_window')}, sep_marker={self.flert_config.get('context_sep_marker')}")
            else:
                logger.warning("flert_config.json not found, using defaults: context_window=2, sep_marker=True")
                self.flert_config = {"context_window": 2, "context_sep_marker": True}

            # 3. Instantiate ModernBertCRF
            local_base_model_path = MODEL_FOLDER / self.crf_config["base_model_name"]
            if not local_base_model_path.exists():
                logger.warning(f"Local base model not found at {local_base_model_path}. Trying HF name: {self.crf_config['base_model_name']}")
                local_base_model_path = self.crf_config["base_model_name"]

            self.model = ModernBertCRF(
                base_model_name=local_base_model_path,
                num_labels=self.crf_config["num_labels"],
                id2label=self.original_id2label,
                label2id=self.original_label2id
            )

            # 4. Load weights
            model_weights_path = self.model_path / "pytorch_model.bin"
            state_dict = torch.load(model_weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded and set to evaluation mode")

            # 5. Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)

            # 6. Build label mapping
            self.label_mapping = self._build_safe_label_mapping()
            logger.info("✅ Anonymization model loaded successfully with OFFICIAL FLERT support")

        except Exception as e:
            import traceback
            logger.error(f"Failed to load mmbert model: {e}")
            logger.error(traceback.format_exc())
            self.method = None

    def anonymize(self, text):
        """Anonymizes text using OFFICIAL FLERT context windowing."""
        if not self.method or not self.model or not self.tokenizer:
            return None, False, "Anonymization model not loaded."

        logger.info("Running anonymization with OFFICIAL FLERT context windowing...")

        # 1. Split into sentences (with spaCy)
        sentences_data = split_dialogue_into_sentences(text)

        if not sentences_data:
            return text, False, "No sentences detected."

        # 2. Extract token lists for prediction
        sentences_tokens = [tokens for _, tokens in sentences_data]

        # 3. Run FLERT prediction WITH CONTEXT WINDOWING
        context_window = self.flert_config.get("context_window", 2) if self.flert_config else 2
        use_sep_marker = self.flert_config.get("context_sep_marker", True) if self.flert_config else True
        
        logger.info(f"FLERT parameters: context_window={context_window}, sep_marker={use_sep_marker}")

        try:
            predictions = predict_dialogue_flert(
                sentences_tokens=sentences_tokens,
                model=self.model,
                tokenizer=self.tokenizer,
                id_to_tag_map=self.label_mapping,
                device=self.device,
                context_window=context_window,
                use_sep_marker=use_sep_marker
            )
        except Exception as e:
            logger.error(f"FLERT prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return text, False, str(e)

        # 4. Reconstruct text (merge same-speaker sentences)
        reconstructed_text = reconstruct_text_from_predictions(sentences_data, predictions, {})
        reconstructed_text = normalize_punctuation(reconstructed_text)
        reconstructed_text = merge_adjacent_tags(reconstructed_text)

        entity_count = sum(1 for tag in ['[PERSON]', '[ORGANISATION]', '[CITY]', '[DATE]', '[TIME]'] 
                          if tag in reconstructed_text)
        logger.info(f"Anonymization complete: {entity_count} entities detected")

        return reconstructed_text, True, "Success"

# ============================================================================
# LLM FUNCTIONS
# ============================================================================

def generate_paraphrase(raw_dialogue, lang='DE', model="gpt-oss-120b", temperature=0.3):
    """Rephrase/anonymize a dialogue using a local LLM model."""
    lang = lang.upper()
    if lang not in PARAPHRASE_PROMPTS:
        logger.warning(f"Language '{lang}' not supported, falling back to 'EN'.")
        lang = 'EN'

    system_prompt = PARAPHRASE_PROMPTS[lang]
    user_prompt = f"{USER_PROMPTS[lang]}\n{raw_dialogue.strip()}"

    try:
        client = OpenAI(api_key=CHAT_AI_API_KEY, base_url=CHAT_AI_ENDPOINT)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        result = response.choices[0].message.content.strip()
        logger.info(f"Paraphrase generated successfully ({lang}, model={model}).")
        return result

    except Exception as e:
        logger.error(f"generate_paraphrase failed: {e}")
        return None

def call_llm_rewriter(text, model_id, system_prompt=None):
    """Calls the external LLM API to rewrite text."""
    import re

    if not CHAT_AI_API_KEY:
        logger.error("LLM API Key not configured. Skipping LLM rewrite.")
        return None, "API Key missing"

    final_system_prompt = system_prompt if system_prompt is not None else LLM_REWRITE_SYSTEM_PROMPT

    try:
        client = OpenAI(api_key=CHAT_AI_API_KEY, base_url=CHAT_AI_ENDPOINT)

        final_model = model_id
        if model_id not in AVAILABLE_LLM_MODELS.values():
            if model_id in AVAILABLE_LLM_MODELS:
                final_model = AVAILABLE_LLM_MODELS[model_id]
            else:
                logger.warning(f"Model ID '{model_id}' not recognized, attempting to use as-is.")

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": text}
        ]

        logger.info(f"Calling LLM model: {final_model} with max_tokens=16384")

        chat_completion = client.chat.completions.create(
            messages=messages,
            model=final_model,
            stream=False,
            temperature=0.3,
            max_tokens=16384
        )

        rewritten_text = None
        raw_content = None

        if (hasattr(chat_completion, 'choices') and 
            chat_completion.choices and 
            hasattr(chat_completion.choices[0], 'message')):

            msg = chat_completion.choices[0].message

            if hasattr(msg, 'content') and msg.content is not None:
                raw_content = str(msg.content)
            elif hasattr(msg, 'text') and msg.text:
                raw_content = str(msg.text)
            else:
                raw_content = str(msg)

        if raw_content:
            if raw_content.startswith("language None"):
                match = re.search(r'language None\s*<asr_text>(.*?)</asr_text>', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    rewritten_text = match.group(1).strip()
                else:
                    rewritten_text = raw_content[len("language None"):].strip()

            elif "<thought>" in raw_content.lower():
                match = re.search(r'</thought>\s*(.*)', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    rewritten_text = match.group(1).strip()
                else:
                    rewritten_text = re.sub(r'^.*?<thought>.*?</thought>\s*', '', raw_content, flags=re.DOTALL | re.IGNORECASE).strip()

            else:
                transcript_start_pattern = r'(?:^|\n)(?!\s*[0-9]+\.\s|\s*[-*]\s|\s*\*\*)(\s*SPEAKER_\d+:.*)'
                match = re.search(transcript_start_pattern, raw_content, re.MULTILINE | re.IGNORECASE)

                if match:
                    rewritten_text = raw_content[match.start():].strip()
                else:
                    paragraphs = re.split(r'\n\s*\n', raw_content)
                    if paragraphs:
                        last_para = paragraphs[-1]
                        if 'SPEAKER_' in last_para:
                            rewritten_text = last_para.strip()
                        else:
                            rewritten_text = last_para.strip()
                    else:
                        rewritten_text = raw_content.strip()

        if rewritten_text:
            rewritten_text = re.sub(r'^```\w*\s*|\s*```$', '', rewritten_text, flags=re.MULTILINE)
            rewritten_text = re.sub(r'```[\s\S]*?```', '', rewritten_text)
            rewritten_text = re.sub(r'<[^>]+>', '', rewritten_text)
            rewritten_text = re.sub(r'[^\S\n]+', ' ', rewritten_text)
            rewritten_text = rewritten_text.strip()

            if rewritten_text.lower() in ["language none", "none", ""]:
                raise ValueError("Extracted text is empty or invalid.")

            if '\n' not in rewritten_text and '\n' in text:
                logger.warning("LLM output collapsed into single line despite instructions.")

        if not rewritten_text:
            raise ValueError("Model returned empty content or unrecognized format.")

        return rewritten_text, "Success"

    except Exception as e:
        logger.error(f"LLM Rewriter failed for model {final_model}: {e}")
        return None, str(e)

def run_adversarial_anonymization(text, model_id, iterations=3):
    """Runs a dual-agent adversarial loop."""
    if not CHAT_AI_API_KEY:
        logger.error("Adversarial loop requires CHAT_AI_API_KEY.")
        return text, []

    client = OpenAI(api_key=CHAT_AI_API_KEY, base_url=CHAT_AI_ENDPOINT)
    final_model = AVAILABLE_LLM_MODELS.get(model_id, model_id)

    DEFENDER_SYSTEM_PROMPT = (
        "You are a Privacy Defender AI. Your goal is to anonymize the provided text to prevent re-identification.\n"
        "Rules:\n"
        "1. Remove or generalize ALL direct identifiers (names, emails, phones, IDs, specific addresses).\n"
        "2. Remove or generalize INDIRECT identifiers (specific job titles, unique combinations of demographics, rare locations, specific dates, unique medical conditions).\n"
        "3. Replace removed info with generic placeholders like [NAME], [LOCATION], [DATE], [JOB_TITLE].\n"
        "4. CRITICAL: Do NOT change the general meaning, tone, or flow of the conversation.\n"
        "5. CRITICAL: Preserve speaker tags (SPEAKER_00, etc.) and original line breaks exactly.\n"
        "6. Do NOT translate the text. Keep the original language.\n"
        "7. If the text is already well-anonymized, make minimal adjustments only where the Attacker pointed out flaws.\n"
        "Return ONLY the anonymized text."
    )

    ATTACKER_SYSTEM_PROMPT = (
        "You are a Privacy Red Team AI. Your goal is to re-identify individuals or infer sensitive attributes from the provided text.\n"
        "Task:\n"
        "1. Analyze the text for any remaining Direct Identifiers (names, emails, phones).\n"
        "2. Analyze for Indirect Identifiers (combinations of age, location, job, specific events that could uniquely identify someone).\n"
        "3. Attempt to infer attributes not explicitly stated but implied.\n"
        "4. Output a structured critique:\n"
        "   - 'Risks Found': List specific phrases or patterns that allow re-identification.\n"
        "   - 'Inferred Attributes': List any attributes you can guess about the speakers.\n"
        "   - 'Recommendation': Specific instructions on how to fix these leaks.\n"
        "5. If the text is perfectly anonymous, state 'No risks found.'\n"
        "Return ONLY the critique in the format described."
    )

    iteration_log = []
    current_text = text

    logger.info(f"Starting Adversarial Loop: {iterations} iterations with model {final_model}")

    for i in range(1, iterations + 1):
        logger.info(f"--- Iteration {i}/{iterations} ---")

        # --- Step A: Defender (Anonymize) ---
        defender_messages = [
            {"role": "system", "content": DEFENDER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please anonymize the following text:\n\n{text if i == 1 else current_text}"}
        ]

        if i > 1:
            last_critique = iteration_log[-1]["attacker_output"]
            defender_messages[1]["content"] += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS ROUND:\n{last_critique}\n\nAddress these specific points."

        try:
            defender_response = client.chat.completions.create(
                model=final_model,
                messages=defender_messages,
                temperature=0.3,
                max_tokens=16384
            )
            anonymized_text = defender_response.choices[0].message.content.strip()

            anonymized_text = re.sub(r'<[^>]+>', '', anonymized_text)
            anonymized_text = re.sub(r'[^\S\n]+', ' ', anonymized_text)

            logger.info(f"Defender completed iteration {i}. Length: {len(anonymized_text)}")

        except Exception as e:
            logger.error(f"Defender failed in iteration {i}: {e}")
            break

        # --- Step B: Attacker (Critique) ---
        attacker_messages = [
            {"role": "system", "content": ATTACKER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this text for re-identification risks:\n\n{anonymized_text}"}
        ]

        try:
            attacker_response = client.chat.completions.create(
                model=final_model,
                messages=attacker_messages,
                temperature=0.5,
                max_tokens=4096
            )
            critique = attacker_response.choices[0].message.content.strip()
            critique = re.sub(r'<[^>]+>', '', critique)

            logger.info(f"Attacker completed iteration {i}. Risks found: {'Yes' if 'Risk' in critique or 'Found' in critique else 'None'}")

        except Exception as e:
            logger.error(f"Attacker failed in iteration {i}: {e}")
            critique = "Error generating critique."

        iteration_log.append({
            "iteration": i,
            "defender_output": anonymized_text,
            "attacker_output": critique
        })

        current_text = anonymized_text

        if "No risks found" in critique.lower() and i < iterations:
            logger.info("Attacker found no risks. Stopping early.")
            break

    return current_text, iteration_log

# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

def process_anonymization(llm_rewrite_enabled=None, llm_model_id=None, skip_bert=False,
                          adversarial_mode=False, include_tags=None, exclude_tags=None,
                          file_list=None):
    """
    Reads raw transcripts, anonymizes them with BERT, and optionally rewrites with LLM.
    Supports adversarial mode which runs a 3-iteration Red Team vs. Blue Team loop.
    """
    if not TRANSCRIPTS_FOLDER.exists() and not skip_bert:
        logger.warning(f"No 'transcripts' folder found at {TRANSCRIPTS_FOLDER}. Skipping anonymization.")
        return {"success": 0, "failed": 0, "llm_success": 0, "llm_failed": 0}

    logger.info(f"Found 'transcripts' folder at {TRANSCRIPTS_FOLDER}. Starting anonymization process...")

    use_llm = llm_rewrite_enabled if llm_rewrite_enabled is not None else LLM_REWRITE_ENABLED
    target_llm_model = llm_model_id if llm_model_id else DEFAULT_CHAT_AI_MODEL

    if use_llm and not CHAT_AI_API_KEY:
        logger.warning("LLM rewrite requested but no API key found. Disabling LLM step.")
        use_llm = False
        adversarial_mode = False

    anonymizer = None
    if not skip_bert:
        anonymizer = AnonymizationEngine(
            method=ANONYMIZATION_METHOD,
            level=ANONYMIZATION_LEVEL,
            model_path=MODEL_FOLDER / "multilingual_DialogPII_NER",
            include_tags=include_tags,
            exclude_tags=exclude_tags
        )

        if not anonymizer.method:
            logger.error("Anonymization engine (BERT) failed to initialize. Aborting.")
            return {"success": 0, "failed": 0, "llm_success": 0, "llm_failed": 0}

    processed_count = 0
    failed_count = 0
    llm_processed_count = 0
    llm_failed_count = 0

    source_folder = TRANSCRIPTS_FOLDER if not skip_bert else ANONYM_FOLDER
    source_suffix = ".txt"

    if skip_bert:
        logger.info("Skipping BERT anonymization. Processing existing files in 'anonym' folder for LLM rewrite.")
        files = []
        for f in source_folder.iterdir():
            if f.is_file() and f.suffix.lower() == source_suffix:
                name = f.name
                if "_llm" in name or "_adversarial_" in name:
                    continue
                files.append(f)
    else:
        logger.info("Processing raw transcripts for BERT anonymization.")
        files = [f for f in source_folder.iterdir()
                 if f.is_file() and f.suffix.lower() == source_suffix and "_anon" not in f.name]

    # Filter by file_list if provided
    if file_list:
        file_stems = [Path(f).stem for f in file_list]
        files = [f for f in files if f.stem in file_stems or any(fs in f.name for fs in file_stems)]

    if not files:
        logger.info(f"No files found to process in {source_folder}.")
        return {"success": processed_count, "failed": failed_count,
                "llm_success": llm_processed_count, "llm_failed": llm_failed_count}

    logger.info(f"Found {len(files)} files to process.")
    if adversarial_mode:
        logger.info("⚠️  ADVERSARIAL MODE ENABLED: Running 3-iteration Red/Blue team loop.")

    for file in files:
        if not validate_path(file, source_folder):
            logger.error(f"Security Alert: Attempted path traversal detected for {file.name}. Skipping.")
            continue

        base_name = file.stem
        if skip_bert and base_name.endswith("_anon"):
            base_name = base_name[:-5]

        logger.info(f"Processing file: {file.name}")

        try:
            with open(file, "r", encoding="utf-8") as f:
                text_content = f.read()

            if not text_content.strip():
                logger.warning(f"File {file.name} is empty. Skipping.")
                continue

            # Step 1: BERT Anonymization
            if not skip_bert:
                anonymized_text, success, msg = anonymizer.anonymize(text_content)

                if not success or not anonymized_text:
                    logger.warning(f"BERT Anonymization failed for {base_name}: {msg}")
                    failed_count += 1
                    continue

                output_filename = f"{base_name}_anon.txt"
                output_path = ANONYM_FOLDER / output_filename

                if not validate_path(output_path, ANONYM_FOLDER):
                    logger.error(f"Security Alert: Output path traversal detected for {output_filename}. Skipping save.")
                    failed_count += 1
                    continue

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(anonymized_text)
                logger.info(f"BERT Anonymized transcript saved to: {output_path}")
                processed_count += 1

                text_for_llm = anonymized_text
            else:
                text_for_llm = text_content
                logger.info(f"Using existing anonymized content from {file.name} for LLM step.")

            # Step 2: Optional LLM Rewrite (Standard or Adversarial)
            if use_llm:
                if adversarial_mode:
                    logger.info(f"Running ADVERSARIAL LOOP (3 iterations) on {base_name} with model {target_llm_model}...")

                    final_text, iteration_log = run_adversarial_anonymization(
                        text_for_llm, target_llm_model, iterations=3
                    )

                    if final_text:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        llm_filename = f"{base_name}_adversarial_{timestamp}.txt"
                        llm_path = LLM_ANONYM_FOLDER / llm_filename

                        if not validate_path(llm_path, LLM_ANONYM_FOLDER):
                            logger.error(f"Security Alert: LLM output path traversal detected for {llm_filename}. Skipping.")
                            llm_failed_count += 1
                            continue

                        with open(llm_path, "w", encoding="utf-8") as f:
                            f.write(final_text)

                        log_filename = f"{base_name}_adversarial_{timestamp}_log.json"
                        log_path = LLM_ANONYM_FOLDER / log_filename
                        with open(log_path, "w", encoding="utf-8") as f:
                            json.dump(iteration_log, f, indent=2, ensure_ascii=False)

                        logger.info(f"Adversarial anonymization saved to: {llm_path}")
                        logger.info(f"Audit log saved to: {log_path}")
                        llm_processed_count += 1
                    else:
                        logger.warning(f"Adversarial loop failed for {base_name}")
                        llm_failed_count += 1
                else:
                    logger.info(f"Running standard LLM rewrite on {base_name} with model {target_llm_model}...")
                    llm_result, status = call_llm_rewriter(text_for_llm, target_llm_model)

                    if llm_result:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        llm_filename = f"{base_name}_llm_{timestamp}.txt"
                        llm_path = LLM_ANONYM_FOLDER / llm_filename

                        if not validate_path(llm_path, LLM_ANONYM_FOLDER):
                            logger.error(f"Security Alert: LLM output path traversal detected for {llm_filename}. Skipping.")
                            llm_failed_count += 1
                            continue

                        with open(llm_path, "w", encoding="utf-8") as f:
                            f.write(llm_result)
                        logger.info(f"LLM Rewritten transcript saved to: {llm_path}")
                        llm_processed_count += 1
                    else:
                        logger.warning(f"LLM rewrite failed for {base_name}: {status}")
                        llm_failed_count += 1

        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_count += 1

    logger.info(f"Anonymization phase complete.")
    if not skip_bert:
        logger.info(f"  BERT Processed: {processed_count}, Failed: {failed_count}")
    if use_llm:
        if adversarial_mode:
            logger.info(f"  LLM Adversarial Processed: {llm_processed_count}, Failed: {llm_failed_count}")
        else:
            logger.info(f"  LLM Rewritten (Standard): {llm_processed_count}, Failed: {llm_failed_count}")

    return {
        "success": processed_count,
        "failed": failed_count,
        "llm_success": llm_processed_count,
        "llm_failed": llm_failed_count
    }

def anonymize_text_locally(text):
    """Wrapper function to anonymize text using the local BERT model."""
    try:
        engine = AnonymizationEngine(
            method="local_mmbert",
            level="standard",
            model_path=MODEL_FOLDER / "multilingual_DialogPII_NER"
        )

        if not engine.method:
            return None, False, "Anonymization engine failed to initialize (method not set)."

        result_text, success, msg = engine.anonymize(text)

        if success:
            return result_text, True, "Success"
        else:
            return None, False, msg

    except Exception as e:
        logger.error(f"Error in anonymize_text_locally: {e}", exc_info=True)
        return None, False, str(e)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Core pipeline functions
    'transcribe_audio_locally',
    'load_models',
    'call_llm_rewriter',
    'generate_paraphrase',
    'anonymize_text_locally',
    'process_anonymization',
    'process_videos',
    'process_audios',

    # LLM functions
    'run_adversarial_anonymization',

    # TTS service functions
    'generate_speech',
    'generate_beep',
    'synthesize_segment',
    'get_available_tts_voices',
    'get_tts_status',
    'TTS_BACKEND',
    'TTS_ENABLED',

    # Paths and config
    'BASE_PATH',
    'pipeline_dir',
    'ANONYM_FOLDER',
    'LLM_ANONYM_FOLDER',
    'MODEL_FOLDER',
    'TRANSCRIPTS_FOLDER',
    'AVAILABLE_LLM_MODELS',
    'CHAT_AI_API_KEY',
    'CHAT_AI_ENDPOINT',
]

# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audio Anonymization Pipeline with Granular Step Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process.py                          # Run all steps (default)
  python process.py --disable-transcription  # Skip audio extraction
  python process.py --disable-diarization    # Skip speaker identification
  python process.py --disable-anonymization  # Skip anonymization
  python process.py --disable-llm            # Disable the LLM element of the anonymization process
  python process.py --llm-only               # Skip BERT and process existing anonymized files with LLM
  python process.py --disable-transcription --disable-diarization  # Multiple disables
        """
    )

    parser.add_argument('--disable-transcription', action='store_true',
                        help='Disable audio extraction from videos (skip process_videos)')
    parser.add_argument('--disable-diarization', action='store_true',
                        help='Disable speaker diarization during transcription')
    parser.add_argument('--disable-anonymization', action='store_true',
                        help='Disable BERT-based anonymization')
    parser.add_argument('--disable-llm', action='store_true',
                        help='Disable LLM-based indirect identifier removal (Run BERT anonymization only)')

    parser.add_argument('--llm-only', action='store_true',
                        help='Skip BERT anonymization and process existing files in "anonym" folder with LLM rewrite only.')

    parser.add_argument('--lang', type=str, default=None,
                        choices=list(SUPPORTED_LANGUAGES.keys()),
                        help=f"Force language (e.g., DE, EN, SP, ES). Default: Auto-detect.")

    parser.add_argument('--llm-model', type=str, default=None,
                        choices=list(AVAILABLE_LLM_MODELS.keys()),
                        help=f"Specific LLM model to use for rewriting.")

    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug-level logging')

    parser.add_argument('--adversarial', action='store_true',
                        help='Enable dual-agent adversarial anonymization (3 iterations: Anonymize -> Attack -> Refine)')

    parser.add_argument('--include-tags', type=str, nargs='+', default=None,
                        help=f"Select specific tags to anonymize. "
                             f"Available tags: {', '.join(AVAILABLE_TAGS)}. "
                             f"If omitted, all tags are enabled. "
                             f"Example: --include-tags PERSON ORG LOC_CITY")

    parser.add_argument('--exclude-tags', type=str, nargs='+', default=None,
                        help=f"Select specific tags to IGNORE (do not anonymize). "
                             f"Available tags: {', '.join(AVAILABLE_TAGS)}. "
                             f"Example: --exclude-tags PROFESSION QUANTITY")

    parser.add_argument('--file', type=str, help='Process a single .wav file')
    parser.add_argument('--files', nargs='+', help='Process multiple .wav files')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    check_gpu_resources()

    run_transcription = not args.disable_transcription
    run_diarization = not args.disable_diarization
    run_anonymization = not args.disable_anonymization
    run_llm = not args.disable_llm
    is_adversarial = args.adversarial

    skip_bert_for_llm = args.llm_only

    if args.llm_only:
        run_llm = True

    logger.info("="*60)
    logger.info("PIPELINE EXECUTION PLAN")
    logger.info("="*60)
    logger.info(f"  Audio Extraction (Videos→WAV):     {'✅ ENABLED' if run_transcription else '❌ DISABLED'}")
    logger.info(f"  Transcription & Diarization:       {'✅ ENABLED' if run_transcription else '❌ DISABLED'}")
    if run_transcription:
        logger.info(f"    └─ Speaker Diarization:        {'✅ ENABLED' if run_diarization else '❌ DISABLED'}")
    logger.info(f"  BERT Anonymization:                {'✅ ENABLED' if run_anonymization and not skip_bert_for_llm else '❌ DISABLED (or Skipped for LLM-only)'}")
    logger.info(f"  LLM Indirect Identifier Removal:   {'✅ ENABLED' if run_llm else '❌ DISABLED'}")
    if skip_bert_for_llm:
        logger.info(f"    └─ Mode: LLM-only (processing existing 'anonym' folder)")
    logger.info("="*60)

    logger.info("Starting Audio Anonymizer full pipeline...")

    input_file_list = []
    if hasattr(args, 'file') and args.file:
        input_file_list = [args.file]
    elif hasattr(args, 'files') and args.files:
        input_file_list = args.files

    # Step 1: Audio Extraction (Videos → WAV)
    if run_transcription:
        process_videos(file_list=input_file_list if input_file_list else None)
    else:
        logger.info("⏭️  Skipping audio extraction (--disable-transcription)")

    # Step 2: Transcription & Diarization (WAV → Transcript)
    if run_transcription:
        process_audios(
            enable_diarization=run_diarization,
            lang_code=args.lang,
            file_list=input_file_list if input_file_list else None
        )
    else:
        logger.info("⏭️  Skipping transcription (--disable-transcription)")

    include_tags = args.include_tags
    exclude_tags = args.exclude_tags

    # Step 3: Anonymization (Transcript → Anonymized)
    if run_anonymization or args.llm_only:
        process_anonymization(
            llm_rewrite_enabled=run_llm,
            llm_model_id=args.llm_model,
            skip_bert=skip_bert_for_llm,
            adversarial_mode=is_adversarial,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            file_list=input_file_list if input_file_list else None
        )
    else:
        logger.info("⏭️  Skipping anonymization (--disable-anonymization)")
        if run_llm and not args.llm_only:
            logger.warning("⚠️  LLM rewrite requested but BERT anonymization is disabled and --llm-only not set. "
                           "LLM step will be skipped as it depends on anonymized input.")

    logger.info("Pipeline finished.")