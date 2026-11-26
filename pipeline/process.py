# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import os
import sys
import time
import re
import subprocess
import logging
import gc
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import BytesIO
from collections import OrderedDict, defaultdict

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import requests
import ffmpeg
import spacy
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from openai import OpenAI
# Faker is optional: pipeline.surrogates falls back to static pools without it
# (HAS_FAKER is set below, once the project root is on sys.path).

# =============================================================================
# cuDNN DISABLE FIX
# =============================================================================
# This MUST run BEFORE any whisperx/pyannote imports that trigger cuDNN init
# =============================================================================
import torch
import warnings

# Only apply verbose filters when running as main script (args exists)
try:
    if args.verbose:  # This will fail during import if args not defined
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
except NameError:
    # Safe to ignore - this only applies when running as main script
    pass

# Disable cuDNN to avoid CUDNN_STATUS_NOT_INITIALIZED with WhisperX + Pyannote
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Re-enable TF-32 for better performance on A40 (Ampere architecture)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =============================================================================

# Now safe to import whisperx and pyannote-dependent modules
import whisperx

# =============================================================================
# LOCAL PROJECT IMPORTS
# =============================================================================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.tts.tts_engine import (
    generate_speech,
    synthesize_segment,
    get_available_tts_voices,
    get_tts_status,
    TTS_BACKEND,
    TTS_ENABLED,
)
from pipeline.redaction import DEFAULT_PAD_S  # noqa: E402
from pipeline.secure_store import (  # noqa: E402
    encrypt_bytes, ensure_private_dir, load_master_key, write_private_file)
from pipeline.surrogates import POOLS as SURROGATES  # noqa: E402,F401  (legacy alias, new shape)
from pipeline.surrogates import apply_surrogates, faker_available, lang_from_locales  # noqa: E402
import pipeline.redaction_cli as redaction_cli  # noqa: E402

HAS_FAKER = faker_available()

# --- Surrogate Consistency Support ---

FAKER_LOCALE_MAP = {
    'EN': 'en_US', 'DE': 'de_DE', 'FR': 'fr_FR', 'ES': 'es_ES',
    'IT': 'it_IT', 'PL': 'pl_PL', 'PT': 'pt_BR', 'FI': 'fi_FI',
    'TR': 'tr_TR', 'AR': 'ar_AA', 'HI': 'hi_IN'
}

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================
load_dotenv()

CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')

# =============================================================================
# LOGGING SETUP
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# GLOBAL CONSTANTS & PATHS
# =============================================================================
BASE_PATH = Path(__file__).parent
MODEL_FOLDER = BASE_PATH / 'pipeline' / 'model'
WHISPERX_MODEL_PATH = MODEL_FOLDER / 'Systran--faster-whisper-large-v3'

script_dir = Path(__file__).resolve().parent.parent  # ATA/ root
interactive_app_path = script_dir / "interactive_app"

if str(interactive_app_path) not in sys.path:
    sys.path.insert(0, str(interactive_app_path))

# Now import audio_utils (this will work now)
try:
    from audio_utils import AudioBeepReplacer, AudioError, probe_media
except ImportError:
    logging.warning("audio_utils not found. Beep replacement and media validation disabled.")
    AudioBeepReplacer = None
    probe_media = None

    class AudioError(Exception):          # placeholder so ``except AudioError`` stays valid
        code = "audio_error"
        message = "audio utilities unavailable"

SEGMENT_MERGE_THRESHOLD = 2.0  # Seconds - set to None for unconditional merge
AUDIO_EDIT_SUPPORT = True      # Enable audio offset manipulation features

# =============================================================================
# END OF IMPORT SECTION
# =============================================================================

# GPU Detection Check

def check_gpu_resources():
    """
    Checks for GPU availability and logs the status.
    """
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

# --- Session Logger ---

class SessionLogger:
    """
    Manages the generation of a summary log file for every script execution.
    """
    def __init__(self, base_path):
        self.base_path = base_path
        self.log_dir = base_path / "logs"
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = self.log_dir / f"session_{self.session_id}.txt"
        
        # Ensure logs directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session data
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
            "adversarial_iterations": 0,
            "errors": []
        }
        
        # Initialize the log file with header
        self._write_header()

    def _write_header(self):
        """Writes the initial header to the log file."""
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("AUDIO ANONYMIZATION PIPELINE - SESSION LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Script Path: {Path(__file__).resolve()}\n")
            f.write(f"Settings logged in finish()\n")  # Placeholder
            f.write("-" * 60 + "\n")

    def log_settings(self, settings_dict):
        """
        Stores configuration settings for later writing in finish().
        Settings are no longer written immediately to avoid fragmentation.
        """
        self.settings = settings_dict  # Store only, don't write yet

    def log_stats_update(self, **kwargs):
        """Updates the stats dictionary."""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value

    def log_error(self, error_msg):
        """Logs an error message to internal storage."""
        self.stats["errors"].append(error_msg)

    def finish(self):
        """
        Writes the complete session log in one operation.
        Consolidates all settings and stats into unified output.
        """
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            # Write all settings at once (consolidated)
            f.write("SETTINGS:\n")
            f.write("-" * 60 + "\n")
            for key, value in self.settings.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Write execution summary
            f.write("-" * 60 + "\n")
            f.write("EXECUTION SUMMARY:\n")
            f.write("-" * 60 + "\n")
            f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Duration: {duration}\n")
            f.write(f"Duration (Seconds): {duration.total_seconds():.2f}\n")
            f.write("\n")
            
            # Write stats
            f.write("FILES PROCESSED:\n")
            f.write(f"  Videos Extracted: {self.stats['videos_processed']}\n")
            f.write(f"  Audios Transcribed: {self.stats['audios_processed']}\n")
            f.write(f"  Transcripts Generated: {self.stats['transcripts_generated']}\n")
            f.write(f"  Anonymizations Success: {self.stats['anonymizations_success']}\n")
            f.write(f"  Anonymizations Failed: {self.stats['anonymizations_failed']}\n")
            f.write(f"  LLM Rewrites Success: {self.stats['llm_rewrites_success']}\n")
            f.write(f"  LLM Rewrites Failed: {self.stats['llm_rewrites_failed']}\n")
            f.write(f"  Adversarial Iterations: {self.stats['adversarial_iterations']}\n")
            
            # Write errors (if any)
            if self.stats["errors"]:
                f.write("\nERRORS ENCOUNTERED:\n")
                for err in self.stats["errors"]:
                    f.write(f"  - {err}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("SESSION COMPLETE\n")
            f.write("=" * 60 + "\n")
        
        logger.info(f"Session log saved to: {self.log_file_path}")
        return self.log_file_path

# --- Configuration & Security ---

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

# The project root is the parent of this file's folder (".../<root>/pipeline/process.py").
# That works for any checkout name and for the Docker layout (/app/pipeline), unlike the
# previous "walk up to a folder named ATA" logic. Override with ATA_BASE_PATH.
BASE_PATH = Path(os.environ.get("ATA_BASE_PATH") or SCRIPT_DIR.parent).resolve()
if not (BASE_PATH / "pipeline").is_dir():
    logger.critical(f"Project root {BASE_PATH} has no 'pipeline' folder. Set ATA_BASE_PATH.")
    logger.critical(f"Current script location: {SCRIPT_DIR}")
    sys.exit(1)

logger.info(f"Base path detected: {BASE_PATH}")

pipeline_dir = BASE_PATH / "pipeline"

# Derived paths
VIDEOS_FOLDER = pipeline_dir / "videos"
AUDIOS_FOLDER = pipeline_dir / "audios"
TRANSCRIPTS_FOLDER = pipeline_dir / "transcripts"
MODEL_FOLDER = pipeline_dir / "model"
ANONYM_FOLDER = pipeline_dir / "anonym"
LLM_ANONYM_FOLDER = pipeline_dir / "LLM-Anon"
ENTITYMAP_FOLDER = pipeline_dir / "entitymaps"          # opt-in, private (see _save_entity_map)
REDACTED_AUDIO_FOLDER = pipeline_dir / "audios_redacted"

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

# Local Model Paths - CORRECTED
WHISPERX_MODEL_PATH = MODEL_FOLDER / "Systran--faster-whisper-large-v3"
DIARIZATION_MODEL_PATH = MODEL_FOLDER / "models--pyannote--speaker-diarization-community-1"

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

# Fallback logic for Whisper Model
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
    logger.info(f"✅ Using default model: large-v3")
logger.info(f"✅ WhisperX Model Path Set: {WHISPERX_MODEL_PATH.name}")

if not MODEL_FOLDER.exists():
    logger.critical(f"CRITICAL: Model folder not found at {MODEL_FOLDER}.")
    sys.exit(1)

if not DIARIZATION_MODEL_PATH.exists():
    logger.warning(f"WARNING: Diarization model not found at {DIARIZATION_MODEL_PATH}.")
    logger.warning("Speaker diarization will be disabled. Using generic speaker labels.")
else:
    logger.info(f"✅ Diarization Model Path Set: {DIARIZATION_MODEL_PATH.name}")

logger.info(f"✅ Paths verified successfully.")
logger.info(f"   Model Folder: {MODEL_FOLDER}")
logger.info(f"   WhisperX Model: {WHISPERX_MODEL_PATH.name}")
logger.info(f"   Diarization Model: {DIARIZATION_MODEL_PATH.name if DIARIZATION_MODEL_PATH.exists() else 'MISSING'}")

# Supported Languages
SUPPORTED_LANGUAGES = {
    'AR': 'Arabic', 'DE': 'German', 'EN': 'English', 'FI': 'Finnish',
    'FR': 'French', 'HI': 'Hindi', 'IT': 'Italian', 'PL': 'Polish',
    'PT': 'Portuguese', 'SP': 'Spanish', 'ES': 'Spanish', 'TR': 'Turkish'
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
    "PERSON", "PERSON_EMAIL", "PERSON_SOCIAL_RELATION",
    "ORG", 
    "LOC_CITY", "LOC_COUNTRY", "LOC_STREET", "LOC_ZIP", "LOC_HOUSENUMBER", "LOC_OTHER",
    "DATETIME", "DATETIME_AGE",
    "CODE", "CODE_PHONE", "CODE_URL",
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
    "GUIDELINES FOR GENERALIZATION:\n"
    "- If a job title is highly specific, generalize to a broader category.\n"
    "- If a location is rare or unique, generalize to a broader geographic area.\n"
    "- If a combination of demographics is unique, generalize accordingly.\n"
    "- If the detail is already generic, DO NOT change it.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. PRESERVE SPEAKER TAGS: Lines starting with 'SPEAKER_XX:' are structural metadata. NEVER modify them.\n"
    "2. PRESERVE EXISTING TAGS: Do not touch [NAME_OTHER], [DATE], [ID], [LOCATION_CITY], etc.\n"
    "3. MINIMAL CHANGE: Only generalize when necessary for privacy.\n"
    "4. PRESERVE STRUCTURE: Keep all newlines, line breaks, and grammar exactly as they are.\n"
    "5. LANGUAGE: Do not translate. Keep the original language.\n\n"
    "Return ONLY the anonymized text. No explanations, no reasoning, no commentary."
)

AVAILABLE_LLM_MODELS = {
    'medgemma': 'medgemma',
    'medgemma27b': 'medgemma27b',
    'gpt-oss-120b': 'gpt-oss-120b',
    'Qwen3.6-27B': 'Qwen3.6-27B',
    'Qwen3.5-27B': 'Qwen3.5-27B',
    'qwen3-asr-1.7b': 'qwen3-asr-1.7b',
    'cle-Kimi-K2.6': 'cle-Kimi-K2.6',
    'cle-Qwen3-Coder-Next-FP8': 'cle-Qwen3-Coder-Next-FP8',
    'cle-Qwen3.5-397B-A17B-FP8': 'cle-Qwen3.5-397B-A17B-FP8'
}

# --- Helper Functions ---

def sanitize_filename(filename):
    """Removes potentially dangerous characters from filenames."""
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized

def validate_path(path, base, check_exists=True):
    """
    Validates paths for security (prevent directory traversal).
    
    Args:
        path: The path to validate
        base: The allowed base directory
        check_exists: Whether to check if the path exists (default: True)
                     Set to False when validating NEW files to be created
    """
    try:
        resolved = path.resolve()
        base_resolved = base.resolve()
        
        if not resolved.is_relative_to(base_resolved):
            return False
        
        if check_exists and not resolved.exists():
            return False
        
        if resolved.is_symlink():
            return False
        
        return True
    except (OSError, RuntimeError) as e:
        logger.debug(f"Path validation error: {e}")
        return False

def merge_consecutive_speaker_segments(segments, max_gap_seconds=2.0):
    """
    Merges consecutive segments spoken by the same speaker.
    Allows configurable silence threshold between segments to control merging.

    Word-level timestamps (``words``) are preserved and concatenated: audio
    redaction needs them to locate anonymized words in time.

    Args:
        segments: List of dicts with 'start', 'end', 'speaker', 'text' and
                  optionally 'words' (WhisperX word alignment).
        max_gap_seconds: Maximum silence allowed between segments to merge them.
                         Set to None to merge ALL consecutive same-speaker segments
                         regardless of timing gap.
    Returns:
        List of merged segment dicts, each with a ``words`` list (possibly empty).
    """
    if not segments:
        return []

    def new_segment(speaker, text, start, end, words):
        return {"speaker": speaker, "text": text, "start": start, "end": end,
                "words": list(words or [])}

    merged_segments = []
    current_segment = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "")
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        words = segment.get("words")

        if current_segment is None:
            current_segment = new_segment(speaker, text, start, end, words)
        elif current_segment["speaker"] == speaker and (
                max_gap_seconds is None or start - current_segment["end"] <= max_gap_seconds):
            current_segment["text"] += " " + text
            current_segment["end"] = end
            current_segment["words"].extend(words or [])
        else:
            merged_segments.append(current_segment)
            current_segment = new_segment(speaker, text, start, end, words)

    if current_segment:
        merged_segments.append(current_segment)

    return merged_segments


# =============================================================================
# SECTION 2: AUDIO OFFSET MODIFICATION FUNCTIONS (NEW)
# =============================================================================

def adjust_segment_offsets(segments, offset_map):
    """
    Applies time offset adjustments to transcript segments for audio editing.
    Supports cuts, inserts, and absolute shifts.
    
    Args:
        segments: List of segment dicts with 'start', 'end', 'speaker', 'text'
        offset_map: Dict specifying time adjustments:
            {
                "absolute_shift": float,          # Shift all segments by N seconds
                "cut_ranges": [(start, end), ...], # Time ranges to remove
                "insert_points": [(position, duration), ...]  # Silent insertions
            }
    
    Returns:
        tuple: (adjusted_segments, removed_count)
        adjusted_segments: Modified segments with corrected timestamps
        removed_count: Number of segments completely removed by cuts
    
    Note:
        For HPC/slurm environments, ensure offset calculations use float precision
        to avoid cumulative rounding errors in long audio files.
    """
    import bisect
    
    adjusted_segments = []
    removed_count = 0
    
    for segment in segments:
        new_start = segment["start"]
        new_end = segment["end"]
        
        # Apply absolute shift if specified
        if "absolute_shift" in offset_map:
            shift = offset_map["absolute_shift"]
            new_start += shift
            new_end += shift
        
        # Handle cut ranges (removal of time spans)
        if "cut_ranges" in offset_map:
            total_cut_before = 0.0
            for cut_start, cut_end in offset_map["cut_ranges"]:
                if new_start >= cut_end:
                    # Segment completely after cut region
                    cut_duration = cut_end - cut_start
                    new_start -= cut_duration
                    new_end -= cut_duration
                    total_cut_before += cut_duration
                elif new_end <= cut_start:
                    # Segment completely before cut region - no adjustment needed
                    pass
                elif new_start >= cut_start and new_end <= cut_end:
                    # Segment completely inside cut region - remove it
                    new_start = None
                    new_end = None
                    removed_count += 1
                    break
                elif new_start < cut_start < new_end < cut_end:
                    # Cut removes beginning of segment
                    cut_from_start = cut_end - cut_start
                    new_end -= cut_from_start
                elif cut_start < new_start < cut_end < new_end:
                    # Cut removes end of segment  
                    cut_from_end = cut_end - cut_start
                    new_start -= cut_from_end
            
            # Apply insertions after cuts (cumulative shift calculation needed)
            if "insert_points" in offset_map:
                for insert_pos, insert_duration in offset_map["insert_points"]:
                    if new_start >= insert_pos:
                        new_start += insert_duration
                    if new_end > insert_pos:
                        new_end += insert_duration
        
        # Store adjusted segment if not fully removed
        if new_start is not None:
            adjusted_segments.append({
                **segment,
                "start": new_start,
                "end": new_end
            })
    
    return adjusted_segments, removed_count

def generate_audio_edit_script(segments, output_file):
    """
    Generates FFmpeg filter script for applying audio edits based on segments.
    Produces a standalone script executable with ffprobe/ffmpeg.
    
    Args:
        segments: List of adjusted segment dicts (from adjust_segment_offsets)
        output_file: Path to save the edit script (bash/FFmpeg filter format)
    
    Returns:
        str: Path to generated script file
    
    Note:
        Script includes validation checks for segment continuity and detects
        gaps/overlaps. Suitable for batch processing on Slurm cluster nodes.
    """
    if not segments:
        logger.warning("No segments provided for audio edit script.")
        return None
    
    # Validate segment continuity
    gaps = []
    overlaps = []
    for i in range(1, len(segments)):
        prev_end = segments[i-1]["end"]
        curr_start = segments[i]["start"]
        if curr_start > prev_end:
            gaps.append((segments[i-1]["speaker"], prev_end, curr_start))
        elif curr_start < prev_end:
            overlaps.append((segments[i-1]["speaker"], prev_end, curr_start))
    
    if gaps:
        logger.warning(f"Detected {len(gaps)} audio gaps between segments")
    if overlaps:
        logger.warning(f"Detected {len(overlaps)} overlapping segments")
    
    # Generate FFmpeg filter chains
    filter_chains = []
    for idx, segment in enumerate(segments):
        # Use atemporal filters to handle variable segment durations
        chain = (
            f"[0:a]atrim=start={segment['start']:.6f},end={segment['end']:.6f},"
            f"asetpts=PTS-STARTPTS[seg{idx}]"
        )
        filter_chains.append(chain)
    
    # Build concat operation
    concat_inputs = ",".join([f"[seg{i}]" for i in range(len(segments))])
    concat_filter = f"{concat_inputs}concat=n={len(segments)}:v=0:a=1[out]"
    
    # Write script with metadata header
    with open(output_file, "w") as f:
        f.write(f"# Audio Edit Script - Generated {datetime.now().isoformat()}\n")
        f.write(f"# Total segments: {len(segments)}\n")
        f.write(f"# Gaps detected: {len(gaps)}, Overlaps: {len(overlaps)}\n\n")
        f.write("# Input: ffmpeg -i input.wav -filter_complex \"FILTERS\" output.wav\n\n")
        for chain in filter_chains:
            f.write(f"# {chain}\n")
        f.write(f"# Final: ffmpeg -i input.wav -filter_complex \"{';'.join(filter_chains)}{concat_filter}\" output.wav\n")
    
    logger.info(f"Audio edit script saved to: {output_file}")
    return str(output_file)

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
    
    weight_extensions = ['.bin', '.safetensors', '.pt', '.pth']
    weight_files = [f for ext in weight_extensions for f in path.glob(f"*{ext}")]
    
    if not weight_files:
        logger.error(f"WhisperX model missing weight files")
        return False
    
    logger.info(f"WhisperX model verified at: {model_path}")
    return True

def verify_diarization_model(model_path):
    """Verifies Pyannote Diarization model."""
    path = Path(model_path)
    if not path.exists():
        logger.error(f"Diarization model directory not found at: {model_path}")
        return False
    
    config_files = list(path.glob("config.*"))
    pyannote_yaml = path / "pyannote.yaml"
    if not config_files:
        logger.error(f"Diarization model missing config files")
        return False
    
    required_subdirs = ['embedding', 'plda', 'segmentation']
    missing_subdirs = [d for d in required_subdirs if not (path / d).exists()]
    
    if missing_subdirs:
        logger.warning(f"Diarization model missing subdirectories: {missing_subdirs}")
    
    logger.info(f"Diarization model verified at: {model_path}")
    return True

def convert_numpy(obj):
    """Converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    else:
        return obj

# --- Extract Audio ---

def process_videos(file_list=None):
    """Extracts audio from video files in VIDEOS_FOLDER."""
    if not VIDEOS_FOLDER.exists():
        logger.warning(f"No 'videos' folder found at {VIDEOS_FOLDER}. Skipping audio extraction.")
        return 0

    files_to_process = []
    
    if file_list:
        logger.info(f"User specified {len(file_list)} video file(s). Targeting specific sources...")
        for fname in file_list:
            fpath = VIDEOS_FOLDER / fname
            
            if not fpath.exists():
                logger.error(f"Requested video '{fname}' not found in {VIDEOS_FOLDER}. Skipping.")
                continue
            
            if not validate_path(fpath, VIDEOS_FOLDER):
                logger.error(f"Security Alert: Path traversal detected for {fname}. Skipping.")
                continue
            
            if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported format: {fname} ({fpath.suffix})")
                continue
                
            files_to_process.append(fpath)
        
        if not files_to_process:
            logger.warning("No valid video files found for the specified inputs.")
            return 0
            
        logger.info(f"Processing {len(files_to_process)} specific video file(s).")
    else:
        logger.info("Scanning 'videos' folder for supported files...")
        files_to_process = [
            f for f in VIDEOS_FOLDER.iterdir() 
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        if not files_to_process:
            logger.info("No supported video files found in VIDEOS_FOLDER.")
            return 0
            
        logger.info(f"Found {len(files_to_process)} video files to process.")

    if not AUDIOS_FOLDER.exists():
        AUDIOS_FOLDER.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    for idx, file in enumerate(files_to_process, 1):
        base_name = sanitize_filename(file.stem)
        audio_filename = f"{base_name}.wav"
        audio_path = AUDIOS_FOLDER / audio_filename

        if audio_path.exists():
            logger.info(f"[{idx}/{len(files_to_process)}] Audio already exists: {audio_filename}. Skipping.")
            continue

        logger.info(f"[{idx}/{len(files_to_process)}] Processing video: {file.name} -> {audio_filename}")
        
        try:
            subprocess.run([
                'ffmpeg', '-i', str(file),
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(audio_path)
            ], check=True, capture_output=True, text=True)
            
            processed_count += 1
            logger.info(f"   ✅ Successfully saved: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"   ❌ FFmpeg error processing {file.name}: {e.stderr}")
        except Exception as e:
            logger.error(f"   ❌ Unexpected error processing {file.name}: {e}")

    if processed_count == 0:
        logger.info("No new files processed.")
    else:
        logger.info(f"Audio extraction complete: {processed_count} file(s) processed.")
        
    return processed_count

def process_audios(enable_diarization=True, lang_code=None, file_list=None, args=None):
    """Process audio files: transcribe and optionally diarize."""
    file_start = time.time()
    force_language = lang_code
    
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

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        model = whisperx.load_model(
            str(WHISPERX_MODEL_PATH), 
            device, 
            compute_type=compute_type, 
            local_files_only=True
        )
        logger.info("WhisperX model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return

    if enable_diarization:
        try:
            from pyannote.audio import Pipeline
            diarize_model = Pipeline.from_pretrained(str(DIARIZATION_MODEL_PATH))
            logger.info("Diarization Pipeline loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Diarization Pipeline: {e}. Disabling diarization.")
            diarize_model = None
    else:
        diarize_model = None

    files_to_process = []
    if file_list:
        files_to_process = [Path(f) for f in file_list]
    elif getattr(args, 'file', None):
        files_to_process = [Path(args.file)]
    elif getattr(args, 'files', None):
        files_to_process = [Path(f) for f in args.files]
    else:
        files_to_process = [f for f in AUDIOS_FOLDER.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]

    if not files_to_process:
        logger.info("No files found to process.")
        return

    logger.info(f"Found {len(files_to_process)} files to process. Starting stream...")

    for idx, input_file in enumerate(files_to_process, 1):
        if not input_file.exists():
            logger.error(f"Requested file '{input_file.name}' not found at {input_file}. Skipping.")
            continue
        
        if input_file.suffix.lower() in SUPPORTED_EXTENSIONS and input_file.suffix.lower() != '.wav':
            if not validate_path(input_file, VIDEOS_FOLDER):
                logger.error(f"Security Alert: Skipping {input_file.name} (path traversal in videos folder).")
                continue
            source_folder = VIDEOS_FOLDER
        elif input_file.suffix.lower() == '.wav':
            if not validate_path(input_file, AUDIOS_FOLDER):
                logger.error(f"Security Alert: Skipping {input_file.name} (path traversal in audios folder).")
                continue
            source_folder = AUDIOS_FOLDER
        else:
            logger.warning(f"Skipping unsupported file format: {input_file.suffix}")
            continue

        logger.info(f"[{idx}/{len(files_to_process)}] Processing: {input_file.name}")
        file_start = time.time()
        transcript_file = None      # defined up-front so the ``finally`` block cannot fail

        actual_audio_path = input_file
        
        if input_file.suffix.lower() != '.wav':
            base_name = sanitize_filename(input_file.stem)
            actual_audio_path = AUDIOS_FOLDER / f"{base_name}.wav"
            
            if not actual_audio_path.exists():
                logger.info(f"   Converting {input_file.name} -> {actual_audio_path.name}")
                try:
                    subprocess.run([
                        'ffmpeg', '-i', str(input_file),
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-y',
                        str(actual_audio_path)
                    ], check=True, capture_output=True, text=True)
                    logger.info(f"   Conversion successful.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"   FFmpeg error: {e.stderr}")
                    continue
                except Exception as e:
                    logger.error(f"   Unexpected conversion error: {e}")
                    continue
            else:
                logger.info(f"   Audio file already exists: {actual_audio_path.name}")
        
        try:
            audio = whisperx.load_audio(str(actual_audio_path))
            
            transcribe_kwargs = {
                "audio": audio,
                "batch_size": BATCH_SIZE,
                "verbose": False,
                "print_progress": False,
                "task": "transcribe"
            }
            if whisper_code:
                transcribe_kwargs["language"] = whisper_code
            
            result = model.transcribe(**transcribe_kwargs)
            
            if result.get("language"):
                try:
                    align_device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=align_device)
                    result = whisperx.align(result["segments"], model_a, metadata, audio, align_device, return_char_alignments=False)
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}")

            if enable_diarization and diarize_model:
                try:
                    diarize_output = diarize_model(str(actual_audio_path))
                    speaker_diarization = diarize_output.speaker_diarization
                    
                    segments_list = []
                    for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                        duration = turn.end - turn.start
                        if duration < 0.5: continue
                        segments_list.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
                    
                    if segments_list:
                        import pandas as pd
                        diarize_df = pd.DataFrame(segments_list)
                        result = whisperx.assign_word_speakers(diarize_df, result)
                    else:
                        logger.warning(f"   No valid speakers found. Using fallback.")
                        for i, seg in enumerate(result["segments"]):
                            seg["speaker"] = f"SPEAKER_{i%2:02d}"
                except Exception as e:
                    logger.error(f"   Diarization failed: {e}. Using fallback.")
                    for i, seg in enumerate(result["segments"]):
                        seg["speaker"] = f"SPEAKER_{i%2:02d}"
            else:
                for i, seg in enumerate(result["segments"]):
                    seg["speaker"] = f"SPEAKER_{i%2:02d}"

            result["segments"] = merge_consecutive_speaker_segments(
                result["segments"], 
                max_gap_seconds=SEGMENT_MERGE_THRESHOLD
            )
            
            base_name = sanitize_filename(input_file.stem)
            transcript_file = TRANSCRIPTS_FOLDER / f"{base_name}.txt"
            
            with open(transcript_file, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        f.write(f"{segment.get('speaker', 'Unknown')}: {text}\n")

            # Word-level timeline for audio redaction (same sensitivity as the transcript).
            redaction_cli.write_words_sidecar(
                TRANSCRIPTS_FOLDER / f"{base_name}{redaction_cli.WORDS_SUFFIX}", result["segments"])
            
        except Exception as e:
            logger.error(f"❌ FAILED: {input_file.name} - {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            logger.info(f"✅ COMPLETED: {input_file.name} -> {transcript_file.name if transcript_file else 'n/a'}")
            logger.info(f"   Duration: {time.time() - file_start:.2f}s")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.1)
    
    # AFTER the loop completes:
    cleanup_gpu_resources()
    logger.info("Stream processing finished.")
    return len(files_to_process)
    
# --- Model management & transcription ---------------------------------------------

#: UI/CLI model key -> folder name under MODEL_FOLDER.
WHISPER_MODEL_DIRS = {
    'tiny': 'Systran--faster-whisper-tiny',
    'base': 'Systran--faster-whisper-base',
    'small': 'Systran--faster-whisper-small',
    'medium': 'Systran--faster-whisper-medium',
    'large': 'Systran--faster-whisper-large-v3',
    'large-turbo': 'Systran--faster-whisper-large-v3-turbo',
}
_MODEL_ALIASES = {'large-v3': 'large', 'large_v3': 'large', 'large-v3-turbo': 'large-turbo'}

_MODEL_LOCK = threading.RLock()          # guards loading/switching of the cached models
# GPU inference is serialised by default; raise ATA_MAX_CONCURRENT_TRANSCRIPTIONS on big GPUs.
_INFER_SEMAPHORE = threading.BoundedSemaphore(
    max(1, int(os.getenv('ATA_MAX_CONCURRENT_TRANSCRIPTIONS', '1'))))
_QUEUE_TIMEOUT_S = float(os.getenv('ATA_QUEUE_TIMEOUT', '900'))

_loaded_whisper_model = None
_loaded_whisper_name = None
_loaded_diarize_model = None
_diarize_attempted = False


class TranscriptionError(RuntimeError):
    """Expected transcription failure. ``code`` is a stable machine-readable key."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class TranscriptionResult:
    """Structured outcome of :func:`transcribe_audio_result`.

    Success and failure are signalled *only* through ``ok``/``error_code``, never
    by inspecting ``text``: a real transcript that happens to contain the word
    "failed" or "Error:" is a perfectly valid result.
    """

    ok: bool
    text: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)   # incl. word timestamps
    language: Optional[str] = None
    duration: Optional[float] = None
    error_code: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def normalize_model_name(name=None):
    """Returns a key of ``WHISPER_MODEL_DIRS`` for ``name`` (default: $WHISPER_MODEL or 'large').

    Raises:
        ValueError: Unknown model name.
    """
    key = str(name or os.getenv('WHISPER_MODEL') or 'large').strip().lower()
    key = _MODEL_ALIASES.get(key, key)
    if key not in WHISPER_MODEL_DIRS:
        raise ValueError(f"Unknown Whisper model '{name}'. Valid: {', '.join(WHISPER_MODEL_DIRS)}")
    return key


def resolve_language(language):
    """Converts a UI/CLI language value to a WhisperX code (case-insensitive).

    Accepts ``None``/``''``/``'auto'`` (-> ``None`` = auto-detect), codes such as
    ``'DE'``/``'de'``/``'SP'`` and native WhisperX codes.

    Raises:
        ValueError: Unsupported language.
    """
    text = '' if language is None else str(language).strip()
    if not text or text.lower() in ('auto', 'autodetect', 'auto-detect'):
        return None
    if text.upper() in WHISPER_LANG_MAP:
        return WHISPER_LANG_MAP[text.upper()]
    if text.lower() in set(WHISPER_LANG_MAP.values()):
        return text.lower()
    raise ValueError(f"Unsupported language '{language}'.")


def _free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_models(model_name=None, explicit=False):
    """Loads (or reuses) the Whisper and diarization models.

    Args:
        model_name: Key of ``WHISPER_MODEL_DIRS`` or None for the default.
        explicit: True if the caller asked for this model by name; a missing
            model is then an error instead of silently using another one.

    Returns:
        ``(whisper_model, diarization_model_or_None)``.

    Raises:
        TranscriptionError: Model missing or failed to load.
    """
    global _loaded_whisper_model, _loaded_whisper_name, _loaded_diarize_model, _diarize_attempted
    with _MODEL_LOCK:
        name = normalize_model_name(model_name)
        path = MODEL_FOLDER / WHISPER_MODEL_DIRS[name]
        if not path.exists():
            if explicit or not WHISPERX_MODEL_PATH.exists():
                raise TranscriptionError('model_unavailable', f"Whisper model '{name}' is not installed.")
            logger.warning(f"Whisper model '{name}' not found; using {WHISPERX_MODEL_PATH.name}.")
            path, name = WHISPERX_MODEL_PATH, 'default'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"

        if _loaded_whisper_model is None or _loaded_whisper_name != name:
            if _loaded_whisper_model is None:
                check_gpu_resources()
            logger.info(f"Loading WhisperX model '{name}' from: {path}")
            _loaded_whisper_model = None          # release the old model before loading
            _free_gpu_memory()
            try:
                _loaded_whisper_model = whisperx.load_model(
                    str(path), device, compute_type=compute_type, local_files_only=True)
                _loaded_whisper_name = name
            except Exception as e:
                logger.error(f"Failed to load WhisperX model: {e}")
                _loaded_whisper_name = None
                raise TranscriptionError('model_load_failed', 'The Whisper model could not be loaded.')
            logger.info("✅ WhisperX model loaded successfully.")

        if not _diarize_attempted:                # diarization is loaded once, independent of Whisper
            _diarize_attempted = True
            if DIARIZATION_MODEL_PATH and DIARIZATION_MODEL_PATH.exists():
                try:
                    logger.info(f"Loading Diarization Pipeline from: {DIARIZATION_MODEL_PATH}")
                    from pyannote.audio import Pipeline
                    # pyannote disables TF32 on import; re-enable for speed on Ampere GPUs.
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    pipeline = Pipeline.from_pretrained(str(DIARIZATION_MODEL_PATH))
                    if pipeline is not None:
                        if device == "cuda":
                            pipeline.to(torch.device(device))
                        else:
                            logger.warning("⚠️  Diarization Pipeline loaded on CPU (will be slower).")
                        pipeline.min_duration_on = 3.0
                        pipeline.min_duration_off = 4.0
                    _loaded_diarize_model = pipeline
                except Exception as e:
                    logger.error(f"Failed to load Diarization Pipeline: {e}")
                    _loaded_diarize_model = None
            else:
                logger.warning("⚠️  Diarization model not found. Speaker diarization will be disabled.")
        return _loaded_whisper_model, _loaded_diarize_model


def load_models(whisper_model_override=None):
    """Loads/reuses the models; kept for backwards compatibility.

    Unlike earlier versions the cache is no longer cleared on every call, and the
    override is honoured. Returns ``(None, None)`` on failure.
    """
    try:
        return _ensure_models(whisper_model_override, explicit=whisper_model_override is not None)
    except (TranscriptionError, ValueError) as e:
        logger.critical(f"Model loading failed: {e}")
        return None, None


def _fail(code, message):
    return TranscriptionResult(ok=False, error_code=code, error=message)


def transcribe_audio_result(audio_path, language=None, whisper_model_name=None, max_duration=None):
    """Transcribes one file and returns a :class:`TranscriptionResult` (never raises
    for expected failures).

    Args:
        audio_path: Path of an audio/video file (validated with ffprobe first).
        language: ``None``/``'auto'`` or a code such as ``'DE'``/``'de'``.
        whisper_model_name: Model key (``'tiny'``..``'large'``, ``'large-turbo'``) or None.
        max_duration: Override of ``ATA_MAX_AUDIO_SECONDS`` (0 disables).

    Returns:
        ``TranscriptionResult``. ``error_code`` is one of ``invalid_language``,
        ``invalid_model``, ``unreadable_media``, ``unsupported_format``,
        ``media_too_long``, ``model_unavailable``, ``model_load_failed``, ``busy``
        or ``transcription_error``.
    """
    try:
        lang_code = resolve_language(language)
    except ValueError as e:
        return _fail('invalid_language', str(e))
    try:
        model_name = normalize_model_name(whisper_model_name) if whisper_model_name else None
    except ValueError as e:
        return _fail('invalid_model', str(e))

    duration = None
    if probe_media is not None:                     # reject hostile/oversized media early
        try:
            duration = probe_media(str(audio_path), max_duration).duration
        except AudioError as e:
            return _fail(e.code, e.message)

    if not _INFER_SEMAPHORE.acquire(timeout=_QUEUE_TIMEOUT_S):
        return _fail('busy', 'The server is busy. Please try again shortly.')
    try:
        whisper_model, diarize_model = _ensure_models(model_name, explicit=whisper_model_name is not None)
        audio = whisperx.load_audio(str(audio_path))
        transcribe_kwargs = {"audio": audio, "batch_size": BATCH_SIZE, "verbose": False,
                             "print_progress": False, "task": "transcribe"}
        if lang_code:
            transcribe_kwargs["language"] = lang_code
            logger.info(f"Forced language: {lang_code}")
        result = whisper_model.transcribe(**transcribe_kwargs)

        detected_language = result.get("language") or lang_code or 'en'
        align_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:                                         # word alignment (needed for redaction)
            align_model, metadata = whisperx.load_align_model(language_code=detected_language,
                                                              device=align_device)
            result = whisperx.align(result["segments"], align_model, metadata, audio,
                                    align_device, return_char_alignments=False)
        except Exception as e:
            logger.warning(f"Alignment failed: {e}")

        if diarize_model:
            try:
                diarization_audio = {'waveform': torch.from_numpy(audio).unsqueeze(0), 'sample_rate': 16000}
                diarize_output = diarize_model(diarization_audio, min_speakers=2, max_speakers=10)
                segments_list = [
                    {'start': turn.start, 'end': turn.end, 'speaker': speaker}
                    for turn, _, speaker in diarize_output.speaker_diarization.itertracks(yield_label=True)
                    if turn.end - turn.start >= 0.5]
                if segments_list:
                    import pandas as pd
                    result = whisperx.assign_word_speakers(pd.DataFrame(segments_list), result)
                    logger.info(f"✅ Diarization complete: {len(segments_list)} speaker segments assigned")
                else:
                    logger.warning("No valid speakers found. Using fallback assignment.")
                    for i, seg in enumerate(result["segments"]):
                        seg["speaker"] = f"SPEAKER_{i % 2:02d}"
            except Exception as e:
                logger.error(f"Diarization failed: {e}. Using fallback assignment.")
                for i, seg in enumerate(result["segments"]):
                    seg["speaker"] = f"SPEAKER_{i % 2:02d}"
        else:
            logger.warning("Diarization disabled. Using fallback speaker assignment.")
            for i, seg in enumerate(result["segments"]):
                seg.setdefault("speaker", f"SPEAKER_{i % 2:02d}")

        segments = merge_consecutive_speaker_segments(
            result["segments"], max_gap_seconds=globals().get('SEGMENT_MERGE_THRESHOLD', 2.0))
        lines = [f"{seg.get('speaker', 'SPEAKER_00')}: {seg.get('text', '').strip()}"
                 for seg in segments if seg.get('text', '').strip()]
        warnings = [] if lines else ['no_speech_detected']   # empty text is valid, not an error
        segments = convert_numpy(segments)
        if duration is None and segments:
            duration = float(segments[-1].get('end') or 0.0)
        logger.info(f"Transcription complete: {len(lines)} segment(s)")
        return TranscriptionResult(True, "\n".join(lines), segments, detected_language, duration,
                                   warnings=warnings)
    except TranscriptionError as e:
        return _fail(e.code, e.message)
    except Exception:
        logger.exception("Transcription failed")
        return _fail('transcription_error', 'Transcription failed because of an internal error.')
    finally:
        _INFER_SEMAPHORE.release()
        _free_gpu_memory()


def transcribe_audio_locally(audio_path, language=None, whisper_model_name=None):
    """Legacy wrapper returning ``(text, segments)``.

    Failures raise :class:`TranscriptionError` instead of returning an error
    string, so transcript content is never mistaken for an error message.
    """
    res = transcribe_audio_result(audio_path, language, whisper_model_name)
    if not res.ok:
        raise TranscriptionError(res.error_code, res.error)
    return res.text, res.segments


# --- Anonymization Engine (V2-style, simplified with DFKI-SLT model) ---

_nlp_multilingual = None

def get_nlp_pipeline(lang_code=None):
    """Loads or retrieves the SpaCy pipeline for sentence splitting."""
    global _nlp_multilingual
    
    if _nlp_multilingual is None:
        try:
            _nlp_multilingual = spacy.blank("xx")
            _nlp_multilingual.add_pipe("sentencizer")
            logger.info("Loaded SpaCy multilingual sentencizer.")
        except Exception as e:
            logger.error(f"Failed to load SpaCy 'xx' model: {e}. Falling back to regex splitting.")
            return None
    return _nlp_multilingual

def split_dialogue_into_sentences(text, nlp_pipeline=None):
    """Splits dialogue text into sentences (each a list of tokens)."""
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

def merge_adjacent_tags(text):
    """Merges adjacent duplicate tags to prevent double-replacement."""
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
    out_str = re.sub("  *", " ", out_str)
    return out_str
    
def reconstruct_text_from_predictions(original_sentences, predictions, speaker_map):
    """Reconstructs the text from predictions, merging consecutive same-speaker sentences.
        Returns:
        tuple: (reconstructed_text, entity_occurrences)
        entity_occurrences = {tag: [original_text, ...]} in document order.
        Consecutive words with the same tag are merged into single occurrences.
        Enables downstream surrogate substitution to map identical entities
        consistently (e.g., "Max Mustermann" → "[PERSON_001]" everywhere).
    """
    reconstructed_blocks = []
    entity_occurrences = {}
    
    current_speaker = None
    current_text_parts = []

    for i, (speaker, tokens) in enumerate(original_sentences):
        labels = predictions[i] if i < len(predictions) else []
        reconstructed_words = []

        # Align tokens with labels, merging consecutive same-tag words
        w_idx = 0
        n_tokens = len(tokens)
        while w_idx < n_tokens:
            tag = labels[w_idx] if w_idx < len(labels) else "O"

            if not tag or tag == "O":
                reconstructed_words.append(tokens[w_idx])
                w_idx += 1
                continue

            # Merge consecutive words with same tag into single entity occurrence
            span_words = [tokens[w_idx]]
            j = w_idx + 1
            while j < n_tokens and (labels[j] if j < len(labels) else "O") == tag:
                span_words.append(tokens[j])
                j += 1

            # Track original entity text for surrogate mapping
            entity_occurrences.setdefault(tag, []).append(" ".join(span_words))
            reconstructed_words.append(tag)
            w_idx = j

        sentence_text = " ".join(reconstructed_words)
        
        # Merge consecutive sentences from same speaker
        if speaker == current_speaker:
            current_text_parts.append(sentence_text)
        else:
            if current_speaker and current_text_parts:
                full_text = " ".join(current_text_parts)
                reconstructed_blocks.append(f"{current_speaker}: {full_text}")
            current_speaker = speaker
            current_text_parts = [sentence_text]

    # Append final block
    if current_speaker and current_text_parts:
        full_text = " ".join(current_text_parts)
        reconstructed_blocks.append(f"{current_speaker}: {full_text}")
        
    return "\n".join(reconstructed_blocks), entity_occurrences

def predict_sentences_simple(sentences_tokens, model, tokenizer, id_to_tag_map, device="cpu"):
    """Simple sentence-level NER prediction without FLERT context windowing."""
    all_predictions = []
    
    for tokens in sentences_tokens:
        if not tokens:
            all_predictions.append([])
            continue
            
        enc = tokenizer(
            tokens, is_split_into_words=True,
            return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        
        word_ids = enc.word_ids(batch_index=0)
        
        with torch.no_grad():
            outputs = model(**enc)
            emissions = outputs["logits"]
            mask = enc["attention_mask"].bool()
            preds = model.decode(emissions, mask)[0]
        
        word_labels = ["O"] * len(tokens)
        seen = set()
        
        for idx, wid in enumerate(word_ids):
            if wid is None or wid in seen or wid >= len(tokens):
                continue
            seen.add(wid)
            
            pred_id = preds[idx]
            key = str(pred_id)
            tag = id_to_tag_map.get(key, "O")
            word_labels[wid] = tag
        
        all_predictions.append(word_labels)
        
    return all_predictions

#: Model tag -> placeholder written into the anonymized text ('' = leave text unchanged).
TAG_PLACEHOLDERS = {
    'PERSON': '[PERSON]', 'PERSON_EMAIL': '[EMAIL]',
    'PERSON_SOCIAL_RELATION': '[NAME_RELATIVE]', 'ORG': '[ORGANISATION]',
    'LOC_CITY': '[CITY]', 'LOC_COUNTRY': '[COUNTRY]',
    'LOC_STREET': '[STREET]', 'LOC_ZIP': '[ZIP]',
    'LOC_HOUSENUMBER': '[HOUSENUMBER]', 'LOC_OTHER': '[LOCATION]',
    'DATETIME': '[DATETIME]', 'DATETIME_AGE': '[AGE]',
    'CODE': '[CODE]', 'CODE_PHONE': '[PHONE]', 'CODE_URL': '[URL]',
    'PROFESSION': '[PROFESSION]', 'PRODUCT': '[PRODUCT]',
    'QUANTITY': '[QUANTITY]', 'MISC': '[MISC]', 'O': ''
}
_ANON_LOCK = threading.Lock()          # the shared model must not run concurrent inference


class AnonymizationEngine:
    """Streamlined AnonymizationEngine compatible with DFKI-SLT/multilingual_DialogPII_NER."""
    
    def __init__(self, method="local_mmbert", level="standard", model_path=None, 
                 include_tags=None, exclude_tags=None):
        self.method = method
        self.level = level
        self.model_path = model_path or (MODEL_FOLDER / "multilingual_DialogPII_NER")
        self.include_tags = include_tags
        self.exclude_tags = exclude_tags
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = None
        self.label_mapping = {}
        self.original_id2label = {}
        
        if ANONYMIZATION_ENABLED:
            logger.info(f"AnonymizationEngine initialized: Method={self.method}, Level={self.level}")
            logger.info(f"Model path: {self.model_path}")
            self._load_model()

    @staticmethod
    def _active_tags(include_tags=None, exclude_tags=None):
        """Set of model tags to anonymize after applying include/exclude filters."""
        active = set(TAG_PLACEHOLDERS) - {'O'}
        if include_tags is not None:          # [] means "none" (e.g. the UI's Clear All)
            active &= {str(t).upper() for t in include_tags}
        if exclude_tags:
            active -= {str(t).upper() for t in exclude_tags}
        return active

    def _build_safe_label_mapping(self, include_tags=None, exclude_tags=None):
        """Maps model label ids to placeholders; filtered-out tags map to '' (text kept).

        Previously the filter was computed and then ignored, so the PII checkboxes
        and --include-tags/--exclude-tags had no effect.
        """
        active = self._active_tags(include_tags, exclude_tags)
        logger.info(f"Anonymizing tags: {sorted(active)}")
        mapping = {}
        for label_id, label_name in self.original_id2label.items():
            clean = re.sub(r"^[BIES]-", "", str(label_name)).upper()
            mapping[str(label_id)] = TAG_PLACEHOLDERS.get(clean, "") if clean in active else ""
        logger.info(f"Built label mapping with {len(mapping)} entries.")
        return mapping

    def _load_model(self):
        """Loads the multilingual_DialogPII_NER model with proper CRF support."""
        
        # Validate required modules early
        required_modules = {'transformers': 'AutoModel, AutoTokenizer', 'torchcrf': 'CRF'}

        for module, feature in required_modules.items():
            try:
                __import__(module)
            except ImportError:
                print(f"ERROR: {module} module not found.")
                print(f"To install: pip install {'pytorch-crf' if module == 'torchcrf' else module}")
                sys.exit(1)

        # Now import safely
        from transformers import AutoModel, AutoTokenizer
        from torchcrf import CRF

        if not self.model_path.exists():
            logger.error(f"Model path not found: {self.model_path}")
            logger.error(f"Available folders in MODEL_FOLDER: {list(MODEL_FOLDER.iterdir()) if MODEL_FOLDER.exists() else 'Folder missing'}")
            self.method = None
            return

        try:
            import torch.nn as nn
            import json
            
            crf_config_path = self.model_path / "crf_config.json"
            if not crf_config_path.exists():
                logger.error(f"crf_config.json not found at {crf_config_path}")
                logger.error(f"Folder contents: {list(self.model_path.iterdir())}")
                self.method = None
                return
            
            with open(crf_config_path, "r") as f:
                self.config = json.load(f)
            
            required_keys = ["base_model_name", "num_labels", "id2label", "label2id"]
            if not all(k in self.config for k in required_keys):
                logger.error(f"Missing required keys in crf_config.json: {required_keys}")
                self.method = None
                return

            self.original_id2label = self.config.get("id2label", {})
            self.original_label2id = self.config.get("label2id", {})
            logger.info(f"Loaded {len(self.original_id2label)} original model labels.")

            class ModernBertCRF(nn.Module):
                def __init__(self, base_model_name, num_labels, id2label, label2id):
                    super().__init__()
                    self.num_labels = num_labels
                    self.id2label = id2label
                    self.label2id = label2id
                    self.transformer = AutoModel.from_pretrained(base_model_name, local_files_only=True)
                    hidden_size = self.transformer.config.hidden_size
                    self.classifier = nn.Linear(hidden_size, num_labels)
                    self.dropout = nn.Dropout(0.1)
                    self.crf = CRF(num_labels, batch_first=True)

                def forward(self, input_ids, attention_mask, labels=None, **kwargs):
                    kwargs.pop("token_type_ids", None)
                    outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                    sequence_output = self.dropout(outputs.last_hidden_state)
                    emissions = self.classifier(sequence_output)
                    if labels is not None:
                        return {"loss": None, "logits": emissions}
                    else:
                        return {"logits": emissions}

                def decode(self, emissions, mask):
                    return self.crf.decode(emissions, mask=mask)

            local_base_model_path = MODEL_FOLDER / self.config["base_model_name"]
            if not local_base_model_path.exists():
                logger.warning(f"Local base model not found. Trying HF name: {self.config['base_model_name']}")
                local_base_model_path = self.config["base_model_name"]
            
            self.model = ModernBertCRF(
                base_model_name=local_base_model_path,
                num_labels=self.config["num_labels"],
                id2label=self.original_id2label,
                label2id=self.original_label2id
            )
            
            model_weights_path = self.model_path / "pytorch_model.bin"
            if not model_weights_path.exists():
                logger.error(f"Model weights not found at {model_weights_path}")
                self.method = None
                return
                
            state_dict = torch.load(model_weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            self.label_mapping = self._build_safe_label_mapping(self.include_tags, self.exclude_tags)
            self._mapping_cache = {}
            logger.info("Model loaded and label mapping initialized.")

        except Exception as e:
            logger.error(f"Failed to load mmbert model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.method = None

    def anonymize(self, text, use_surrogates=False, surrogate_seed=None,
                  surrogate_locales='de_DE,en_US', include_tags=None, exclude_tags=None):
        """
        Anonymizes text using simple sentence-level splitting.

        Args:
            text: Transcript with ``SPEAKER_nn:`` lines.
            use_surrogates: Replace placeholders with realistic fake values.
            surrogate_seed: Seed for reproducible surrogates.
            surrogate_locales: Locale list, e.g. ``'de_DE,en_US'``; the first one selects the language.
            include_tags / exclude_tags: Per-call tag filters (model tag names such as
                ``'PERSON'``, ``'LOC_CITY'``); they override the engine-level filters.

        Returns:
            Always a 5-tuple ``(text, success, message, entity_map, surrogate_registry)``.
            Earlier versions returned 4 values on error paths, which broke callers
            that unpack 5. ``entity_map`` holds the *original* strings per tag in
            document order (sensitive); ``surrogate_registry`` holds fake values only.
        """
        if not self.method or not self.model or not self.tokenizer:
            return None, False, "Anonymization model not loaded.", {}, {}

        mapping = self.label_mapping
        if include_tags is not None or exclude_tags is not None:
            key = (frozenset(include_tags or ()), frozenset(exclude_tags or ()))
            if key not in self._mapping_cache:
                self._mapping_cache[key] = self._build_safe_label_mapping(include_tags, exclude_tags)
            mapping = self._mapping_cache[key]

        sentences_data = split_dialogue_into_sentences(text)
        if not sentences_data:
            return text, False, "No sentences detected.", {}, {}

        sentences_tokens = [tokens for _, tokens in sentences_data]
        try:
            with _ANON_LOCK:
                predictions = predict_sentences_simple(
                    sentences_tokens=sentences_tokens, model=self.model, tokenizer=self.tokenizer,
                    id_to_tag_map=mapping, device=self.device)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return text, False, str(e), {}, {}

        reconstructed_text, entity_map = reconstruct_text_from_predictions(sentences_data, predictions, {})
        reconstructed_text = normalize_punctuation(reconstructed_text)
        reconstructed_text = merge_adjacent_tags(reconstructed_text)

        surrogate_registry = {}
        if use_surrogates:
            reconstructed_text, surrogate_registry = apply_surrogate_substitution(
                reconstructed_text, entity_map, use_surrogates=True,
                seed=surrogate_seed, locales=surrogate_locales)
        return reconstructed_text, True, "Success", entity_map, surrogate_registry

# --- LLM Rewrite Features ---

def generate_paraphrase(raw_dialogue, lang='DE', model="gpt-oss-120b", temperature=0.3):
    """Rephrase/anonymize a dialogue using a local LM Studio model."""
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
    """Calls the external LLM API to rewrite text with robust output parsing."""
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
            messages=messages, model=final_model,
            stream=False, temperature=0.3, max_tokens=16384
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
                        rewritten_text = last_para.strip() if 'SPEAKER_' in last_para else last_para.strip()
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
        logger.error(f"LLM Rewriter failed for model {final_model}: {type(e).__name__}")
        if hasattr(e, 'response'):
            logger.error(f"  HTTP status: {e.response.status_code}")
            logger.error(f"  Endpoint: {CHAT_AI_ENDPOINT}")
            logger.error(f"  Response: {e.response.text[:200]}")
        elif hasattr(e, 'request'):
            logger.error(f"  Endpoint: {CHAT_AI_ENDPOINT}")
            logger.error(f"  Request error: {e.request}")
        else:
            logger.error(f"  Endpoint: {CHAT_AI_ENDPOINT}")
            logger.error(f"  Error details: {str(e)[:200]}")
        return None, str(e)


# --- ADVERSARIAL ANONYMIZATION ---

def run_adversarial_anonymization(text, model_id, iterations=3):
    """
    Runs a dual-agent adversarial loop:
    1. Defender (Anonymizer): Removes PII.
    2. Attacker (Re-identifier): Attempts to find PII or infer attributes.
    3. Defender iterates based on Attacker's critique.
    
    Args:
        text: The input transcript (already BERT-anonymized or raw).
        model_id: The LLM model to use for both agents.
        iterations: Number of red/blue team cycles (default 3).
        
    Returns:
        tuple: (final_text, iteration_log)
    """
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
        "5. CRITICAL: Preserve speaker tags (SPEAKER_XX, etc.) and original line breaks exactly.\n"
        "6. Do NOT translate the text. Keep the original language.\n"
        "7. If the text is already well-anonymized, make minimal adjustments only where the Attacker pointed out flaws.\n"
        "Return ONLY the anonymized text."
    )

    ATTACKER_SYSTEM_PROMPT = (
        "You are a Privacy Red Team AI. Your goal is to re-identify individuals or infer sensitive attributes from the provided text.\n"
        "Task:\n"
        "1. Analyze the text for any remaining Direct Identifiers (names, emails, phones).\n"
        "2. Analyze for Indirect Identifiers (combinations of age, location, job, specific events that could uniquely identify someone).\n"
        "3. Attempt to infer attributes not explicitly stated but implied (e.g., 'I live near the Eiffel Tower' -> inferred City: Paris).\n"
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
        
        defender_messages = [
            {"role": "system", "content": DEFENDER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please anonymize the following text:\n\n{text if i == 1 else current_text}"}
        ]
        
        if i > 1:
            last_critique = iteration_log[-1]["attacker_output"]
            defender_messages[1]["content"] += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS ROUND:\n{last_critique}\n\nAddress these specific points."

        try:
            defender_response = client.chat.completions.create(
                model=final_model, messages=defender_messages,
                temperature=0.3, max_tokens=16384
            )
            anonymized_text = defender_response.choices[0].message.content.strip()
            anonymized_text = re.sub(r'<[^>]+>', '', anonymized_text)
            anonymized_text = re.sub(r'[^\S\n]+', ' ', anonymized_text)
            logger.info(f"Defender completed iteration {i}. Length: {len(anonymized_text)}")

        except Exception as e:
            logger.error(f"Defender failed in iteration {i}: {e}")
            break

        attacker_messages = [
            {"role": "system", "content": ATTACKER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this text for re-identification risks:\n\n{anonymized_text}"}
        ]

        try:
            attacker_response = client.chat.completions.create(
                model=final_model, messages=attacker_messages,
                temperature=0.5, max_tokens=4096
            )
            critique = attacker_response.choices[0].message.content.strip()
            critique = re.sub(r'<[^>]+>', '', critique)
            logger.info(f"Attacker completed iteration {i}. Risks found: {'None' if 'no risks found' in critique.lower() else 'Yes'}")

        except Exception as e:
            logger.error(f"Attacker failed in iteration {i}: {e}")
            critique = "Error generating critique."

        iteration_log.append({
            "iteration": i,
            "defender_output": anonymized_text,
            "attacker_output": critique
        })

        current_text = anonymized_text

        if "no risks found" in critique.lower() and i < iterations:     # was mixed-case: never matched
            logger.info("Attacker found no risks. Stopping early.")
            break

    return current_text, iteration_log

def normalize_punctuation(text) -> str:
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

        sentences = re.split(r'([.!?])', content)
        final_parts = []
        capitalize_next = False
        
        for part in sentences:
            if part in ['.', '!', '?']:
                final_parts.append(part)
                capitalize_next = True
            elif part.strip():
                if capitalize_next:
                    if part[0].isalpha():
                        part = part[0].upper() + part[1:]
                    capitalize_next = False
                final_parts.append(part)
        
        content = "".join(final_parts)
        normalized_lines.append(f"{speaker}: {content}")

    return "\n".join(normalized_lines)

# ============================================================================
# SURROGATE POOLS FOR FAKE DATA GENERATION
# ============================================================================

# The static surrogate pools moved to pipeline/surrogates.py (``POOLS``). ``SURROGATES`` is kept
# as an import alias (see imports); its shape changed from {lang: {TAG: [...]}} to {lang: {kind: [...]}}.

# --- Surrogate Substitution ---

TAG_TO_SURROGATE_CATEGORY = {
    'PERSON': 'PERSON',
    'PERSON_EMAIL': 'EMAIL',
    'PERSON_SOCIAL_RELATION': 'PERSON',
    'ORG': 'ORGANISATION',
    'LOC_CITY': 'CITY',
    'LOC_COUNTRY': 'CITY',
    'LOC_STREET': 'STREET',
    'LOC_ZIP': 'ZIP',
    'LOC_HOUSENUMBER': 'ZIP',
    'LOC_OTHER': 'CITY',
    'DATETIME': 'AGE',
    'DATETIME_AGE': 'AGE',
    'CODE_PHONE': 'PHONE',
    'CODE_URL': 'URL',
    'PROFESSION': 'PROFESSION',
    'PRODUCT': 'PRODUCT',
}

def apply_surrogate_substitution(text, entity_map, use_surrogates=False, seed=None, locales='en_US'):
    """Replaces ``[TAG]`` placeholders with realistic fake values.

    Compatibility wrapper around :mod:`pipeline.surrogates` (Faker when installed,
    otherwise static pools). Same entity -> same fake; see that module for details.

    Args:
        text: Anonymized text containing placeholders.
        entity_map: ``{tag: [original strings in document order]}`` from the anonymizer
            (enables consistency across mentions); may be empty.
        use_surrogates: No-op unless True.
        seed: Seed for reproducible output.
        locales: Locale list such as ``'de_DE,en_US'``; the first entry selects the language.

    Returns:
        ``(text, registry)`` where ``registry`` maps tag -> list of fake values
        (never contains the originals).
    """
    if not use_surrogates or not text or '[' not in text:
        return text, {}
    result = apply_surrogates(text, entity_map, lang=lang_from_locales(locales), seed=seed)
    for warning in result.warnings:
        logger.warning(f"Surrogate note: {warning}")
    logger.info(f"Applied surrogate substitution: {result.replaced} placeholder(s) replaced "
                f"(backend={result.backend})")
    return result.text, result.registry


_MASTER_KEY = None


def _save_entity_map(base_name, entity_map):
    """Persists the entity map (it contains the ORIGINAL PII!) - only when requested.

    Stored in a 0700 folder with 0600 files, encrypted (AES-256-GCM) when a persistent
    master key is configured, so the file cannot sit unprotected next to the shareable
    anonymized output as it did before. Returns the path written.
    """
    global _MASTER_KEY
    if _MASTER_KEY is None:
        _MASTER_KEY = load_master_key(required=False)
    safe = sanitize_filename(base_name)
    folder = ensure_private_dir(ENTITYMAP_FOLDER)
    payload = json.dumps(entity_map, indent=2, ensure_ascii=False).encode("utf-8")
    if not _MASTER_KEY.ephemeral:
        path = folder / f"{safe}_entitymap.json.enc"
        write_private_file(path, encrypt_bytes(payload, _MASTER_KEY.key, b"ATA:entitymap:" + safe.encode()))
    else:
        path = folder / f"{safe}_entitymap.json"
        write_private_file(path, payload)
        logger.warning("No master key configured: entity map saved UNENCRYPTED (mode 0600). "
                       "Configure secrets/encryption_key.txt to encrypt it.")
    return path


def process_anonymization(llm_rewrite_enabled=None, llm_model_id=None, skip_bert=False, 
                          adversarial_mode=False, include_tags=None, exclude_tags=None, file_list=None,
                          enable_surrogates=False, surrogate_seed=None,
                          surrogate_locales='de_DE,en_US', save_entity_map=False):
    """
    Reads raw transcripts, anonymizes them with BERT, and optionally rewrites with LLM.
    Supports adversarial_mode which runs a 3-iteration Red Team vs. Blue Team loop.
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
    
    files_to_process = []
    
    if file_list:
        logger.info(f"User specified {len(file_list)} file(s). Targeting specific transcripts...")
        for fname in file_list:
            base_name = Path(fname).stem
            target_name = f"{base_name}.txt"
            fpath = source_folder / target_name
            
            if not fpath.exists():
                logger.error(f"Requested transcript '{target_name}' not found in {source_folder}. Skipping.")
                continue
            if not validate_path(fpath, source_folder):
                logger.error(f"Security Alert: Path traversal detected for {target_name}. Skipping.")
                continue
            
            if skip_bert:
                if "_llm" in target_name or "_adversarial_" in target_name:
                    logger.info(f"Skipping {target_name} as it appears already processed by LLM.")
                    continue
            
            files_to_process.append(fpath)
        
        if not files_to_process:
            logger.warning("No valid transcripts found for the specified input files.")
            return {"success": 0, "failed": 0, "llm_success": 0, "llm_failed": 0}
            
        logger.info(f"Processing {len(files_to_process)} specific file(s) as requested.")

    else:
        logger.info("No specific files requested. Scanning folder for eligible files...")
        if skip_bert:
            for f in source_folder.iterdir():
                if f.is_file() and f.suffix.lower() == source_suffix:
                    name = f.name
                    if "_llm" in name or "_adversarial_" in name:
                        continue
                    files_to_process.append(f)
        else:
            for f in source_folder.iterdir():
                if f.is_file() and f.suffix.lower() == source_suffix and "_anon" not in f.name:
                    files_to_process.append(f)
        
        if not files_to_process:
            logger.info(f"No files found to process in {source_folder}.")
            return {"success": 0, "failed": 0, "llm_success": 0, "llm_failed": 0}
            
        logger.info(f"Found {len(files_to_process)} files to process.")

    if adversarial_mode:
        logger.info("⚠️  ADVERSARIAL MODE ENABLED: Running 3-iteration Red/Blue team loop.")

    for file in files_to_process:
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

            if not skip_bert:
                anon_start = time.time()
                anonymized_text, success, msg, entity_map, surrogate_registry = anonymizer.anonymize(
                    text_content,
                    use_surrogates=enable_surrogates,
                    surrogate_seed=surrogate_seed,
                    surrogate_locales=surrogate_locales
                )
                
                if not success or not anonymized_text:
                    logger.warning(f"BERT Anonymization failed for {base_name}: {msg}")
                    failed_count += 1
                    continue
                
                logger.info(f"  Entities detected: {len(entity_map)}")
                logger.info(f"  Tags found: {', '.join(entity_map.keys()) if entity_map else 'None'}")
                
                # The entity map contains the original PII: only written on request (--save-entity-map).
                if save_entity_map:
                    entity_map_path = _save_entity_map(base_name, entity_map)
                    logger.info(f"Entity map saved (private): {entity_map_path}")

                # Save surrogate registry if generated
                if surrogate_registry and enable_surrogates:
                    surrogate_path = ANONYM_FOLDER / f"{base_name}_anon_surrogates.json"
                    write_private_file(surrogate_path, json.dumps(
                        surrogate_registry, indent=2, ensure_ascii=False).encode("utf-8"))
                    logger.info(f"Surrogate registry saved: {surrogate_path}")

                output_filename = f"{base_name}_anon.txt"
                output_path = ANONYM_FOLDER / output_filename
                
                if not validate_path(output_path, ANONYM_FOLDER, check_exists=False):
                    logger.error(f"Security Alert: Output path traversal detected for {output_filename}. Skipping save.")
                    failed_count += 1
                    continue

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(anonymized_text)
                logger.info(f"BERT Anonymized transcript saved to: {output_path}")
                processed_count += 1
                
                anon_duration = time.time() - anon_start
                logger.info(f"  BERT Anonymization took: {anon_duration:.2f}s")

                text_for_llm = anonymized_text
            else:
                text_for_llm = text_content
                logger.info(f"Using existing anonymized content from {file.name} for LLM step.")

            llm_start = None 
            if use_llm:
                llm_start = time.time()
                if adversarial_mode:
                    logger.info(f"Running ADVERSARIAL LOOP (3 iterations) on {base_name} with model {target_llm_model}...")
                    
                    final_text, iteration_log = run_adversarial_anonymization(
                        text_for_llm, 
                        target_llm_model, 
                        iterations=3
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
                        
                        if not validate_path(llm_path, LLM_ANONYM_FOLDER, check_exists=False):
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

            if llm_start:  # Only calculate if timing was set
                llm_duration = time.time() - llm_start
                logger.info(f"  LLM Rewrite took: {llm_duration:.2f}s")

        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_count += 1
    
    logger.info(f"Anonymization phase complete.")
    if not skip_bert:
        logger.info(f"  BERT Processed: {processed_count}, Failed: {failed_count}")
    if use_llm:
        logger.info(f"  LLM Rewritten (Standard): {llm_processed_count if not adversarial_mode else 'N/A'}, Failed: {llm_failed_count if not adversarial_mode else 'N/A'}")
        if adversarial_mode:
            logger.info(f"  LLM Adversarial Processed: {llm_processed_count}, Failed: {llm_failed_count}")

    return {
        "success": processed_count,
        "failed": failed_count,
        "llm_success": llm_processed_count,
        "llm_failed": llm_failed_count
    }

# ============================================================================
# USER STUDY NOTICE
# ============================================================================

def display_user_study_notice():
    """Brief notice for user study recruitment."""
    print("\n" + "=" * 60)
    print("📢 PARTICIPATE IN OUR USER STUDY!")
    print("   Do you find this tool helpful? Contribute to our anonymous user study and help us make it better!")
    print("   Link: https://survey.charite.de/InterviewTranscriber_UserSurvey/")
    print("   Contact: luke.flanagan@bih-charite.de")
    print("=" * 60 + "\n")

# --- Main Execution ---

if __name__ == "__main__":
    
    pipeline_start = time.time()
    
    parser = argparse.ArgumentParser(
        description="Audio Anonymization Pipeline with Granular Step Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process.py                          # Run all steps (default)
  python process.py --disable-transcription  # Skip audio extraction
  python process.py --disable-diarization    # Skip speaker identification
  python process.py --disable-anonymization  # Skip anonymization
  python process.py --disable-llm            # Disable LLM-based indirect identifier removal
  python process.py --llm-only               # Skip BERT and process existing anonymized files with LLM only
  python process.py --adversarial            # Enable dual-agent adversarial anonymization (3 iterations)
  python process.py --lang DE                # Force German language
  python process.py --include-tags PERSON ORG LOC_CITY   # Select specific tags to anonymize
  python process.py --exclude-tags PROFESSION QUANTITY   # Exclude specific tags from anonymization
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
    parser.add_argument('--file', type=str, help='Process a single file')
    parser.add_argument('--files', nargs='+', help='Process multiple files')
    parser.add_argument('--enable-tts', action='store_true',
                    help='[not implemented in the CLI yet - use the web interface] Multi-speaker TTS generation. '
                         'Voice count is derived from diarization; assignments are randomized.')
    parser.add_argument('--merge-threshold', type=float, default=SEGMENT_MERGE_THRESHOLD,
                    help=f"Maximum gap in seconds between segments to merge (default: {SEGMENT_MERGE_THRESHOLD}). "
                         "Set to 0 for immediate consecutive merge only.")
    parser.add_argument('--audio-edit', type=str, default=None,
                        help='[not implemented yet] Apply audio offset modifications based on JSON config file')
    parser.add_argument('--enable-surrogates', action='store_true',
                        help='Enable Faker-based surrogate substitution (e.g., "Max Mustermann" → "Axel Schneider")')
    parser.add_argument('--surrogate-seed', type=int, default=None,
                        help='Seed for Faker RNG (for reproducible surrogates). Default: random.')
    parser.add_argument('--surrogate-language', type=str, default=None,
                        help='Locale(s) for surrogate generation (comma-separated); the first selects the language. '
                             'Default: derived from --lang, else de_DE,en_US')
    
    parser.add_argument('--save-entity-map', action='store_true',
                        help='Also save the entity map (contains ORIGINAL PII) to pipeline/entitymaps/ '
                             '(0700 folder; encrypted if a master key is configured). Off by default.')
    parser.add_argument('--redact-audio', action='store_true',
                        help='After anonymization, mask the anonymized words in the audio with beeps '
                             '(writes audios_redacted/<name>_redacted.<fmt> and anonym/<name>_redactions.json).')
    parser.add_argument('--redact-only', action='store_true',
                        help='Only run audio redaction on existing transcripts/anonym files (no transcription). '
                             'Lightweight alternative without the ML stack: python -m pipeline.redaction_cli')
    parser.add_argument('--redact-mode', choices=['placeholders', 'changes'], default='placeholders',
                        help="'placeholders': mask words replaced by [TAG] (default); 'changes': mask every difference.")
    parser.add_argument('--redact-pad', type=float, default=DEFAULT_PAD_S,
                        help=f'Padding in seconds added around each masked span (default {DEFAULT_PAD_S}).')
    parser.add_argument('--redact-format', choices=['wav', 'mp3', 'flac'], default='wav')
    parser.add_argument('--redactions-from', type=str, default=None,
                        help='Use hand-edited spans (JSON, schema v1) instead of deriving them; needs exactly one --file.')
    parser.add_argument('--redactions-include-words', action='store_true',
                        help='Keep the original words in the redactions JSON (not recommended).')

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.surrogate_language is None:
        args.surrogate_language = FAKER_LOCALE_MAP.get((args.lang or '').upper(), 'de_DE,en_US')
    if args.merge_threshold is not None:
        SEGMENT_MERGE_THRESHOLD = args.merge_threshold      # was parsed but never applied

    session_logger = SessionLogger(BASE_PATH)
    session_logger.log_settings({
        "script_version": "V4",
        "whisper_model": WHISPERX_MODEL_PATH.name,
        "diarization_model": DIARIZATION_MODEL_PATH.name if DIARIZATION_MODEL_PATH.exists() else "MISSING",
        "anonymization_method": ANONYMIZATION_METHOD,
        "llm_rewrite_enabled": LLM_REWRITE_ENABLED,
        "adversarial_mode": args.adversarial,
        "language": args.lang if args.lang else "auto-detect",
        "include_tags": args.include_tags,
        "exclude_tags": args.exclude_tags,
    })
    
    check_gpu_resources()
    
    run_transcription = not args.disable_transcription and not args.redact_only
    run_diarization = not args.disable_diarization
    run_anonymization = not args.disable_anonymization and not args.redact_only
    run_llm = not args.disable_llm and not args.redact_only
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
    logger.info(f"  Segment Merge Threshold:         {SEGMENT_MERGE_THRESHOLD}s" if args.merge_threshold == SEGMENT_MERGE_THRESHOLD else f"  Segment Merge Threshold:         {args.merge_threshold}s")
    if args.audio_edit:
        logger.info(f"  Audio Edit Config:             {args.audio_edit}")
    logger.info(f"  BERT Anonymization:                {'✅ ENABLED' if run_anonymization and not skip_bert_for_llm else '❌ DISABLED (or Skipped for LLM-only)'}")
    logger.info(f"  LLM Indirect Identifier Removal:   {'✅ ENABLED' if run_llm else '❌ DISABLED'}")
    if skip_bert_for_llm:
        logger.info(f"    └─ Mode: LLM-only (processing existing 'anonym' folder)")
    if is_adversarial:
        logger.info(f"    └─ Adversarial Mode:           ✅ ENABLED (3 iterations)")
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
            file_list=input_file_list if input_file_list else None,
            enable_surrogates=args.enable_surrogates,
            surrogate_seed=args.surrogate_seed,
            surrogate_locales=args.surrogate_language,
            save_entity_map=args.save_entity_map
        )
    elif not args.redact_only:
        logger.info("⏭️  Skipping anonymization (--disable-anonymization)")
        if run_llm and not args.llm_only:
            logger.warning("⚠️  LLM rewrite requested but BERT anonymization is disabled...")
    
    # Step 4: Audio redaction (words timeline + anonymized text -> beeped audio)
    if args.redact_audio or args.redact_only:
        bases = ([Path(f).stem for f in input_file_list] if input_file_list
                 else redaction_cli.discover_bases(TRANSCRIPTS_FOLDER))
        if args.redactions_from and len(bases) != 1:
            parser.error("--redactions-from needs exactly one --file")
        redaction_cli.redact_batch(
            bases, transcripts=TRANSCRIPTS_FOLDER, anonym=ANONYM_FOLDER, audios=AUDIOS_FOLDER,
            out_audio=REDACTED_AUDIO_FOLDER,
            options=redaction_cli.rd.RedactionOptions.from_mapping(
                {"mode": args.redact_mode, "pad_s": args.redact_pad}),
            fmt=args.redact_format,
            spans_file=Path(args.redactions_from) if args.redactions_from else None,
            include_words=args.redactions_include_words)

    logger.info("Pipeline finished.")
    pipeline_duration = time.time() - pipeline_start
    logger.info(f"Pipeline total duration: {pipeline_duration:.2f}s ({pipeline_duration/60:.1f} minutes)")
    #logger.info(f"   Output size: {file_size:.1f} KB")
    # Show user study notice
    display_user_study_notice()
    session_logger.finish()


# ============================================================================
# MODULE EXPORTS - For interoperability with app.py and external modules
# ============================================================================
__all__ = [
    # Core functions
    'check_gpu_resources',
    'SessionLogger',
    'process_videos',
    'process_audios',
    'load_models',
    'transcribe_audio_locally',
    'transcribe_audio_result',
    'TranscriptionResult',
    'TranscriptionError',
    'resolve_language',
    'normalize_model_name',
    'TAG_PLACEHOLDERS',
    'AnonymizationEngine',
    'process_anonymization',
    'generate_audio_edit_script',
    'AUDIO_EDIT_SUPPORT',
    
    # LLM functions
    'call_llm_rewriter',
    'generate_paraphrase',
    'run_adversarial_anonymization',
    
    # TTS functions (from tts_engine)
    'generate_speech',
    'synthesize_segment',
    'get_available_tts_voices',
    'get_tts_status',

    # Surrogate functions
    'apply_surrogate_substitution',
    'SURROGATES',
    'TAG_TO_SURROGATE_CATEGORY',
    
    # Configuration variables
    'TTS_BACKEND',
    'TTS_ENABLED',
    'CHAT_AI_API_KEY',
    'CHAT_AI_ENDPOINT',
    'DEFAULT_CHAT_AI_MODEL',
    'LLM_REWRITE_ENABLED',
    'AVAILABLE_LLM_MODELS',
    'AVAILABLE_TAGS',
    
    # Path constants
    'BASE_PATH',
    'pipeline_dir',
    'MODEL_FOLDER',
    'ANONYM_FOLDER',
    'LLM_ANONYM_FOLDER',
    'TRANSCRIPTS_FOLDER',
    'VIDEOS_FOLDER',
    'AUDIOS_FOLDER',
    
    # Language support
    'WHISPER_LANG_MAP',
    'SUPPORTED_LANGUAGES',
]