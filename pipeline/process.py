# ============================================================================
# ENVIRONMENT CONFIGURATION (must precede all ML library imports)
# ============================================================================

import os

# Pin HF Hub and Transformers to offline mode. Both libraries attempt
# metadata API calls (commit refs, PR discussions, safetensors index probes)
# even when local_files_only=True is passed downstream. Setting these
# environment variables before import suppresses all network activity,
# eliminating spurious 404s and connection-timeout delays.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# Reduce transformers logging verbosity to WARNING so that benign load-time
# messages do not clutter production logs.
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'warning')

# ============================================================================
# STANDARD IMPORTS
# ============================================================================

import sys
import subprocess
import gc
import logging
import re
import time
import argparse
import json
import traceback
import numpy as np
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Dict, List
import logging
from pathlib import Path
import json
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig
from torchcrf import CRF

# Import torch FIRST, configure TF32 BEFORE importing pyannote/whisperx
import torch

# Enable TF32 for CUDA matmul and cuDNN convolution kernels IMMEDIATELY
# after torch import. pyannote-audio emits a ReproducibilityWarning when
# TF32 is disabled because it both slows inference (~3× on Ampere+) and
# can marginally reduce diarization accuracy. TF32 uses the first 19 bits
# of the FP32 mantissa, which is well within the tolerance band for
# ASR/diarization pipelines.
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ML libraries imported after torch and TF32 are configured
import whisperx
import ffmpeg

load_dotenv()
logger = logging.getLogger(__name__)

# GPU Detection Check

def check_gpu_resources():
    """
    Checks for GPU availability and logs the status.
    This is called only when processing starts, not during --help.
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
            f.write("-" * 60 + "\n")
            f.write("SETTINGS:\n")
            f.write("-" * 60 + "\n")

    def log_settings(self, settings_dict):
        """Logs the configuration settings used for this run."""
        self.settings = settings_dict
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            for key, value in settings_dict.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    def log_stats_update(self, **kwargs):
        """Updates the stats dictionary and writes a brief update to the log."""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key] = value
        
        # Optional: Write a brief progress update if needed
        # For now, we just accumulate stats for the final report

    def log_error(self, error_msg):
        """Logs an error message."""
        self.stats["errors"].append(error_msg)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] {error_msg}\n")

    def finish(self):
        """Finalizes the log file with end time, duration, and summary."""
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


# --- Configuration & Security ---

if os.getenv('ENABLE_BETTER_EXCEPTIONS') == 'true':
    import better_exceptions
    better_exceptions.hook()

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
ANNONYM_FOLDER = pipeline_dir / "annonym"
LLM_ANONNYM_FOLDER = pipeline_dir / "LLM-Anon"

# Create directories if they don't exist
for folder in [TRANSCRIPTS_FOLDER, ANNONYM_FOLDER, MODEL_FOLDER,LLM_ANONNYM_FOLDER]:
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

# Local Model Paths
WHISPERX_MODEL_PATH = MODEL_FOLDER / "Systran--faster-whisper-large-v3"


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



# CRITICAL FALLBACK LOGIC
if not WHISPERX_MODEL_PATH.exists():
    logger.warning("❌ large-v3 model NOT found. Searching for alternatives...")
    
    # List of preferred fallback models in order of quality
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
        # Last resort: Any folder starting with the prefix
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

# Verify the final path
if not WHISPERX_MODEL_PATH.exists():
    logger.critical(f"CRITICAL: Final model path {WHISPERX_MODEL_PATH} does not exist.")
    sys.exit(1)

logger.info(f"✅ WhisperX Model Path Set: {WHISPERX_MODEL_PATH.name}")
DIARIZATION_MODEL_PATH = MODEL_FOLDER / "models--pyannote--speaker-diarization-community-1"

if not MODEL_FOLDER.exists():
    logger.critical(f"CRITICAL: Model folder not found at {MODEL_FOLDER}.")
    logger.critical(f"Current script location: {SCRIPT_DIR}")
    sys.exit(1)

if not WHISPERX_MODEL_PATH.exists():
    logger.critical(f"CRITICAL: WhisperX model not found at {WHISPERX_MODEL_PATH}.")
    logger.critical(f"Available folders in model directory: {list(MODEL_FOLDER.iterdir())}")
    sys.exit(1)

if not WHISPERX_MODEL_PATH.exists():
    # Fallback to any available whisper model if large-v3 is missing
    whisper_folders = [f for f in MODEL_FOLDER.iterdir() if f.is_dir() and f.name.startswith("models--Systran--faster-whisper-")]
    if whisper_folders:
        WHISPERX_MODEL_PATH = whisper_folders[0]
        logger.warning(f"Using fallback model: {WHISPERX_MODEL_PATH.name}")
    else:
        logger.critical(f"CRITICAL: No WhisperX model found in {MODEL_FOLDER}.")
        sys.exit(1)

if not DIARIZATION_MODEL_PATH.exists():
    logger.warning(f"WARNING: Diarization model not found at {DIARIZATION_MODEL_PATH}.")
    logger.warning("Speaker diarization will be disabled. Using generic speaker labels.")
    # We can still proceed without diarization, but warn the user

logger.info(f"✅ Paths verified successfully.")
logger.info(f"   Model Folder: {MODEL_FOLDER}")
logger.info(f"   WhisperX Model: {WHISPERX_MODEL_PATH.name}")
logger.info(f"   Diarization Model: {DIARIZATION_MODEL_PATH.name if DIARIZATION_MODEL_PATH.exists() else 'MISSING'}")

# Supported Languages

SUPPORTED_LANGUAGES = {
    'AR': 'Arabic',
    'DE': 'German',
    'EN': 'English',
    'FI': 'Finnish',
    'FR': 'French',
    'HI': 'Hindi',
    'IT': 'Italian',
    'PL': 'Polish',
    'PT': 'Portuguese',
    'SP': 'Spanish',
    'ES': 'Spanish', # Added ES alias
    'TR': 'Turkish'
}

WHISPER_LANG_MAP = {
    'AR': 'ar', 'DE': 'de', 'EN': 'en', 'FI': 'fi', 'FR': 'fr',
    'HI': 'hi', 'IT': 'it', 'PL': 'pl', 'PT': 'pt', 
    'SP': 'es', 'ES': 'es',
    'TR': 'tr'
}

# Anonymization Configuration
ANONYMIZATION_ENABLED = True
ANONYMIZATION_LEVEL = "standard"  # Options: 'basic', 'standard', 'strict'
ANONYMIZATION_METHOD = "local_mmbert"  # Options: 'local_bert', 'local_spacy', 'local_ensemble', 'remote_chat_ai'

# Available Tags for Selection/Deselection
# These correspond to the PII entities the model can detect.
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
    "You are an expert anonymizer. Your goal is to protect privacy by removing or generalizing identifiers.\n"
    "\n"
    "TWO STRATEGIES:\n"
    "1. **REPLACE** specific values with placeholders (e.g., 'John Smith' → '[NAME_OTHER]', '123 Main St' → '[ADDRESS]').\n"
    "2. **GENERALIZE** descriptions when the combination of details could identify someone (e.g., 'senior neurosurgeon at St. Mary's Hospital' → 'doctor at a hospital').\n"
    "\n"
    "GUIDELINES FOR GENERALIZATION:\n"
    "- If a job title is highly specific (e.g., 'Chief of Cardiology at City Hospital'), generalize to 'medical professional at a hospital'.\n"
    "- If a location is rare or unique (e.g., 'the small village of X with population 500'), generalize to 'a small rural community'.\n"
    "- If a combination of demographics is unique (e.g., '55-year-old female engineer from Berlin'), generalize to 'an older professional from a major city'.\n"
    "- If the detail is already generic (e.g., 'I work in healthcare'), DO NOT change it.\n"
    "\n"
    "CRITICAL CONSTRAINTS:\n"
    "1. PRESERVE SPEAKER TAGS: Lines starting with 'SPEAKER_XX:' are structural metadata. NEVER modify them.\n"
    "2. PRESERVE EXISTING TAGS: Do not touch [NAME_OTHER], [DATE], [ID], [LOCATION_CITY], etc. They are already safe.\n"
    "3. MINIMAL CHANGE: Only generalize when necessary for privacy. If a detail is already generic, leave it alone.\n"
    "4. PRESERVE STRUCTURE: Keep all newlines, line breaks, and grammar exactly as they are.\n"
    "5. LANGUAGE: Do not translate. Keep the original language.\n"
    "\n"
    "EXAMPLES:\n"
    "\n"
    "Example 1 (Replace):\n"
    "Input:  'SPEAKER_00: My name is Anna Schmidt and I live at 123 Hauptstraße.'\n"
    "Output: 'SPEAKER_00: My name is [NAME_OTHER] and I live at [ADDRESS].'\n"
    "\n"
    "Example 2 (Generalize):\n"
    "Input:  'SPEAKER_01: I am the head of the rare disease research unit at University Hospital Zurich.'\n"
    "Output: 'SPEAKER_01: I am a researcher at a university hospital.'\n"
    "(Note: Specific role and location were generalized to protect identity.)\n"
    "\n"
    "Example 3 (Preserve Existing Tags):\n"
    "Input:  'SPEAKER_00: I was born in [LOCATION_CITY] and I work as a [PROFESSION].'\n"
    "Output: 'SPEAKER_00: I was born in [LOCATION_CITY] and I work as a [PROFESSION].'\n"
    "(Note: No changes - tags are already anonymized.)\n"
    "\n"
    "Example 4 (No Change Needed):\n"
    "Input:  'SPEAKER_01: I enjoy hiking and reading books.'\n"
    "Output: 'SPEAKER_01: I enjoy hiking and reading books.'\n"
    "(Note: No identifiers present, no change needed.)\n"
    "\n"
    "Example 5 (Partial Generalization):\n"
    "Input:  'SPEAKER_00: I work as a senior data scientist at Veranda GmbH in Berlin.'\n"
    "Output: 'SPEAKER_00: I work as a data professional at a company in a major city.'\n"
    "(Note: Job title and company were generalized; 'Berlin' became 'major city'.)\n"
    "\n"
    "Return ONLY the anonymized text. No explanations, no reasoning, no commentary."
)


AVAILABLE_LLM_MODELS = {
    'medgemma': 'medgemma',            # Try this first (matches 'medgemma' working hint)
    'medgemma27b': 'medgemma27b',      # Try 'medgemma27b' instead of 'medgemma-27b-it'
    'gpt-oss-120b': 'gpt-oss-120b',    # This one definitely works
    'Qwen3.6-27B': 'Qwen3.6-27B',      # Match the table name exactly, no prefix
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

def validate_path(path, base):
    """Ensures the resolved path is within the base directory."""
    resolved = path.resolve()
    base_resolved = base.resolve()
    try:
        resolved.relative_to(base_resolved)
        return True
    except ValueError:
        return False

def merge_consecutive_speaker_segments(segments, max_gap_seconds=1.5, min_pause_words=3):
    """
    Improved: Only merge if SAME TOPIC (short pauses between sentences)
    
    Args:
        segments: List of dicts with 'start', 'end', 'speaker', 'text'
        max_gap_seconds: Maximum silence allowed between segments to merge them.
        min_pause_words: Minimum word count to consider as "pause" before backchannel.
        
    Returns:
        List of merged segment dictionaries.
    """
    if not segments:
        return []

    merged_segments = []
    current_segment = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)
        end = segment.get("end", 0)

        if current_segment is None:
            # First segment
            current_segment = {
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end
            }
        else:
            # Calculate gap
            gap = start - current_segment["end"]
            
            # NEW: Detect topic shift indicators (backchannels, discourse markers)
            prev_text = current_segment["text"].strip().lower()
            curr_text = text.strip().lower()
            
            # Heuristic: Potential topic/speaker shift if:
            # 1. Gap > 1.5 seconds (longer silence = different utterance)
            # 2. Current text starts with backchannel word (response indicator)
            # 3. Previous text ended with sentence terminator (question/statement complete)
            
            backchannel_indicators = ["yeah", "yes", "no", "right", "okay", "ok", "mm-hmm", "uh-huh"]
            is_backchannel = any(curr_text.startswith(bc + " ") or curr_text == bc for bc in backchannel_indicators)
            is_prev_complete = prev_text.endswith((".", "?", "!", "..."))
            
            is_topic_shift = (
                gap > 1.5 or          # Longer silence = likely different utterance
                is_backchannel or     # Response/backchannel = likely separate utterance  
                is_prev_complete      # Previous sentence complete = new statement
            )
            
            if (current_segment["speaker"] == speaker and 
                gap <= max_gap_seconds and 
                not is_topic_shift):  # ← NEW: Only merge if no topic shift
                
                # Merge: extend text and end time
                current_segment["text"] += " " + text
                current_segment["end"] = end
            else:
                # Either different speaker OR significant gap OR topic shift
                merged_segments.append(current_segment)
                current_segment = {
                    "speaker": speaker,
                    "text": text,
                    "start": start,
                    "end": end
                }

    # Append the last segment
    if current_segment:
        merged_segments.append(current_segment)

    return merged_segments

def rule_based_punctuation(text):
    """
    Adds basic punctuation using patterns.
    Faster than ML-based restoration, works offline.
    
    Args:
        text: Raw transcript string without punctuation.
        
    Returns:
        Text with added periods, capitalization, and basic structure.
    """
    if not text:
        return text
    
    # Step 1: Capitalize first letter of text
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    
    # Step 2: Insert periods before backchannel/discourse words (common sentence boundaries)
    backchannel_pattern = r'\b(okay|alright|well|so|though|however|actually|basically|literally|pretty|really)\b'
    text = re.sub(backchannel_pattern, r'. \1', text, flags=re.IGNORECASE)
    
    # Step 3: Fix question words at start of clauses
    question_pattern = r'(?<!\?)\b(how|what|when|where|why|who|which|whose)\b'
    text = re.sub(question_pattern, r'\1', text, flags=re.IGNORECASE)  # Keep as is, model handles questions
    
    # Step 4: Add periods before speaker-switching indicators
    transition_pattern = r'\b(but|and|then|also|plus|moreover)\b'
    text = re.sub(transition_pattern, r'. \1', text, flags=re.IGNORECASE)
    
    # Step 5: End sentences before quoted speech or self-correction
    self_correction = r'\b(that\'s|i mean|you know|like)\b'
    text = re.sub(self_correction, r'. \1', text, flags=re.IGNORECASE)
    
    # Step 6: Capitalize sentence starts (after period + space)
    text = re.sub(r'(\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    
    # Step 7: Remove excessive spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Step 8: Ensure final period (unless it already ends with ? ! or .)
    if text and not text.endswith(('.', '?', '!', ')', '"')):
        text += '.'
    
    return text

def split_long_blocks(segments, max_sentences=2, max_words=40):
    """
    Prevent overly long blocks by splitting on sentence boundaries.
    
    Args:
        segments: List of segment dictionaries.
        max_sentences: Maximum sentences per block before forcing split.
        max_words: Maximum words per block before forcing split.
        
    Returns:
        List of split segment dictionaries.
    """
    import re as _re
    
    split_segments = []
    
    for segment in segments:
        text = segment.get("text", "").strip()
        words = text.split()
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        
        # Only split if exceeding thresholds
        if len(words) <= max_words:
            split_segments.append(segment)
            continue
        
        # Split on sentence boundaries
        sentences = _re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        sentence_count = 0
        chunk_start = start
        
        for sent in sentences:
            tentative = (current_chunk + " " + sent).strip() if current_chunk else sent
            
            if (len(tentative.split()) <= max_words and 
                sentence_count < max_sentences):
                current_chunk = tentative
                sentence_count += 1
            else:
                # Finalize current chunk if non-empty
                if current_chunk:
                    split_segments.append({
                        **segment,
                        "text": current_chunk,
                        "start": chunk_start,
                        "end": end if not split_segments else segment.get("end", end)
                    })
                    chunk_start = end  # Estimate new start (rough approximation)
                
                # Start new chunk
                current_chunk = sent
                sentence_count = 1
        
        # Append remainder
        if current_chunk:
            split_segments.append({
                **segment,
                "text": current_chunk,
                "start": chunk_start,
                "end": end
            })
    
    return split_segments

def detect_backchannels(segments, backchannel_words=["yeah", "yes", "no", "right", "okay", "ok", "mm-hmm", "uh-huh"], log_detected=True):
    """
    Flags segments that might be backchannel responses needing separation.
    
    Args:
        segments: List of segment dictionaries.
        backchannel_words: List of words that typically indicate backchannel responses.
        log_detected: If True, log detected backchannels to logger.
        
    Returns:
        List of indices where splits may be needed.
    """
    split_indices = []
    
    for i, segment in enumerate(segments):
        text = segment.get("text", "").strip().lower()
        
        # Check if starts with backchannel word followed by additional content
        for word in backchannel_words:
            # Pattern: "yeah [something else]" = likely separate utterance
            if (text.startswith(word + " ") or text == word) and len(text) > len(word) + 5:
                if log_detected:
                    logger.debug(f"Potential backchannel at segment {i}: '{text[:50]}...'")
                split_indices.append(i)
                break
    
    return split_indices

def cleanup_gpu_resources(*objects_to_delete):
    """
    Aggressively clears GPU memory, runs garbage collection, 
    and explicitly deletes passed objects to free System RAM.
    
    Args:
        *objects_to_delete: Variable number of tensor/dataframe objects to delete immediately.
    """
    # 1. Explicitly delete large objects passed to the function
    # This breaks reference cycles immediately, helping gc later
    for obj in objects_to_delete:
        try:
            del obj
        except NameError:
            pass # Object might already be deleted
            
    # 2. Force Python Garbage Collection to reclaim System RAM
    gc.collect()
    
    # 3. Clear GPU VRAM cache
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
        logger.error(f"Diarization model missing config files (config.* or pyannote.yaml)")
        return False
    
    required_subdirs = ['embedding', 'plda', 'segmentation']
    missing_subdirs = [d for d in required_subdirs if not (path / d).exists()]
    
    if missing_subdirs:
        logger.warning(f"Diarization model missing subdirectories: {missing_subdirs}"
            f"This might cause loading failures if the model structure is non-standard.")
    
    logger.info(f"Diarization model verified at: {model_path}")
    return True


def convert_numpy(obj):
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
    """
    Extracts audio from video files in VIDEOS_FOLDER.
    If file_list is provided, only processes those specific filenames.
    Otherwise, scans the folder for all supported video formats.
    """
    if not VIDEOS_FOLDER.exists():
        logger.warning(f"No 'videos' folder found at {VIDEOS_FOLDER}. Skipping audio extraction.")
        return 0

    # Determine which files to process
    files_to_process = []
    
    if file_list:
        # User specified files via --file or --files
        logger.info(f"User specified {len(file_list)} video file(s). Targeting specific sources...")
        for fname in file_list:
            fpath = VIDEOS_FOLDER / fname
            
            # Check existence
            if not fpath.exists():
                logger.error(f"Requested video '{fname}' not found in {VIDEOS_FOLDER}. Skipping.")
                continue
            
            # Validate path security
            if not validate_path(fpath, VIDEOS_FOLDER):
                logger.error(f"Security Alert: Path traversal detected for {fname}. Skipping.")
                continue
            
            # Validate extension
            if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning(f"Skipping unsupported format: {fname} ({fpath.suffix})")
                continue
                
            files_to_process.append(fpath)
        
        if not files_to_process:
            logger.warning("No valid video files found for the specified inputs.")
            return 0
            
        logger.info(f"Processing {len(files_to_process)} specific video file(s).")
    else:
        # Default: scan folder for all supported videos
        logger.info("Scanning 'videos' folder for supported files...")
        files_to_process = [
            f for f in VIDEOS_FOLDER.iterdir() 
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        
        if not files_to_process:
            logger.info("No supported video files found in VIDEOS_FOLDER.")
            return 0
            
        logger.info(f"Found {len(files_to_process)} video files to process.")

    # Create audio output folder if it doesn't exist
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

def process_audios(enable_diarization=True, lang_code=None, file_list=None):
    """
    Process audio files with WhisperX following official pipeline patterns.
    
    CORRECTED PER OFFICIAL WHISPERX GUIDANCE:
    1. Diarization runs on audio file path (not numpy array)
    2. assign_word_speakers() expects sorted DataFrame
    3. Alignment happens via whisperx.align(), not transcribe()
    4. VAD preprocessing enabled to reduce hallucinations
    5. GPU memory flushed between major stages
    """
    global args
    check_gpu_resources() 

    # Determine language
    force_language = lang_code if lang_code is not None else getattr(args, 'lang', None)
    
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
    else:
        logger.info("✅ Speaker diarization ENABLED.")

    # Load Transcription Model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        
        model = whisperx.load_model(
            str(WHISPERX_MODEL_PATH), 
            device, 
            compute_type=compute_type, 
            local_files_only=True
        )
        logger.info("WhisperX transcription model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return

    # Load Diarization Model (if enabled)
    diarize_model = None
    if enable_diarization and DIARIZATION_MODEL_PATH.exists():
        try:
            from pyannote.audio import Pipeline
            diarize_pipeline = Pipeline.from_pretrained(str(DIARIZATION_MODEL_PATH))
            
            # Configure diarization thresholds (reduce over-segmentation)
            if hasattr(diarize_pipeline, 'min_duration_on'):
                diarize_pipeline.min_duration_on = 1.0  # Reduced from 4.0
            if hasattr(diarize_pipeline, 'min_duration_off'):
                diarize_pipeline.min_duration_off = 0.5  # Reduced from 2.0
            
            diarize_model = diarize_pipeline
            logger.info("Diarization Pipeline loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load Diarization Pipeline: {e}. Disabling diarization.")
            diarize_model = None
    elif enable_diarization and not DIARIZATION_MODEL_PATH.exists():
        logger.warning(f"Diarization model not found at {DIARIZATION_MODEL_PATH}. Disabling.")
        enable_diarization = False

    # Determine Files to Process
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

    logger.info(f"Found {len(files_to_process)} files to process.")

    for idx, input_file in enumerate(files_to_process, 1):
        if not input_file.exists():
            logger.error(f"Requested file '{input_file.name}' not found. Skipping.")
            continue
        
        # Validate path
        if input_file.suffix.lower() == '.wav':
            if not validate_path(input_file, AUDIOS_FOLDER):
                logger.error(f"Security Alert: Path traversal detected. Skipping.")
                continue
            audio_path = input_file
        else:
            # Video file - convert first (existing code)
            base_name = sanitize_filename(input_file.stem)
            audio_path = AUDIOS_FOLDER / f"{base_name}.wav"
            if not audio_path.exists():
                try:
                    subprocess.run([
                        'ffmpeg', '-i', str(input_file),
                        '-acodec', 'pcm_s16le',
                        '-ar', '16000',
                        '-ac', '1',
                        '-y',
                        str(audio_path)
                    ], check=True, capture_output=True, text=True)
                    logger.info(f"Converted: {input_file.name} -> {audio_path.name}")
                except Exception as e:
                    logger.error(f"Conversion failed: {e}")
                    continue
            else:
                logger.info(f"Audio already exists: {audio_path.name}")
        
        logger.info(f"[{idx}/{len(files_to_process)}] Processing: {input_file.name}")

        try:
            # =================================================================
            # STEP 1: TRANSCRIBE (with VAD preprocessing - official recommendation)
            # =================================================================
            logger.info("Step 1/4: Transcribing audio...")
            audio = whisperx.load_audio(str(audio_path))
            
            transcribe_kwargs = {
                "batch_size": 16,  # Reduced from 32 per official guidance
                "verbose": False,
                "print_progress": False,
                "vad_filter": True,  # ← NEW: Official recommendation
            }
            if whisper_code:
                transcribe_kwargs["language"] = whisper_code
            
            result = model.transcribe(audio, **transcribe_kwargs)
            logger.info(f"  ✓ Transcription complete ({len(result['segments'])} segments)")
            
            # Flush GPU memory after transcription
            del audio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # =================================================================
            # STEP 2: ALIGN (for word-level timestamps - official WhisperX flow)
            # =================================================================
            logger.info("Step 2/4: Aligning segments...")
            if result.get("language"):
                try:
                    align_device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_a, metadata = whisperx.load_align_model(
                        language_code=result["language"], 
                        device=align_device
                    )
                    
                    # Pre-split long segments (keep existing logic)
                    import re as _re
                    MAX_ALIGN_WORDS = 30
                    pre_split_segments = []
                    for seg in result["segments"]:
                        text = seg.get("text", "").strip()
                        words = text.split()
                        if len(words) <= MAX_ALIGN_WORDS:
                            pre_split_segments.append(seg)
                        else:
                            sentences = _re.split(r'(?<=[.!?])\s+', text)
                            current_chunk = ""
                            for sent in sentences:
                                tentative = (current_chunk + " " + sent).strip() if current_chunk else sent
                                if len(tentative.split()) <= MAX_ALIGN_WORDS:
                                    current_chunk = tentative
                                else:
                                    if current_chunk:
                                        pre_split_segments.append({
                                            "start": seg["start"],
                                            "end": seg["end"],
                                            "text": current_chunk
                                        })
                                    current_chunk = sent
                            if current_chunk:
                                pre_split_segments.append({
                                    "start": seg["start"],
                                    "end": seg["end"],
                                    "text": current_chunk
                                })
                    
                    result["segments"] = pre_split_segments
                    
                    # Official whisperx.align() call
                    result = whisperx.align(
                        result["segments"], 
                        model_a, 
                        metadata, 
                        audio,  # ← Pass numpy array (correct)
                        align_device, 
                        return_char_alignments=False
                    )
                    
                    logger.info(f"  ✓ Alignment complete ({len(result['segments'])} aligned segments)")
                    
                    # Free alignment model memory
                    del model_a
                    del metadata
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}. Proceeding without alignment.")
            else:
                logger.warning("  ⚠ No language detected, skipping alignment.")

            # =================================================================
            # STEP 3: DIARIZE (on audio FILE PATH - official requirement)
            # =================================================================
            if enable_diarization and diarize_model:
                logger.info("Step 3/4: Running diarization...")
                try:
                    # CRITICAL FIX: Pass AUDIO FILE PATH, not numpy array
                    # Pyannote.pipeline expects file path string
                    diarize_output = diarize_model(str(audio_path))
                    
                    # Convert pyannote.Annotation to DataFrame (official pattern)
                    segments_list = []
                    for turn, _, speaker in diarize_output.itertracks(yield_label=True):
                        duration = turn.end - turn.start
                        
                        # Filter short fragments (< 0.5s)
                        if duration < 0.5:
                            continue
                        
                        segments_list.append({
                            'start': turn.start,
                            'end': turn.end,
                            'speaker': speaker
                        })
                    
                    if segments_list:
                        # CRITICAL FIX: Sort by start time before DataFrame creation
                        # pyannote may return tracks ordered by speaker cluster
                        segments_list = sorted(segments_list, key=lambda x: x['start'])
                        
                        import pandas as pd
                        diarize_df = pd.DataFrame(segments_list)
                        
                        logger.info(f"  ✓ Diarization: {len(diarize_df)} speaker turns detected")
                        
                        # assign_word_speakers() expects DataFrame (not Annotation)
                        result = whisperx.assign_word_speakers(diarize_df, result)
                        logger.info("  ✓ Speaker assignment complete")
                        
                        # Clean up diarization data
                        del diarize_df
                        del segments_list
                    else:
                        logger.warning("  ⚠ No valid diarization segments found. Using fallback.")
                        for i, seg in enumerate(result["segments"]):
                            seg["speaker"] = f"SPEAKER_{i%2:02d}"
                
                except Exception as e:
                    logger.error(f"  ✗ Diarization failed: {e}. Using fallback.")
                    for i, seg in enumerate(result["segments"]):
                        seg["speaker"] = f"SPEAKER_{i%2:02d}"
            else:
                # Fallback: Generic speaker labels
                for i, seg in enumerate(result["segments"]):
                    seg["speaker"] = f"SPEAKER_{i%2:02d}"

            # =================================================================
            # STEP 4: POST-PROCESSING (your improved merging/punctuation)
            # =================================================================
            logger.info("Step 4/4: Applying post-processing...")
            
            # Merge with topic-aware logic
            result["segments"] = merge_consecutive_speaker_segments(result["segments"])
            
            # Split long blocks
            result["segments"] = split_long_blocks(result["segments"])
            
            # Detect backchannels (informational only)
            backchannel_indices = detect_backchannels(result["segments"], log_detected=False)
            if backchannel_indices:
                logger.info(f"  ⚠ Detected {len(backchannel_indices)} potential backchannel responses")
            
            # Save with punctuation
            base_name = sanitize_filename(input_file.stem)
            transcript_file = TRANSCRIPTS_FOLDER / f"{base_name}.txt"
            
            with open(transcript_file, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    text = segment.get("text", "").strip()
                    if text:
                        text_with_punct = rule_based_punctuation(text)
                        f.write(f"{segment.get('speaker', 'Unknown')}: {text_with_punct}\n")
            
            logger.info(f"  ✓ Saved: {transcript_file.name}")
            
            # Log sample
            logger.info("  📝 Sample output:")
            for i, seg in enumerate(result["segments"][:3]):
                display_text = seg.get("text", "")[:60] + "..." if len(seg.get("text", "")) > 60 else seg.get("text", "")
                logger.info(f"     [{i}] {seg.get('speaker')}: {display_text}")

        except Exception as e:
            logger.error(f"❌ FAILED: {input_file.name} - {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            # Final cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.1)

    # Final cleanup
    del model
    if diarize_model:
        del diarize_model
    cleanup_gpu_resources()
    
    logger.info("Stream processing finished.")
    return len(files_to_process)

# Global variables to hold loaded models
_loaded_whisper_model = None
_loaded_diarize_model = None
    
def load_models():
    """Loads models locally on GPU if available."""
    global _loaded_whisper_model, _loaded_diarize_model
    
    # 1. Check GPU and determine device
    check_gpu_resources()
    
    # Define device variables locally here so they are available for the rest of the function
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    if _loaded_whisper_model and _loaded_diarize_model:
        return _loaded_whisper_model, _loaded_diarize_model

    # 1. Load WhisperX Model (Direct Path Loading with faster_whisper)
    try:
        logger.info(f"Loading WhisperX model directly from: {WHISPERX_MODEL_PATH}")
        
        # Import faster_whisper directly
        from faster_whisper import WhisperModel
        
        # Determine compute type
        compute_type = "float16" if device_str == "cuda" else "float32"
        
        # Load the model directly from the folder path
        _loaded_whisper_model = WhisperModel(
            str(WHISPERX_MODEL_PATH), 
            device=device_str, 
            compute_type=compute_type,
            local_files_only=True
        )
        
        logger.info("✅ WhisperX model loaded successfully (direct path).")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        logger.critical("Hint: Ensure the folder contains 'model.bin', 'config.json', and 'tokenizer.json'.")
        logger.critical(f"Folder contents: {list(WHISPERX_MODEL_PATH.iterdir()) if WHISPERX_MODEL_PATH.exists() else 'Folder missing'}")
        return None, None
    
    # 2. Load Diarization Pipeline
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

def transcribe_audio_locally(audio_path, language=None):
    """
    Transcribes a single audio file using local models.
    Logs progress to console AND returns text for web interface.
    Includes punctuation restoration and improved block splitting.
    """
    global _loaded_whisper_model, _loaded_diarize_model
    
    logger.info(f"--- Starting transcription for: {os.path.basename(audio_path)} ---")
    
    # 1. Ensure models are loaded
    if not _loaded_whisper_model or not _loaded_diarize_model:
        logger.info("Loading models (if not already loaded)...")
        _loaded_whisper_model, _loaded_diarize_model = load_models()
    
    if not _loaded_whisper_model:
        error_msg = "Error: WhisperX model not loaded."
        logger.error(error_msg)
        return error_msg, []
    
    try:
        # 2. Load Audio
        logger.info(f"Loading audio file: {audio_path}")
        audio = whisperx.load_audio(audio_path)
        logger.info(f"Audio loaded. Duration: {len(audio)/16000:.2f} seconds.")
        
        # 3. Transcribe
        logger.info("Running WhisperX transcription...")
        segments, info = _loaded_whisper_model.transcribe(
            audio, 
            beam_size=BATCH_SIZE, 
            language=language,
            vad_filter=True
        )
        
        # Convert generator to list of Segment objects
        raw_segments = list(segments)
        
        # CRITICAL FIX: Convert Segment objects to dictionaries
        segments_list = []
        for seg in raw_segments:
            seg_dict = {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": getattr(seg, 'words', None)
            }
            segments_list.append(seg_dict)
        
        detected_language = info.language if info else language
        logger.info(f"Transcription completed. Detected language: {detected_language}")
        
        # Reconstruct result dict
        result = {"segments": segments_list, "language": detected_language}
        
        # 4. Align
        if result.get("language"):
            try:
                logger.info("Aligning transcription segments...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                align_model, metadata = whisperx.load_align_model(
                    language_code=result["language"], device=device
                )

                # Pre-split overly long segments to reduce backtrack failures.
                MAX_ALIGN_WORDS = 30
                pre_split_segments = []
                for seg in result["segments"]:
                    text = seg.get("text", "").strip()
                    words = text.split()
                    if len(words) <= MAX_ALIGN_WORDS:
                        pre_split_segments.append(seg)
                    else:
                        # Split on sentence boundaries first, then by word count
                        import re as _re
                        sentences = _re.split(r'(?<=[.!?])\s+', text)
                        current_chunk = ""
                        for sent in sentences:
                            tentative = (current_chunk + " " + sent).strip()
                            if len(tentative.split()) <= MAX_ALIGN_WORDS:
                                current_chunk = tentative
                            else:
                                if current_chunk:
                                    pre_split_segments.append({
                                        "start": seg["start"],
                                        "end": seg["end"],
                                        "text": current_chunk
                                    })
                                current_chunk = sent
                        if current_chunk:
                            pre_split_segments.append({
                                "start": seg["start"],
                                "end": seg["end"],
                                "text": current_chunk
                            })

                result["segments"] = pre_split_segments
                result = whisperx.align(
                    result["segments"], align_model, metadata,
                    audio, device, return_char_alignments=False
                )
                logger.info(f"Alignment completed for {len(result.get('segments', []))} segments.")
            except Exception as e:
                logger.warning(f"Alignment failed: {e}. Proceeding without alignment.")

        # 5. Diarize
        if _loaded_diarize_model:
            logger.info("Running speaker diarization...")
            try:
                _loaded_diarize_model.min_duration_on = 4.0
                _loaded_diarize_model.min_duration_off = 2

                diarize_output = _loaded_diarize_model(
                    audio_path, min_speakers=2, max_speakers=10
                )
                speaker_diarization = diarize_output.speaker_diarization

                import pandas as pd
                segments_list = []

                for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                    duration = turn.end - turn.start

                    if duration < 0.5:
                        logger.debug(f"Discarding short diarization turn: {duration:.3f}s")
                        continue

                    segments_list.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker
                    })

                logger.info(f"Diarization extracted {len(segments_list)} speaker segments.")
                segments_list = sorted(segments_list, key=lambda x: x.get('start', 0))

                diarize_df = pd.DataFrame(segments_list)
                result = whisperx.assign_word_speakers(diarize_df, result)
                logger.info("Speakers assigned to segments.")

            except Exception as e:
                logger.error(f"Diarization failed: {e}")
                logger.warning("Falling back to generic speaker labels.")
                for i, segment in enumerate(result["segments"]):
                    segment["speaker"] = f"SPEAKER_{i % 2:02d}"
        else:
            for i, seg in enumerate(result["segments"]):
                seg["speaker"] = f"SPEAKER_{i % 2:02d}"

        # 6. MERGE CONSECUTIVE SEGMENTS (IMPROVED VERSION) ← NEW
        logger.info("Merging consecutive speaker segments with topic-aware logic...")
        result["segments"] = merge_consecutive_speaker_segments(result["segments"])
        
        # 7. SPLIT LONG BLOCKS (NEW STEP) ← NEW
        logger.info("Splitting long blocks to improve readability...")
        result["segments"] = split_long_blocks(result["segments"])
        
        # 8. DETECT BACKCHANNELS (LOG ONLY) ← NEW
        backchannel_indices = detect_backchannels(result["segments"], log_detected=True)
        if backchannel_indices:
            logger.info(f"Detected {len(backchannel_indices)} potential backchannel responses")
        
        # 9. RECONSTRUCT TEXT WITH PUNCTUATION ← NEW
        logger.info("Applying rule-based punctuation restoration...")
        result_lines = []
        for seg in result["segments"]:
            if seg['text'].strip():
                # Apply punctuation restoration to each segment
                text_with_punct = rule_based_punctuation(seg['text'])
                line = f"{seg['speaker']}: {text_with_punct}"
                result_lines.append(line)
                logger.info(f"  >> {line[:80]}{'...' if len(line) > 80 else ''}")
        
        result_text = "\n".join(result_lines)
        
        if not result_text:
            result_text = "No speech detected."
            logger.warning("No speech detected in audio.")
        else:
            logger.info(f"--- Transcription Complete. Total lines: {len(result_lines)} ---")

        resultOffset=convert_numpy(result["segments"])
        
        return result_text, resultOffset

    except Exception as e:
        error_msg = f"Transcription failed: {e}"
        logger.error(error_msg, exc_info=True)
        return error_msg, []

# --- Anonymization Engine Class (Custom CRF Implementation) ---

def parse_transcript_into_blocks(transcript_text):
    """
    Parses a transcript string into a list of (speaker, text_block) tuples.
    Consecutive lines from the same speaker are merged into one block.
    """
    lines = transcript_text.strip().split('\n')
    blocks = []
    current_speaker = None
    current_text = []

    for line in lines:
        # Match pattern: SPEAKER_XX: text
        match = re.match(r'^(SPEAKER_\d+):\s*(.*)$', line)
        if match:
            speaker = match.group(1)
            text = match.group(2)
            
            if speaker == current_speaker:
                # Continue current block
                current_text.append(text)
            else:
                # New speaker: save previous block if exists
                if current_speaker and current_text:
                    blocks.append((current_speaker, " ".join(current_text)))
                
                # Start new block
                current_speaker = speaker
                current_text = [text]
        else:
            # Handle lines that don't match the pattern (e.g., empty lines or errors)
            # If we are in a block, append it as is (or ignore)
            if current_speaker:
                current_text.append(line)
            # If no speaker context, ignore or log warning

    # Append the last block
    if current_speaker and current_text:
        blocks.append((current_speaker, " ".join(current_text)))

    return blocks

def predict_dialogue_with_context(sentences_tokens, model, tokenizer, id_to_tag_map, device="cuda", context_window=1):
    """
    Predicts NER labels using FLERT-style context windowing.
    """
    from collections import Counter
    from collections import defaultdict
    
    all_predictions = []
    sep_token = tokenizer.sep_token if hasattr(tokenizer, 'sep_token') else '[SEP]'
    
    for i, target_tokens in enumerate(sentences_tokens):
        # Build context window
        left_ctx = sentences_tokens[max(0, i - context_window):i]
        right_ctx = sentences_tokens[i + 1:i + 1 + context_window]
        
        # Flatten tokens with SEP markers
        flat_tokens = []
        for ctx_sent in left_ctx:
            flat_tokens.extend(ctx_sent)
            flat_tokens.append(sep_token)
        
        # Record target boundaries
        tgt_start_idx = len(flat_tokens)
        flat_tokens.extend(target_tokens)
        tgt_end_idx = len(flat_tokens)
        
        # Add right context
        if right_ctx:
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
        
        # STEP 1: Collect subword predictions grouped by word ID
        word_predictions = defaultdict(list)
        for idx, wid in enumerate(word_ids):
            if wid is None or wid < tgt_start_idx or wid >= tgt_end_idx:
                continue
            pred_id = preds[idx]
            tag = id_to_tag_map.get(str(pred_id), "O")
            target_position = wid - tgt_start_idx
            if 0 <= target_position < len(target_tokens):
                word_predictions[target_position].append(tag)
        
        # STEP 2: Aggregate subword predictions (majority vote)
        target_labels = ["O"] * len(target_tokens)
        for pos in range(len(target_tokens)):
            if pos in word_predictions and word_predictions[pos]:
                votes = word_predictions[pos]
                # Use most frequent non-"O" label, or "O" if all are "O"
                non_o_votes = [v for v in votes if v != "O"]
                if non_o_votes:
                    target_labels[pos] = Counter(non_o_votes).most_common(1)[0][0]
                else:
                    target_labels[pos] = "O"
            else:
                target_labels[pos] = "O"
        
        all_predictions.append(target_labels)
    
    return all_predictions

import spacy
import re

# --- Global SpaCy NLP Objects (Lazy Loaded) ---
_nlp_multilingual = None

def get_nlp_pipeline(lang_code=None):
    """
    Loads or retrieves the SpaCy pipeline for sentence splitting.
    Uses 'xx' (multilingual) blank model by default, or specific lang if provided.
    """
    global _nlp_multilingual
    
    # Determine language code for spaCy
    # Map your codes to spacy codes if necessary, otherwise use 'xx' for generic
    spacy_lang = 'xx' # Default to multilingual blank
    
    # Optional: Map specific codes if you have downloaded them
    # if lang_code == 'DE': spacy_lang = 'de'
    # elif lang_code == 'EN': spacy_lang = 'en'
    
    if spacy_lang == 'xx':
        if _nlp_multilingual is None:
            try:
                _nlp_multilingual = spacy.blank("xx")
                _nlp_multilingual.add_pipe("sentencizer")
                logger.info("Loaded SpaCy multilingual sentencizer.")
            except Exception as e:
                logger.error(f"Failed to load SpaCy 'xx' model: {e}. Falling back to regex splitting.")
                return None
        return _nlp_multilingual
    else:
        # Logic for specific language models if implemented
        return None

def split_dialogue_into_sentences(text, nlp_pipeline=None):
    """
    Splits dialogue text into a list of sentences (each a list of tokens).
    Handles SPEAKER_XX: prefixes and splits multi-sentence turns.
    
    Args:
        text: Raw transcript string.
        nlp_pipeline: Optional pre-loaded spaCy pipeline.
        
    Returns:
        list of tuples: [(speaker_id, [token1, token2, ...]), ...]
    """
    sentences_with_speakers = []
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
        
        # Use original text directly - don't split into sentences!
        # The NER model needs full context for person detection
        tokens = content.split()
        if tokens:
            sentences_with_speakers.append((speaker, tokens, content))  # Keep original text!
                
    return sentences_with_speakers


def merge_adjacent_tags(text):
    # split into tags, whitespace, and normal text
    parts = re.split(r"(\s+|\[[A-Z_]+\])", text)

    merged = []
    last_tag = None
    only_whitespace_since_tag = False

    for part in parts:
        if not part:
            continue

        # tag?
        if re.fullmatch(r"\[[A-Z_]+\]", part):
            if part == last_tag and only_whitespace_since_tag:
                # skip duplicate adjacent tag
                continue

            merged.append(part)
            last_tag = part
            only_whitespace_since_tag = True

        # whitespace?
        elif part.isspace():
            merged.append(part)

        else:
            # normal text
            merged.append(part)
            last_tag = None
            only_whitespace_since_tag = False

    out_str="".join(merged)
    out_str=re.sub("  *", " ", out_str)
    return out_str
    
def reconstruct_text_from_predictions(original_sentences, predictions, speaker_map):
    """Reconstructs text preserving speaker structure and original spacing."""
    reconstructed_blocks = []
    
    for i, (speaker, tokens, original_text) in enumerate(original_sentences):
        labels = predictions[i] if i < len(predictions) else []
        
        # Use ORIGINAL text instead of rejoining tokens
        reconstructed_text = original_text
        
        # Work backwards to avoid index shifts when replacing
        for w_idx in reversed(range(len(tokens))):
            if w_idx < len(labels) and labels[w_idx] not in ["O", ""]:
                tag = labels[w_idx]
                word = tokens[w_idx]
                reconstructed_text = re.sub(
                    r'\b' + re.escape(word) + r'\b', 
                    tag, 
                    reconstructed_text,
                    count=1
                )
        
        reconstructed_blocks.append(f"{speaker}: {reconstructed_text}")

    return "\n".join(reconstructed_blocks)

def predict_sentences(sentences_tokens, model, tokenizer, id_to_tag_map, device="cpu"):
    """
    Predicts NER labels for a list of sentences (tokens).
    Matches the official inference example from the model card.
    """
    all_predictions = []
    
    for sentence_idx, tokens in enumerate(sentences_tokens):
        enc = tokenizer(
            tokens, 
            is_split_into_words=True,
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        word_ids = enc.word_ids(batch_index=0)
        
        with torch.no_grad():
            outputs = model(**enc)
            emissions = outputs["logits"]
            mask = enc["attention_mask"].bool()
            preds = model.decode(emissions, mask)[0]
        
        # FIXED: Initialize all labels to "O" first
        word_labels = ["O"] * len(tokens)
        
        # FIXED: Track which token indices we've actually assigned
        assigned_indices = set()
        
        for idx, wid in enumerate(word_ids):
            if wid is None:
                continue
            if wid in assigned_indices:
                continue
            if wid >= len(tokens):
                continue
                
            assigned_indices.add(wid)
            pred_id = preds[idx]
            tag = id_to_tag_map.get(str(pred_id), "O")
            word_labels[wid] = tag
        
        all_predictions.append(word_labels)
        
    return all_predictions

def load_id2label_mapping(model_path):
    """Load the id2label mapping from the model directory."""
    id2label_file = Path(model_path) / "id2label.json"
    
    if not id2label_file.exists():
        logger.error(f"id2label.json not found at {id2label_file}")
        return None
    
    try:
        # Read the file (format: one "id label" pair per line)
        content = id2label_file.read_text(encoding="utf-8")
        id2label = {}
        for line in content.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                id2label[int(parts[0])] = parts[1]
        logger.info(f"Loaded {len(id2label)} label mappings")
        return id2label
    except Exception as e:
        logger.error(f"Failed to load id2label mapping: {e}")
        return None

class AnonymizationEngine:
    """
    Anonymization Engine using DFKI-SLT/multilingual_DialogPII_NER model.
    Follows official HuggingFace usage pattern with ModernBertCRF + FLERT context windowing.
    """

    def __init__(self, method="local_mmbert", level="standard", model_path=None, 
                 include_tags=None, exclude_tags=None):
        self.method = method
        self.level = level
        # FIXED: Point to TRAINED checkpoint (not untrained default)
        self.model_path = model_path or (MODEL_FOLDER / "multilingual_DialogPII_NER")
        self.include_tags = include_tags
        self.exclude_tags = exclude_tags
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.id2label = {}
        self.label2id = {}

        if ANONYMIZATION_ENABLED:
            logger.info(f"AnonymizationEngine initialized: Method={self.method}, Level={self.level}")
            logger.info(f"Model path: {self.model_path}")
            self._load_model()

    def _build_safe_label_mapping(self):
        """Constructs mapping from model label IDs to anonymization tags."""
        all_target_tags = {
            'PERSON': '[PERSON]',
            'PERSON_EMAIL': '[EMAIL]',
            'PERSON_SOCIAL_RELATION': '[NAME_RELATIVE]',
            'ORG': '[ORGANISATION]',
            'LOC_CITY': '[CITY]',
            'LOC_COUNTRY': '[COUNTRY]',
            'LOC_STREET': '[STREET]',
            'LOC_ZIP': '[ZIP]',
            'LOC_HOUSENUMBER': '[HOUSENUMBER]',
            'LOC_OTHER': '[LOCATION]',
            'DATETIME': '[DATETIME]',
            'DATETIME_AGE': '[AGE]',
            'CODE': '[CODE]',
            'CODE_PHONE': '[PHONE]',
            'CODE_URL': '[URL]',
            'PROFESSION': '[PROFESSION]',
            'PRODUCT': '[PRODUCT]',
            'QUANTITY': '[QUANTITY]',
            'MISC': '[MISC]',
        }

        active_tags = set(all_target_tags.keys())

        if self.include_tags:
            include_set = set(t.upper() for t in self.include_tags)
            active_tags = active_tags.intersection(include_set)

        if self.exclude_tags:
            exclude_set = set(t.upper() for t in self.exclude_tags)
            active_tags = active_tags.difference(exclude_set)

        mapping = {}
        for label_id, label_name in self.id2label.items():
            clean_label = label_name.replace("B-", "").replace("I-", "").replace("S-", "").replace("E-","")
            clean_label_upper = clean_label.upper()

            if clean_label_upper in active_tags:
                mapping[str(label_id)] = all_target_tags[clean_label_upper]
            else:
                mapping[str(label_id)] = "O"

        logger.info(f"Built label mapping with {len(mapping)} entries, active tags: {active_tags}")
        return mapping

    def _load_model(self):
        """Loads the multilingual_DialogPII_NER model with CRF layer."""
        if not self.model_path.exists():
            logger.error(f"Model path not found: {self.model_path}")
            logger.error(f"Contents of MODEL_FOLDER: {list(MODEL_FOLDER.iterdir()) if MODEL_FOLDER.exists() else 'Folder missing'}")
            self.method = None
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            from torchcrf import CRF
            import torch.nn as nn
            import json

            # 1. Load CRF Config
            crf_config_path = self.model_path / "crf_config.json"
            if not crf_config_path.exists():
                logger.error(f"crf_config.json not found at {crf_config_path}")
                self.method = None
                return

            with open(crf_config_path, "r") as f:
                crf_config = json.load(f)

            required_keys = ["base_model_name", "num_labels", "id2label", "label2id"]
            if not all(k in crf_config for k in required_keys):
                logger.error(f"Missing required keys in crf_config.json: {required_keys}")
                self.method = None
                return

            self.id2label = {int(k): v for k, v in crf_config.get("id2label", {}).items()}
            self.label2id = crf_config.get("label2id", {})

            logger.info(f"Loaded {len(self.id2label)} label mappings from crf_config.json")

            # 2. Try to load FLERT config
            flert_config_path = self.model_path / "flert_config.json"
            self.flert_config = None
            if flert_config_path.exists():
                with open(flert_config_path, "r") as f:
                    self.flert_config = json.load(f)
                logger.info(f"FLERT config loaded: context_window={self.flert_config.get('context_window', 2)}")
            else:
                self.flert_config = {"context_window": 2, "context_sep_marker": True}
                logger.warning("flert_config.json not found, using defaults")

            # 3. Define Model Architecture (EXACTLY as per HuggingFace docs)
            class ModernBertCRF(nn.Module):
                def __init__(self, base_model_name, num_labels, id2label, label2id):
                    super().__init__()
                    self.num_labels = num_labels
                    self.id2label = id2label
                    self.label2id = label2id

                    # Load base transformer
                    try:
                        self.transformer = AutoModel.from_pretrained(
                            base_model_name, 
                            local_files_only=True
                        )
                    except Exception as e:
                        logger.warning(f"Local base model '{base_model_name}' not found, trying HF...")
                        self.transformer = AutoModel.from_pretrained(base_model_name)

                    hidden_size = self.transformer.config.hidden_size
                    self.classifier = nn.Linear(hidden_size, num_labels)
                    self.dropout = nn.Dropout(0.1)
                    # CRF layer for Viterbi decoding
                    self.crf = CRF(num_labels, batch_first=True)

                def forward(self, input_ids, attention_mask, labels=None, **kwargs):
                    kwargs.pop("token_type_ids", None)
                    outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                    sequence_output = self.dropout(outputs.last_hidden_state)
                    emissions = self.classifier(sequence_output)

                    if labels is not None:
                        mask = attention_mask.bool()
                        labels_for_crf = labels.clone()
                        labels_for_crf[labels_for_crf == -100] = 0
                        loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
                        return {"loss": loss, "logits": emissions}
                    else:
                        return {"logits": emissions}

                def decode(self, emissions, mask):
                    # CRITICAL FIX: Use CRF Viterbi decoding (NOT argmax!)
                    return self.crf.decode(emissions, mask=mask)

            # 4. Instantiate Model
            local_base_model_path = self.model_path / crf_config["base_model_name"]
            if not local_base_model_path.exists():
                logger.warning(f"Local base model not found at {local_base_model_path}")
                base_model_name = crf_config["base_model_name"]
            else:
                base_model_name = str(local_base_model_path)

            self.model = ModernBertCRF(
                base_model_name=base_model_name,
                num_labels=crf_config["num_labels"],
                id2label=self.id2label,
                label2id=self.label2id
            )

            # Load weights
            model_weights_path = self.model_path / "pytorch_model.bin"
            if not model_weights_path.exists():
                logger.error(f"pytorch_model.bin not found at {model_weights_path}")
                self.method = None
                return

            state_dict = torch.load(model_weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), 
                local_files_only=True
            )

            # Build label mapping
            self.label_mapping = self._build_safe_label_mapping()

            logger.info("✅ Anonymization model loaded successfully with CRF decoding")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.method = None

    def split_dialogue_into_sentences(self, text, nlp_pipeline=None):
        sentences_with_speakers = []
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(r"^(SPEAKER_\d+)\s*:\s*(.*)", line)
            if not match:
                continue

            speaker = match.group(1)
            content = match.group(2)  # ← Save original text!

            if not content:
                continue

            tokens = content.split()
            if tokens:
                sentences_with_speakers.append((speaker, tokens, content))  # ← Include original_text

        return sentences_with_speakers

    def predict_dialogue_with_context(self, sentences_tokens, context_window=2):
        """
        Predicts NER labels using FLERT-style context windowing.
        """
        from collections import Counter
        from collections import defaultdict
        
        sep_token = self.tokenizer.sep_token if hasattr(self.tokenizer, 'sep_token') else '[SEP]'
        all_predictions = []

        for i, target_tokens in enumerate(sentences_tokens):
            # FIX: Add dummy left context for first sentence to stabilize tokenization
            if i == 0:
                left_ctx = [[]]  # Dummy empty sentence with SEP
            else:
                left_ctx = sentences_tokens[max(0, i - context_window):i]
            
            right_ctx = sentences_tokens[i + 1:i + 1 + context_window]

            # Flatten tokens with SEP markers
            flat_tokens = []
            for ctx_sent in left_ctx:
                flat_tokens.extend(ctx_sent)
                flat_tokens.append(sep_token)

            # Record target boundaries
            tgt_start_idx = len(flat_tokens)
            flat_tokens.extend(target_tokens)
            tgt_end_idx = len(flat_tokens)

            # Add right context
            if right_ctx:
                flat_tokens.append(sep_token)
                for ctx_sent in right_ctx:
                    flat_tokens.extend(ctx_sent)

            # Tokenize
            enc = self.tokenizer(
                flat_tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=False
            ).to(self.device)

            word_ids = enc.word_ids(batch_index=0)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**enc)
                emissions = outputs["logits"]
                mask = enc["attention_mask"].bool()
                preds = self.model.decode(emissions, mask)[0]

            # STEP 1: Collect subword predictions grouped by word ID
            word_predictions = defaultdict(list)
            for idx, wid in enumerate(word_ids):
                if wid is None or wid < tgt_start_idx or wid >= tgt_end_idx:
                    continue
                pred_id = preds[idx]
                tag = self.label_mapping.get(str(pred_id), "O")
                target_position = wid - tgt_start_idx
                if 0 <= target_position < len(target_tokens):
                    word_predictions[target_position].append(tag)

            # STEP 2: Aggregate subword predictions (majority vote)
            target_labels = ["O"] * len(target_tokens)
            for pos in range(len(target_tokens)):
                if pos in word_predictions and word_predictions[pos]:
                    votes = word_predictions[pos]
                    # Use most frequent non-"O" label, or "O" if all are "O"
                    non_o_votes = [v for v in votes if v != "O"]
                    if non_o_votes:
                        target_labels[pos] = Counter(non_o_votes).most_common(1)[0][0]
                    else:
                        target_labels[pos] = "O"
                else:
                    target_labels[pos] = "O"

            all_predictions.append(target_labels)

        return all_predictions

    def reconstruct_text_from_predictions(self, original_sentences, predictions):
        reconstructed_blocks = []

        for i, (speaker, tokens, original_text) in enumerate(original_sentences):
            labels = predictions[i] if i < len(predictions) else ["O"] * len(tokens)
            
            # Use ORIGINAL TEXT to preserve formatting
            reconstructed_text = original_text  
            
            # Work backwards to avoid index shifts
            for w_idx in reversed(range(len(tokens))):
                if w_idx < len(labels) and labels[w_idx] not in ["O", "", None]:
                    tag = labels[w_idx]
                    word = tokens[w_idx]
                    reconstructed_text = re.sub(
                        r'\b' + re.escape(word) + r'\b',
                        tag,
                        reconstructed_text,
                        count=1
                    )

            reconstructed_blocks.append(f"{speaker}: {reconstructed_text}")

        return "\n".join(reconstructed_blocks)

    def anonymize(self, text):
        """Anonymizes text using FLERT-style context windowing."""
        if not self.method or not self.model or not self.tokenizer:
            return None, False, "Anonymization model not loaded"

        logger.info("Running anonymization with FLERT context windowing...")

        # 1. Split into sentences
        sentences_data = self.split_dialogue_into_sentences(text)

        if not sentences_data:
            return text, False, "No sentences detected"

        # Extract token lists for prediction
        sentences_tokens = [tokens for _, tokens, _ in sentences_data]

        # 2. Run inference WITH CONTEXT WINDOWING
        try:
            context_window = self.flert_config.get("context_window", 2) if self.flert_config else 2
            predictions = self.predict_dialogue_with_context(
                sentences_tokens=sentences_tokens,
                context_window=context_window
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return text, False, str(e)

        # 3. Reconstruct text
        reconstructed_text = self.reconstruct_text_from_predictions(sentences_data, predictions)

        return reconstructed_text, True, "Success"

def generate_paraphrase(raw_dialogue, lang='DE', model="gpt-oss-120b", temperature=0.3):
    """
    Rephrase/anonymize a dialogue using a local LM Studio model.

    Args:
        raw_dialogue (str):  The dialogue text with SPEAKER_XX: labels.
        lang (str):          Language code, 'DE' or 'EN' (default: 'DE').
        model (str):         LM Studio model string to use.
        temperature (float): Sampling temperature (default: 0.3).

    Returns:
        str: The paraphrased dialogue, or None on failure.
    """
    
    lang = lang.upper()
    if lang not in PARAPHRASE_PROMPTS:
        logger.warning(f"Language '{lang}' not supported, falling back to 'EN'.")
        lang = 'EN'

    system_prompt = PARAPHRASE_PROMPTS[lang]
    user_prompt = f"{USER_PROMPTS[lang]}\n{raw_dialogue.strip()}"

    try:
        client = OpenAI(api_key=CHAT_AI_API_KEY, base_url=CHAT_AI_ENDPOINT)
        #client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
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
    """
    Calls the external LLM API to rewrite text.
    Robustly strips thinking blocks by finding the first 'clean' SPEAKER line.
    """
    import re

    if not CHAT_AI_API_KEY:
        logger.error("LLM API Key not configured. Skipping LLM rewrite.")
        return None, "API Key missing"

    final_system_prompt = system_prompt if system_prompt is not None else LLM_REWRITE_SYSTEM_PROMPT

    try:
        client = OpenAI(api_key=CHAT_AI_API_KEY, base_url=CHAT_AI_ENDPOINT)
        
        # Resolve model ID
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

        # --- 1. Extract Raw Content ---
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

        # --- 2. Parse Specific Patterns ---
        if raw_content:
            # Pattern A: Qwen-asr style "language None<asr_text>..."
            if raw_content.startswith("language None"):
                match = re.search(r'language None\s*<asr_text>(.*?)</asr_text>', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    rewritten_text = match.group(1).strip()
                else:
                    rewritten_text = raw_content[len("language None"):].strip()
            
            # Pattern B: XML Thinking Blocks (<thought>...</thought>)
            elif "<thought>" in raw_content.lower():
                match = re.search(r'</thought>\s*(.*)', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    rewritten_text = match.group(1).strip()
                else:
                    rewritten_text = re.sub(r'^.*?<thought>.*?</thought>\s*', '', raw_content, flags=re.DOTALL | re.IGNORECASE).strip()

            # Pattern C: Generic Reasoning Blocks (The main fix)
            # Strategy: Find the FIRST line that starts with SPEAKER_ but is NOT part of a list (e.g., "1. SPEAKER_")
            # and is NOT bolded (e.g., "**SPEAKER_").
            else:
                # Regex explanation:
                # (?:^|\n) : Start of string or newline
                # (?!\s*[0-9]+\.\s|\s*[-*]\s|\s*\*\*) : Negative lookahead to exclude lines starting with numbers, bullets, or bold stars
                # (\s*SPEAKER_\d+:.*) : Capture the SPEAKER line (allowing leading whitespace)
                transcript_start_pattern = r'(?:^|\n)(?!\s*[0-9]+\.\s|\s*[-*]\s|\s*\*\*)(\s*SPEAKER_\d+:.*)'
                
                match = re.search(transcript_start_pattern, raw_content, re.MULTILINE | re.IGNORECASE)
                
                if match:
                    # Extract from the match start
                    rewritten_text = raw_content[match.start():].strip()
                else:
                    # Fallback 1: If no clean SPEAKER line found, try to find the last paragraph
                    # (Often the model puts the result in the last paragraph)
                    paragraphs = re.split(r'\n\s*\n', raw_content)
                    if paragraphs:
                        # Check if the last paragraph contains SPEAKER lines
                        last_para = paragraphs[-1]
                        if 'SPEAKER_' in last_para:
                            rewritten_text = last_para.strip()
                        else:
                            # Fallback 2: Just take the last paragraph anyway
                            rewritten_text = last_para.strip()
                    else:
                        rewritten_text = raw_content.strip()

        # --- 3. Final Cleanup (Preserve Newlines & Strip Markdown) ---
        if rewritten_text:
            # 1. Remove Markdown Code Blocks
            rewritten_text = re.sub(r'^```\w*\s*|\s*```$', '', rewritten_text, flags=re.MULTILINE)
            rewritten_text = re.sub(r'```[\s\S]*?```', '', rewritten_text)
            
            # 2. Remove any remaining HTML/XML tags but PRESERVE newlines
            rewritten_text = re.sub(r'<[^>]+>', '', rewritten_text)
            
            # 3. Normalize multiple spaces but KEEP newlines
            rewritten_text = re.sub(r'[^\S\n]+', ' ', rewritten_text)
            
            # 4. Remove any leading/trailing whitespace
            rewritten_text = rewritten_text.strip()

            # Safety check
            if rewritten_text.lower() in ["language none", "none", ""]:
                raise ValueError("Extracted text is empty or invalid.")

            # Warning if newlines were lost
            if '\n' not in rewritten_text and '\n' in text:
                logger.warning("LLM output collapsed into single line despite instructions.")

        if not rewritten_text:
            raise ValueError("Model returned empty content or unrecognized format.")

        return rewritten_text, "Success"

    except Exception as e:
        logger.error(f"LLM Rewriter failed for model {final_model}: {e}")
        return None, str(e)

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

    # --- Prompts ---
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
        "3. Attempt to infer attributes not explicitly stated but implied (e.g., 'I live near the Eiffel Tower' -> inferred City: Paris).\n"
        "4. Output a structured critique:\n"
        "   - 'Risks Found': List specific phrases or patterns that allow re-identification.\n"
        "   - 'Inferred Attributes': List any attributes you can guess about the speakers.\n"
        "   - 'Recommendation': Specific instructions on how to fix these leaks (e.g., 'Generalize the street name to [STREET]').\n"
        "5. If the text is perfectly anonymous, state 'No risks found.'\n"
        "Return ONLY the critique in the format described."
    )

    iteration_log = []
    current_text = text

    logger.info(f"Starting Adversarial Loop: {iterations} iterations with model {final_model}")

    for i in range(1, iterations + 1):
        logger.info(f"--- Iteration {i}/{iterations} ---")
        
        # --- Step A: Defender (Anonymize) ---
        # On the first iteration, we anonymize the raw text. 
        # On subsequent iterations, we anonymize based on the previous critique.
        defender_messages = [
            {"role": "system", "content": DEFENDER_SYSTEM_PROMPT},
            {"role": "user", "content": f"Please anonymize the following text:\n\n{text if i == 1 else current_text}"}
        ]
        
        # If not the first iteration, append the attacker's critique to the user prompt
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
            
            # Clean up potential formatting artifacts
            anonymized_text = re.sub(r'<[^>]+>', '', anonymized_text)
            anonymized_text = re.sub(r'\s+', ' ', anonymized_text).strip() # Caution: preserve newlines? 
            # Better preservation:
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
                temperature=0.5, # Slightly higher temp for creative attack vectors
                max_tokens=4096
            )
            critique = attacker_response.choices[0].message.content.strip()
            critique = re.sub(r'<[^>]+>', '', critique)

            logger.info(f"Attacker completed iteration {i}. Risks found: {'Yes' if 'Risk' in critique or 'Found' in critique else 'None'}")

        except Exception as e:
            logger.error(f"Attacker failed in iteration {i}: {e}")
            critique = "Error generating critique."

        # --- Log State ---
        iteration_log.append({
            "iteration": i,
            "defender_output": anonymized_text,
            "attacker_output": critique
        })

        # Update current text for next round
        current_text = anonymized_text

        # Early exit if Attacker finds nothing
        if "No risks found" in critique.lower() and i < iterations:
            logger.info("Attacker found no risks. Stopping early.")
            break

    return current_text, iteration_log

import re

def normalize_punctuation(text):
    """
    Fixes punctuation spacing issues:
    - Removes space before apostrophes: 's → 's
    - Removes space around hyphens: stand - up → stand-up
    - Removes space before commas, periods, etc.
    - Ensures space after punctuation
    """
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

        # 1. Fix apostrophe spacing: "word 's" → "word's"
        content = re.sub(r"\s+'(\w)", r"'\1", content)
        
        # 2. Fix hyphen spacing: "word - word" → "word-word"
        content = re.sub(r'\s+-\s+', '-', content)
        
        # 3. Remove space before punctuation: "word ," → "word,"
        content = re.sub(r'\s+([,.!?;:])', r'\1', content)
        
        # 4. Ensure space after punctuation if missing
        content = re.sub(r'([.!?;:])([A-Za-z])', r'\1 \2', content)
        
        # 5. Capitalize first letter
        if content:
            content = content[0].upper() + content[1:]

        # 6. Handle sentence-ending capitalization
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

# --- Step 3: Anonymize Existing Transcripts ---

def process_anonymization(llm_rewrite_enabled=None, llm_model_id=None, skip_bert=False, adversarial_mode=False,
                          include_tags=None, exclude_tags=None, file_list=None):
    """
    Reads raw transcripts, anonymizes them with BERT, and optionally rewrites with LLM.
    Supports a new 'adversarial_mode' which runs a 3-iteration Red Team vs. Blue Team loop.
    
    Args:
        llm_rewrite_enabled: If True, run LLM on anonymized text.
        llm_model_id: Specific LLM model to use.
        skip_bert: If True, skip BERT anonymization and process existing files in ANNONYM_FOLDER.
        adversarial_mode: If True, run the 3-iteration Defender/Attacker loop instead of single-pass LLM.
    """
    if not TRANSCRIPTS_FOLDER.exists() and not skip_bert:
        logger.warning(f"No 'transcripts' folder found at {TRANSCRIPTS_FOLDER}. Skipping anonymization.")
        return {"success": 0, "failed": 0, "llm_success": 0, "llm_failed": 0}

    logger.info(f"Found 'transcripts' folder at {TRANSCRIPTS_FOLDER}. Starting anonymization process...")

    # Determine LLM settings
    use_llm = llm_rewrite_enabled if llm_rewrite_enabled is not None else LLM_REWRITE_ENABLED
    target_llm_model = llm_model_id if llm_model_id else DEFAULT_CHAT_AI_MODEL

    # Check if LLM can run (requires API key)
    if use_llm and not CHAT_AI_API_KEY:
        logger.warning("LLM rewrite requested but no API key found. Disabling LLM step.")
        use_llm = False
        adversarial_mode = False # Cannot run adversarial without API

    # Initialize Anonymization Engine (BERT) only if not skipping
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

    # Determine source folder based on skip_bert flag
    source_folder = TRANSCRIPTS_FOLDER if not skip_bert else ANNONYM_FOLDER
    source_suffix = ".txt"
    
    # Check for user-specified files first ---
    files_to_process = []
    
    if file_list:
        logger.info(f"User specified {len(file_list)} file(s). Targeting specific transcripts...")
        for fname in file_list:
            # Handle both .wav inputs (from args.file) and .txt inputs (direct transcript names)
            base_name = Path(fname).stem
            target_name = f"{base_name}.txt"
            fpath = source_folder / target_name
            
            if not fpath.exists():
                logger.error(f"Requested transcript '{target_name}' not found in {source_folder}. Skipping.")
                continue
            if not validate_path(fpath, source_folder):
                logger.error(f"Security Alert: Path traversal detected for {target_name}. Skipping.")
                continue
            
            # Apply filters if relevant (e.g., if running LLM-only, ensure we aren't double-processing)
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
        # Default behavior: Scan folder
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
        # Validate that the file path is strictly within source_folder
        if not validate_path(file, source_folder):
            logger.error(f"Security Alert: Attempted path traversal detected for {file.name}. Skipping.")
            continue
        
        base_name = file.stem
        # Adjust base_name if coming from anonym folder (remove _anon suffix for consistent naming)
        if skip_bert and base_name.endswith("_anon"):
            base_name = base_name[:-5] # Remove "_anon"
            
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

                # Save BERT result
                output_filename = f"{base_name}_anon.txt"
                output_path = ANNONYM_FOLDER / output_filename
                
                # Additional safety: Ensure output path is also within ANNONYM_FOLDER
                if not validate_path(output_path, ANNONYM_FOLDER):
                    logger.error(f"Security Alert: Output path traversal detected for {output_filename}. Skipping save.")
                    failed_count += 1
                    continue

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(anonymized_text)
                logger.info(f"BERT Anonymized transcript saved to: {output_path}")
                processed_count += 1
                
                # Use the newly created anonymized text for LLM step
                text_for_llm = anonymized_text
            else:
                # If skipping BERT, use the content directly from the anonym folder
                text_for_llm = text_content
                logger.info(f"Using existing anonymized content from {file.name} for LLM step.")

            # Step 2: Optional LLM Rewrite (Standard or Adversarial)
            if use_llm:
                if adversarial_mode:
                    logger.info(f"Running ADVERSARIAL LOOP (3 iterations) on {base_name} with model {target_llm_model}...")
                    
                    # Call the new adversarial function
                    final_text, iteration_log = run_adversarial_anonymization(
                        text_for_llm, 
                        target_llm_model, 
                        iterations=3
                    )
                    
                    if final_text:
                        # Generate timestamp for unique filenames
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Save the final anonymized result with timestamp
                        llm_filename = f"{base_name}_adversarial_{timestamp}.txt"
                        llm_path = LLM_ANONNYM_FOLDER / llm_filename
                        
                        # Additional safety: Ensure LLM output path is within LLM_ANONNYM_FOLDER
                        if not validate_path(llm_path, LLM_ANONNYM_FOLDER):
                            logger.error(f"Security Alert: LLM output path traversal detected for {llm_filename}. Skipping.")
                            llm_failed_count += 1
                            continue
                            
                        with open(llm_path, "w", encoding="utf-8") as f:
                            f.write(final_text)
                        
                        # Save the iteration log for audit purposes with matching timestamp
                        log_filename = f"{base_name}_adversarial_{timestamp}_log.json"
                        log_path = LLM_ANONNYM_FOLDER / log_filename
                        with open(log_path, "w", encoding="utf-8") as f:
                            json.dump(iteration_log, f, indent=2, ensure_ascii=False)
                            
                        logger.info(f"Adversarial anonymization saved to: {llm_path}")
                        logger.info(f"Audit log saved to: {log_path}")
                        llm_processed_count += 1
                    else:
                        logger.warning(f"Adversarial loop failed for {base_name}")
                        llm_failed_count += 1

                else:
                    # Standard single-pass LLM rewrite
                    logger.info(f"Running standard LLM rewrite on {base_name} with model {target_llm_model}...")
                    llm_result, status = call_llm_rewriter(text_for_llm, target_llm_model)
                    
                    if llm_result:
                        # Also add timestamp to standard LLM output to prevent overwrites
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        llm_filename = f"{base_name}_llm_{timestamp}.txt"
                        llm_path = LLM_ANONNYM_FOLDER / llm_filename
                        
                        # Additional safety: Ensure LLM output path is within LLM_ANONNYM_FOLDER
                        if not validate_path(llm_path, LLM_ANONNYM_FOLDER):
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
        logger.info(f"  LLM Rewritten (Standard): {llm_processed_count if not adversarial_mode else 'N/A'}, Failed: {llm_failed_count if not adversarial_mode else 'N/A'}")
        if adversarial_mode:
            logger.info(f"  LLM Adversarial Processed: {llm_processed_count}, Failed: {llm_failed_count}")

    # Return stats for the SessionLogger
    return {
        "success": processed_count,
        "failed": failed_count,
        "llm_success": llm_processed_count,
        "llm_failed": llm_failed_count
    }

def anonymize_text_locally(text):
    """
    Wrapper function to anonymize text using the local BERT model.
    Returns (anonymized_text, success_status, error_message)
    """
    try:
        # Initialize the engine using the paths defined in this file
        # We use the global configuration from process.py
        engine = AnonymizationEngine(
            method="local_mmbert", 
            level="standard", 
            model_path=MODEL_FOLDER / "multilingual_DialogPII_NER"
        )
        
        if not engine.method:
            return None, False, "Anonymization engine failed to initialize (method not set)."

        # Call the class method
        result_text, success, msg = engine.anonymize(text)
        
        if success:
            return result_text, True, "Success"
        else:
            return None, False, msg
            
    except Exception as e:
        logger.error(f"Error in anonymize_text_locally: {e}", exc_info=True)
        return None, False, str(e)

# ============================================================================
# TTS SERVICE LAYER (Updated for Piper + Coqui TTS)
# ============================================================================

import re
import random
from pathlib import Path

try:
    from pydub import AudioSegment
    from pydub.generators import Sine
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None
    logger.warning(
        "Pydub not available — using ffmpeg fallback for beep generation. "
        "Install with: pip install pydub  (requires ffmpeg in PATH)"
    )

# TTS configuration from environment (UPDATED)
TTS_BACKEND = os.getenv('TTS_BACKEND', 'piper').lower()
TTS_ENABLED = os.getenv('TTS_ENABLED', 'true').lower() == 'true'
TTS_DEFAULT_LANG = os.getenv('TTS_DEFAULT_LANG', 'en')

# >>> ADD VALIDATION HERE <<<
if TTS_BACKEND not in ['piper', 'coqui_xtts']:
    logger.warning(f"Invalid TTS_BACKEND '{TTS_BACKEND}'. Defaulting to 'piper'")
    TTS_BACKEND = 'piper'

# Piper TTS settings
TTS_BIN_PATH = Path(os.getenv('TTS_BIN_PATH', ''))
TTS_VOICE_DIR = Path(os.getenv('TTS_VOICE_DIR', str(pipeline_dir / 'model' / 'piper-voices')))
TTS_VOICE_PATH = Path(os.getenv('TTS_VOICE_PATH', ''))
TTS_SAMPLE_RATE = int(os.getenv('TTS_SAMPLE_RATE', '22050'))

# Coqui XTTS settings
TTS_MODEL_NAME = os.getenv('TTS_MODEL_NAME', 'tts_models/multilingual/multi-dataset/xtts_v2')
XTTS_MODEL_PATH = os.getenv('XTTS_Model_Path', str(pipeline_dir / 'model' / 'coqui-xtts'))
XTTS_REFERENCE_AUDIO = os.getenv('XTTS_Reference_Audio_Path', '')

# Global TTS engine instance
_tts_engine = None
_tts_engine_lang = None


def generate_beep(duration_ms=400, freq=1000):
    """
    Generate a beep sound. Uses pydub if available, ffmpeg fallback otherwise.
    Returns AudioSegment object (if pydub) or file path string (if ffmpeg).
    """
    if PYDUB_AVAILABLE and AudioSegment is not None:
        try:
            from pydub.generators import Sine
            return Sine(freq).to_audio_segment(duration=duration_ms).apply_gain(-12)
        except Exception as e:
            logger.error(f"Pydub beep generation failed: {e}")
    
    # Fallback: use ffmpeg (returns temp file path)
    try:
        import tempfile
        temp_path = tempfile.mktemp(suffix='.wav')
        duration_sec = float(duration_ms) / 1000.0
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i',
            f'sine=frequency={freq}:duration={duration_ms/1000}',
            '-ar', '16000',
            '-ac', '1',
            temp_path
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
        # Add tts module to path
        tts_module_dir = pipeline_dir / 'tts'
        if str(tts_module_dir) not in sys.path:
            sys.path.insert(0, str(tts_module_dir))
        
        # Import backend module
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
        logger.info("Hint: Run install script again to ensure TTS dependencies are installed")
        return None
    except TTSError as e:
        logger.error(f"TTS backend initialization failed: {e}")
        logger.info(f"Tip: Check that {'Piper binary' if TTS_BACKEND == 'piper' else 'Coqui model'} exists")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize TTS engine (unexpected error): {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

# ============================================================================
# EXPORTED TTS HELPER FUNCTIONS
# ============================================================================

def get_tts_status():
    """Return TTS configuration status for web interface."""
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
    """
    Main speech generation wrapper - delegates to backend.
    Args:
        text: Text to synthesize
        language: Language code
        output_dir: If provided, saves to file; if None and return_bytes=True, uses /tmp
        speaker_id: Speaker ID for multi-speaker models
        return_bytes: If True, returns BytesIO object instead of file path
    
    Returns:
        If return_bytes=True: BytesIO object with audio data
        Else: str path to generated audio file
    """
    logger.debug(f"DEBUG: input types - text={type(text)}, lang={type(language)}, dir={type(output_dir)}")
    logger.debug(f"DEBUG: return_bytes={return_bytes}")
    
    engine = _get_tts_engine()
    
    if not engine:
        logger.error("TTS engine not available")
        return None
    
    if return_bytes:
        # ✅ IN-MEMORY MODE: Use temp file + immediate cleanup
        import io
        import tempfile
        import os
        
        # Create secure temp file in /tmp (string path, NOT BytesIO)
        fd, temp_path = tempfile.mkstemp(suffix='.wav', prefix='piper_tts_', dir='/tmp')
        os.close(fd)  # Close file descriptor immediately
        logger.debug(f"📝 Created temp file: {temp_path}")
        
        try:
            # ✅ Synthesize to TEMP FILE STRING PATH (Piper accepts string paths)
            engine.synthesize(
                text=text,
                output_path=temp_path,  # ← This is a STRING, not BytesIO!
                speaker_id=int(speaker_id),
                language=str(language)
            )
            
            # Read into memory buffer AFTER file is written
            buffer = io.BytesIO()
            with open(temp_path, 'rb') as f:
                buffer.write(f.read())
            buffer.seek(0)
            logger.debug(f"✅ Audio buffered ({buffer.tell()} bytes)")
            
            return buffer
            
        except Exception as e:
            logger.error(f"TTS synthesis to buffer failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
            
        finally:
            # ✅ SECURITY: Always delete temp file immediately
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.debug(f"🗑️ Temporary file deleted: {temp_path}")
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")
    
    else:
        # DISK MODE: Write to file (legacy behavior for CLI usage)
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
            import traceback
            logger.debug(traceback.format_exc())
            return None

def synthesize_segment(text, language='en'):
    """
    Synthesize a single text segment (wrapper for generate_speech).
    Returns audio file path.
    """
    return generate_speech(text, language=language)

__all__ = [
    # Core pipeline functions
    'transcribe_audio_locally',
    'load_models',
    'call_llm_rewriter',
    'generate_paraphrase',
    'anonymize_text_locally',
    'process_anonymization',
    
    # TTS service functions
    'get_tts_backend',
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
    'ANNONYM_FOLDER',
    'LLM_ANONNYM_FOLDER',
    'MODEL_FOLDER',
    'TRANSCRIPTS_FOLDER',
]

# --- Main Execution ---

if __name__ == "__main__":
    import argparse

    # 1. Set up the Argument Parser FIRST
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
    
    # 2. Define all arguments
    parser.add_argument('--disable-transcription', action='store_true',
                        help='Disable audio extraction from videos (skip process_videos)')
    parser.add_argument('--disable-diarization', action='store_true',
                        help='Disable speaker diarization during transcription')
    parser.add_argument('--disable-anonymization', action='store_true',
                        help='Disable BERT-based anonymization')
    parser.add_argument('--disable-llm', action='store_true',
                    help='Disable LLM-based indirect identifier removal (Run BERT anonymization only)')
    
    parser.add_argument('--llm-only', action='store_true',
                        help='Skip BERT anonymization and process existing files in "annonym" folder with LLM rewrite only.')
    
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
    
    # 3. PARSE ARGUMENTS NOW (This handles --help correctly)
    args = parser.parse_args()
    
    # 4. NOW it is safe to check args.verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 5. NOW it is safe to check GPU resources (after --help would have exited)
    check_gpu_resources() 
    
    # 6. Determine which steps to run
    run_transcription = not args.disable_transcription
    run_diarization = not args.disable_diarization
    run_anonymization = not args.disable_anonymization
    run_llm = not args.disable_llm
    is_adversarial = args.adversarial

    # Handle --llm-only flag logic
    skip_bert_for_llm = args.llm_only
    
    if args.llm_only:
        run_llm = True
    
    # Log the execution plan
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
        logger.info(f"    └─ Mode: LLM-only (processing existing 'annonym' folder)")
    logger.info("="*60)
    
    # Execute pipeline steps conditionally
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
    # Pass the SAME input_file_list here. 
    # Note: If you used --file video.mp4, this will look for video.txt internally.
    if run_anonymization or args.llm_only:
        process_anonymization(
            llm_rewrite_enabled=run_llm,
            llm_model_id=args.llm_model,
            skip_bert=skip_bert_for_llm,
            adversarial_mode=is_adversarial,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            file_list=input_file_list if input_file_list else None  # <--- FORCE PASS THE LIST
        )
    else:
        logger.info("⏭️  Skipping anonymization (--disable-anonymization)")
        if run_llm and not args.llm_only:
            logger.warning("⚠️  LLM rewrite requested but BERT anonymization is disabled...")
    
    logger.info("Pipeline finished.")
