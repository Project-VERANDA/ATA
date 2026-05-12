import os
import sys
import whisperx
import ffmpeg
import torch
import subprocess
import gc
import logging
import re
import time
import argparse
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

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

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*40)
logger.info("GPU DETECTION CHECK")
logger.info("="*40)
cuda_available = torch.cuda.is_available()
device_count = torch.cuda.device_count()
logger.info(f"DEBUG: API Key loaded? {'YES' if os.getenv('CHAT_AI_API_KEY') else 'NO'}")

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
SUPPORTED_EXTENSIONS = ('.mp4', '.mp3', '.mkv')
DEVICE = "cuda"
BATCH_SIZE = 32
COMPUTE_TYPE = "float16"
MIN_SPEAKERS = 2
MAX_SPEAKERS = 4

# Local Model Paths
WHISPERX_MODEL_PATH = MODEL_FOLDER / "models--Systran--faster-whisper-large-v3"

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

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Audio Processing Pipeline")
    parser.add_argument('--lang', type=str, default=None, 
                        choices=list(SUPPORTED_LANGUAGES.keys()),
                        help=f"Force language (e.g., DE, EN, SP, ES). Default: Auto-detect.")
    return parser.parse_args()

# Anonymization Configuration
ANONYMIZATION_ENABLED = True
ANONYMIZATION_LEVEL = "standard"  # Options: 'basic', 'standard', 'strict'
ANONYMIZATION_METHOD = "local_mmbert"  # Options: 'local_bert', 'local_spacy', 'local_ensemble', 'remote_chat_ai'

# Remote Chat AI API Configuration
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY', '')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')
DEFAULT_CHAT_AI_MODEL = os.getenv('CHAT_AI_MODEL', 'gpt-oss-120b')
LLM_REWRITE_ENABLED = True

LLM_REWRITE_SYSTEM_PROMPT = (
    "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous.\n"
    "Your task is to rewrite the provided text to remove any **indirect identifiers**.\n"
    "Indirect identifiers include: specific job titles, unique combinations of demographics, rare locations, specific dates,\n"
    "unique medical conditions, or any detail that could allow someone to identify the speaker when combined with other data.\n"
    "Replace these specific details with generic placeholders like [INDIRECT_ID] or generalize the description.\n"
    "You follow the instructions and format precisely and you try to change as little as possible,\n"
    "keeping the original text in tact as much as possible. Only generalize information and do not invent new information.\n"
    "Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not.\n"
    "Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not.\n"
    "Example: 'my husband and I' -> 'I' is also valid as it only removes information.\n"
    "Do not change the general meaning or flow of the conversation.\n"
    "CRITICAL INSTRUCTIONS:\n"
    "- Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc. These must be preserved exactly as they appear.\n"
    "- Do NOT translate any text. Keep ALL text in its original language exactly as it appears.\n"
    "- Preserve the other tags exactly as they are. For example, [AGE], [NAME_OTHER]. These must be preserved exactly as they appear.\n"
    "- Preserve the original grammar, sentence structure, and language completely.\n"
    "- **CRITICAL: PRESERVE ALL NEWLINES AND LINE BREAKS EXACTLY AS THEY APPEAR IN THE INPUT.**\n"
    "- Do not merge lines into a single paragraph. If the input has multiple lines, the output MUST have the same number of lines in the same order.\n"
    "IMPORTANT: Return ONLY the anonymized text. Do not include any explanations, reasoning, or additional commentary."
)


AVAILABLE_LLM_MODELS = {
    'medgemma': 'medgemma',
    'medgemma27b': 'medgemma27b',
    'gpt-oss-120b': 'gpt-oss-120b',
    'Qwen3.5-27B': 'Qwen3.5-27B',
    'Qwen3.5-397B-A17B': 'Qwen3.5-397B-A17B',
    'qwen3-asr-1.7b': 'qwen3-asr-1.7b',
    'cle-Kimi-K2.5': 'cle-Kimi-K2.5',
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

def merge_consecutive_speaker_segments(segments, max_gap_seconds=2.0):
    """
    Merges consecutive segments spoken by the same speaker.
    If a speaker speaks again after a short gap (< max_gap_seconds), 
    the segments are merged into one continuous block.
    
    Args:
        segments: List of dicts with 'start', 'end', 'speaker', 'text'
        max_gap_seconds: Maximum silence allowed between segments to merge them.
    """
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
            # First segment
            current_segment = {
                "speaker": speaker,
                "text": text,
                "start": start,
                "end": end
            }
        else:
            # Check if same speaker and gap is small enough
            if current_segment["speaker"] == speaker:
                gap = start - current_segment["end"]
                if gap <= max_gap_seconds:
                    # Merge: extend text and end time
                    current_segment["text"] += " " + text
                    current_segment["end"] = end
                else:
                    # Gap too large: finalize current, start new
                    merged_segments.append(current_segment)
                    current_segment = {
                        "speaker": speaker,
                        "text": text,
                        "start": start,
                        "end": end
                    }
            else:
                # Different speaker: finalize current, start new
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

# --- Extract Audio ---

def process_videos():
    if not VIDEOS_FOLDER.exists():
        logger.warning(f"No 'videos' folder found at {VIDEOS_FOLDER}. Skipping audio extraction.")
        return

    logger.info(f"Found 'videos' folder at {VIDEOS_FOLDER}. Processing supported files...")
    
    if not AUDIOS_FOLDER.exists():
        AUDIOS_FOLDER.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    for file in VIDEOS_FOLDER.iterdir():
        if not file.is_file():
            continue
            
        if file.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.debug(f"Skipping unsupported file: {file.name}")
            continue

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
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(audio_path)
            ], check=True, capture_output=True, text=True)
            
            processed_count += 1
            logger.info(f"Successfully saved: {audio_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error processing {file.name}: {e.stderr}")
        except Exception as e:
            logger.error(f"Unexpected error processing {file.name}: {e}")

    if processed_count == 0:
        logger.info("No new files processed.")
    return processed_count

def process_audios(enable_diarization=True):
    """
    Process audio files with optional diarization control.
    
    Args:
        enable_diarization: If False, skip speaker diarization and use generic labels.
    """
    # Parse command line args
    args = parse_args()
    force_language = args.lang
    
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
    
    # Log diarization status
    if not enable_diarization:
        logger.info("⚠️  Speaker diarization DISABLED. Using generic speaker labels.")
        diarize_model = None
    else:
        logger.info("✅ Speaker diarization ENABLED.")
    
    # --- 1. LOAD MODELS ONCE ---
    # Load WhisperX Model (Direct Path)
    try:
        logger.info(f"Loading WhisperX model directly from: {WHISPERX_MODEL_PATH}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "float32"
        
        model = whisperx.load_model(
            str(WHISPERX_MODEL_PATH), 
            device, 
            compute_type=compute_type, 
            local_files_only=True
        )
        logger.info("WhisperX model loaded successfully (Direct Path).")
        
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return

    # Load Diarization Pipeline (Only if enabled)
    if enable_diarization:
        try:
            logger.info(f"Loading Diarization Pipeline directly from: {DIARIZATION_MODEL_PATH}...")
            from pyannote.audio import Pipeline
            
            try:
                diarize_pipeline = Pipeline.from_pretrained(str(DIARIZATION_MODEL_PATH))
            except TypeError:
                logger.warning("Standard load failed. Trying fallback with use_auth_token=None...")
                diarize_pipeline = Pipeline.from_pretrained(str(DIARIZATION_MODEL_PATH), use_auth_token=None)
            
            diarize_model = diarize_pipeline
            logger.info("Diarization Pipeline loaded successfully (Direct Path).")
            
        except Exception as e:
            logger.critical(f"Failed to load Diarization Pipeline: {e}")
            logger.critical("Diarization will be skipped. Using generic speaker labels.")
            diarize_model = None
    else:
        diarize_model = None

    # --- 2. PROCESS FILES ONE BY ONE ---
    files = [f for f in AUDIOS_FOLDER.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]
    total_files = len(files)
    
    if total_files == 0:
        logger.info("No .wav files found in audios folder.")
        return

    logger.info(f"Found {total_files} files to process. Starting stream...")

    for idx, file in enumerate(files, 1):
        if not validate_path(file, AUDIOS_FOLDER):
            logger.error(f"Security Alert: Skipping {file.name} (path traversal).")
            continue

        logger.info(f"[{idx}/{total_files}] Processing: {file.name}")

        try:
            # --- STEP A: Transcribe ---
            audio = whisperx.load_audio(str(file))
            
            # Prepare transcription arguments
            transcribe_kwargs = {
                "audio": audio,
                "batch_size": BATCH_SIZE,
                "verbose": False,
                "print_progress": False,
                "task": "transcribe"
            }
            
            if whisper_code:
                transcribe_kwargs["language"] = whisper_code
                logger.info(f"  -> Forced Language: {whisper_code}")
            else:
                logger.info(f"  -> Language: Auto-Detect")

            result = model.transcribe(**transcribe_kwargs)
            
            # --- STEP B: Align ---
            if result.get("language"):
                try:
                    align_device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=align_device)
                    result = whisperx.align(result["segments"], model_a, metadata, audio, align_device, return_char_alignments=False)
                except Exception as e:
                    logger.warning(f"Alignment failed for {file.name}: {e}")
            else:
                logger.warning(f"No language detected for {file.name}.")

            # --- STEP C: Diarize (Conditional) ---
            if enable_diarization and diarize_model:
                try:
                    logger.info(f"  -> Running local diarization...")
                    diarize_output = diarize_model(str(file))
                    
                    speaker_diarization = diarize_output.speaker_diarization
                    
                    segments_list = []
                    for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                        duration = turn.end - turn.start
                        if duration < 0.5: continue
                        
                        segments_list.append({'start': turn.start, 'end': turn.end, 'speaker': speaker})
                    
                    if not segments_list:
                        logger.warning(f"  -> No valid speakers found. Using fallback.")
                        for i, seg in enumerate(result["segments"]):
                            seg["speaker"] = f"SPEAKER_{i%2:02d}"
                    else:
                        import pandas as pd
                        diarize_df = pd.DataFrame(segments_list)
                        result = whisperx.assign_word_speakers(diarize_df, result)
                        
                except Exception as e:
                    logger.error(f"  -> Diarization failed: {e}. Using fallback.")
                    for i, seg in enumerate(result["segments"]):
                        seg["speaker"] = f"SPEAKER_{i%2:02d}"
            else:
                logger.info(f"  -> Skipping diarization (disabled or model unavailable).")
                for i, seg in enumerate(result["segments"]):
                    seg["speaker"] = f"SPEAKER_{i%2:02d}"

            # --- STEP D: Merge & Save ---
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
            # Cleanup
            for var in ['audio', 'result', 'model_a', 'metadata', 'diarize_output', 'speaker_diarization', 'segments_list', 'diarize_df']:
                if var in locals(): del locals()[var]
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0.1)

    # Final Cleanup
    if 'model' in locals(): del model
    if 'diarize_model' in locals(): del diarize_model
    cleanup_gpu_resources()
    logger.info("Stream processing finished.")
    return total_files

# Global variables to hold loaded models
_loaded_whisper_model = None
_loaded_diarize_model = None
    
def load_models():
    """Loads models locally on GPU if available."""
    global _loaded_whisper_model, _loaded_diarize_model
    
    if _loaded_whisper_model and _loaded_diarize_model:
        return _loaded_whisper_model, _loaded_diarize_model

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    logger.info(f"=== DEVICE CHECK ===")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if device_str == "cuda":
        logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ NO GPU DETECTED. Falling back to CPU.")
    logger.info("====================")

    # 1. Load WhisperX Model (Direct Path Loading with faster_whisper)
    try:
        logger.info(f"Loading WhisperX model directly from: {WHISPERX_MODEL_PATH}")
        
        # Import faster_whisper directly
        from faster_whisper import WhisperModel
        
        # Determine compute type
        compute_type = "float16" if device_str == "cuda" else "float32"
        
        # Load the model directly from the folder path
        # This bypasses the HuggingFace cache system entirely
        # We pass the absolute path to the folder containing model.bin, config.json, etc.
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

def transcribe_audio_locally(audio_path, language='de'):
    """
    Transcribes a single audio file using local models.
    Logs progress to console AND returns text for web interface.
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
        return error_msg
    
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
                result = whisperx.align(result["segments"], align_model, metadata, audio, device, return_char_alignments=False)
                logger.info("Alignment completed.")
            except Exception as e:
                logger.warning(f"Alignment failed: {e}. Proceeding without alignment.")

        # 5. Diarize
        if _loaded_diarize_model:
            logger.info("Running speaker diarization...")
            try:
                # Apply aggressive thresholds to prevent over-segmentation
                _loaded_diarize_model.min_duration_on = 4.0
                _loaded_diarize_model.min_duration_off = 2
                
                diarize_output = _loaded_diarize_model(audio_path, min_speakers=2, max_speakers=10)
                speaker_diarization = diarize_output.speaker_diarization
                
                import pandas as pd
                segments_list = []
                for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                    segments_list.append({
                        'start': turn.start,
                        'end': turn.end,
                        'speaker': speaker
                    })
                
                logger.info(f"Diarization extracted {len(segments_list)} speaker segments.")
                
                # CRITICAL FIX: Sort segments by start time before assigning speakers
                # This ensures chronological order for proper merging later
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

        # CRITICAL FIX: Ensure segments are sorted by time BEFORE merging
        # Diarization models sometimes return segments out of order or grouped by speaker
        if 'segments' in result:
            result["segments"] = sorted(result["segments"], key=lambda x: x.get('start', 0))
            logger.info(f"Segments sorted by time. Count: {len(result['segments'])}")

        # 6. MERGE CONSECUTIVE SEGMENTS
        logger.info("Merging consecutive speaker segments...")
        merged_segments = []
        prev_segment = None
        
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "").strip()
            
            # FILTER: Discard extremely short segments (likely noise/hallucinations)
            # Unless the text is a known tag or very short valid word (like "I", "a")
            if len(text) < 4 and text.lower() not in ["i", "a", "ok", "no", "yes", "hi"]:
                logger.debug(f"Discarding short segment: '{text}' (Length: {len(text)})")
                continue
            
            # Only merge if it's the SAME speaker AND it's immediately following
            if prev_segment and prev_segment["speaker"] == speaker:
                prev_segment["text"] += " " + text
            else:
                if prev_segment:
                    merged_segments.append(prev_segment)
                prev_segment = {"speaker": speaker, "text": text}
        
        if prev_segment:
            merged_segments.append(prev_segment)
        
        logger.info(f"Merged into {len(merged_segments)} final segments.")

        # 7. BUILD RETURN STRING & LOG FINAL RESULT
        result_lines = []
        for seg in merged_segments:
            if seg['text'].strip():
                line = f"{seg['speaker']}: {seg['text'].strip()}"
                result_lines.append(line)
                logger.info(f"  >> {line}")  # Log each line to console
        
        result_text = "\n".join(result_lines)
        
        if not result_text:
            result_text = "No speech detected."
            logger.warning("No speech detected in audio.")
        else:
            logger.info(f"--- Transcription Complete. Total lines: {len(result_lines)} ---")

        return result_text

    except Exception as e:
        error_msg = f"Transcription failed: {e}"
        logger.error(error_msg, exc_info=True)  # Log full traceback to console
        return error_msg

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

def predict_dialogue_flert(sentences_tokens, model, tokenizer, id_to_tag_map, device="cpu", context_window=2):
    """
    Predicts NER labels for a sequence of sentences using a FLERT-style context window.
    
    Args:
        sentences_tokens (list[list[str]]): List of sentences, where each sentence is a list of word tokens (strings).
        model: The loaded ModernBertCRF model instance.
        tokenizer: The associated tokenizer.
        id_to_tag_map (dict): Mapping from model prediction ID (int/str) to the final anonymization tag (str).
        device (str): 'cuda' or 'cpu'.
        context_window (int): Number of sentences to include before and after the target.
        
    Returns:
        list[list[str]]: A list of label lists, one per sentence. Each inner list contains the 
                         anonymization tags (or empty string) for the corresponding words.
    """
    sep = tokenizer.sep_token
    all_predictions = []

    for i, target_tokens in enumerate(sentences_tokens):
        # 1. Construct Context Window
        left = sentences_tokens[max(0, i - context_window):i]
        right = sentences_tokens[i + 1:i + 1 + context_window]

        flat_tokens = []
        
        # Add Left Context
        for s in left:
            flat_tokens.extend(s)
        if left:
            flat_tokens.append(sep)

        # Record Target Boundaries
        tgt_start = len(flat_tokens)
        flat_tokens.extend(target_tokens)
        tgt_end = len(flat_tokens)

        # Add Right Context
        if right:
            flat_tokens.append(sep)
        for s in right:
            flat_tokens.extend(s)

        # 2. Tokenize for Model Input
        # is_split_into_words=True tells the tokenizer that flat_tokens are already words
        enc = tokenizer(
            flat_tokens, 
            is_split_into_words=True,
            return_tensors="pt", 
            truncation=False
        ).to(device)
        
        word_ids = enc.word_ids(batch_index=0)

        # 3. Run Inference
        with torch.no_grad():
            outputs = model(**enc)
            emissions = outputs["logits"]
            mask = enc["attention_mask"].bool()
            # Decode CRF predictions
            preds = model.decode(emissions, mask)[0]

        # 4. Extract Predictions for Target Sentence Only
        target_labels = []
        seen_word_indices = set()
        
        for idx, wid in enumerate(word_ids):
            # Skip special tokens (None) or duplicates
            if wid is None or wid in seen_word_indices:
                continue
            
            # Check if current token belongs to the target sentence
            if tgt_start <= wid < tgt_end:
                pred_id = preds[idx]
                
                # Map Prediction ID -> Anonymization Tag
                # Ensure key is string if the map uses string keys
                key = str(pred_id)
                tag = id_to_tag_map.get(key, "")
                
                target_labels.append(tag)
                seen_word_indices.add(wid)
        
        all_predictions.append(target_labels)

    return all_predictions

class AnonymizationEngine:
    """
    Encapsulates logic for text anonymization using the local mmbert model with FLERT context.
    Ensures label integrity by mapping model outputs dynamically without altering internal weights.
    """
    
    def __init__(self, method="local_mmbert", level="standard", model_path=None):
        self.method = method
        self.level = level
        self.model_path = model_path or (MODEL_FOLDER / "mmbert_multilingual_pii_ner")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = None
        self.label_mapping = {} # Will store {model_id: anonymization_tag}
        
        if ANONYMIZATION_ENABLED:
            logger.info(f"AnonymizationEngine initialized: Method={self.method}, Level={self.level}")
            self._load_model()

    def _build_safe_label_mapping(self):
        """
        Constructs a dynamic mapping from the model's original training labels to anonymization tags.
        This preserves the model's training integrity while allowing custom output formats.
        """
        target_tags = {
            'PERSON': '[NAME_OTHER]',
            'PERSON_EMAIL': '[CONTACT_EMAIL]',
            'PERSON_SOCIAL_RELATION': '[NAME_RELATIVE]',
            'ORG': '[LOCATION_ORGANISATION]',
            'LOC_CITY': '[LOCATION_CITY]',
            'LOC_COUNTRY': '[LOCATION_COUNTRY]',
            'LOC_STREET': '[LOCATION_STREET]',
            'LOC_ZIP': '[LOCATION_ZIP]',
            'LOC_HOUSENUMBER': '[LOCATION_STREET]',
            'LOC_OTHER': '[LOCATION_OTHER]',
            'DATETIME': '[DATE]',
            'DATETIME_AGE': '[AGE]',
            'CODE': '[ID]',
            'CODE_PHONE': '[CONTACT_PHONE]',
            'CODE_URL': '[CONTACT_URL]',
            'PROFESSION': '[PROFESSION]',
            'PRODUCT': '[ID]',
            'QUANTITY': '[ID]',
            'MISC': '[ID]',
            'O': ''
        }
        
        mapping = {}
        # Iterate over the model's original ID -> Label Name mapping
        for label_id, label_name in self.original_id2label.items():
            # Normalize label name (remove BIO prefixes if present in training set)
            clean_label = label_name.replace("B-", "").replace("I-", "")
            
            if clean_label in target_tags:
                mapping[str(label_id)] = target_tags[clean_label]
            else:
                # Fallback for unknown labels: Map to generic ID or log warning
                logger.warning(f"Model label '{label_name}' (ID: {label_id}) not in target set. Mapping to [UNKNOWN_PII].")
                mapping[str(label_id)] = "[UNKNOWN_PII]"
                
        return mapping

    def _load_model(self):
        """Loads the mmbert model and initializes the safe label mapping."""
        if not self.model_path.exists():
            logger.error(f"Model path not found: {self.model_path}")
            self.method = None
            return

        try:
            from transformers import AutoModel, AutoTokenizer
            from torchcrf import CRF
            import torch.nn as nn
            import json
            
            # 1. Load Config
            crf_config_path = self.model_path / "crf_config.json"
            if not crf_config_path.exists():
                logger.error(f"crf_config.json not found at {crf_config_path}")
                self.method = None
                return
            
            with open(crf_config_path, "r") as f:
                self.config = json.load(f)
            
            # Store original label mappings
            self.original_id2label = self.config.get("id2label", {})
            self.original_label2id = self.config.get("label2id", {})
            
            logger.info(f"Loaded {len(self.original_id2label)} original model labels.")

            # 2. Define Model Architecture
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
                        mask = attention_mask.bool()
                        labels_for_crf = labels.clone()
                        labels_for_crf[labels_for_crf == -100] = 0
                        loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction='mean')
                        return {"loss": loss, "logits": emissions}
                    else:
                        return {"logits": emissions}

                def decode(self, emissions, mask):
                    return self.crf.decode(emissions, mask=mask)

            # 3. Instantiate Model
            local_base_model_path = MODEL_FOLDER / "mmBERT-base-local"
            if not local_base_model_path.exists():
                logger.error(f"Local base model not found at {local_base_model_path}.")
                self.method = None
                return
            
            self.model = ModernBertCRF(
                base_model_name=str(local_base_model_path),
                num_labels=self.config["num_labels"],
                id2label=self.original_id2label,
                label2id=self.original_label2id
            )
            
            model_weights_path = self.model_path / "pytorch_model.bin"
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # 4. Build Dynamic Mapping
            self.label_mapping = self._build_safe_label_mapping()
            logger.info("Model loaded and safe label mapping initialized.")

        except Exception as e:
            import traceback
            logger.error(f"Failed to load mmbert model: {e}")
            logger.error(traceback.format_exc())
            self.method = None

    def anonymize(self, text):
        """
        Anonymizes text using the FLERT context window approach while preserving
        original line breaks and speaker structure.
        
        Processes the input line-by-line to maintain formatting, applying FLERT
        context windows across the sequence of lines.
        """
        if not self.method or not self.model or not self.tokenizer:
            return None, False, "Anonymization model not loaded."

        logger.info("Running anonymization with FLERT context window (line-preserving)...")

        # 1. Split into Lines (Preserve Structure)
        # Split by newline but keep empty lines to maintain structure
        lines = text.split('\n')
        lines = [line for line in lines if line.strip()] # Filter empty lines for processing
        
        if not lines:
            return text, False, "No lines detected."

        # 2. Tokenize Lines (Word-level)
        # Each line becomes a "sentence" for the FLERT context window
        tokenized_lines = [line.split() for line in lines]

        # 3. Run FLERT Prediction
        try:
            line_labels = predict_dialogue_flert(
                sentences_tokens=tokenized_lines,
                model=self.model,
                tokenizer=self.tokenizer,
                id_to_tag_map=self.label_mapping,
                device=self.device,
                context_window=2
            )
        except Exception as e:
            logger.error(f"FLERT prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return text, False, str(e)

        # 4. Reconstruct Text (Preserve Line Breaks)
        anonymized_lines = []
        for i, words in enumerate(tokenized_lines):
            labels = line_labels[i]
            reconstructed_words = []
            
            for w_idx, word in enumerate(words):
                # Get label for this word position
                tag = labels[w_idx] if w_idx < len(labels) else ""
                
                if tag:
                    reconstructed_words.append(tag)
                else:
                    reconstructed_words.append(word)
            
            # Join words back into a line
            anonymized_lines.append(" ".join(reconstructed_words))

        # Join lines back with newline characters to preserve original structure
        final_text = "\n".join(anonymized_lines)
        return final_text, True, "Success"

def call_llm_rewriter(text, model_id, system_prompt=None):
    """
    Calls the external LLM API to rewrite text.
    Uses a high max_tokens limit (16384) to handle long transcripts.
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
                # Fallback: convert whole message to string
                raw_content = str(msg)

        # --- 2. Parse Specific Patterns ---
        if raw_content:
            # Pattern A: Qwen-asr style "language None<asr_text>..." or just "language None..."
            if raw_content.startswith("language None"):
                match = re.search(r'language None\s*<asr_text>(.*?)</asr_text>', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    rewritten_text = match.group(1).strip()
                else:
                    rewritten_text = raw_content[len("language None"):].strip()
            
            # Pattern B: MedGemma / Thinking blocks ( ... )
            elif "<think>" in raw_content:
                match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    after_thinking = raw_content[match.end():].strip()
                    if after_thinking:
                        rewritten_text = after_thinking
                    else:
                        rewritten_text = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL | re.IGNORECASE).strip()
                else:
                    rewritten_text = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # Pattern C: Standard text (gpt-oss-120b) or others
            else:
                rewritten_text = raw_content.strip()

        # --- 3. Final Cleanup (Preserve Newlines & Strip Markdown) ---
        if rewritten_text:
            # 1. Remove Markdown Code Blocks (```text ... ``` or ``` ... ```)
            # This regex matches opening ``` with optional language tag, captures content, and matches closing ```
            rewritten_text = re.sub(r'^```\w*\s*|\s*```$', '', rewritten_text, flags=re.MULTILINE)
            # Also handle inline or mid-text code blocks if they exist
            rewritten_text = re.sub(r'```[\s\S]*?```', '', rewritten_text)
            
            # 2. Remove any remaining HTML/XML tags but PRESERVE newlines
            rewritten_text = re.sub(r'<[^>]+>', '', rewritten_text)
            
            # 3. Normalize multiple spaces but KEEP newlines
            rewritten_text = re.sub(r'[^\S\n]+', ' ', rewritten_text)
            
            # 4. Remove any leading/trailing whitespace that might be leftover from block removal
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

# --- Step 3: Anonymize Existing Transcripts ---

def process_anonymization(llm_rewrite_enabled=None, llm_model_id=None, skip_bert=False, adversarial_mode=False):
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
            model_path=MODEL_FOLDER / "mmbert_multilingual_pii_ner"
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
    
    if skip_bert:
        logger.info("Skipping BERT anonymization. Processing existing files in 'annonym' folder for LLM rewrite.")
        # Filter for files that haven't been processed by LLM yet
        # Exclude: _llm.txt, _adversarial_*.txt
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

    if not files:
        logger.info(f"No files found to process in {source_folder}.")
        return {"success": processed_count, "failed": failed_count, "llm_success": llm_processed_count, "llm_failed": llm_failed_count}

    logger.info(f"Found {len(files)} files to process.")
    if adversarial_mode:
        logger.info("⚠️  ADVERSARIAL MODE ENABLED: Running 3-iteration Red/Blue team loop.")

    for file in files:
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
            model_path=MODEL_FOLDER / "mmbert_multilingual_pii_ner"
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

# --- Main Execution ---

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Audio Anonymization Pipeline with Granular Step Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process.py                          # Run all steps (default)
  python process.py --disable-transcription  # Skip audio extraction
  python process.py --disable-diarization    # Skip speaker identification
  python process.py --disable-anonymization  # Skip BERT anonymization
  python process.py --llm-only               # Skip BERT and process existing anonymized files with LLM
  python process.py --disable-transcription --disable-diarization  # Multiple disables
        """
    )
    
    # Step Control Arguments (enabled by default)
    parser.add_argument('--disable-transcription', action='store_true',
                        help='Disable audio extraction from videos (skip process_videos)')
    parser.add_argument('--disable-diarization', action='store_true',
                        help='Disable speaker diarization during transcription')
    parser.add_argument('--disable-anonymization', action='store_true',
                        help='Disable BERT-based anonymization')
    parser.add_argument('--disable-llm', action='store_true',
                        help='Disable LLM-based indirect identifier removal')
    
    # New flag for LLM-only mode
    parser.add_argument('--llm-only', action='store_true',
                        help='Skip BERT anonymization and process existing files in "annonym" folder with LLM rewrite only.')
    
    # Language Override
    parser.add_argument('--lang', type=str, default=None, 
                        choices=list(SUPPORTED_LANGUAGES.keys()),
                        help=f"Force language (e.g., DE, EN, SP, ES). Default: Auto-detect.")
    
    # LLM Model Selection
    parser.add_argument('--llm-model', type=str, default=None, 
                        choices=list(AVAILABLE_LLM_MODELS.keys()),
                        help=f"Specific LLM model to use for rewriting.")
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true',
                        help='Enable debug-level logging')
    
    # Adversial Anonymizer
    parser.add_argument('--adversarial', action='store_true',
                    help='Enable dual-agent adversarial anonymization (3 iterations: Anonymize -> Attack -> Refine)')
    
    args = parser.parse_args()
    
    # Configure logging verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine which steps to run
    run_transcription = not args.disable_transcription
    run_diarization = not args.disable_diarization
    run_anonymization = not args.disable_anonymization
    run_llm = not args.disable_llm
    is_adversarial = args.adversarial

    
    # Handle --llm-only flag logic
    # If --llm-only is set, we skip BERT but still run LLM
    # This overrides --disable-anonymization if both are present (llm-only takes precedence for the specific workflow)
    skip_bert_for_llm = args.llm_only
    
    # If --llm-only is used, we force run_llm to True and ensure we don't run BERT
    if args.llm_only:
        run_llm = True
        # Note: We don't necessarily disable the BERT step globally, but we tell the function to skip it
        # However, if the user explicitly said --disable-anonymization, that's fine too.
        # The key is that process_anonymization will receive skip_bert=True
    
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
    
    # Step 1: Audio Extraction (Videos → WAV)
    if run_transcription:
        process_videos()
    else:
        logger.info("⏭️  Skipping audio extraction (--disable-transcription)")
    
    # Step 2: Transcription & Diarization (WAV → Transcript)
    if run_transcription:
        # Pass diarization flag to process_audios
        process_audios(enable_diarization=run_diarization)
    else:
        logger.info("⏭️  Skipping transcription (--disable-transcription)")
    
    # Step 3: Anonymization (Transcript → Anonymized)
    # If --llm-only is set, we call process_anonymization with skip_bert=True
    # This will look in ANNONYM_FOLDER instead of TRANSCRIPTS_FOLDER
    if run_anonymization or args.llm_only:
        process_anonymization(
            llm_rewrite_enabled=run_llm,
            llm_model_id=args.llm_model,
            skip_bert=skip_bert_for_llm,
            adversarial_mode=is_adversarial  # Pass the new flag
        )
    else:
        logger.info("⏭️  Skipping anonymization (--disable-anonymization)")
        # If anonymization is disabled but LLM is enabled, warn the user
        if run_llm and not args.llm_only:
            logger.warning("⚠️  LLM rewrite requested but BERT anonymization is disabled and --llm-only not set. "
                          "LLM step will be skipped as it depends on anonymized input.")
    
    logger.info("Pipeline finished.")
