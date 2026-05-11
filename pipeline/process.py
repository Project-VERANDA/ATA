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
from collections import defaultdict
from pathlib import Path
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


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
    "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous"
    "Your task is to rewrite the provided text to remove any **indirect identifiers**."
    "Indirect identifiers include: specific job titles, unique combinations of demographics, rare locations, specific dates,"
    "unique medical conditions, or any detail that could allow someone to identify the speaker when combined with other data."
    "Replace these specific details with generic placeholders like [INDIRECT_ID] or generalize the description. "
    "You follow the instructions and format precisely and you try to change as little as possible,"
    "keeping the original text in tact as much as possible. Only generalize"
    "information and do not invent new information."
    "Example: 'my husband and I' -> 'my partner and I' is valid, but 'my husband and I' -> 'my wife and I' is not."
    "Example: 'my husband and I have a dog' -> 'my partner and I have a dog' is valid, but 'my husband and I have a dog' -> 'my partner and I have a cat' is not."
    "Example: 'my husband and I' -> 'I' is also valid as it only removes information."
    "Do not change the general meaning or flow of the conversation. "
    "CRITICAL INSTRUCTIONS:"
    "- Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc. These must be preserved exactly as they appear."
    "- Do NOT translate any text. Keep ALL text in its original language exactly as it appears."
    "- Preserve the other tags exactly as they are. For example, [AGE], [NAME_OTHER]. These must be preserved exactly as they appear."
    "- Preserve the original grammar, sentence structure, and language completely."
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

class AnonymizationEngine:
    """
    Encapsulates all logic related to text anonymization using the local mmbert model
    with its custom ModernBertCRF architecture.
    """
    
    def __init__(self, method="local_mmbert", level="standard", model_path=None):
        self.method = method
        self.level = level
        # Point to the folder containing crf_config.json and pytorch_model.bin
        self.model_path = model_path or (MODEL_FOLDER / "mmbert_multilingual_pii_ner")
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = None
        
        if ANONYMIZATION_ENABLED:
            logger.info(f"AnonymizationEngine initialized: Method={self.method}, Level={self.level}, Path={self.model_path}")
            self._load_model()

    def _load_model(self):
        """Loads the mmbert model with custom CRF architecture."""
        if not self.model_path.exists():
            logger.error(f"Model path not found: {self.model_path}")
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
                self.config = json.load(f)
            
            logger.info(f"Loaded CRF config: base_model={self.config.get('base_model_name')}, num_labels={self.config.get('num_labels')}")

            # 2. Define the Custom Model Class
            class ModernBertCRF(nn.Module):
                def __init__(self, base_model_name, num_labels, id2label, label2id):
                    super().__init__()
                    self.num_labels = num_labels
                    self.id2label = id2label
                    self.label2id = label2id
                    # Load the base transformer (e.g., bert-base-multilingual-cased)
                    self.transformer = AutoModel.from_pretrained(base_model_name, local_files_only=True)
                    hidden_size = self.transformer.config.hidden_size
                    self.classifier = nn.Linear(hidden_size, num_labels)
                    self.dropout = nn.Dropout(0.1)
                    self.crf = CRF(num_labels, batch_first=True)

                def forward(self, input_ids, attention_mask, labels=None, **kwargs):
                    # Remove token_type_ids if present (common issue with some tokenizers)
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

            # 3. Instantiate the Model
            logger.info(f"Instantiating ModernBertCRF model...")
            
            # Use local base model path instead of hub repo name
            local_base_model_path = MODEL_FOLDER / "mmBERT-base-local"
            if not local_base_model_path.exists():
                logger.error(f"Local base model not found at {local_base_model_path}. Please download it first.")
                self.method = None
                return
            
            logger.info(f"Using local base model: {local_base_model_path}")
            
            self.model = ModernBertCRF(
                base_model_name=str(local_base_model_path),  # Changed to local path
                num_labels=self.config["num_labels"],
                id2label=self.config["id2label"],
                label2id=self.config["label2id"]
            )
            # 4. Load Weights
            model_weights_path = self.model_path / "pytorch_model.bin"
            logger.info(f"Loading weights from {model_weights_path}...")
            state_dict = torch.load(model_weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 5. Load Tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            logger.info("mmbert model with CRF loaded successfully.")

        except ImportError as e:
            logger.error(f"Missing dependency (likely torchcrf). Install with: pip install torchcrf. Error: {e}")
            self.method = None
        except Exception as e:
            import traceback
            logger.error(f"Failed to load mmbert model: {e}")
            logger.error(f"Full Traceback:\n{traceback.format_exc()}")
            self.method = None

    def _get_labels(self):
        """Maps the model's output labels to our anonymization tags."""
        return {
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
            'MISC': '[ID]'
        }

    def anonymize(self, text):
        """
        Anonymizes text using the local mmbert model with chunking to handle long transcripts.
        Splits text into overlapping chunks, processes them, and reassembles the result.
        """
        if not self.method or not self.model or not self.tokenizer:
            return None, False, "Anonymization model not loaded."

        logger.info(f"Running anonymization via mmbert (CRF) with chunking...")

        # Configuration
        MAX_TOKENS = 512
        OVERLAP_TOKENS = 50  # Ensures entities spanning chunk boundaries are caught
        
        # 1. Pre-tokenize to get exact token boundaries
        # We disable truncation here to analyze the full length
        full_encoding = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=False, 
            return_offsets_mapping=True,
            add_special_tokens=False 
        )
        
        input_ids = full_encoding['input_ids'][0]
        offsets = full_encoding['offsets_mapping'][0]
        total_tokens = len(input_ids)
        
        # If text is short enough, process in one go
        if total_tokens <= MAX_TOKENS:
            chunks = [(0, total_tokens)]
        else:
            # Create overlapping chunks
            chunks = []
            start = 0
            while start < total_tokens:
                # Calculate end of current chunk
                end = min(start + MAX_TOKENS, total_tokens)
                
                # If not the last chunk, create overlap
                if end < total_tokens:
                    # Move end back to create overlap, but ensure chunk is still substantial
                    overlap_start = end - OVERLAP_TOKENS
                    if overlap_start <= start:
                        # Prevent infinite loop if overlap is larger than chunk size
                        overlap_start = start + 1
                    
                    end = overlap_start
                
                chunks.append((start, end))
                start = end # Next chunk starts where this one ended (minus overlap logic handled by slicing)
                
                # Break if we are at the end
                if end >= total_tokens:
                    break

        logger.info(f"Split text into {len(chunks)} chunks (Total tokens: {total_tokens}).")

        # 2. Process each chunk
        anonymized_chunks = []
        
        for i, (start_idx, end_idx) in enumerate(chunks):
            # Slice the token IDs for this chunk
            chunk_ids = input_ids[start_idx:end_idx].unsqueeze(0)
            
            # Create attention mask
            attention_mask = torch.ones_like(chunk_ids)
            
            # Move to device
            chunk_ids = chunk_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            try:
                with torch.no_grad():
                    outputs = self.model(chunk_ids, attention_mask=attention_mask)
                    emissions = outputs["logits"]
                    mask = attention_mask.bool()
                    
                    # CRF Decoding
                    predictions = self.model.decode(emissions, mask)
                
                # Map predictions to labels
                pred_ids = predictions[0]
                tokens = self.tokenizer.convert_ids_to_tokens(chunk_ids[0])
                id2label = self.config["id2label"]
                labels = [id2label[str(pid)] for pid in pred_ids]
                
                # Reconstruct text for this chunk
                label_map = self._get_labels()
                result_tokens = []
                j = 0
                while j < len(tokens):
                    token = tokens[j]
                    label = labels[j]
                    
                    # Skip special tokens if any slipped in
                    if token in ['[CLS]', '[SEP]', '[PAD]', '<pad>', '<cls>', '<sep>']:
                        j += 1
                        continue
                    
                    # Handle B- and I- tags
                    if label.startswith('B-') or label.startswith('I-'):
                        entity_type = label.split('-')[1]
                        replacement_tag = label_map.get(entity_type, '[UNKNOWN_PII]')
                        result_tokens.append(replacement_tag)
                        
                        # Skip subsequent I- tags for this entity
                        k = j + 1
                        while k < len(labels) and labels[k].startswith('I-') and labels[k].split('-')[1] == entity_type:
                            k += 1
                        j = k
                    else:
                        # Clean token formatting
                        clean_token = token.replace('##', '').replace('▁', ' ')
                        result_tokens.append(clean_token)
                        j += 1
                
                # Join and clean spacing
                chunk_text = "".join(result_tokens).replace("  ", " ").strip()
                # Clean up spacing around brackets
                chunk_text = re.sub(r'\s+\[', '[', chunk_text)
                chunk_text = re.sub(r'\]\s+', ']', chunk_text)
                
                anonymized_chunks.append(chunk_text)
                logger.debug(f"Processed chunk {i+1}/{len(chunks)}")

            except Exception as e:
                logger.error(f"Chunk {i+1} failed: {e}")
                # Fallback: Return the original text segment for this chunk to prevent data loss
                fallback_tokens = [t.replace('##', '').replace('▁', ' ') for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]']]
                anonymized_chunks.append("".join(fallback_tokens).strip())

        # 3. Reassemble chunks
        if not anonymized_chunks:
            return text, False, "No chunks processed."

        final_result = anonymized_chunks[0]
        
        for i in range(1, len(anonymized_chunks)):
            current_chunk = anonymized_chunks[i]
            previous_result = final_result
            
            # Smart Stitching: Find the longest overlap between the end of the previous result
            # and the start of the current chunk.
            max_overlap_search = min(len(previous_result), len(current_chunk), 200)
            overlap_len = 0
            
            for length in range(max_overlap_search, 0, -1):
                if previous_result.endswith(current_chunk[:length]):
                    overlap_len = length
                    break
            
            if overlap_len > 0:
                # Append only the non-overlapping part
                final_result += current_chunk[overlap_len:]
            else:
                # If no overlap found, just append with a space
                final_result += " " + current_chunk

        logger.info(f"Anonymization complete. Final length: {len(final_result)}")
        return final_result, True, "Success"

def call_llm_rewriter(text, model_id, system_prompt=None):
    """
    Calls the external LLM API to rewrite text.
    Handles specific quirks of Qwen-asr and MedGemma.
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

        logger.info(f"Calling LLM model: {final_model}")
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=final_model,
            stream=False,
            temperature=0.3,
            max_tokens=4096
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
            # If it starts with "language None", strip it and look for the rest
            if raw_content.startswith("language None"):
                # Try to find text after "language None"
                # Case 1: <asr_text> tags exist
                match = re.search(r'language None\s*<asr_text>(.*?)</asr_text>', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    rewritten_text = match.group(1).strip()
                else:
                    # Case 2: Just raw text after "language None"
                    # Remove the prefix and take the rest
                    rewritten_text = raw_content[len("language None"):].strip()
            
            # Pattern B: MedGemma / Thinking blocks (<think> ... </think>)
            elif "<think>" in raw_content:
                # Extract content between <think> and </think>
                match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL | re.IGNORECASE)
                if match:
                    # If there is text AFTER the thinking block, take that. 
                    # Usually the model puts the answer after the thinking.
                    after_thinking = raw_content[match.end():].strip()
                    if after_thinking:
                        rewritten_text = after_thinking
                    else:
                        # If no text after, maybe the thinking IS the answer (unlikely for rewrite)
                        # Or maybe the thinking block contains the answer?
                        # Let's try to extract the last block of text if available
                        pass
                else:
                    # Fallback: Remove <think> tags entirely and keep the rest
                    rewritten_text = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL | re.IGNORECASE).strip()

            # Pattern C: Standard text (gpt-oss-120b)
            else:
                rewritten_text = raw_content.strip()

        # --- 3. Final Cleanup ---
        if rewritten_text:
            # Remove any remaining tags
            rewritten_text = re.sub(r'<[^>]+>', '', rewritten_text)
            # Remove extra whitespace
            rewritten_text = re.sub(r'\s+', ' ', rewritten_text).strip()
            
            # Safety check: if it's just "language None" or empty, fail
            if rewritten_text.lower() in ["language none", "none", ""]:
                raise ValueError("Extracted text is empty or invalid.")

        if not rewritten_text:
            raise ValueError("Model returned empty content or unrecognized format.")

        return rewritten_text, "Success"

    except Exception as e:
        logger.error(f"LLM Rewriter failed for model {final_model}: {e}")
        return None, str(e)

# --- Step 3: Anonymize Existing Transcripts (Active) ---

def process_anonymization(llm_rewrite_enabled=None, llm_model_id=None):
    """
    Reads raw transcripts, anonymizes them with BERT, and optionally rewrites with LLM.
    
    Args:
        llm_rewrite_enabled: If True, run LLM on anonymized text.
        llm_model_id: Specific LLM model to use.
    """
    if not TRANSCRIPTS_FOLDER.exists():
        logger.warning(f"No 'transcripts' folder found at {TRANSCRIPTS_FOLDER}. Skipping anonymization.")
        return

    logger.info(f"Found 'transcripts' folder at {TRANSCRIPTS_FOLDER}. Starting anonymization process...")

    # Determine LLM settings
    use_llm = llm_rewrite_enabled if llm_rewrite_enabled is not None else LLM_REWRITE_ENABLED
    target_llm_model = llm_model_id if llm_model_id else DEFAULT_CHAT_AI_MODEL

    # Check if LLM can run (requires API key)
    if use_llm and not CHAT_AI_API_KEY:
        logger.warning("LLM rewrite requested but no API key found. Disabling LLM step.")
        use_llm = False

    # Initialize Anonymization Engine (BERT)
    anonymizer = AnonymizationEngine(
        method=ANONYMIZATION_METHOD,
        level=ANONYMIZATION_LEVEL,
        model_path=MODEL_FOLDER / "mmbert_multilingual_pii_ner"
    )

    if not anonymizer.method:
        logger.error("Anonymization engine (BERT) failed to initialize. Aborting.")
        return

    processed_count = 0
    failed_count = 0
    llm_processed_count = 0

    for file in TRANSCRIPTS_FOLDER.iterdir():
        # Skip directories
        if not file.is_file():
            continue
        
        # Skip non-text files
        if not file.suffix.lower() == ".txt":
            continue
        
        # Validate that the file path is strictly within TRANSCRIPTS_FOLDER
        if not validate_path(file, TRANSCRIPTS_FOLDER):
            logger.error(f"Security Alert: Attempted path traversal detected for {file.name}. Skipping.")
            continue
        
        # Skip already anonymized files
        if "_anon" in file.name:
            logger.debug(f"Skipping already anonymized file: {file.name}")
            continue

        base_name = file.stem
        logger.info(f"Processing transcript: {file.name}")

        try:
            with open(file, "r", encoding="utf-8") as f:
                transcript_text = f.read()

            if not transcript_text.strip():
                logger.warning(f"Transcript {file.name} is empty. Skipping.")
                continue

            # Step 1: BERT Anonymization
            anonymized_text, success, msg = anonymizer.anonymize(transcript_text)
            
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

            # Step 2: Optional LLM Rewrite
            if use_llm:
                logger.info(f"Running LLM rewrite on {base_name} with model {target_llm_model}...")
                llm_result, status = call_llm_rewriter(anonymized_text, target_llm_model)
                
                if llm_result:
                    llm_filename = f"{base_name}_llm.txt"
                    llm_path = LLM_ANONNYM_FOLDER / llm_filename
                    
                    # Additional safety: Ensure LLM output path is within LLM_ANONNYM_FOLDER
                    if not validate_path(llm_path, LLM_ANONNYM_FOLDER):
                        logger.error(f"Security Alert: LLM output path traversal detected for {llm_filename}. Skipping.")
                        failed_count += 1
                        continue
                        
                    with open(llm_path, "w", encoding="utf-8") as f:
                        f.write(llm_result)
                    logger.info(f"LLM Rewritten transcript saved to: {llm_path}")
                    llm_processed_count += 1
                else:
                    logger.warning(f"LLM rewrite failed for {base_name}: {status}")

        except Exception as e:
            logger.error(f"Error processing transcript {file.name}: {e}")
            failed_count += 1

    logger.info(f"Anonymization phase complete.")
    logger.info(f"  BERT Processed: {processed_count}, Failed: {failed_count}")
    if use_llm:
        logger.info(f"  LLM Rewritten: {llm_processed_count}")

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
  python process.py --disable-llm            # Skip LLM rewriting
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
    
    args = parser.parse_args()
    
    # Configure logging verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine which steps to run (enabled by default, disabled if flag is set)
    run_transcription = not args.disable_transcription
    run_diarization = not args.disable_diarization
    run_anonymization = not args.disable_anonymization
    run_llm = not args.disable_llm
    
    # Log the execution plan
    logger.info("="*60)
    logger.info("PIPELINE EXECUTION PLAN")
    logger.info("="*60)
    logger.info(f"  Audio Extraction (Videos→WAV):     {'✅ ENABLED' if run_transcription else '❌ DISABLED'}")
    logger.info(f"  Transcription & Diarization:       {'✅ ENABLED' if run_transcription else '❌ DISABLED'}")
    if run_transcription:
        logger.info(f"    └─ Speaker Diarization:          {'✅ ENABLED' if run_diarization else '❌ DISABLED'}")
    logger.info(f"  BERT Anonymization:                {'✅ ENABLED' if run_anonymization else '❌ DISABLED'}")
    logger.info(f"  LLM Indirect Identifier Removal:   {'✅ ENABLED' if run_llm else '❌ DISABLED'}")
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
    if run_anonymization:
        process_anonymization(
            llm_rewrite_enabled=run_llm,
            llm_model_id=args.llm_model
        )
    else:
        logger.info("⏭️  Skipping anonymization (--disable-anonymization)")
        # If anonymization is disabled but LLM is enabled, warn the user
        if run_llm:
            logger.warning("⚠️  LLM rewrite requested but BERT anonymization is disabled. "
                          "LLM step will be skipped as it depends on anonymized input.")
    
    logger.info("Pipeline finished.")