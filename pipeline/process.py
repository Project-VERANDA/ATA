import os
import sys
import whisperx
import ffmpeg
import torch
import subprocess
import gc
import logging
import re
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

# Traverse up until we find a folder named 'MAIN' or hit the root
while BASE_PATH.name != "MAIN" and BASE_PATH != BASE_PATH.parent:
    BASE_PATH = BASE_PATH.parent

if BASE_PATH.name != "MAIN":
    logger.critical(f"Could not locate 'MAIN' folder. Script expects to be run from within .../MAIN/")
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

if not DIARIZATION_MODEL_PATH.exists():
    logger.warning(f"WARNING: Diarization model not found at {DIARIZATION_MODEL_PATH}.")
    logger.warning("Speaker diarization will be disabled. Using generic speaker labels.")
    # We can still proceed without diarization, but warn the user

logger.info(f"✅ Paths verified successfully.")
logger.info(f"   Model Folder: {MODEL_FOLDER}")
logger.info(f"   WhisperX Model: {WHISPERX_MODEL_PATH.name}")
logger.info(f"   Diarization Model: {DIARIZATION_MODEL_PATH.name if DIARIZATION_MODEL_PATH.exists() else 'MISSING'}")

# Anonymization Configuration
ANONYMIZATION_ENABLED = True
ANONYMIZATION_LEVEL = "standard"  # Options: 'basic', 'standard', 'strict'
ANONYMIZATION_METHOD = "local_mmbert"  # Options: 'local_bert', 'local_spacy', 'local_ensemble', 'remote_chat_ai'

# Remote Chat AI API Configuration
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY', '')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://llm.cloud.cci.charite.de/v1')
DEFAULT_CHAT_AI_MODEL = os.getenv('CHAT_AI_MODEL', 'cle-Qwen3.5-397B-A17B-FP8')
LLM_REWRITE_ENABLED = True

LLM_REWRITE_SYSTEM_PROMPT = (
    "You are an expert privacy auditor specializing in de-identification. "
    "Your task is to rewrite the provided text to remove any **indirect identifiers**. "
    "Indirect identifiers include: specific job titles, unique combinations of demographics, rare locations, specific dates, "
    "unique medical conditions, or any detail that could allow someone to identify the speaker when combined with other data. "
    "Replace these specific details with generic placeholders like [INDIRECT_ID] or generalize the description. "
    "IMPORTANT: Preserve the original speaker tags (e.g., SPEAKER_00, SPEAKER_01) exactly as they appear. "
    "Do not change the general meaning or flow of the conversation. "
    "Return ONLY the rewritten text. Do not include any introductory or concluding remarks."
)


AVAILABLE_LLM_MODELS = {
    'medgemma': 'medgemma-1.5-4b-it',
    'medgemma27b': 'medgemma-27b-it',
    'gpt-oss-120b': 'gpt-oss-120b',
    'Qwen3.5-27B': 'Qwen3.5-27B',
    'Qwen3.5-397B-A17B': 'Qwen3.5-397B-A17B',
    'qwen3-asr-1.7b': 'Qwen3-ASR-1.7B',
    'cle-Kimi-K2.5': 'Kimi-K2.5',
    'cle-Qwen3.5-397B-A17B-FP8': 'Qwen3.5-397B-A17B-FP8'
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

def merge_consecutive_speaker_segments(segments):
    """Merges consecutive segments spoken by the same speaker."""
    merged_segments = []
    prev_segment = None

    for segment in segments:
        speaker = segment.get("speaker", "Unknown")
        text = segment.get("text", "")

        if prev_segment and prev_segment["speaker"] == speaker:
            prev_segment["text"] += " " + text
        else:
            if prev_segment:
                merged_segments.append(prev_segment)
            prev_segment = {"speaker": speaker, "text": text}

    if prev_segment:
        merged_segments.append(prev_segment)

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

# --- Step 1: Extract Audio ---

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

# --- Step 2: Transcribe & Diarize ---

def process_audios():
    """
    Processes all audio files in the AUDIOS_FOLDER using the globally loaded models.
    Ensures GPU usage via load_models().
    """
    global _loaded_whisper_model, _loaded_diarize_model

    if not AUDIOS_FOLDER.exists():
        logger.warning(f"No 'audios' folder found at {AUDIOS_FOLDER}. Skipping transcription.")
        return

    logger.info(f"Found 'audios' folder at {AUDIOS_FOLDER}. Starting transcription process...")

    # ---------------------------------------------------------
    # STEP 1: Load Models (Triggers GPU Check & .to(device))
    # ---------------------------------------------------------
    logger.info("Loading models (this will trigger GPU detection logs)...")
    _loaded_whisper_model, _loaded_diarize_model = load_models()

    if not _loaded_whisper_model:
        logger.critical("Failed to load WhisperX model. Aborting transcription.")
        return
    
    if not _loaded_diarize_model:
        logger.warning("Diarization model failed to load. Proceeding with generic speaker labels.")

    logger.info("Models loaded successfully. Starting file processing...")

    # ---------------------------------------------------------
    # STEP 2: Process Files
    # ---------------------------------------------------------
    processed_count = 0
    failed_count = 0

    for file in AUDIOS_FOLDER.iterdir():
        if not file.is_file() or not file.suffix.lower() == ".wav":
            continue

        if not validate_path(file, AUDIOS_FOLDER):
            logger.error(f"Security Alert: Attempted path traversal detected for {file.name}. Skipping.")
            continue

        audio_path = file
        logger.info(f"Processing: {file.name}")

        try:
            # 1. Load Audio
            audio = whisperx.load_audio(str(audio_path))
            logger.debug(f"Audio loaded. Duration: {len(audio)/16000:.2f}s")

            # 2. Transcribe (Using Global Model)
            # Note: _loaded_whisper_model is already on GPU if available
            result = _loaded_whisper_model.transcribe(audio, batch_size=BATCH_SIZE, verbose=False, print_progress=False)
            logger.info(f"Transcription completed. Detected language: {result.get('language', 'unknown')}")

            # 3. Align
            if result.get("language"):
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
                    logger.debug("Alignment completed.")
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}. Proceeding without alignment.")

            # 4. Diarize (Using Global Model)
            if _loaded_diarize_model:
                try:
                    logger.debug("Running speaker diarization...")
                    # Pass the file path to the pipeline (Pyannote handles loading internally)
                    # The pipeline is already on GPU thanks to load_models()
                    diarize_output = _loaded_diarize_model(str(audio_path), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
                    
                    speaker_diarization = diarize_output.speaker_diarization
                    
                    # Convert to list of segments
                    segments_list = []
                    for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                        segments_list.append({
                            'start': turn.start,
                            'end': turn.end,
                            'speaker': speaker
                        })
                    
                    logger.info(f"Diarization extracted {len(segments_list)} speaker segments")
                    
                    # Assign speakers to transcribed segments
                    import pandas as pd
                    diarize_df = pd.DataFrame(segments_list)
                    result = whisperx.assign_word_speakers(diarize_df, result)
                    logger.debug("Speakers assigned.")
                    
                except Exception as e:
                    logger.error(f"Diarization failed: {e}")
                    logger.warning("Falling back to generic speaker labels.")
                    for i, segment in enumerate(result["segments"]):
                        segment["speaker"] = f"SPEAKER_{i%2:02d}"
            else:
                # Fallback if no model loaded
                logger.warning("No diarization model loaded. Using generic speaker labels.")
                for i, segment in enumerate(result["segments"]):
                    segment["speaker"] = f"SPEAKER_{i%2:02d}"

            # 5. Merge Consecutive Segments
            merged_segments = merge_consecutive_speaker_segments(result["segments"])
            logger.debug(f"Merged into {len(merged_segments)} final segments.")

            # 6. Save Transcript
            base_name = sanitize_filename(file.stem)
            transcript_file = TRANSCRIPTS_FOLDER / f"{base_name}.txt"
            
            try:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    for segment in merged_segments:
                        speaker = segment.get("speaker", "Unknown")
                        text = segment.get("text", "")
                        f.write(f"{speaker}: {text}\n")
                logger.info(f"Transcription saved: {transcript_file}")
                processed_count += 1
            except PermissionError:
                logger.error(f"Permission denied writing to {transcript_file}")
                failed_count += 1
            except IOError as e:
                logger.error(f"I/O error writing transcript: {e}")
                failed_count += 1

            # 7. Cleanup GPU Resources for this file
            cleanup_gpu_resources(
                audio, 
                result, 
                model_a if 'model_a' in locals() else None, 
                metadata if 'metadata' in locals() else None, 
                diarize_output if 'diarize_output' in locals() else None, 
                speaker_diarization if 'speaker_diarization' in locals() else None, 
                segments_list if 'segments_list' in locals() else None, 
                diarize_df if 'diarize_df' in locals() else None
            )
            
            # Explicitly delete references
            del audio, result
            if 'model_a' in locals(): del model_a
            if 'metadata' in locals(): del metadata
            if 'diarize_output' in locals(): del diarize_output
            if 'speaker_diarization' in locals(): del speaker_diarization
            if 'segments_list' in locals(): del segments_list
            if 'diarize_df' in locals(): del diarize_df

        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            failed_count += 1
            
            # Safe cleanup for error cases
            cleanup_gpu_resources()

    # End of file loop
    logger.info(f"Transcription phase complete.")
    logger.info(f"  Processed: {processed_count}, Failed: {failed_count}")
    
    # Optional: Clear global models if you want to free VRAM after the whole batch
    del _loaded_whisper_model, _loaded_diarize_model
    cleanup_gpu_resources()


# Global variables to hold loaded models (so we don't reload every time)
_loaded_whisper_model = None
_loaded_diarize_model = None

def load_models():
    """Loads models locally on GPU if available."""
    global _loaded_whisper_model, _loaded_diarize_model
    
    if _loaded_whisper_model and _loaded_diarize_model:
        return _loaded_whisper_model, _loaded_diarize_model

    # 1. Determine Device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    
    logger.info(f"=== DEVICE CHECK ===")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if device_str == "cuda":
        logger.info(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"Device Object: {device}")
    else:
        logger.warning("⚠️ NO GPU DETECTED. Falling back to CPU. Performance will be slow.")
    logger.info("====================")

    # 2. Load WhisperX Model (GPU)
    try:
        logger.info(f"Loading WhisperX model on {device_str}...")
        _loaded_whisper_model = whisperx.load_model(
            "large-v3", 
            device_str, 
            compute_type="float16" if device_str == "cuda" else "float32", 
            download_root=str(MODEL_FOLDER),
            local_files_only=True
        )
        logger.info("✅ WhisperX model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return None, None
    
    # 3. Load Diarization Pipeline (GPU)
    if DIARIZATION_MODEL_PATH.exists():
        try:
            logger.info(f"Loading Diarization Pipeline on {device_str}...")
            from pyannote.audio import Pipeline
            
            # Load the pipeline
            _loaded_diarize_model = Pipeline.from_pretrained(
                str(DIARIZATION_MODEL_PATH)
            )
            
            # Set pipeline to use GPU if nvidia GPU detected.
            if device_str == "cuda":
                _loaded_diarize_model.to(device)  # Uses the torch.device object
                logger.info("✅ Diarization Pipeline moved to GPU using .to(torch.device('cuda')).")
            else:
                logger.warning("⚠️ Diarization Pipeline loaded on CPU.")
                
        except Exception as e:
            logger.error(f"Failed to load Diarization Pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            _loaded_diarize_model = None
    else:
        logger.warning("Diarization model path not found. Skipping.")
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
        result = _loaded_whisper_model.transcribe(audio, batch_size=32, language=language)
        logger.info(f"Transcription completed. Detected language: {result.get('language', 'unknown')}")
        
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
                diarize_output = _loaded_diarize_model(audio_path, min_speakers=2, max_speakers=4)
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

        # 6. MERGE CONSECUTIVE SEGMENTS
        logger.info("Merging consecutive speaker segments...")
        merged_segments = []
        prev_segment = None
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            text = segment.get("text", "")
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

class AnonymizationEngine:
    """
    Encapsulates all logic related to text anonymization using the local mmbert model
    with its custom ModernBertCRF architecture.
    """
    
    def __init__(self, method="local_mmbert", level="standard", model_path=None):
        self.method = method
        self.level = level
        # Point to the folder containing crf_config.json and pytorch_model.bin
        self.model_path = model_path or (MODEL_FOLDER / "mmbert_multilingual_pii_ner" / "jhu-clsp-mmBERT-base-multilingual-pii")
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
        """Main entry point for anonymization using the custom CRF model."""
        if not self.method or not self.model or not self.tokenizer:
            return None, False, "Anonymization model not loaded."

        logger.info(f"Running anonymization via mmbert (CRF)...")

        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # Remove token_type_ids if present to match model expectation
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                emissions = outputs["logits"]
                mask = inputs["attention_mask"].bool()
                
                # CRF Decoding
                predictions = self.model.decode(emissions, mask)

            # Map predictions to labels
            # predictions is a list of lists (batch_size x seq_len)
            pred_ids = predictions[0] # Take first batch item
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            id2label = self.config["id2label"]
            labels = [id2label[str(pid)] for pid in pred_ids]
            
            # Reconstruct text
            label_map = self._get_labels()
            result_tokens = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                label = labels[i]
                
                # Skip special tokens
                if token in ['[CLS]', '[SEP]', '[PAD]', '<pad>', '<cls>', '<sep>']:
                    i += 1
                    continue
                
                # Handle B- and I- tags
                if label.startswith('B-') or label.startswith('I-'):
                    entity_type = label.split('-')[1]
                    replacement_tag = label_map.get(entity_type, '[UNKNOWN_PII]')
                    result_tokens.append(replacement_tag)
                    
                    # Skip subsequent I- tags for this entity
                    j = i + 1
                    while j < len(labels) and labels[j].startswith('I-') and labels[j].split('-')[1] == entity_type:
                        j += 1
                    i = j
                else:
                    # Clean token
                    clean_token = token.replace('##', '').replace('▁', ' ')
                    result_tokens.append(clean_token)
                    i += 1
            
            # Join and clean spacing
            anonymized_text = "".join(result_tokens).replace("  ", " ").strip()
            anonymized_text = re.sub(r'\s+\[', '[', anonymized_text)
            anonymized_text = re.sub(r'\]\s+', ']', anonymized_text)
            
            logger.info(f"Anonymization complete. Length: {len(anonymized_text)}")
            return anonymized_text, True, "Success"

        except Exception as e:
            logger.error(f"Anonymization failed: {e}", exc_info=True)
            return None, False, str(e)

def call_llm_rewriter(text, model_id, system_prompt=None):
    """
    Calls the external LLM API to rewrite text.
    If system_prompt is None, it defaults to the global LLM_REWRITE_SYSTEM_PROMPT.
    """
    if not CHAT_AI_API_KEY:
        logger.error("LLM API Key not configured. Skipping LLM rewrite.")
        return None, "API Key missing"

    # Use the provided prompt or fall back to the global constant
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

        # Safety check: Ensure content exists and is not None
        if not chat_completion.choices or not chat_completion.choices[0].message:
            raise ValueError("API response missing choices or message object.")
            
        content = chat_completion.choices[0].message.content
        
        if content is None:
            raise ValueError("API returned None for message content. The model may have failed silently.")
            
        rewritten_text = content.strip()
        
        if not rewritten_text:
            raise ValueError("API returned an empty string after stripping.")

        return rewritten_text, "Success"

    except Exception as e:
        logger.error(f"LLM Rewriter failed: {e}")
        return None, str(e)

# --- Step 3: Anonymize Existing Transcripts (Active) ---

def process_anonymization(llm_rewrite_enabled=None, llm_model_id=None):
    """
    Reads raw transcripts, anonymizes them with BERT, and optionally rewrites with LLM.
    """
    if not TRANSCRIPTS_FOLDER.exists():
        logger.warning(f"No 'transcripts' folder found at {TRANSCRIPTS_FOLDER}. Skipping anonymization.")
        return

    logger.info(f"Found 'transcripts' folder at {TRANSCRIPTS_FOLDER}. Starting anonymization process...")

    # Determine LLM settings
    use_llm = llm_rewrite_enabled if llm_rewrite_enabled is not None else LLM_REWRITE_ENABLED
    target_llm_model = llm_model_id if llm_model_id else DEFAULT_CHAT_AI_MODEL

    if use_llm and not CHAT_AI_API_KEY:
        logger.warning("LLM rewrite requested but no API key found. Disabling LLM step.")
        use_llm = False

    # Initialize Anonymization Engine (BERT)
    anonymizer = AnonymizationEngine(
        method=ANONYMIZATION_METHOD,
        level=ANONYMIZATION_LEVEL,
        model_path=MODEL_FOLDER / "mmbert_multilingual_pii_ner" / "jhu-clsp-mmBERT-base-multilingual-pii"
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
        
        # SECURITY FIX: Validate that the file path is strictly within TRANSCRIPTS_FOLDER
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
                        logger.error(f"Security Alert: LLM output path traversal detected for {llm_filename}. Skipping save.")
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
            model_path=MODEL_FOLDER / "mmbert_multilingual_pii_ner" / "jhu-clsp-mmBERT-base-multilingual-pii"
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
    
    parser = argparse.ArgumentParser(description="Run the Anonymization Pipeline")
    parser.add_argument('--enable-llm', action='store_true', help='Enable LLM indirect identifier removal')
    parser.add_argument('--llm-model', type=str, default=None, help='Specific LLM model key (e.g., medgemma) to use')
    args = parser.parse_args()
    
    logger.info("Starting Audio Anonymizer full pipeline...")
    
    process_videos()
    process_audios()
    process_anonymization(
        llm_rewrite_enabled=args.enable_llm,
        llm_model_id=args.llm_model
    )
    
    logger.info("Pipeline finished.")