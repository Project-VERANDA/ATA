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

# Create directories if they don't exist
for folder in [TRANSCRIPTS_FOLDER, ANNONYM_FOLDER, MODEL_FOLDER]:
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

# Anonymization Configuration
ANONYMIZATION_ENABLED = True
ANONYMIZATION_LEVEL = "standard"  # Options: 'basic', 'standard', 'strict'
ANONYMIZATION_METHOD = "local_ensemble"  # Options: 'local_bert', 'local_spacy', 'local_ensemble', 'remote_chat_ai'

# Remote Chat AI API Configuration
CHAT_AI_API_KEY = os.getenv('CHAT_AI_API_KEY', '')
CHAT_AI_ENDPOINT = os.getenv('CHAT_AI_ENDPOINT', 'https://chat-ai.academiccloud.de/v1')
DEFAULT_CHAT_AI_MODEL = os.getenv('CHAT_AI_MODEL', 'llama-3.1-8b-instruct')

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

def cleanup_gpu_resources():
    """Explicitly clears GPU memory and runs garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("GPU resources cleaned up.")

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
    if not AUDIOS_FOLDER.exists():
        logger.warning(f"No 'audios' folder found at {AUDIOS_FOLDER}. Skipping transcription.")
        return

    logger.info(f"Found 'audios' folder at {AUDIOS_FOLDER}. Starting transcription process...")

    # Verify local models exist before proceeding
    logger.info("Verifying local model files...")
    
    whisperx_model_valid = verify_whisperx_model(WHISPERX_MODEL_PATH)
    diarization_model_valid = verify_diarization_model(DIARIZATION_MODEL_PATH)
    
    if not whisperx_model_valid or not diarization_model_valid:
        logger.critical("One or more required local models are missing or invalid.")
        return

    # Load WhisperX Model
    try:
        logger.info(f"Loading WhisperX model from: {WHISPERX_MODEL_PATH}")
        
        model = whisperx.load_model(
            "large-v3", 
            DEVICE, 
            compute_type=COMPUTE_TYPE, 
            download_root=str(MODEL_FOLDER),
            local_files_only=True
        )
        logger.info("WhisperX model loaded successfully.")
        print(f"DEBUG: SCRIPT_DIR = {SCRIPT_DIR}")
        print(f"DEBUG: BASE_PATH = {BASE_PATH}")
        print(f"DEBUG: MODEL_FOLDER = {MODEL_FOLDER}")
        print(f"DEBUG: WHISPERX_MODEL_PATH = {WHISPERX_MODEL_PATH}")
        print(f"DEBUG: EXISTS? {WHISPERX_MODEL_PATH.exists()}")
        if WHISPERX_MODEL_PATH.exists():
            print(f"DEBUG: Contents: {list(WHISPERX_MODEL_PATH.iterdir())[:5]}")
        else:
            print(f"DEBUG: Parent exists? {WHISPERX_MODEL_PATH.parent.exists()}")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return

    # Load Diarization Pipeline from local model
    try:
        logger.info(f"Loading Diarization Pipeline from local model: {DIARIZATION_MODEL_PATH}")
        from pyannote.audio import Pipeline
        
        diarize_pipeline = Pipeline.from_pretrained(
            str(DIARIZATION_MODEL_PATH)
        )
        
        diarize_model = diarize_pipeline
        logger.info("Diarization Pipeline loaded successfully (offline mode).")
        
    except Exception as e:
        logger.critical(f"Failed to load local Diarization Pipeline: {e}")
        if 'model' in locals():
            del model
        cleanup_gpu_resources()
        return

    for file in AUDIOS_FOLDER.iterdir():
        if not file.is_file() or not file.suffix.lower() == ".wav":
            continue

        if not validate_path(file, AUDIOS_FOLDER):
            logger.error(f"Security Alert: Attempted path traversal detected for {file.name}. Skipping.")
            continue

        audio_path = file
        logger.info(f"Processing: {audio_path}")

        try:
            # Load audio for transcription
            audio = whisperx.load_audio(str(audio_path))
            result = model.transcribe(audio, batch_size=BATCH_SIZE, verbose=False, print_progress=False)

            # Align Whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
            result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

            # Diarization - extract speaker_diarization Annotation and convert to DataFrame
            diarize_output = diarize_model(str(audio_path), min_speakers=MIN_SPEAKERS, max_speakers=MAX_SPEAKERS)
            
            # Extract the speaker_diarization Annotation object
            speaker_diarization = diarize_output.speaker_diarization
            
            # Convert Annotation to list of segment dictionaries
            segments_list = []
            for turn, _, speaker in speaker_diarization.itertracks(yield_label=True):
                segments_list.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker
                })
            
            logger.info(f"Diarization extracted {len(segments_list)} speaker segments")
            
            # Convert list to Pandas DataFrame (Required by whisperx)
            import pandas as pd
            diarize_df = pd.DataFrame(segments_list)
            # Now assign speakers to transcribed segments
            result = whisperx.assign_word_speakers(diarize_df, result)
            result["segments"] = merge_consecutive_speaker_segments(result["segments"])

            base_name = sanitize_filename(file.stem)
            transcript_file = TRANSCRIPTS_FOLDER / f"{base_name}.txt"
            
            try:
                with open(transcript_file, "w", encoding="utf-8") as f:
                    for segment in result["segments"]:
                        speaker = segment.get("speaker", "Unknown")
                        text = segment.get("text", "")
                        f.write(f"{speaker}: {text}\n")
                logger.info(f"Transcription saved: {transcript_file}")
            except PermissionError:
                logger.error(f"Permission denied writing to {transcript_file}")
            except IOError as e:
                logger.error(f"I/O error writing transcript: {e}")

            del model_a, diarize_output, speaker_diarization, segments_list, diarize_df, result
            cleanup_gpu_resources()

        except Exception as e:
            logger.error(f"Error processing {file.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            cleanup_gpu_resources()

    del model, diarize_model
    cleanup_gpu_resources()
    logger.info("Transcription phase complete. Run 'process_anonymization()' next to anonymize the transcript(s).")


# Global variables to hold loaded models (so we don't reload every time)
_loaded_whisper_model = None
_loaded_diarize_model = None

def load_models():
    """Loads models locally. Called once on import or first use."""
    global _loaded_whisper_model, _loaded_diarize_model
    
    if _loaded_whisper_model and _loaded_diarize_model:
        return _loaded_whisper_model, _loaded_diarize_model

    # 1. Define paths
    logger.info(f"Loading WhisperX model from: {WHISPERX_MODEL_PATH}")
    
    # 2. Verify paths
    if not verify_whisperx_model(WHISPERX_MODEL_PATH):
        logger.critical("WhisperX model verification failed.")
        return None, None
    
    if not DIARIZATION_MODEL_PATH.exists():
        logger.warning(f"Diarization model not found at {DIARIZATION_MODEL_PATH}.")
        logger.warning("Will proceed without speaker diarization.")
        _loaded_diarize_model = None
    else:
        if not verify_diarization_model(DIARIZATION_MODEL_PATH):
            logger.warning("Diarization model verification failed. Will proceed without it.")
            _loaded_diarize_model = None
    
    # 3. Load WhisperX (local_files_only=True)
    try:
        _loaded_whisper_model = whisperx.load_model(
            "large-v3", 
            DEVICE, 
            compute_type=COMPUTE_TYPE, 
            download_root=str(MODEL_FOLDER),
            local_files_only=True
        )
        logger.info("WhisperX model loaded successfully.")
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return None, None
    
    # 4. Load Pyannote Pipeline (local_files_only=True) if available
    if _loaded_diarize_model is None and DIARIZATION_MODEL_PATH.exists():
        try:
            from pyannote.audio import Pipeline
            _loaded_diarize_model = Pipeline.from_pretrained(
                str(DIARIZATION_MODEL_PATH)
            )
            logger.info("Diarization Pipeline loaded successfully (offline mode).")
        except Exception as e:
            logger.warning(f"Failed to load local Diarization Pipeline: {e}")
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

## --- Anonymization Engine Class ---

# class AnonymizationEngine:
#     """
#     Encapsulates all logic related to text anonymization.
#     Handles local models (BERT, spaCy, Ensemble) and remote API calls.
#     """
    
#     def __init__(self, method="local_ensemble", level="standard", api_key="", endpoint="", default_model=""):
#         self.method = method
#         self.level = level
#         self.api_key = api_key or CHAT_AI_API_KEY
#         self.endpoint = endpoint or CHAT_AI_ENDPOINT
#         self.default_model = default_model or DEFAULT_CHAT_AI_MODEL
#         self.local_availability = self._check_local_availability()
        
#         if ANONYMIZATION_ENABLED:
#             logger.info(f"AnonymizationEngine initialized: Method={self.method}, Level={self.level}")
#             self._validate_configuration()

#     def _check_local_availability(self):
#         """Checks which local anonymizers are available."""
#         availability = {'bert': False, 'spacy': False, 'ensemble': False}
        
#         try:
#             from bert_anonymizer.anonymizer import anonymize_text_with_bert
#             availability['bert'] = True
#             logger.debug("Local BERT anonymizer available")
#         except (ImportError, OSError) as e:
#             logger.debug(f"BERT Anonymizer not available: {e}")
        
#         try:
#             from spacy_anonymizer.anonymizer import anonymize_text_with_spacy, is_spacy_model_available
#             availability['spacy'] = is_spacy_model_available()
#             if availability['spacy']:
#                 logger.debug("Local spaCy anonymizer available")
#         except (ImportError, OSError) as e:
#             logger.debug(f"spaCy Anonymizer not available: {e}")
        
#         availability['ensemble'] = availability['bert'] and availability['spacy']
#         if availability['ensemble']:
#             logger.debug("Local Ensemble anonymizer available")
            
#         return availability

#     def _validate_configuration(self):
#         """Validates that the chosen method is actually available."""
#         if self.method == 'local_ensemble' and not self.local_availability['ensemble']:
#             logger.warning("Method 'local_ensemble' requested but unavailable. Falling back to 'remote_chat_ai' if key exists.")
#             if self.api_key:
#                 self.method = 'remote_chat_ai'
#             else:
#                 logger.error("No fallback available. Anonymization will be skipped.")
#                 self.method = None
#         elif self.method == 'local_bert' and not self.local_availability['bert']:
#             logger.warning("Method 'local_bert' requested but unavailable.")
#             self.method = None
#         elif self.method == 'local_spacy' and not self.local_availability['spacy']:
#             logger.warning("Method 'local_spacy' requested but unavailable.")
#             self.method = None
#         elif self.method == 'remote_chat_ai' and not self.api_key:
#             logger.warning("Method 'remote_chat_ai' requested but no API key configured.")
#             self.method = None

#     def anonymize(self, text):
#         """
#         Main entry point for anonymization.
#         Routes to the appropriate handler based on configuration.
        
#         Returns:
#             tuple: (anonymized_text, success_status, message)
#         """
#         if not self.method:
#             return None, False, "Anonymization method not configured or unavailable."

#         logger.info(f"Running anonymization via {self.method}...")

#         try:
#             if self.method == 'local_ensemble':
#                 return self._run_local_ensemble(text)
#             elif self.method == 'local_bert':
#                 return self._run_local_bert(text)
#             elif self.method == 'local_spacy':
#                 return self._run_local_spacy(text)
#             elif self.method == 'remote_chat_ai':
#                 return self._run_remote_llm(text)
#             else:
#                 return None, False, f"Unknown method: {self.method}"
#         except Exception as e:
#             logger.error(f"Anonymization failed unexpectedly: {e}")
#             return None, False, str(e)

#     def _run_local_ensemble(self, text):
#         try:
#             from ensemble_anonymizer.anonymizer import anonymize_text_with_ensemble
#             result = anonymize_text_with_ensemble(text)
#             return result, True, "Success"
#         except Exception as e:
#             logger.error(f"Ensemble error: {e}")
#             return None, False, f"Ensemble error: {e}"

#     def _run_local_bert(self, text):
#         try:
#             from bert_anonymizer.anonymizer import anonymize_text_with_bert
#             result, _ = anonymize_text_with_bert(text)
#             return result, True, "Success"
#         except Exception as e:
#             logger.error(f"BERT error: {e}")
#             return None, False, f"BERT error: {e}"

#     def _run_local_spacy(self, text):
#         try:
#             from spacy_anonymizer.anonymizer import anonymize_text_with_spacy
#             result, _ = anonymize_text_with_spacy(text)
#             return result, True, "Success"
#         except Exception as e:
#             logger.error(f"spaCy error: {e}")
#             return None, False, f"spaCy error: {e}"

#     def _run_remote_llm(self, text):
#         import requests
        
#         prompts = {
#             'basic': """
#             Anonymize the following text by replacing only the most obvious personally identifiable information (PII) using these specific labels:
#             - Patient names with [NAME_PATIENT]
#             - Doctor names with [NAME_DOCTOR]
#             - Phone numbers with [CONTACT_PHONE]
#             - Email addresses with [CONTACT_EMAIL]
#             - Street addresses with [LOCATION_STREET]
#             - Cities with [LOCATION_CITY]
            
#             CRITICAL INSTRUCTIONS: 
#             - Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc.
#             - Do NOT translate any text. Keep ALL text in its original language.
#             - Return ONLY the anonymized text.
#             """,
#             'standard': """
#             Anonymize the following text by replacing personally identifiable information (PII) using these specific labels:
#             - Patient names with [NAME_PATIENT]
#             - Doctor names with [NAME_DOCTOR]
#             - Relative names with [NAME_RELATIVE]
#             - Other names with [NAME_OTHER]
#             - Professions with [PROFESSION]
#             - Phone numbers with [CONTACT_PHONE]
#             - Email addresses with [CONTACT_EMAIL]
#             - Fax numbers with [CONTACT_FAX]
#             - Street addresses with [LOCATION_STREET]
#             - Cities with [LOCATION_CITY]
#             - ZIP codes with [LOCATION_ZIP]
#             - Hospitals with [LOCATION_HOSPITAL]
#             - Organizations with [LOCATION_ORGANISATION]
#             - Dates with [DATE]
#             - Ages with [AGE]
#             - ID numbers with [ID]
            
#             CRITICAL INSTRUCTIONS: 
#             - Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc.
#             - Do NOT translate any text. Keep ALL text in its original language.
#             - Return ONLY the anonymized text.
#             """,
#             'strict': """
#             Thoroughly anonymize the following text by replacing all personally identifiable information (PII) using these specific labels:
#             - Patient names with [NAME_PATIENT]
#             - Doctor names with [NAME_DOCTOR]
#             - Relative names with [NAME_RELATIVE]
#             - External names with [NAME_EXT]
#             - Usernames with [NAME_USERNAME]
#             - Other names with [NAME_OTHER]
#             - Titles with [NAME_TITLE]
#             - Professions with [PROFESSION]
#             - Dates with [DATE]
#             - Ages with [AGE]
#             - Street addresses with [LOCATION_STREET]
#             - Cities with [LOCATION_CITY]
#             - ZIP codes with [LOCATION_ZIP]
#             - Countries with [LOCATION_COUNTRY]
#             - States with [LOCATION_STATE]
#             - Hospitals with [LOCATION_HOSPITAL]
#             - Organizations with [LOCATION_ORGANISATION]
#             - Other locations with [LOCATION_OTHER]
#             - ID numbers with [ID]
#             - Phone numbers with [CONTACT_PHONE]
#             - Email addresses with [CONTACT_EMAIL]
#             - Fax numbers with [CONTACT_FAX]
#             - URLs with [CONTACT_URL]
#             - Other contact information with [CONTACT_OTHER]
            
#             CRITICAL INSTRUCTIONS: 
#             - Do NOT anonymize or modify speaker identification tags like SPEAKER_00, SPEAKER_01, etc.
#             - Do NOT translate any text. Keep ALL text in its original language.
#             - Return ONLY the anonymized text.
#             """
#         }
        
#         prompt = prompts.get(self.level, prompts['standard'])
        
#         headers = {
#             'Authorization': f'Bearer {self.api_key}',
#             'Content-Type': 'application/json'
#         }
        
#         payload = {
#             "model": self.default_model,
#             "messages": [
#                 {"role": "system", "content": prompt},
#                 {"role": "user", "content": f"Text to anonymize:\n\n{text}"}
#             ],
#             "max_tokens": 8000,
#             "temperature": 0.1
#         }
        
#         max_retries = 3
#         timeout = 360
        
#         for attempt in range(max_retries):
#             try:
#                 response = requests.post(
#                     f"{self.endpoint}/chat/completions",
#                     headers=headers,
#                     json=payload,
#                     timeout=timeout
#                 )
#                 break
#             except requests.exceptions.Timeout:
#                 if attempt < max_retries - 1:
#                     logger.warning(f"Request timeout on attempt {attempt + 1}, retrying...")
#                     time.sleep(2)
#                     continue
#                 else:
#                     return None, False, "Timeout after retries"
#             except requests.exceptions.RequestException as e:
#                 return None, False, f"Connection error: {str(e)}"
        
#         if response.status_code != 200:
#             return None, False, f"API error: {response.status_code}"
        
#         response_data = response.json()
#         anonymized_text = response_data['choices'][0]['message']['content'].strip()
        
#         Remove any <thought> tags
#         anonymized_text = re.sub(r'<thought>.*?</thought>', '', anonymized_text, flags=re.DOTALL).strip()
        
#         return anonymized_text, True, "Success"

    # def save_anonymized_file(self, base_name, anonymized_text):
    #     """Saves the anonymized text to the 'annonym' folder."""
    #     if not anonymized_text:
    #         logger.warning("No text to save.")
    #         return False
            
    #     output_filename = f"{base_name}_anon.txt"
    #     output_path = ANONYM_OUTPUT_DIR / output_filename
        
    #     try:
    #         with open(output_path, "w", encoding="utf-8") as f:
    #             f.write(anonymized_text)
    #         logger.info(f"Anonymized transcript saved to: {output_path}")
    #         return True
    #     except Exception as e:
    #         logger.error(f"Failed to save anonymized file: {e}")
    #         return False


# --- Step 3: Anonymize Existing Transcripts ---

# def process_anonymization():
#     """
#     Reads raw transcripts from the 'transcripts' folder, anonymizes them,
#     and saves the results to the 'annonym' folder.
#     This step is independent of audio processing.
#     """
#     if not TRANSCRIPTS_FOLDER.exists():
#         logger.warning(f"No 'transcripts' folder found at {TRANSCRIPTS_FOLDER}. Skipping anonymization.")
#         return

#     logger.info(f"Found 'transcripts' folder at {TRANSCRIPTS_FOLDER}. Starting anonymization process...")

#     Initialize Anonymization Engine
#     if not ANONYMIZATION_ENABLED:
#         logger.info("Anonymization is disabled in configuration.")
#         return

#     anonymizer = AnonymizationEngine(
#         method=ANONYMIZATION_METHOD,
#         level=ANONYMIZATION_LEVEL,
#         api_key=CHAT_AI_API_KEY,
#         endpoint=CHAT_AI_ENDPOINT,
#         default_model=DEFAULT_CHAT_AI_MODEL
#     )

#     if not anonymizer.method:
#         logger.error("Anonymization engine failed to initialize a valid method. Aborting.")
#         return

#     processed_count = 0
#     failed_count = 0

#     for file in TRANSCRIPTS_FOLDER.iterdir():
#         if not file.is_file() or not file.suffix.lower() == ".txt":
#             continue
        
#         Skip files that are already anonymized (optional safety check)
#         if "_anon" in file.name:
#             logger.debug(f"Skipping already anonymized file: {file.name}")
#             continue

#         base_name = file.stem
#         logger.info(f"Processing transcript: {file.name}")

#         try:
#             with open(file, "r", encoding="utf-8") as f:
#                 transcript_text = f.read()

#             if not transcript_text.strip():
#                 logger.warning(f"Transcript {file.name} is empty. Skipping.")
#                 continue

#             anonymized_text, success, msg = anonymizer.anonymize(transcript_text)
            
#             if success and anonymized_text:
#                 anonymizer.save_anonymized_file(base_name, anonymized_text)
#                 processed_count += 1
#             else:
#                 logger.warning(f"Anonymization failed for {base_name}: {msg}")
#                 failed_count += 1

#         except Exception as e:
#             logger.error(f"Error processing transcript {file.name}: {e}")
#             failed_count += 1

#     logger.info(f"Anonymization phase complete. Processed: {processed_count}, Failed: {failed_count}")

# --- Main Execution ---

if __name__ == "__main__":
    logger.info("Starting Audio Anonymizer full pipeline...")
    
    process_videos()
    process_audios()
    #process_anonymization()
    
    logger.info("Pipeline finished.")