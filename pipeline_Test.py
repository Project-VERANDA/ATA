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
BASE_PATH_STR = "/mnt/Data_Mount/VERANDA_DataMount/pipeline"
BASE_PATH = Path(BASE_PATH_STR).resolve()

# Derived paths
VIDEOS_FOLDER = BASE_PATH / "videos"
AUDIOS_FOLDER = BASE_PATH / "audios"
TRANSCRIPTS_FOLDER = BASE_PATH / "transcripts"
MODEL_FOLDER = BASE_PATH / "model"
ANNONYM_FOLDER = BASE_PATH / "annonym"

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

# Local Model Paths
WHISPERX_MODEL_PATH = MODEL_FOLDER / "models--Systran--faster-whisper-large-v3"
DIARIZATION_MODEL_PATH = MODEL_FOLDER / "models--pyannote--speaker-diarization-community-1"

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
    if not config_files:
        logger.error(f"Diarization model missing config file")
        return False
    
    required_subdirs = ['embedding', 'plda', 'segmentation']
    missing_subdirs = [d for d in required_subdirs if not (path / d).exists()]
    
    if missing_subdirs:
        logger.warning(f"Diarization model missing subdirectories: {missing_subdirs}")
    
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
        
    except Exception as e:
        logger.critical(f"Failed to load WhisperX model: {e}")
        return

    # Load Diarization Pipeline from local model
    try:
        logger.info("Loading Diarization Pipeline from local model...")
        from pyannote.audio import Pipeline
        
        diarize_pipeline = Pipeline.from_pretrained(
            str(DIARIZATION_MODEL_PATH)
        )
        
        diarize_model = diarize_pipeline
        logger.info("Diarization Pipeline loaded successfully (offline mode).")
        
    except Exception as e:
        logger.critical(f"Failed to load local Diarization Pipeline: {e}")
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
    logger.info("All processing complete.")

# --- Main Execution ---

if __name__ == "__main__":
    logger.info("Starting VERANDA Pipeline (Offline Mode - Fixed Diarization)...")
    
    process_videos()
    process_audios()
    
    logger.info("Pipeline finished.")
