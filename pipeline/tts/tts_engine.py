"""
TTS Engine Module - Unified Text-to-Speech functionality
Compatible with Piper and Coqui XTTS backends
"""

import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path
from io import BytesIO

# Add pipeline path for imports
pipeline_dir = Path(__file__).resolve().parent.parent
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

logger = logging.getLogger(__name__)

# TTS Configuration
TTS_BACKEND = os.getenv('TTS_BACKEND', 'piper')
TTS_ENABLED = os.getenv('TTS_ENABLED', 'false').lower() == 'true'

# TTS Paths (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TTS_BIN_DIR = PROJECT_ROOT / "pipeline" / "tts" / "bin"
PIPER_EXECUTABLE = TTS_BIN_DIR / "piper"

# Default voice paths (adjust based on your actual voice files)
PIPER_VOICES_DIR = PROJECT_ROOT / "pipeline" / "tts" / "voices"


def get_tts_status():
    """Return current TTS configuration and status."""
    return {
        'enabled': TTS_ENABLED,
        'backend': TTS_BACKEND,
        'piper_exists': PIPER_EXECUTABLE.exists(),
        'voices_dir_exists': PIPER_VOICES_DIR.exists() if TTS_BACKEND == 'piper' else None
    }


def get_available_tts_voices(language='en'):
    """Return list of available voices for given language."""
    voices = []
    
    if TTS_BACKEND == 'piper':
        if not PIPER_VOICES_DIR.exists():
            logger.warning(f"Piper voices directory not found: {PIPER_VOICES_DIR}")
            return voices
        
        for voice_file in PIPER_VOICES_DIR.glob(f"*{language}*.onnx"):
            voice_name = voice_file.stem
            # Extract quality from filename (e.g., en_US-libritts-high)
            quality = 'high' if 'high' in voice_name.lower() else 'medium'
            voices.append({
                'id': voice_name,
                'name': voice_name.replace('_', ' ').title(),
                'language': language,
                'quality': quality,
                'offline': True
            })
    
    elif TTS_BACKEND == 'coqui_xtts':
        # Coqui XTTS supports voice cloning
        voices = [
            {'id': 'default', 'name': 'Default Voice', 'language': language, 'cloning_supported': True},
            {'id': 'female_1', 'name': 'Female Voice 1', 'language': language, 'cloning_supported': True},
            {'id': 'male_1', 'name': 'Male Voice 1', 'language': language, 'cloning_supported': True},
        ]
    
    return voices


def generate_beep(duration_ms=400, freq=1000, return_bytes=False):
    """Generate a beep sound using ffmpeg or Piper."""
    try:
        buffer = BytesIO()
        
        proc = subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i',
            f'sine=frequency={freq}:duration={duration_ms/1000}',
            '-c:a', 'libmp3lame' if not return_bytes else 'pcm_s16le',
            '-'
        ], capture_output=True, check=True)
        
        if return_bytes:
            buffer.write(proc.stdout)
            buffer.seek(0)
            return buffer
        else:
            # Save to temp file
            tmp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
            tmp_file.write(proc.stdout)
            tmp_file.close()
            return tmp_file.name
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Beep generation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Beep generation error: {e}")
        return None


def generate_speech(text, language='en', voice_id=None, return_bytes=False):
    """
    Generate speech using configured TTS backend.
    
    Args:
        text: Text to synthesize
        language: Language code (e.g., 'en', 'de')
        voice_id: Specific voice to use (optional)
        return_bytes: Return BytesIO buffer instead of file path
    
    Returns:
        BytesIO buffer or file path containing audio data
    """
    if not TTS_ENABLED:
        logger.warning("TTS disabled in .env (TTS_ENABLED=false)")
        return None
    
    try:
        if TTS_BACKEND == 'piper':
            return _generate_speech_piper(text, language, voice_id, return_bytes)
        elif TTS_BACKEND == 'coqui_xtts':
            return _generate_speech_coqui(text, language, voice_id, return_bytes)
        else:
            logger.error(f"Unknown TTS backend: {TTS_BACKEND}")
            return None
            
    except Exception as e:
        logger.error(f"Speech generation failed: {e}", exc_info=True)
        return None


def _generate_speech_piper(text, language, voice_id, return_bytes):
    """Generate speech using Piper TTS."""
    if not PIPER_EXECUTABLE.exists():
        logger.error(f"Piper executable not found: {PIPER_EXECUTABLE}")
        return None
    
    # Auto-select voice if not specified
    if voice_id is None:
        voices = get_available_tts_voices(language)
        if voices:
            voice_id = voices[0]['id']
        else:
            logger.error(f"No voices available for language: {language}")
            return None
    
    voice_file = PIPER_VOICES_DIR / f"{voice_id}.onnx"
    if not voice_file.exists():
        logger.error(f"Voice file not found: {voice_file}")
        return None
    
    # Create temp file for output
    if return_bytes:
        buffer = BytesIO()
        stdout_pipe = subprocess.PIPE
    else:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_file.close()
        stdout_pipe = tmp_file.name
    
    # Build Piper command
    cmd = [
        str(PIPER_EXECUTABLE),
        '-m', str(voice_file),
        '--sentence-splitter', 'en',
        '--output-formatter', 'json',
    ]
    
    try:
        # Run Piper with text as input
        proc = subprocess.run(
            cmd,
            input=text.encode('utf-8'),
            stdout=stdout_pipe,
            stderr=subprocess.PIPE,
            timeout=30
        )
        
        if proc.returncode != 0:
            logger.error(f"Piper TTS error: {proc.stderr.decode('utf-8')}")
            return None
        
        if return_bytes:
            buffer.write(proc.stdout)
            buffer.seek(0)
            return buffer
        else:
            return tmp_file.name
            
    except subprocess.TimeoutExpired:
        logger.error("Piper TTS timed out")
        return None
    except Exception as e:
        logger.error(f"Piper TTS error: {e}")
        return None


def _generate_speech_coqui(text, language, voice_id, return_bytes):
    """Generate speech using Coqui XTTS."""
    try:
        from TTS.api import TTS
        
        # Initialize Coqui TTS
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        
        if return_bytes:
            buffer = BytesIO()
            tts.tts_to_file(
                text=text,
                language=language,
                file_path=buffer,
                split_sentences=True
            )
            buffer.seek(0)
            return buffer
        else:
            tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp_file.close()
            tts.tts_to_file(
                text=text,
                language=language,
                file_path=tmp_file.name,
                split_sentences=True
            )
            return tmp_file.name
            
    except ImportError:
        logger.error("Coqui TTS library not installed (pip install TTS)")
        return None
    except Exception as e:
        logger.error(f"Coqui XTTS error: {e}")
        return None


def synthesize_segment(text, speaker='SPEAKER_00', emotion=None):
    """
    Synthesize a single speaker segment with optional emotion.
    Note: Emotion support depends on TTS backend capabilities.
    """
    # Strip speaker tag for synthesis
    import re
    clean_text = re.sub(r'^SPEAKER_\d+:\s*', '', text)
    
    if not clean_text.strip():
        return None
    
    return generate_speech(clean_text, return_bytes=True)


# Exports for interoperability
__all__ = [
    'TTS_BACKEND',
    'TTS_ENABLED', 
    'get_tts_status',
    'get_available_tts_voices',
    'generate_beep',
    'generate_speech',
    'synthesize_segment',
]