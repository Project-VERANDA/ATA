# /mnt/Data_Mount/VERANDA_DataMount/Experimental/ATA/pipeline/tts/tts_engine.py

import os
import sys
import subprocess
from pathlib import Path
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Load .env first
from dotenv import load_dotenv
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

# TTS Configuration from .env
TTS_BACKEND = os.getenv('TTS_BACKEND', 'piper')
TTS_ENABLED = os.getenv('TTS_ENABLED', 'false').lower() == 'true'

# TTS Paths from .env (with fallbacks)
TTS_BIN_PATH = Path(os.getenv('TTS_BIN_PATH', './pipeline/tts/bin')).resolve()
TTS_VOICE_DIR = Path(os.getenv('TTS_VOICE_DIR', './pipeline/tts/voices')).resolve()
TTS_VOICE_PATH = Path(os.getenv('TTS_VOICE_PATH', '')).resolve() if os.getenv('TTS_VOICE_PATH') else None

# Piper executable
PIPER_EXECUTABLE = TTS_BIN_PATH / "piper"

def get_tts_status():
    """Return current TTS configuration and status."""
    return {
        'enabled': TTS_ENABLED,
        'backend': TTS_BACKEND,
        'piper_exists': PIPER_EXECUTABLE.exists(),
        'voices_dir_exists': TTS_VOICE_DIR.exists() if TTS_BACKEND == 'piper' else None,
        'default_voice_exists': TTS_VOICE_PATH.exists() if TTS_VOICE_PATH else None
    }

def get_available_tts_voices(language='en'):
    """Return list of available voices for given language."""
    voices = []
    
    if TTS_BACKEND == 'piper':
        if not TTS_VOICE_DIR.exists():
            logger.warning(f"Piper voices directory not found: {TTS_VOICE_DIR}")
            return voices
        
        for voice_file in TTS_VOICE_DIR.glob(f"*{language}*.onnx"):
            voice_name = voice_file.stem
            quality = 'high' if 'high' in voice_name.lower() else 'medium' if 'medium' in voice_name.lower() else 'low'
            voices.append({
                'id': voice_name,
                'name': voice_name.replace('_', ' ').title(),
                'language': language,
                'quality': quality,
                'offline': True
            })
    
    return voices

def generate_beep(duration_ms=400, freq=1000, return_bytes=False):
    """Generate a beep sound using ffmpeg."""
    try:
        buffer = BytesIO()
        proc = subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i',
            f'sine=frequency={freq}:duration={duration_ms/1000}',
            '-c:a', 'pcm_s16le', '-'
        ], capture_output=True, check=True)
        buffer.write(proc.stdout)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Beep generation failed: {e}")
        return None

def generate_speech(text, language='en', voice_id=None, return_bytes=False):
    """
    Generate speech using configured TTS backend.
    """
    if not TTS_ENABLED:
        logger.warning("TTS disabled in .env (TTS_ENABLED=false)")
        return None
    
    try:
        if TTS_BACKEND == 'piper':
            return _generate_speech_piper(text, language, voice_id, return_bytes)
        else:
            logger.error(f"Unsupported TTS backend: {TTS_BACKEND}")
            return None
    except Exception as e:
        logger.error(f"Speech generation failed: {e}", exc_info=True)
        return None

def _generate_speech_piper(text, language, voice_id, return_bytes):
    """Generate speech using Piper TTS."""
    # Check Piper executable
    if not PIPER_EXECUTABLE.exists():
        logger.error(f"Piper executable not found: {PIPER_EXECUTABLE}")
        return None
    
    # Determine voice to use
    if voice_id is None:
        if TTS_VOICE_PATH and TTS_VOICE_PATH.exists():
            voice_path = TTS_VOICE_PATH
        else:
            voices = get_available_tts_voices(language)
            if voices:
                voice_path = TTS_VOICE_DIR / f"{voices[0]['id']}.onnx"
            else:
                logger.error(f"No voices available for language: {language}")
                return None
    else:
        voice_path = TTS_VOICE_DIR / f"{voice_id}.onnx"
    
    if not voice_path.exists():
        logger.error(f"Voice file not found: {voice_path}")
        return None
    
    # Run Piper
    if return_bytes:
        buffer = BytesIO()
        stdout_pipe = subprocess.PIPE
    else:
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_file.close()
        stdout_pipe = tmp_file.name
    
    cmd = [
        str(PIPER_EXECUTABLE),
        '-m', str(voice_path),
        '--sentence-splitter', language,
        '--output-formatter', 'json',
    ]
    
    try:
        import tempfile
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

def synthesize_segment(text, speaker='SPEAKER_00', emotion=None):
    """Synthesize a single speaker segment with optional emotion."""
    import re
    clean_text = re.sub(r'^SPEAKER_\d+:\s*', '', text)
    if not clean_text.strip():
        return None
    return generate_speech(clean_text, return_bytes=True)

__all__ = [
    'TTS_BACKEND', 'TTS_ENABLED',
    'get_tts_status', 'get_available_tts_voices',
    'generate_beep', 'generate_speech', 'synthesize_segment'
]