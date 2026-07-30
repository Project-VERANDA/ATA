"""
TTS Engine Module - Unified Text-to-Speech functionality
Compatible with Piper backend
Location: ATA/pipeline/tts/tts_engine.py
"""

import os
import subprocess
from pathlib import Path
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Get project root (ATA directory) - go up 2 levels from tts_engine.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger.info(f"TTS Engine: Project root = {PROJECT_ROOT}")

# Load .env
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)
logger.info(f"TTS Engine: .env loaded from {env_path}, exists={env_path.exists()}")

# Configuration from .env
TTS_BACKEND = os.getenv('TTS_BACKEND', 'piper')
TTS_ENABLED = os.getenv('TTS_ENABLED', 'false').lower() == 'true'

# Paths relative to PROJECT_ROOT (ATA directory)
TTS_DIR = PROJECT_ROOT / "pipeline" / "tts"
TTS_BIN_DIR = TTS_DIR / "bin"
TTS_VOICE_DIR = TTS_DIR / "voices"
PIPER_EXECUTABLE = TTS_BIN_DIR / "piper"

logger.info(f"TTS Enabled: {TTS_ENABLED}")
logger.info(f"TTS Backend: {TTS_BACKEND}")
logger.info(f"Piper path: {PIPER_EXECUTABLE}")
logger.info(f"Piper exists: {PIPER_EXECUTABLE.exists()}")
logger.info(f"Voices dir: {TTS_VOICE_DIR}")
logger.info(f"Voices dir exists: {TTS_VOICE_DIR.exists()}")

def get_tts_status():
    """Return current TTS configuration and status."""
    return {
        'enabled': TTS_ENABLED,
        'backend': TTS_BACKEND,
        'piper_exists': PIPER_EXECUTABLE.exists(),
        'voices_dir_exists': TTS_VOICE_DIR.exists(),
        'voices_count': len(list(TTS_VOICE_DIR.glob("*.onnx"))) if TTS_VOICE_DIR.exists() else 0
    }

def get_available_tts_voices(language='en'):
    """Return list of available voices for given language."""
    voices = []
    if TTS_BACKEND == 'piper' and TTS_VOICE_DIR.exists():
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
    Returns: BytesIO buffer (if return_bytes=True) or None on error
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
    # Verify Piper exists
    if not PIPER_EXECUTABLE.exists():
        logger.error(f"Piper executable not found: {PIPER_EXECUTABLE}")
        logger.error(f"Contents of {TTS_BIN_DIR}: {list(TTS_BIN_DIR.iterdir()) if TTS_BIN_DIR.exists() else 'DIR MISSING'}")
        return None
    
    # Determine voice to use
    if voice_id is None:
        # Try default voice from .env
        default_voice_path = os.getenv('TTS_VOICE_PATH')
        if default_voice_path:
            voice_path = Path(default_voice_path).expanduser().resolve()
        else:
            # Find first available voice
            voices = get_available_tts_voices(language)
            if voices:
                voice_path = TTS_VOICE_DIR / f"{voices[0]['id']}.onnx"
            else:
                logger.error(f"No voices available for language: {language}")
                logger.error(f"Contents of {TTS_VOICE_DIR}: {list(TTS_VOICE_DIR.iterdir()) if TTS_VOICE_DIR.exists() else 'DIR MISSING'}")
                return None
    else:
        voice_path = TTS_VOICE_DIR / f"{voice_id}.onnx"
    
    if not voice_path.exists():
        logger.error(f"Voice file not found: {voice_path}")
        return None
    
    logger.info(f"Using Piper: {PIPER_EXECUTABLE} with voice: {voice_path.name}")
    
    # Run Piper - stdin/stdout mode
    buffer = BytesIO()
    cmd = [
        str(PIPER_EXECUTABLE),
        '-m', str(voice_path),
        '-f', '-',      # Read from stdin
        '-o', '-'       # Write to stdout
    ]
    
    try:
        proc = subprocess.run(
            cmd,
            input=text.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        
        if proc.returncode != 0:
            err_msg = proc.stderr.decode('utf-8')
            logger.error(f"Piper TTS error (code {proc.returncode}): {err_msg[:200]}")
            return None
        
        buffer.write(proc.stdout)
        buffer.seek(0)
        logger.info(f"Piper generated {len(buffer.getvalue())} bytes of audio")
        return buffer
        
    except subprocess.TimeoutExpired:
        logger.error("Piper TTS timed out")
        return None
    except FileNotFoundError as e:
        logger.error(f"Piper command failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Piper TTS error: {e}", exc_info=True)
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