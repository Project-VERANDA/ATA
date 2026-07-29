"""
TTS Engine Module - Unified Text-to-Speech functionality
Compatible with Piper backend
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Load .env first
from dotenv import load_dotenv

# Find ATA project root by traversing up from this file
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent.parent  # Goes: tts -> pipeline -> ATA
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

logger.info(f"TTS Engine: Project root detected at {PROJECT_ROOT}")
logger.info(f"TTS Engine: .env path={env_path}, exists={env_path.exists()}")

# TTS Configuration from .env
TTS_BACKEND = os.getenv('TTS_BACKEND', 'piper')
TTS_ENABLED = os.getenv('TTS_ENABLED', 'false').lower() == 'true'

# TTS Paths - use absolute paths rooted at ATA
TTS_BIN_DIR = PROJECT_ROOT / "pipeline" / "tts" / "bin"
TTS_VOICE_DIR = PROJECT_ROOT / "pipeline" / "tts" / "voices"

# Piper executable (no /piper appended - executable IS "piper")
PIPER_EXECUTABLE = TTS_BIN_DIR / "piper"

logger.info(f"TTS Engine: TTS_BACKEND={TTS_BACKEND}")
logger.info(f"TTS Engine: TTS_ENABLED={TTS_ENABLED}")
logger.info(f"TTS Engine: TTS_BIN_DIR={TTS_BIN_DIR}")
logger.info(f"TTS Engine: TTS_VOICE_DIR={TTS_VOICE_DIR}")
logger.info(f"TTS Engine: PIPER_EXECUTABLE={PIPER_EXECUTABLE}")
logger.info(f"TTS Engine: PIPER exists? {PIPER_EXECUTABLE.exists()}")

def get_tts_status():
    return {
        'enabled': TTS_ENABLED,
        'backend': TTS_BACKEND,
        'piper_exists': PIPER_EXECUTABLE.exists(),
        'voices_dir_exists': TTS_VOICE_DIR.exists() if TTS_BACKEND == 'piper' else None,
        'piper_bin_dir': str(TTS_BIN_DIR),
        'voices_dir': str(TTS_VOICE_DIR)
    }

def get_available_tts_voices(language='en'):
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
    if not TTS_ENABLED:
        logger.warning("TTS disabled in .env")
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
    # Verify Piper exists
    if not PIPER_EXECUTABLE.exists():
        logger.error(f"Piper executable not found: {PIPER_EXECUTABLE}")
        logger.error(f"Contents of {TTS_BIN_DIR}: {list(TTS_BIN_DIR.iterdir()) if TTS_BIN_DIR.exists() else 'DIR MISSING'}")
        return None
    
    # Determine voice to use
    if voice_id is None:
        # Check if default voice file exists (from .env TTS_VOICE_PATH)
        default_voice_path = os.getenv('TTS_VOICE_PATH')
        if default_voice_path:
            voice_path = Path(default_voice_path).resolve()
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
        logger.error(f"Contents of {TTS_VOICE_DIR}: {list(TTS_VOICE_DIR.iterdir()) if TTS_VOICE_DIR.exists() else 'DIR MISSING'}")
        return None
    
    logger.info(f"Using Piper: {PIPER_EXECUTABLE} with voice: {voice_path}")
    
    # Run Piper - simple stdin/stdout mode
    buffer = BytesIO()
    cmd = [str(PIPER_EXECUTABLE), '-m', str(voice_path), '-f', '-', '-o', '-']
    
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
            logger.error(f"Piper TTS error (return code {proc.returncode}): {err_msg}")
            return None
        
        buffer.write(proc.stdout)
        buffer.seek(0)
        logger.info(f"Piper generated {len(buffer.getvalue())} bytes of audio")
        return buffer
        
    except subprocess.TimeoutExpired:
        logger.error("Piper TTS timed out")
        return None
    except Exception as e:
        logger.error(f"Piper TTS error: {e}", exc_info=True)
        return None

def synthesize_segment(text, speaker='SPEAKER_00', emotion=None):
    import re
    clean_text = re.sub(r'^SPEAKER_\d+:\s*', '', text)
    if not clean_text.strip():
        return None
    return generate_speech(clean_text, return_bytes=True)

__all__ = ['TTS_BACKEND', 'TTS_ENABLED', 'get_tts_status', 'get_available_tts_voices',
           'generate_beep', 'generate_speech', 'synthesize_segment']