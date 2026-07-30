"""
TTS Engine Module - Multi-Speaker Support
Each speaker gets assigned a unique voice
"""

import os
import subprocess
from pathlib import Path
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger.info(f"TTS Engine: Project root = {PROJECT_ROOT}")

# Load .env
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

# Configuration
TTS_BACKEND = os.getenv('TTS_BACKEND', 'piper')
TTS_ENABLED = os.getenv('TTS_ENABLED', 'false').lower() == 'true'
TTS_DIR = PROJECT_ROOT / "pipeline" / "tts"
TTS_VOICE_DIR = TTS_DIR / "voices"
PIPER_EXECUTABLE = TTS_DIR / "bin" / "piper"

# Speaker-to-Voice Mapping (auto-cycle if more speakers than voices)
SPEAKER_VOICE_MAP = {
    'SPEAKER_00': 'en_US-amy-medium',
    'SPEAKER_01': 'en_US-lessac-medium',
    'SPEAKER_02': 'en_US-kusal-medium',
    'SPEAKER_03': 'en_US-ryan-medium',
    'SPEAKER_04': 'en_US-joe-medium',
    'SPEAKER_05': 'en_US-libritts-high',
    'DEFAULT': 'en_US-amy-medium'
}

# Available voices (loaded dynamically)
AVAILABLE_VOICES = []

def _load_available_voices():
    """Load list of available voice files."""
    global AVAILABLE_VOICES
    AVAILABLE_VOICES = []
    if TTS_VOICE_DIR.exists():
        for voice_file in TTS_VOICE_DIR.glob("*.onnx"):
            voice_name = voice_file.stem
            # Skip JSON metadata files
            if not voice_name.endswith('.json'):
                AVAILABLE_VOICES.append(voice_name)
    logger.info(f"Available voices: {AVAILABLE_VOICES}")
    return AVAILABLE_VOICES

def get_speaker_voice(speaker_id: str) -> str:
    """
    Get voice assignment for a specific speaker.
    Falls back to cycling through available voices if speaker not mapped.
    """
    # Try explicit mapping first
    if speaker_id in SPEAKER_VOICE_MAP:
        voice_name = SPEAKER_VOICE_MAP[speaker_id]
        if f"{voice_name}.onnx" in [f"{v}.onnx" for v in AVAILABLE_VOICES]:
            return voice_name
    
    # Cycle through available voices by speaker number
    if speaker_id.startswith('SPEAKER_'):
        try:
            num = int(speaker_id.split('_')[1])
            _load_available_voices()
            if AVAILABLE_VOICES:
                voice_index = num % len(AVAILABLE_VOICES)
                voice_name = AVAILABLE_VOICES[voice_index]
                logger.info(f"Speaker {speaker_id} → {voice_name} (index {num} % {len(AVAILABLE_VOICES)})")
                return voice_name
        except (ValueError, IndexError):
            pass
    
    # Ultimate fallback
    _load_available_voices()
    return AVAILABLE_VOICES[0] if AVAILABLE_VOICES else SPEAKER_VOICE_MAP['DEFAULT']

def generate_speech(text: str, language: str = 'en', 
                   voice_id: str = None, speaker: str = None,
                   return_bytes: bool = False):
    """
    Generate speech using Piper TTS with speaker-aware voice selection.
    
    Args:
        text: Text to synthesize
        language: Language code (default: 'en')
        voice_id: Explicit voice override (optional)
        speaker: Speaker ID (e.g., 'SPEAKER_00') for automatic voice mapping
        return_bytes: Return BytesIO buffer instead of file path
    
    Returns:
        BytesIO buffer with WAV audio, or None on error
    """
    if not TTS_ENABLED:
        logger.warning("TTS disabled in .env")
        return None
    
    # Determine which voice to use
    if voice_id is None and speaker is not None:
        voice_id = get_speaker_voice(speaker)
    elif voice_id is None:
        voice_id = SPEAKER_VOICE_MAP['DEFAULT']
    
    return _generate_speech_piper(text, voice_id, return_bytes)

def _generate_speech_piper(text: str, voice_id: str, return_bytes: bool):
    """Generate speech using Piper."""
    if not PIPER_EXECUTABLE.exists():
        logger.error(f"Piper executable not found: {PIPER_EXECUTABLE}")
        return None
    
    voice_path = TTS_VOICE_DIR / f"{voice_id}.onnx"
    if not voice_path.exists():
        logger.error(f"Voice file not found: {voice_path}")
        # Try to reload available voices
        _load_available_voices()
        if voice_id not in AVAILABLE_VOICES:
            return None
        # Try first available voice
        if AVAILABLE_VOICES:
            voice_path = TTS_VOICE_DIR / f"{AVAILABLE_VOICES[0]}.onnx"
            logger.info(f"Falling back to {AVAILABLE_VOICES[0]}")
    
    buffer = BytesIO()
    cmd = [
        str(PIPER_EXECUTABLE),
        '-m', str(voice_path),
        '-f', '-',
        '-o', '-'
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
            logger.error(f"Piper TTS error: {err_msg[:200]}")
            return None
        
        buffer.write(proc.stdout)
        buffer.seek(0)
        logger.info(f"Generated {len(buffer.getvalue())} bytes with voice {voice_id}")
        return buffer
        
    except Exception as e:
        logger.error(f"Piper TTS error: {e}", exc_info=True)
        return None

def synthesize_segment(text: str, speaker: str = 'SPEAKER_00', emotion: str = None):
    """
    Synthesize a single speaker segment.
    Automatically selects appropriate voice for the speaker.
    
    Args:
        text: Text with optional speaker prefix (e.g., "SPEAKER_00: Hello")
        speaker: Speaker ID (overrides prefix if both provided)
        emotion: Currently unused (reserved for future enhancement)
    
    Returns:
        BytesIO buffer with WAV audio
    """
    import re
    # Extract speaker from text if present
    speaker_match = re.match(r'^SPEAKER_\d+:\s*', text)
    if speaker_match and speaker == 'SPEAKER_00':
        speaker = speaker_match.group(0).rstrip(': ').strip()
        clean_text = text[len(speaker_match.group(0)):]
    else:
        clean_text = re.sub(r'^SPEAKER_\d+:\s*', '', text)
    
    if not clean_text.strip():
        logger.warning(f"Empty text for speaker {speaker}")
        return None
    
    return generate_speech(clean_text, speaker=speaker, return_bytes=True)

def get_tts_status():
    """Return TTS configuration and available voices."""
    _load_available_voices()
    return {
        'enabled': TTS_ENABLED,
        'backend': TTS_BACKEND,
        'piper_exists': PIPER_EXECUTABLE.exists(),
        'voices_dir_exists': TTS_VOICE_DIR.exists(),
        'available_voices': AVAILABLE_VOICES,
        'speaker_mapping': SPEAKER_VOICE_MAP
    }

def get_available_tts_voices(language: str = 'en'):
    """List available voices for a language."""
    _load_available_voices()
    return [{'id': v, 'language': language} for v in AVAILABLE_VOICES if language in v]

__all__ = [
    'TTS_BACKEND', 'TTS_ENABLED',
    'SPEAKER_VOICE_MAP',
    'get_tts_status', 'get_available_tts_voices',
    'get_speaker_voice', 'generate_speech', 'synthesize_segment'
]