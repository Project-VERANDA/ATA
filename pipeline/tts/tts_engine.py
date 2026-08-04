"""
TTS Engine Module - Multi-Speaker Support
Rewritten to use Python Piper API directly (no subprocess needed)
with CLI fallback for compatibility.
"""

import os
import subprocess
import io
import wave
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root: go up 3 levels from tts_engine.py
# tts_engine.py -> tts/ -> pipeline/ -> ATA/ (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Configuration
TTS_BACKEND = "piper"
TTS_ENABLED = True
TTS_DIR = PROJECT_ROOT / "pipeline" / "tts"
PIPER_VOICE_DIR = TTS_DIR / "voices"
PIPER_EXECUTABLE = TTS_DIR / "bin" / "piper"

# Speaker-to-Voice Mapping
SPEAKER_VOICE_MAP = {
    'SPEAKER_00': 'en_US-amy-medium',
    'SPEAKER_01': 'en_US-lessac-medium',
    'SPEAKER_02': 'en_US-kusal-medium',
    'SPEAKER_03': 'en_US-ryan-medium',
    'SPEAKER_04': 'en_US-joe-medium',
    'SPEAKER_05': 'en_US-libritts-high',
    'DEFAULT': 'en_US-amy-medium'
}


def _load_available_voices():
    """Load list of available voice files."""
    voices = []
    if PIPER_VOICE_DIR.exists():
        for voice_file in PIPER_VOICE_DIR.glob("*.onnx"):
            voice_name = voice_file.stem
            if voice_name != 'VOICE_MAPPING':
                voices.append(voice_name)
    return voices


def get_speaker_voice(speaker_id: str) -> str:
    """Get voice assignment for a specific speaker."""
    if speaker_id in SPEAKER_VOICE_MAP:
        voice_name = SPEAKER_VOICE_MAP[speaker_id]
        if (PIPER_VOICE_DIR / f"{voice_name}.onnx").exists():
            return voice_name

    if speaker_id.startswith('SPEAKER_'):
        try:
            num = int(speaker_id.split('_')[1])
            voices = _load_available_voices()
            if voices:
                return voices[num % len(voices)]
        except (ValueError, IndexError):
            pass

    voices = _load_available_voices()
    return voices[0] if voices else SPEAKER_VOICE_MAP['DEFAULT']


def generate_speech(text: str, language: str = 'en', voice_id: str = None,
                    speaker: str = None, return_bytes: bool = False):
    """Generate speech using Piper TTS.

    Tries Python API first (fastest, no subprocess overhead).
    Falls back to CLI if Python API fails.
    """
    if not TTS_ENABLED:
        return None

    if voice_id is None and speaker is not None:
        voice_id = get_speaker_voice(speaker)
    elif voice_id is None:
        voice_id = SPEAKER_VOICE_MAP['DEFAULT']

    # --- PRIMARY: Python Piper API ---
    audio_buffer = _generate_speech_piper_python(text, voice_id)
    if audio_buffer is not None:
        logger.info(f"TTS via Python API: {voice_id}, {len(text)} chars")
        return audio_buffer

    # --- FALLBACK: CLI subprocess ---
    logger.warning("Python Piper API failed, trying CLI fallback...")
    audio_buffer = _generate_speech_piper_cli(text, voice_id)
    if audio_buffer is not None:
        logger.info(f"TTS via CLI fallback: {voice_id}, {len(text)} chars")
        return audio_buffer

    logger.error(f"All TTS methods failed for voice={voice_id}")
    return None


def _generate_speech_piper_python(text: str, voice_id: str) -> io.BytesIO | None:
    """Generate speech using the piper-tts Python package directly.

    This is the preferred method — no subprocess, no temp files,
    no argument mismatches.
    """
    try:
        from piper import PiperVoice
    except ImportError:
        logger.debug("piper-tts Python package not available")
        return None

    voice_path = PIPER_VOICE_DIR / f"{voice_id}.onnx"
    if not voice_path.exists():
        logger.error(f"Voice model not found: {voice_path}")
        return None

    try:
        voice = PiperVoice.load(str(voice_path))
        chunks = list(voice.synthesize(text))

        if not chunks:
            logger.error("PiperVoice.synthesize() returned no chunks")
            return None

        # Build WAV in memory
        buffer = io.BytesIO()
        sample_rate = 22050
        sample_width = chunks[0].sample_width
        channels = chunks[0].sample_channels

        with wave.open(buffer, 'wb') as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sample_rate)
            for chunk in chunks:
                wav.writeframes(chunk.audio_int16_bytes)

        buffer.seek(0)
        return buffer

    except Exception as e:
        logger.error(f"Python Piper synthesis failed: {e}", exc_info=True)
        return None


def _generate_speech_piper_cli(text: str, voice_id: str) -> io.BytesIO | None:
    """Generate speech via the Piper CLI (subprocess fallback).

    Sends text via stdin (compatible with both the native Piper
    binary and the Python wrapper script).
    """
    if not PIPER_EXECUTABLE.exists():
        logger.error(f"Piper executable not found: {PIPER_EXECUTABLE}")
        return None

    voice_path = PIPER_VOICE_DIR / f"{voice_id}.onnx"
    if not voice_path.exists():
        logger.error(f"Voice model not found: {voice_path}")
        return None

    tmp_wav = Path('/tmp/piper_output.wav')

    # NOTE: Do NOT use -f flag. Send text via stdin instead.
    # This works with both the native Piper binary AND the Python wrapper.
    cmd = [
        str(PIPER_EXECUTABLE),
        '-m', str(voice_path),
        '-o', str(tmp_wav),
        '--sample_rate', '22050'
    ]

    try:
        env = os.environ.copy()
        bin_dir = str(TTS_DIR / 'bin')
        env['LD_LIBRARY_PATH'] = bin_dir + ':' + env.get('LD_LIBRARY_PATH', '')

        proc = subprocess.run(
            cmd,
            input=text,          # ← Send text via stdin, NOT -f flag
            text=True,           # ← Treat as text, not bytes
            capture_output=True,
            timeout=60,
            env=env
        )

        if proc.returncode != 0:
            logger.error(
                f"Piper CLI failed (exit {proc.returncode}): "
                f"{proc.stderr[:500] if proc.stderr else 'no stderr'}"
            )
            return None

        if tmp_wav.exists() and tmp_wav.stat().st_size > 0:
            buffer = io.BytesIO()
            buffer.write(tmp_wav.read_bytes())
            buffer.seek(0)
            tmp_wav.unlink()
            return buffer

        logger.error("Piper CLI produced no output file")
        return None

    except subprocess.TimeoutExpired:
        logger.error("Piper CLI timed out after 60s")
        return None
    except Exception as e:
        logger.error(f"Piper CLI error: {e}", exc_info=True)
        return None


def synthesize_segment(text: str, speaker: str = 'SPEAKER_00', emotion: str = None):
    """Synthesize a single speaker segment."""
    import re
    clean_text = re.sub(r'^SPEAKER_\d+:\s*', '', text)
    if not clean_text.strip():
        return None
    return generate_speech(clean_text, speaker=speaker, return_bytes=True)


def get_tts_status():
    """Return TTS configuration and available voices."""
    return {
        'enabled': TTS_ENABLED,
        'backend': TTS_BACKEND,
        'piper_exists': PIPER_EXECUTABLE.exists(),
        'voices_dir_exists': PIPER_VOICE_DIR.exists(),
        'available_voices': _load_available_voices(),
        'speaker_mapping': SPEAKER_VOICE_MAP
    }


def get_available_tts_voices(language: str = 'en'):
    """List available voices for a language."""
    voices = _load_available_voices()
    return [{'id': v, 'language': language} for v in voices if language in v]


__all__ = [
    'TTS_BACKEND', 'TTS_ENABLED', 'SPEAKER_VOICE_MAP',
    'get_tts_status', 'get_available_tts_voices',
    'get_speaker_voice', 'generate_speech', 'synthesize_segment'
]