"""TTS module - multi-speaker text-to-speech."""
from .tts_engine import *

__all__ = [
    'TTS_BACKEND', 'TTS_ENABLED', 'SPEAKER_VOICE_MAP',
    'get_tts_status', 'get_available_tts_voices',
    'get_speaker_voice', 'generate_speech', 'synthesize_segment'
]
