"""TTS module - unified text-to-speech functionality."""
from .tts_engine import *

__all__ = [
    'TTS_BACKEND', 'TTS_ENABLED',
    'get_tts_status', 'get_available_tts_voices',
    'generate_beep', 'generate_speech', 'synthesize_segment'
]