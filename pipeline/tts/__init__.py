"""
TTS Package - Text-to-Speech functionality
"""
from .tts_engine import (
    generate_speech,
    generate_beep,
    synthesize_segment,
    get_available_tts_voices,
    get_tts_status,
    TTS_BACKEND,
    TTS_ENABLED
)

__all__ = [
    'generate_speech',
    'generate_beep',
    'synthesize_segment',
    'get_available_tts_voices',
    'get_tts_status',
    'TTS_BACKEND',
    'TTS_ENABLED'
]