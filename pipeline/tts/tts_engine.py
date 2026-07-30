"""
TTS Engine Module - Multi-Speaker Support
"""

import os
import subprocess
from pathlib import Path

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
    """Generate speech using Piper TTS."""
    if not TTS_ENABLED:
        return None
    
    if voice_id is None and speaker is not None:
        voice_id = get_speaker_voice(speaker)
    elif voice_id is None:
        voice_id = SPEAKER_VOICE_MAP['DEFAULT']
    
    return _generate_speech_piper(text, voice_id, return_bytes)

def _generate_speech_piper(text: str, voice_id: str, return_bytes: bool):
    """Generate speech using Piper."""
    if not PIPER_EXECUTABLE.exists():
        return None
    
    voice_path = PIPER_VOICE_DIR / f"{voice_id}.onnx"
    
    if not voice_path.exists():
        return None
    
    import io
    buffer = io.BytesIO()
    
    # Use stdout mode (-o -), no config file needed
    cmd = [
        str(PIPER_EXECUTABLE),
        '-m', str(voice_path),
        '-o', '-'
    ]
    
    try:
        env = os.environ.copy()
        bin_dir = str(TTS_DIR / 'bin')
        env['LD_LIBRARY_PATH'] = bin_dir + ':' + env.get('LD_LIBRARY_PATH', '')
        
        proc = subprocess.run(cmd, input=text.encode('utf-8'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=60, env=env)
        
        if proc.returncode != 0:
            return None
        
        buffer.write(proc.stdout)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
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
