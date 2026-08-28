"""
TTS Engine Module - Multi-Speaker Support
Production-ready with local Piper API
"""

import os
import subprocess
import io
import wave
import logging
import random
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

# Project root: go up 3 levels from tts_engine.py
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

def extract_unique_speakers(word_offsets: list) -> tuple:
    """
    Extract unique speakers from WhisperX diarization segments.
    
    Returns:
        tuple: (speaker_count, list_of_speaker_ids)
    """
    speakers = set()
    for seg in word_offsets:
        speaker = seg.get('speaker', 'SPEAKER_00')
        speakers.add(speaker)
    speaker_list = sorted(list(speakers))
    return len(speaker_list), speaker_list

def assign_randomized_voices(speaker_list: list) -> dict:
    """
    Assign random voices to speakers from available pool.
    
    Returns:
        dict: speaker_id -> voice_name mapping
    """
    available_voices = _load_available_voices()
    
    if len(available_voices) < len(speaker_list):
        logger.warning(f"Not enough voices available ({len(available_voices)}) for {len(speaker_list)} speakers. Using cycling fallback.")
        voice_pool = available_voices if available_voices else ['en_US-amy-medium']
    else:
        voice_pool = available_voices[:len(speaker_list)]
    
    # Randomize
    random.shuffle(voice_pool)
    
    speaker_to_voice = {}
    for i, speaker in enumerate(speaker_list):
        if i < len(voice_pool):
            speaker_to_voice[speaker] = voice_pool[i]
        else:
            speaker_to_voice[speaker] = voice_pool[0]
    
    logger.info(f"Assigned voices: {speaker_to_voice}")
    return speaker_to_voice

def generate_speech(text: str, language: str = 'en', voice_id: str = None,
                    speaker: str = None, return_bytes: bool = False):
    """Generate speech using Piper TTS (fully offline)."""
    if not TTS_ENABLED:
        return None

    if voice_id is None and speaker is not None:
        voice_id = get_speaker_voice(speaker)
    elif voice_id is None:
        voice_id = SPEAKER_VOICE_MAP['DEFAULT']

    # --- PRIMARY: Python Piper API ---
    audio_buffer = _generate_speech_piper_python(text, voice_id)
    if audio_buffer is not None:
        logger.debug(f"TTS via Python API: {voice_id}, {len(text)} chars")
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
    """Generate speech using the piper-tts Python package directly."""
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
    """Generate speech via the Piper CLI (subprocess fallback)."""
    if not PIPER_EXECUTABLE.exists():
        logger.error(f"Piper executable not found: {PIPER_EXECUTABLE}")
        return None

    voice_path = PIPER_VOICE_DIR / f"{voice_id}.onnx"
    if not voice_path.exists():
        logger.error(f"Voice model not found: {voice_path}")
        return None

    tmp_wav = Path('/tmp/piper_output.wav')

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
            input=text,
            text=True,
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

def generate_multi_speaker_tts(transcript_text: str, word_offsets: list = None, 
                                audio_source_path: str = None) -> tuple:
    """
    Generate multi-speaker TTS with randomized voice assignments.
    
    Args:
        transcript_text: Full transcript with SPEAKER_X labels
        word_offsets: WhisperX diarization segments (optional)
        audio_source_path: Original audio file path (optional, for naming output)
    
    Returns:
        tuple: (output_audio_path, voice_mapping_dict)
    """
    if not TTS_ENABLED:
        logger.warning("TTS not enabled. Skipping multi-speaker TTS generation.")
        return None, {}
    
    # Detect speakers from offsets or parse transcript
    if word_offsets:
        speaker_count, speaker_list = extract_unique_speakers(word_offsets)
    else:
        import re
        speaker_matches = re.findall(r'SPEAKER_(\d+)', transcript_text)
        unique_speakers = set(speaker_matches)
        speaker_list = sorted([f'SPEAKER_{i}' for i in unique_speakers])
        speaker_count = len(speaker_list)
    
    logger.info(f"Detected {speaker_count} speakers: {speaker_list}")
    
    # Assign randomized voices
    speaker_to_voice = assign_randomized_voices(speaker_list)
    
    # Split transcript into speaker segments
    lines = transcript_text.strip().split('\n')
    segments_by_speaker = defaultdict(list)
    
    for line in lines:
        import re
        match = re.match(r'^(SPEAKER_\d+):\s*(.*)$', line.strip())
        if match:
            speaker = match.group(1)
            text = match.group(2)
            if text.strip():
                segments_by_speaker[speaker].append(text.strip())
    
    # Merge consecutive same-speaker segments
    merged_segments = []
    current_speaker = None
    current_text = []
    
    for speaker in sorted(segments_by_speaker.keys()):
        voice_id = speaker_to_voice.get(speaker, SPEAKER_VOICE_MAP['DEFAULT'])
        
        for segment_text in segments_by_speaker[speaker]:
            merged_segments.append({
                'speaker': speaker,
                'voice': voice_id,
                'text': segment_text
            })
    
    # Generate audio for each segment
    import tempfile
    from pathlib import Path
    
    base_name = Path(audio_source_path).stem if audio_source_path else 'multi_speaker_tts'
    final_output = Path(TTS_DIR) / f"{base_name}_final.wav"
    
    # Create merged WAV
    merged_buffer = io.BytesIO()
    sample_rate = 22050
    sample_width = 2
    channels = 1
    
    with wave.open(merged_buffer, 'wb') as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        
        for segment in merged_segments:
            logger.info(f"Generating TTS for {segment['speaker']} using voice: {segment['voice']}")
            
            audio_buffer = generate_speech(
                text=segment['text'],
                voice_id=segment['voice'],
                return_bytes=True
            )
            
            if audio_buffer:
                with wave.open(audio_buffer, 'rb') as src_wav:
                    wav.writeframes(src_wav.readframes(src_wav.getnframes()))
                
                # Add small pause between segments (100ms)
                silence = b'\x00' * (sample_rate * sample_width * channels // 10)
                wav.writeframes(silence)
    
    merged_buffer.seek(0)
    Path(final_output).parent.mkdir(parents=True, exist_ok=True)
    final_output.write_bytes(merged_buffer.read())
    
    logger.info(f"Multi-speaker TTS output saved to: {final_output}")
    return str(final_output), speaker_to_voice

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
    'get_speaker_voice', 'generate_speech', 'synthesize_segment',
    'extract_unique_speakers', 'assign_randomized_voices', 'generate_multi_speaker_tts'
]