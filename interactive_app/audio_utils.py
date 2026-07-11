"""
Audio utilities for beep replacement (Python 3.13 compatible)
Uses ffmpeg instead of pydub when available
"""

import os
import subprocess
import tempfile


class AudioBeepReplacer:
    def __init__(self, beep_freq=1000, beep_gain_db=-30):
        self.beep_freq = beep_freq
        self.beep_gain_db = beep_gain_db

    def replace_offsets_with_beeps(self, input_wav, offsets, output_wav=None):
        """Replace text offsets with beep sounds using ffmpeg."""
        # Check if pydub is available first
        try:
            from pydub import AudioSegment
            from pydub.generators import Sine
            return self._beep_with_pydub(input_wav, offsets, output_wav)
        except ImportError:
            # Fall back to ffmpeg
            return self._beep_with_ffmpeg(input_wav, offsets, output_wav)

    def _beep_with_pydub(self, input_wav, offsets, output_wav=None):
        """Use pydub for beep generation."""
        audio = AudioSegment.from_wav(input_wav)

        if output_wav is None:
            base, _ = os.path.splitext(input_wav)
            output_wav = f"{base}_beeped.wav"

        result = AudioSegment.empty()
        current_pos_ms = 0

        for item in offsets:
            start, end = (item['start'], item['end']) if isinstance(item, dict) else item

            start_ms = int(start * 1000)
            end_ms = int(end * 1000)

            result += audio[current_pos_ms:start_ms]

            beep = (
                Sine(self.beep_freq)
                .to_audio_segment(duration=end_ms - start_ms)
                .apply_gain(self.beep_gain_db)
            )

            result += beep
            current_pos_ms = end_ms

        result += audio[current_pos_ms:]
        result.export(output_wav, format="wav")

        return output_wav

    def _beep_with_ffmpeg(self, input_wav, offsets, output_wav=None):
        """Use ffmpeg for beep generation (Python 3.13 compatible)."""
        import wave
        import struct
        import numpy as np
        
        if output_wav is None:
            base, _ = os.path.splitext(input_wav)
            output_wav = f"{base}_beeped.wav"
        
        # Load original audio (simple PCM WAV handling)
        try:
            with wave.open(input_wav, 'rb') as wav:
                params = wav.getparams()
                n_channels = params.nchannels
                sampwidth = params.sampwidth
                framerate = params.framerate
                n_frames = params.nframes
                
                audio_data = bytearray(wav.readframes(n_frames))
        except Exception as e:
            # Fallback: use ffmpeg to convert to raw audio first
            tmp_raw = tempfile.mktemp(suffix='.raw')
            subprocess.run([
                'ffmpeg', '-y', '-i', input_wav,
                '-f', 's16le', '-ac', '1', '-ar', '16000',
                tmp_raw
            ], capture_output=True)
            
            audio_data = bytearray(open(tmp_raw, 'rb').read())
            framerate = 16000
            n_channels = 1
            sampwidth = 2
        
        # Convert bytes to numpy array
        audio_bytes = bytes(audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        
        # Calculate sample positions
        def ms_to_samples(ms):
            return int(ms * framerate / 1000)
        
        result_samples = []
        current_pos = 0
        
        for item in offsets:
            start, end = (item['start'], item['end']) if isinstance(item, dict) else item
            start_sample = ms_to_samples(start * 1000)
            end_sample = ms_to_samples(end * 1000)
            
            # Add audio before this segment
            result_samples.extend(audio_array[current_pos:start_sample])
            
            # Generate beep
            beep_duration = end_sample - start_sample
            beep_samples = np.sin(2 * np.pi * self.beep_freq * np.arange(beep_duration) / framerate)
            beep_samples *= 32767 * (10 ** (self.beep_gain_db / 20))
            beep_samples = beep_samples.astype(np.int16)
            
            result_samples.extend(beep_samples)
            current_pos = end_sample
        
        # Add remaining audio
        result_samples.extend(audio_array[current_pos:])
        
        # Convert back to bytes and save
        result_array = np.array(result_samples).astype(np.int16)
        result_bytes = result_array.tobytes()
        
        with wave.open(output_wav, 'wb') as wav_out:
            wav_out.setnchannels(n_channels)
            wav_out.setsampwidth(sampwidth)
            wav_out.setframerate(framerate)
            wav_out.writeframes(result_bytes)
        
        return output_wav