from pydub import AudioSegment
from pydub.generators import Sine
import os


class AudioBeepReplacer:
    def __init__(self, beep_freq=1000, beep_gain_db=-6):
        self.Sine = Sine
        self.beep_freq = beep_freq
        self.beep_gain_db = beep_gain_db

    def replace_offsets_with_beeps(self, input_wav, offsets, output_wav=None):
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
                self.Sine(self.beep_freq)
                .to_audio_segment(duration=end_ms - start_ms)
                .apply_gain(self.beep_gain_db)
            )

            result += beep
            current_pos_ms = end_ms

        result += audio[current_pos_ms:]
        result.export(output_wav, format="wav")

        return output_wav
