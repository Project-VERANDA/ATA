"""Tests for interactive_app.audio_utils (require ffmpeg/ffprobe and numpy)."""
import os
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../", "interactive_app"))
import audio_utils as au  # noqa: E402

HAVE_FFMPEG = au.ffmpeg_available()


def make_tone(path, seconds=4.0, rate=16000, channels=1, extra=()):
    """Creates a 440 Hz test tone with ffmpeg."""
    layout = "mono" if channels == 1 else "stereo"
    subprocess.run([au.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=%d:duration=%s" % (rate, seconds),
                    "-ac", str(channels), "-channel_layout", layout, *extra, path], check=True)


def read_wav(path):
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return data.reshape(-1, w.getnchannels()), w.getframerate()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "in.wav")
        self.out = os.path.join(self.tmp.name, "out.wav")


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class Rendering(Base):
    def test_masks_only_the_span_sample_accurately(self):
        make_tone(self.src)
        res = au.render_redacted_audio(self.src, self.out, [{"start": 1.0, "end": 2.0}],
                                       beep_freq=1000, beep_gain_db=-6)
        src, rate = read_wav(self.src)
        out, _ = read_wav(self.out)
        self.assertEqual(out.shape, src.shape)
        self.assertEqual(res["spans_applied"], 1)
        inside = slice(int(1.0 * rate) + 200, int(2.0 * rate) - 200)
        self.assertTrue(np.array_equal(src[:int(1.0 * rate) - 5], out[:int(1.0 * rate) - 5]))
        self.assertTrue(np.array_equal(src[int(2.0 * rate) + 5:], out[int(2.0 * rate) + 5:]))
        self.assertFalse(np.array_equal(src[inside], out[inside]))
        spectrum = np.abs(np.fft.rfft(out[inside, 0].astype(float)))
        self.assertAlmostEqual(np.argmax(spectrum) * rate / (inside.stop - inside.start), 1000, delta=30)

    def test_stereo_and_sample_rate_preserved(self):
        make_tone(self.src, rate=44100, channels=2)
        au.render_redacted_audio(self.src, self.out, [(0.5, 1.0)])
        out, rate = read_wav(self.out)
        self.assertEqual((rate, out.shape[1]), (44100, 2))

    def test_overlapping_and_out_of_range_spans(self):
        make_tone(self.src, seconds=2)
        res = au.render_redacted_audio(self.src, self.out, [(0.5, 1.0), (0.8, 1.2), (1.9, 99)])
        self.assertEqual(res["spans_applied"], 2)

    def test_other_formats(self):
        make_tone(self.src, seconds=2)
        flac = os.path.join(self.tmp.name, "o.flac")
        au.render_redacted_audio(self.src, flac, [(0.5, 1.0)], fmt="flac")
        self.assertGreater(os.path.getsize(flac), 100)
        mp3 = os.path.join(self.tmp.name, "in.mp3")
        try:
            make_tone(mp3, seconds=2)
        except subprocess.CalledProcessError:
            return                                   # libmp3lame unavailable
        au.render_redacted_audio(mp3, self.out, [(0.5, 1.0)])          # non-WAV input
        self.assertGreater(os.path.getsize(self.out), 100)

    def test_metadata_is_stripped(self):
        make_tone(self.src, seconds=1, extra=("-metadata", "title=SecretName",
                                              "-metadata", "artist=Max Mustermann"))
        au.render_redacted_audio(self.src, self.out, [(0.2, 0.4)])
        with open(self.out, "rb") as fh:
            blob = fh.read()
        self.assertNotIn(b"SecretName", blob)
        self.assertNotIn(b"Mustermann", blob)

    def test_legacy_class_api(self):
        make_tone(self.src, seconds=1)
        path = au.AudioBeepReplacer(beep_freq=1000, beep_gain_db=-6).replace_offsets_with_beeps(
            self.src, [{"start": 0.2, "end": 0.4}], self.out)
        self.assertEqual(path, self.out)
        self.assertTrue(au.AudioBeepReplacer().replace_offsets_with_beeps(
            self.src, [(0.1, 0.2)]).endswith("_beeped.wav"))

    def test_failure_leaves_no_partial_file_and_bad_args_rejected(self):
        bad = os.path.join(self.tmp.name, "bad.wav")
        with open(bad, "wb") as fh:
            fh.write(b"not audio")
        with self.assertRaises(au.AudioError):
            au.render_redacted_audio(bad, self.out, [(0, 1)])
        self.assertEqual([f for f in os.listdir(self.tmp.name) if f.startswith("out")], [])
        make_tone(self.src, seconds=1)
        with self.assertRaises(au.AudioError):
            au.render_redacted_audio(self.src, self.out, [(0, 1)], fmt="exe")
        with self.assertRaises(au.AudioError):
            au.render_redacted_audio(self.src, self.out, [{"start": float("nan"), "end": 1}])


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class ProbeAndWaveform(Base):
    def test_probe_values_and_duration_limit(self):
        make_tone(self.src, seconds=3, rate=22050, channels=2)
        info = au.probe_media(self.src)
        self.assertAlmostEqual(info.duration, 3.0, delta=0.05)
        self.assertEqual((info.sample_rate, info.channels), (22050, 2))
        with self.assertRaises(au.AudioError) as ctx:
            au.probe_media(self.src, max_duration=1)
        self.assertEqual(ctx.exception.code, "media_too_long")

    def test_rejects_garbage_and_dangerous_containers(self):
        junk = os.path.join(self.tmp.name, "junk.wav")
        with open(junk, "wb") as fh:
            fh.write(os.urandom(2048))
        with self.assertRaises(au.AudioError):
            au.probe_media(junk)
        playlist = os.path.join(self.tmp.name, "list.wav")      # concat could read arbitrary local files
        with open(playlist, "w") as fh:
            fh.write("ffconcat version 1.0\nfile '/etc/passwd'\n")
        with self.assertRaises(au.AudioError):
            au.probe_media(playlist)

    def test_waveform(self):
        make_tone(self.src, seconds=4)
        wf = au.compute_waveform(self.src, bars=100, start=1.0, end=3.0)
        self.assertEqual((wf["start"], wf["end"], len(wf["levels"])), (1.0, 3.0, 100))
        self.assertGreater(min(wf["levels"]), 0.8)                    # steady tone -> flat
        with self.assertRaises(au.AudioError):
            au.compute_waveform(self.src, start=3.0, end=3.0)
        silent = os.path.join(self.tmp.name, "s.wav")
        subprocess.run([au.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "anullsrc=r=16000:cl=mono", "-t", "1", silent], check=True)
        self.assertEqual(max(au.compute_waveform(silent, bars=100)["levels"]), 0)

    def test_webm_without_duration_header(self):
        webm = os.path.join(self.tmp.name, "r.webm")
        proc = subprocess.run([au.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                               "sine=frequency=300:duration=2", "-c:a", "libopus", "-f", "webm", "-"],
                              capture_output=True)
        if proc.returncode != 0:
            self.skipTest("libopus unavailable")
        with open(webm, "wb") as fh:
            fh.write(proc.stdout)
        self.assertAlmostEqual(au.probe_media(webm).duration, 2.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
