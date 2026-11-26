"""Audio utilities: safe probing, waveform levels and beep redaction.

Public API (stable):

* ``AudioBeepReplacer.replace_offsets_with_beeps(input, offsets, output)`` - the
  original class and signature, now backed by the streaming renderer.
* ``probe_media(path)`` - validates an upload and returns its properties.
* ``compute_waveform(path, bars, start, end)`` - RMS levels for the editor.
* ``render_redacted_audio(input, output, spans, ...)`` - overwrite time ranges
  with a sine beep and write WAV, MP3 or FLAC.

Design notes:

* Audio is processed as a *stream* (decode -> numpy block -> encode): constant
  memory for any file length, any ffmpeg-readable input, and sample rate and
  channel count are preserved.
* Inputs are untrusted uploads. ffmpeg is never started through a shell, only
  ``file:``/``pipe`` protocols are allowed, the detected container must be on an
  allow-list, every call has a timeout and metadata/chapters/cover art are
  stripped from the output.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

FFMPEG = os.environ.get("ATA_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("ATA_FFPROBE", "ffprobe")

#: Maximum accepted media length in seconds (0 disables the check).
MAX_AUDIO_SECONDS = float(os.environ.get("ATA_MAX_AUDIO_SECONDS", 4 * 3600))
#: Wall-clock limit for a single ffmpeg job in seconds.
FFMPEG_TIMEOUT_S = float(os.environ.get("ATA_FFMPEG_TIMEOUT", 3600))
PROBE_TIMEOUT_S = 60.0

#: Demuxers accepted for uploads. Others (concat, hls, image2, ...) are rejected
#: because they can reference other files or network resources.
ALLOWED_FORMATS = frozenset({
    "wav", "w64", "mp3", "mov", "mp4", "m4a", "3gp", "matroska", "webm", "ogg",
    "flac", "aac", "aiff", "caf", "amr", "asf", "avi", "ac3", "eac3",
})
SUPPORTED_OUTPUT_FORMATS = ("wav", "mp3", "flac")
BLOCK_FRAMES = 65536
WAVEFORM_RATE = 8000


class AudioError(Exception):
    """Audio processing problem. ``code`` is a stable machine-readable key."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class MediaInfo:
    """Properties of an audio (or audio-bearing) file."""

    duration: float
    sample_rate: int
    channels: int
    format_names: Tuple[str, ...]


# --- Process helpers ----------------------------------------------------------
def _input_arg(path: str) -> str:
    """Prefixes ``file:`` so ffmpeg never interprets the path as a URL/option."""
    return "file:" + os.path.abspath(path)


def _base_cmd(binary: str) -> List[str]:
    """Common quiet flags. ``-nostdin`` exists in ffmpeg only, not in ffprobe."""
    if binary == FFPROBE:
        return [binary, "-hide_banner", "-loglevel", "error"]
    return [binary, "-nostdin", "-hide_banner", "-loglevel", "error"]


class _Watchdog:
    """Kills processes after a deadline so a stalled ffmpeg cannot hang a request."""

    def __init__(self, timeout: float):
        self._timer = threading.Timer(timeout, self._fire)
        self._procs: List[subprocess.Popen] = []
        self.fired = False

    def watch(self, proc: subprocess.Popen) -> None:
        self._procs.append(proc)

    def _fire(self) -> None:
        self.fired = True
        for proc in self._procs:
            try:
                proc.kill()
            except OSError:
                pass

    def __enter__(self) -> "_Watchdog":
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._timer.cancel()


def ffmpeg_available() -> bool:
    """True if both ffmpeg and ffprobe can be executed."""
    try:
        for binary in (FFMPEG, FFPROBE):
            subprocess.run([binary, "-version"], capture_output=True, timeout=10, check=True)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


# --- Probing ------------------------------------------------------------------
def probe_media(path: str, max_duration: Optional[float] = None) -> MediaInfo:
    """Validates a media file and returns its properties.

    Args:
        path: File to inspect.
        max_duration: Maximum length in seconds (default ``MAX_AUDIO_SECONDS``, 0 disables).

    Raises:
        AudioError: Unreadable file, unsupported container, no audio stream or too long.
    """
    cmd = _base_cmd(FFPROBE) + [
        "-protocol_whitelist", "file,pipe", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels:format=format_name,duration",
        "-of", "json", "-i", _input_arg(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=PROBE_TIMEOUT_S, check=True)
        data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    except subprocess.TimeoutExpired:
        raise AudioError("probe_timeout", "Inspecting the media file timed out.") from None
    except (subprocess.CalledProcessError, ValueError):
        raise AudioError("unreadable_media", "The file is not a readable audio/video file.") from None
    except OSError:
        raise AudioError("ffmpeg_missing", "ffprobe is not installed on the server.") from None

    streams = data.get("streams") or []
    if not streams:
        raise AudioError("no_audio_stream", "The file contains no audio stream.")
    fmt = data.get("format") or {}
    names = tuple(n for n in str(fmt.get("format_name", "")).split(",") if n)
    if not names or not ALLOWED_FORMATS.intersection(names):
        raise AudioError("unsupported_format", "Unsupported media container.")
    try:
        sample_rate = int(streams[0].get("sample_rate") or 16000)
        channels = int(streams[0].get("channels") or 1)
    except (TypeError, ValueError):
        raise AudioError("unreadable_media", "Invalid audio stream parameters.") from None
    if not (8000 <= sample_rate <= 192000) or not (1 <= channels <= 32):
        raise AudioError("unsupported_audio", "Unsupported sample rate or channel count.")

    try:
        duration = float(fmt.get("duration"))
        if duration != duration or duration <= 0:      # NaN / non-positive
            raise ValueError
    except (TypeError, ValueError):
        duration = measure_duration_by_decoding(path)   # e.g. MediaRecorder WebM
    limit = MAX_AUDIO_SECONDS if max_duration is None else max_duration
    if limit and duration > limit:
        raise AudioError("media_too_long", "Media exceeds the maximum length of %d minutes."
                         % int(limit // 60))
    return MediaInfo(duration, sample_rate, channels, names)


def measure_duration_by_decoding(path: str) -> float:
    """Counts decoded samples; used when the container has no duration field."""
    cmd = _base_cmd(FFMPEG) + ["-protocol_whitelist", "file,pipe", "-i", _input_arg(path),
                               "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(WAVEFORM_RATE),
                               "-f", "s16le", "pipe:1"]
    total = 0
    with tempfile.TemporaryFile() as err, _Watchdog(FFMPEG_TIMEOUT_S) as dog:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        dog.watch(proc)
        while True:
            chunk = proc.stdout.read(1 << 20)
            if not chunk:
                break
            total += len(chunk)
        proc.stdout.close()
        proc.wait()
        if dog.fired or proc.returncode != 0 or total == 0:
            raise AudioError("unreadable_media", "Could not determine the media duration.")
    return total / 2.0 / WAVEFORM_RATE


# --- Waveform -----------------------------------------------------------------
def compute_waveform(path: str, bars: int = 800, start: Optional[float] = None,
                     end: Optional[float] = None) -> Dict[str, Any]:
    """Computes normalised RMS levels for drawing a waveform.

    Args:
        path: Audio file.
        bars: Number of output bars (clamped to 50..4000).
        start: Window start in seconds (default 0).
        end: Window end in seconds (default end of file).

    Returns:
        ``{"duration", "start", "end", "bars", "levels"}``; ``levels`` are floats
        in [0, 1] (0 = silence, 1 = loudest bar in the window).
    """
    bars = max(50, min(int(bars), 4000))
    info = probe_media(path)
    s = min(max(float(start or 0.0), 0.0), info.duration)
    e = info.duration if end is None else min(max(float(end), s), info.duration)
    if e - s < 0.01:
        raise AudioError("invalid_range", "The requested waveform window is empty.")
    expected = max(1, int((e - s) * WAVEFORM_RATE))
    sums = np.zeros(bars, dtype=np.float64)
    counts = np.zeros(bars, dtype=np.int64)

    cmd = _base_cmd(FFMPEG) + ["-protocol_whitelist", "file,pipe", "-ss", "%.3f" % s,
                               "-t", "%.3f" % (e - s), "-i", _input_arg(path), "-vn",
                               "-map", "0:a:0", "-ac", "1", "-ar", str(WAVEFORM_RATE),
                               "-f", "s16le", "pipe:1"]
    pos = 0
    leftover = b""
    with tempfile.TemporaryFile() as err, _Watchdog(FFMPEG_TIMEOUT_S) as dog:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        dog.watch(proc)
        while True:
            chunk = proc.stdout.read(1 << 18)
            if not chunk:
                break
            chunk = leftover + chunk
            usable = len(chunk) - (len(chunk) % 2)
            leftover = chunk[usable:]
            samples = np.frombuffer(chunk[:usable], dtype="<i2").astype(np.float64) / 32768.0
            idx = np.minimum(((np.arange(pos, pos + samples.size) * bars) // expected), bars - 1)
            sums += np.bincount(idx, weights=samples * samples, minlength=bars)
            counts += np.bincount(idx, minlength=bars)
            pos += samples.size
        proc.stdout.close()
        proc.wait()
        if dog.fired or proc.returncode != 0:
            raise AudioError("waveform_failed", "Could not compute the waveform.")
    rms = np.sqrt(sums / np.maximum(counts, 1))
    peak = float(rms.max())
    levels = (rms / peak) if peak > 0 else rms
    return {"duration": round(info.duration, 3), "start": round(s, 3), "end": round(e, 3),
            "bars": bars, "levels": [round(float(v), 3) for v in levels]}


# --- Rendering ----------------------------------------------------------------
SpanLike = Union[Mapping[str, Any], Sequence[float], Any]


def _span_bounds(span: SpanLike) -> Tuple[float, float]:
    """Extracts ``(start, end)`` from a dict, an object or a ``(start, end)`` pair."""
    if isinstance(span, Mapping):
        return float(span["start"]), float(span["end"])
    if hasattr(span, "start") and hasattr(span, "end"):
        return float(span.start), float(span.end)
    return float(span[0]), float(span[1])


def _sample_ranges(spans: Iterable[SpanLike], rate: int, total_frames: int) -> List[Tuple[int, int]]:
    """Converts spans to sorted, non-overlapping sample ranges within the file."""
    ranges: List[Tuple[int, int]] = []
    for span in spans:
        start, end = _span_bounds(span)
        if not (np.isfinite(start) and np.isfinite(end)):
            raise AudioError("invalid_span", "Span times must be finite numbers.")
        lo = max(0, int(round(min(start, end) * rate)))
        hi = min(total_frames, int(round(max(start, end) * rate)))
        if hi > lo:
            ranges.append((lo, hi))
    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for lo, hi in ranges:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _encoder_args(fmt: str) -> List[str]:
    if fmt == "wav":
        return ["-c:a", "pcm_s16le", "-f", "wav"]
    if fmt == "mp3":
        return ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"]
    if fmt == "flac":
        return ["-c:a", "flac", "-f", "flac"]
    raise AudioError("unsupported_output",
                     "Output format must be one of %s." % (SUPPORTED_OUTPUT_FORMATS,))


def render_redacted_audio(input_path: str, output_path: str, spans: Iterable[SpanLike], *,
                          fmt: str = "wav", beep_freq: float = 1000.0,
                          beep_gain_db: float = -30.0, fade_ms: float = 5.0,
                          max_duration: Optional[float] = None) -> Dict[str, Any]:
    """Replaces the given time ranges with a sine beep and writes a new file.

    Args:
        input_path: Source media file (any ffmpeg-readable audio/video).
        output_path: Destination; written atomically via a ``.part`` file.
        spans: Dicts with ``start``/``end``, objects with those attributes or
            ``(start, end)`` pairs, in seconds.
        fmt: ``"wav"`` (default), ``"mp3"`` or ``"flac"``.
        beep_freq: Beep frequency in Hz.
        beep_gain_db: Beep level in dBFS (negative).
        fade_ms: Fade in/out to avoid clicks.
        max_duration: Override of the duration limit.

    Returns:
        Summary dict (path, duration, sample rate, channels, spans applied).

    Raises:
        AudioError: Invalid media/format, or ffmpeg failure.
    """
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise AudioError("unsupported_output",
                         "Output format must be one of %s." % (SUPPORTED_OUTPUT_FORMATS,))
    info = probe_media(input_path, max_duration)
    rate = info.sample_rate
    channels = info.channels if info.channels in (1, 2) else 2
    ranges = _sample_ranges(spans, rate, int(info.duration * rate) + rate)
    amplitude = 32767.0 * (10.0 ** (min(beep_gain_db, 0.0) / 20.0))
    fade = max(1, int(rate * fade_ms / 1000.0))

    decode = _base_cmd(FFMPEG) + ["-protocol_whitelist", "file,pipe", "-i", _input_arg(input_path),
                                  "-vn", "-sn", "-dn", "-map", "0:a:0", "-ac", str(channels),
                                  "-ar", str(rate), "-f", "s16le", "pipe:1"]
    part = os.path.abspath(output_path) + ".part"
    encode = _base_cmd(FFMPEG) + ["-f", "s16le", "-ar", str(rate), "-ac", str(channels),
                                  "-i", "pipe:0", "-map_metadata", "-1", "-map_chapters", "-1",
                                  "-fflags", "+bitexact", "-flags:a", "+bitexact",
                                  ] + _encoder_args(fmt) + ["-y", part]
    frame_bytes = 2 * channels
    offset = 0
    ok = False
    try:
        with tempfile.TemporaryFile() as err_dec, tempfile.TemporaryFile() as err_enc, \
                _Watchdog(FFMPEG_TIMEOUT_S) as dog:
            dec = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=err_dec)
            enc = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=err_enc)
            dog.watch(dec)
            dog.watch(enc)
            try:
                while True:
                    raw = dec.stdout.read(BLOCK_FRAMES * frame_bytes)
                    if not raw:
                        break
                    raw = raw[:len(raw) - (len(raw) % frame_bytes)]
                    block = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).copy()
                    frames = block.shape[0]
                    for lo, hi in ranges:
                        a, b = max(lo, offset), min(hi, offset + frames)
                        if b <= a:
                            continue
                        idx = np.arange(a, b, dtype=np.float64)
                        tone = np.sin(2.0 * np.pi * beep_freq * idx / rate)
                        env = np.minimum(1.0, (idx - lo + 1) / fade) * np.minimum(1.0, (hi - idx) / fade)
                        block[a - offset:b - offset, :] = (tone * env * amplitude).astype("<i2")[:, None]
                    enc.stdin.write(block.tobytes())
                    offset += frames
            except BrokenPipeError:
                pass                                   # encoder died; reported below
            finally:
                try:
                    enc.stdin.close()
                except OSError:
                    pass
                dec.stdout.close()
            dec.wait()
            enc.wait()
            if dog.fired:
                raise AudioError("render_timeout", "Rendering took too long and was cancelled.")
            if dec.returncode != 0 or enc.returncode != 0 or offset == 0:
                for name, handle in (("decoder", err_dec), ("encoder", err_enc)):
                    handle.seek(0)
                    logger.error("ffmpeg %s failed: %s", name,
                                 handle.read(2000).decode("utf-8", "replace").strip())
                raise AudioError("render_failed", "The audio could not be rendered.")
        os.replace(part, output_path)
        ok = True
    finally:
        if not ok and os.path.exists(part):
            os.unlink(part)
    return {"path": output_path, "duration": offset / float(rate), "sample_rate": rate,
            "channels": channels, "format": fmt, "spans_applied": len(ranges)}


# --- Backwards-compatible class -----------------------------------------------
class AudioBeepReplacer:
    """Original API: replace time offsets with beeps (README "Programmatic Usage").

    Args:
        beep_freq: Beep frequency in Hz.
        beep_gain_db: Beep level in dBFS.
    """

    def __init__(self, beep_freq: float = 1000, beep_gain_db: float = -30):
        self.beep_freq = beep_freq
        self.beep_gain_db = beep_gain_db

    def replace_offsets_with_beeps(self, input_wav: str, offsets: Iterable[SpanLike],
                                   output_wav: Optional[str] = None) -> str:
        """Writes a copy of ``input_wav`` with ``offsets`` replaced by beeps.

        Args:
            input_wav: Source file (any ffmpeg-readable format, not just WAV).
            offsets: ``{"start", "end"}`` dicts or ``(start, end)`` pairs (seconds).
            output_wav: Destination; the extension selects WAV (default), MP3 or FLAC.
                Defaults to ``<input>_beeped.wav``.

        Returns:
            Path of the written file.
        """
        if output_wav is None:
            output_wav = os.path.splitext(input_wav)[0] + "_beeped.wav"
        ext = os.path.splitext(output_wav)[1].lower().lstrip(".")
        fmt = ext if ext in SUPPORTED_OUTPUT_FORMATS else "wav"
        render_redacted_audio(input_wav, output_wav, offsets, fmt=fmt,
                              beep_freq=self.beep_freq, beep_gain_db=self.beep_gain_db)
        return output_wav
