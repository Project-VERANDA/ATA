"""Redaction span derivation and validation (pure Python, no heavy imports).

Decides *which time ranges of an audio file must be masked* once a transcript
has been anonymized. It is free of Flask, torch, whisperx and ffmpeg so the CLI,
the Flask app and any future worker (React back end) share one implementation
that can be unit-tested without the ML stack.

Data flow::

    WhisperX segments --words_from_segments--> [Word]
    [Word] + anonymized text --derive_spans--> [Span]   (auto-detected)
    client-edited spans --sanitize_spans--> [Span]       (validated, merged)
    [Span] --audio_utils.render_redacted_audio--> beeped audio file

All times are seconds (float). The JSON contract is versioned by
``SCHEMA_VERSION`` and documented in ``docs/redaction_api.md``.
"""
from __future__ import annotations

import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1

# --- Defaults and hard limits -------------------------------------------------
DEFAULT_PAD_S = 0.08          # Extra audio masked on each side of a span.
DEFAULT_MERGE_GAP_S = 0.15    # Auto spans closer than this are merged.
DEFAULT_MIN_SPAN_S = 0.10     # Auto spans shorter than this are widened.
MANUAL_MIN_SPAN_S = 0.02      # Smallest span a user may draw by hand.
MIN_GUARD_S = 0.02            # Padding never drops below this, even when
                              # neighbouring words touch (leak-safe bias).
MAX_SPANS = 2000              # Upper bound for any request or document.
MAX_LABEL_CHARS = 200
MAX_ID_CHARS = 40
MAX_TOKENS = 400_000          # Guard against pathological inputs.

VALID_MODES = ("placeholders", "changes")

# ``[PERSON]``, ``[NAME_RELATIVE]`` ... as produced by the anonymizer.
PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]")
# Matches placeholders or runs of other non-bracket, non-space characters.
_TOKEN_RE = re.compile(r"\[[A-Z][A-Z0-9_]*\]|[^\s\[\]]+")
# Leading speaker labels written by the pipeline ("SPEAKER_00:", "Unknown:").
_SPEAKER_PREFIX_RE = re.compile(r"^[ \t]*(?:SPEAKER_\d+|Unknown)[ \t]*:[ \t]*", re.MULTILINE)
# "don ' t" / "don't" -> "dont" so both sides tokenize identically.
_APOSTROPHE_RE = re.compile(r"(?<=\w)\s*['\u2019]\s*(?=\w)")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,%d}$" % MAX_ID_CHARS)


class RedactionError(ValueError):
    """Raised for invalid input. ``code`` is a stable, machine-readable key."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --- Data model ---------------------------------------------------------------
@dataclass
class Word:
    """One transcribed word with timing in seconds."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class Span:
    """A time range to mask.

    Attributes:
        start: Start in seconds.
        end: End in seconds (> start).
        label: Original words covered. Sensitive; omit from persisted output.
        tag: Placeholder(s) that replaced the words, e.g. ``"[PERSON]"``.
        source: ``"auto"`` (derived) or ``"manual"`` (user-drawn/edited).
        id: Stable identifier used by clients.
    """

    start: float
    end: float
    label: str = ""
    tag: str = ""
    source: str = "auto"
    id: str = ""

    def to_dict(self, include_words: bool = True) -> Dict[str, Any]:
        """Serialises to the public JSON shape (times rounded to 1 ms)."""
        return {"id": self.id, "start": round(self.start, 3), "end": round(self.end, 3),
                "tag": self.tag, "source": self.source,
                # ``word`` is the legacy key used by the original editor.
                "word": self.label if include_words else (self.tag or "")}


@dataclass
class RedactionOptions:
    """Tunable behaviour for :func:`derive_spans`.

    Attributes:
        pad_s: Seconds of padding on each side of a span.
        merge_gap_s: Spans separated by less than this are merged.
        min_span_s: Spans shorter than this are widened symmetrically.
        mode: ``"placeholders"`` masks only words replaced by a ``[TAG]`` (and
            optionally deleted words); ``"changes"`` masks every difference.
        include_deletions: Also mask words that were deleted without a tag.
    """

    pad_s: float = DEFAULT_PAD_S
    merge_gap_s: float = DEFAULT_MERGE_GAP_S
    min_span_s: float = DEFAULT_MIN_SPAN_S
    mode: str = "placeholders"
    include_deletions: bool = True

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "RedactionOptions":
        """Builds options from untrusted JSON, validating every field."""
        opts = cls()
        if not data:
            return opts
        if not isinstance(data, Mapping):
            raise RedactionError("invalid_options", "options must be an object")
        opts.pad_s = _bounded(data.get("pad_s", data.get("pad")), "pad_s", 0.0, 1.0, opts.pad_s)
        opts.merge_gap_s = _bounded(data.get("merge_gap_s", data.get("merge_gap")),
                                    "merge_gap_s", 0.0, 2.0, opts.merge_gap_s)
        opts.min_span_s = _bounded(data.get("min_span_s", data.get("min_span")),
                                   "min_span_s", MANUAL_MIN_SPAN_S, 2.0, opts.min_span_s)
        mode = data.get("mode", opts.mode)
        if mode not in VALID_MODES:
            raise RedactionError("invalid_options", "mode must be one of %s" % (VALID_MODES,))
        opts.mode = mode
        incl = data.get("include_deletions", opts.include_deletions)
        if not isinstance(incl, bool):
            raise RedactionError("invalid_options", "include_deletions must be boolean")
        opts.include_deletions = incl
        return opts

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the options for inclusion in a document."""
        return {"pad_s": self.pad_s, "merge_gap_s": self.merge_gap_s,
                "min_span_s": self.min_span_s, "mode": self.mode,
                "include_deletions": self.include_deletions}


@dataclass
class DeriveResult:
    """Output of :func:`derive_spans`."""

    spans: List[Span]
    warnings: List[str] = field(default_factory=list)


# --- Small helpers ------------------------------------------------------------
def _as_float(value: Any, name: str) -> float:
    """Converts a JSON value to a finite float or raises ``RedactionError``."""
    if isinstance(value, bool) or value is None:
        raise RedactionError("invalid_number", "%s must be a number" % name)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RedactionError("invalid_number", "%s must be a number" % name) from None
    if not math.isfinite(number):
        raise RedactionError("invalid_number", "%s must be finite" % name)
    return number


def _bounded(value: Any, name: str, low: float, high: float, default: float) -> float:
    """Returns ``value`` as a float within [low, high], or ``default`` if None."""
    if value is None:
        return default
    number = _as_float(value, name)
    if not low <= number <= high:
        raise RedactionError("invalid_options", "%s must be between %s and %s" % (name, low, high))
    return number


def _clean_label(text: Any) -> str:
    """Bounds a free-text label and strips control characters."""
    if not isinstance(text, str):
        if isinstance(text, (list, tuple)):  # legacy: list of words
            text = " ".join(str(t) for t in text)
        else:
            return ""
    return "".join(ch for ch in text if ch.isprintable()).strip()[:MAX_LABEL_CHARS]


def _finite_or_none(value: Any) -> Optional[float]:
    """Returns ``value`` as a finite float, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# --- Step 1: word timeline ----------------------------------------------------
def words_from_segments(segments: Iterable[Mapping[str, Any]]) -> Tuple[List[Word], List[str]]:
    """Flattens WhisperX segments into a monotonic word timeline.

    WhisperX omits ``start``/``end`` for tokens it cannot align (often numbers);
    those words get linearly interpolated times. Segments without any ``words``
    fall back to splitting the segment text evenly across its duration (flagged
    with a warning because the boundaries are approximate).

    Args:
        segments: WhisperX-style dicts (``words``, ``start``, ``end``, ``text``, ``speaker``).

    Returns:
        ``(words, warnings)``.
    """
    words: List[Word] = []
    warnings: List[str] = []
    approximate = False
    for seg in segments:
        if not isinstance(seg, Mapping):
            continue
        seg_start = _finite_or_none(seg.get("start"))
        seg_end = _finite_or_none(seg.get("end"))
        entries: List[List[Any]] = []  # [text, start|None, end|None]
        for item in seg.get("words") or []:
            if not isinstance(item, Mapping):
                continue
            text = str(item.get("word", item.get("text", ""))).strip()
            if text:
                entries.append([text, _finite_or_none(item.get("start")),
                                _finite_or_none(item.get("end"))])
        if not entries:
            text = str(seg.get("text", "")).strip()
            if text and seg_start is not None and seg_end is not None and seg_end > seg_start:
                pieces = text.split()
                step = (seg_end - seg_start) / len(pieces)
                entries = [[p, seg_start + k * step, seg_start + (k + 1) * step]
                           for k, p in enumerate(pieces)]
                approximate = True
            else:
                continue
        _interpolate_missing(entries, seg_start, seg_end, words[-1].end if words else 0.0)
        words.extend(Word(t, float(a), float(b), seg.get("speaker")) for t, a, b in entries)
    if approximate:
        warnings.append("segment_without_word_timestamps")
    return words, warnings


def _interpolate_missing(entries: List[List[Any]], seg_start: Optional[float],
                         seg_end: Optional[float], floor: float) -> None:
    """Fills missing start/end values in place and enforces ``end >= start``."""
    n = len(entries)
    i = 0
    while i < n:
        if entries[i][1] is not None and entries[i][2] is not None:
            i += 1
            continue
        j = i
        while j < n and (entries[j][1] is None or entries[j][2] is None):
            j += 1
        left = entries[i - 1][2] if i > 0 else (seg_start if seg_start is not None else floor)
        right = entries[j][1] if j < n else seg_end
        count = j - i
        if right is None or right <= left:
            right = left + 0.05 * count  # no usable anchor: assume 50 ms per word
        step = (right - left) / count
        for k in range(count):
            entries[i + k][1] = left + k * step
            entries[i + k][2] = left + (k + 1) * step
        i = j
    for entry in entries:
        if entry[2] < entry[1]:
            entry[2] = entry[1]


# --- Step 2: diff words against the anonymized text ---------------------------
def normalize_token(token: str) -> str:
    """Case-folds and strips everything except letters and digits."""
    token = unicodedata.normalize("NFKC", token).casefold()
    return "".join(ch for ch in token if ch.isalnum())


def tokenize_text(text: str) -> List[str]:
    """Splits anonymized text into tokens, isolating ``[TAG]`` placeholders.

    Speaker labels are removed and apostrophes inside words are joined so that
    ``don ' t`` and ``don't`` compare equal.
    """
    text = _APOSTROPHE_RE.sub("", _SPEAKER_PREFIX_RE.sub("", text))
    return _TOKEN_RE.findall(text)


def has_placeholders(text: str) -> bool:
    """Returns True if ``text`` contains at least one ``[TAG]`` placeholder."""
    return bool(text) and PLACEHOLDER_RE.search(text) is not None


_Op = Tuple[str, int, int, int, int]   # (tag, a_start, a_stop, b_start, b_stop)


def _resync(a: Sequence[str], b: Sequence[str], i: int, j: int,
            window: int, k: int) -> Optional[Tuple[int, int]]:
    """Finds the nearest point after a mismatch where both sequences agree again.

    Tries offsets ``(di, dj)`` by increasing ``di + dj`` and accepts the first where
    ``k`` consecutive tokens match - or fewer if the run is cut off by a placeholder
    (an expected mismatch, e.g. ``[PERSON] lives in [CITY]``) or by the end of a text.
    """
    n, m = len(a), len(b)
    for dist in range(1, window + 1):
        for di in range(dist + 1):
            dj = dist - di
            if i + di > n or j + dj > m:
                continue
            if i + di == n and j + dj == m:
                return di, dj                      # both exhausted: tail differs
            run = 0
            while (run < k and i + di + run < n and j + dj + run < m
                   and a[i + di + run] == b[j + dj + run]):
                run += 1
            if run == k:
                return di, dj
            if run >= 1 and (i + di + run == n or j + dj + run == m
                             or b[j + dj + run].startswith("[")):
                return di, dj
    return None


def _align(a: Sequence[str], b: Sequence[str], window: int = 200,
           k: int = 3) -> Tuple[List[_Op], bool]:
    """Linear-time alignment for sequences that are mostly identical.

    ``difflib.SequenceMatcher`` degrades on long, repetitive speech transcripts
    (quadratic time, or one giant wrong block once "popular" tokens are ignored).
    Anonymization only replaces short stretches, so a greedy walk with local
    re-synchronisation is faster and more predictable.

    Returns:
        ``(opcodes, diverged)``; opcodes mimic ``SequenceMatcher.get_opcodes``.
        ``diverged`` is True if the texts could not be re-synchronised.
    """
    ops: List[_Op] = []
    n, m = len(a), len(b)
    i = j = 0

    def gap(i0: int, i1: int, j0: int, j1: int) -> None:
        if i1 > i0 and j1 > j0:
            ops.append(("replace", i0, i1, j0, j1))
        elif i1 > i0:
            ops.append(("delete", i0, i1, j0, j1))
        elif j1 > j0:
            ops.append(("insert", i0, i1, j0, j1))

    while i < n and j < m:
        if a[i] == b[j]:
            i0, j0 = i, j
            while i < n and j < m and a[i] == b[j]:
                i += 1
                j += 1
            ops.append(("equal", i0, i, j0, j))
            continue
        found = _resync(a, b, i, j, window, k)
        if found is None:
            if n - i <= 3000 and m - j <= 3000:   # bounded fallback
                sm = difflib.SequenceMatcher(None, a[i:], b[j:], autojunk=False)
                ops.extend((op, i + x1, i + x2, j + y1, j + y2)
                           for op, x1, x2, y1, y2 in sm.get_opcodes())
            else:
                gap(i, n, j, m)
            return ops, True
        gap(i, i + found[0], j, j + found[1])
        i += found[0]
        j += found[1]
    gap(i, n, j, m)
    return ops, False


def derive_spans(words: Sequence[Word], redacted_text: str,
                 options: Optional[RedactionOptions] = None,
                 duration: Optional[float] = None) -> DeriveResult:
    """Derives mask spans by diffing the original words with the redacted text.

    Words replaced by a ``[TAG]`` placeholder (or deleted) become spans from the
    first to the last affected word. Punctuation/capitalisation changes never
    produce a span. Spans are padded (at most half the gap to the neighbouring
    word plus a small guard), merged, widened to a minimum length and clamped.

    Args:
        words: Timeline from :func:`words_from_segments`.
        redacted_text: Text containing ``[TAG]`` placeholders.
        options: Behaviour switches; defaults when omitted.
        duration: Audio length in seconds, used for clamping.

    Returns:
        :class:`DeriveResult` with spans in chronological order. Warnings may
        include ``unmapped_placeholder_insertion`` and ``alignment_diverged``.

    Raises:
        RedactionError: Unreasonably large input.
    """
    opts = options or RedactionOptions()
    warnings: List[str] = []
    b_raw = tokenize_text(redacted_text or "")
    if len(words) > MAX_TOKENS or len(b_raw) > MAX_TOKENS:
        raise RedactionError("input_too_large", "transcript too long")

    a_norm: List[str] = []
    a_idx: List[int] = []                       # filtered index -> word index
    for idx, word in enumerate(words):
        norm = normalize_token(word.text)
        if norm:
            a_norm.append(norm)
            a_idx.append(idx)
    b_norm: List[str] = []
    b_tok: List[str] = []                       # raw token, for tag detection
    for tok in b_raw:
        norm = tok if PLACEHOLDER_RE.fullmatch(tok) else normalize_token(tok)
        if norm:
            b_norm.append(norm)
            b_tok.append(tok)

    opcodes, diverged = _align(a_norm, b_norm)
    if diverged:
        warnings.append("alignment_diverged")
    raw_spans: List[Tuple[int, int, str, str]] = []   # (first, stop_excl, label, tag)
    for op, i1, i2, j1, j2 in opcodes:
        if op == "equal":
            continue
        tags = [t for t in b_tok[j1:j2] if PLACEHOLDER_RE.fullmatch(t)]
        if op == "insert":
            if tags:
                warnings.append("unmapped_placeholder_insertion")
            continue
        wanted = (bool(tags) or (op == "delete" and opts.include_deletions)
                  if opts.mode == "placeholders" else True)
        if wanted:
            first, last = a_idx[i1], a_idx[i2 - 1]
            raw_spans.append((first, last + 1,
                              " ".join(w.text for w in words[first:last + 1]), " ".join(tags)))

    spans: List[Span] = []
    for first, stop, label, tag in raw_spans:
        start = words[first].start
        end = max(w.end for w in words[first:stop])
        # No neighbour at the file edge -> nothing to protect -> full padding.
        gap_before = max(0.0, start - words[first - 1].end) if first > 0 else float("inf")
        if stop < len(words):
            gap_after = max(0.0, words[stop].start - end)
        else:
            gap_after = max(0.0, duration - end) if duration is not None else float("inf")
        start -= min(opts.pad_s, max(gap_before / 2.0, MIN_GUARD_S))
        end += min(opts.pad_s, max(gap_after / 2.0, MIN_GUARD_S))
        spans.append(Span(start, end, _clean_label(label), tag, "auto"))

    spans = _finalize(spans, duration, opts.merge_gap_s, opts.min_span_s)
    for n, span in enumerate(spans, 1):
        span.id = "a%d" % n
    return DeriveResult(spans, sorted(set(warnings)))


# --- Step 3: normalisation shared by auto and manual spans --------------------
def _finalize(spans: List[Span], duration: Optional[float], merge_gap: float,
              min_span: float) -> List[Span]:
    """Widens, clamps, sorts and merges spans."""
    out: List[Span] = []
    for sp in spans:
        start, end = sp.start, sp.end
        if end - start < min_span:
            centre = (start + end) / 2.0
            start, end = centre - min_span / 2.0, centre + min_span / 2.0
        start = max(0.0, start)
        if duration is not None:
            end = min(end, duration)
            start = min(start, end)
        if end - start > 1e-6:
            out.append(Span(start, end, sp.label, sp.tag, sp.source, sp.id))
    out.sort(key=lambda s: (s.start, s.end))
    merged: List[Span] = []
    for sp in out:
        if merged and sp.start - merged[-1].end <= merge_gap:
            prev = merged[-1]
            prev.end = max(prev.end, sp.end)
            prev.label = _clean_label(", ".join(x for x in (prev.label, sp.label) if x))
            prev.tag = " ".join(dict.fromkeys((prev.tag + " " + sp.tag).split()))
            if sp.source == "manual":
                prev.source = "manual"
        else:
            merged.append(sp)
    return merged


def sanitize_spans(raw: Any, duration: Optional[float], *, merge_gap: float = 0.0,
                   min_span: float = MANUAL_MIN_SPAN_S,
                   max_spans: int = MAX_SPANS) -> List[Span]:
    """Validates spans received from a client and merges overlaps.

    Unlike :func:`derive_spans` no padding is added: user-chosen boundaries are
    final. Overlapping spans are always merged (``merge_gap=0``).

    Raises:
        RedactionError: Non-list input, too many spans, or non-numeric/non-finite times.
    """
    if not isinstance(raw, (list, tuple)):
        raise RedactionError("invalid_segments", "segments must be a list")
    if len(raw) > max_spans:
        raise RedactionError("too_many_segments", "at most %d segments are allowed" % max_spans)
    spans: List[Span] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise RedactionError("invalid_segments", "each segment must be an object")
        start = _as_float(item.get("start"), "start")
        end = _as_float(item.get("end"), "end")
        if end < start:
            start, end = end, start
        raw_id = item.get("id")
        source = item.get("source") if item.get("source") in ("auto", "manual") else "manual"
        spans.append(Span(start, end, _clean_label(item.get("word", item.get("label", ""))),
                          _clean_label(item.get("tag", ""))[:80], source,
                          raw_id if isinstance(raw_id, str) and _SAFE_ID_RE.match(raw_id) else ""))
    result = _finalize(spans, duration, merge_gap, min_span)
    # Guarantee unique ids (merging or client mistakes can duplicate them).
    used: Dict[str, int] = {}
    for sp in result:
        used[sp.id] = used.get(sp.id, 0) + 1
    seen = set()
    counter = 0
    for sp in result:
        if not sp.id or used[sp.id] > 1 or sp.id in seen:
            while True:
                counter += 1
                candidate = "m%d" % counter
                if candidate not in used and candidate not in seen:
                    sp.id = candidate
                    break
        seen.add(sp.id)
    return result


# --- JSON document (CLI files, API responses) ---------------------------------
def build_document(spans: Sequence[Span], duration: Optional[float],
                   options: Optional[RedactionOptions] = None,
                   warnings: Optional[Sequence[str]] = None,
                   include_words: bool = True) -> Dict[str, Any]:
    """Builds the versioned JSON document shared by CLI and API.

    ``include_words=False`` omits the original words (use for files written to
    disk: they would otherwise re-identify the redacted content).
    """
    return {"schema_version": SCHEMA_VERSION,
            "duration": None if duration is None else round(duration, 3),
            "options": (options or RedactionOptions()).to_dict(),
            "spans": [s.to_dict(include_words) for s in spans],
            "warnings": list(warnings or [])}


def spans_from_document(doc: Any, duration: Optional[float] = None) -> List[Span]:
    """Loads and validates spans from a document produced by :func:`build_document`.

    Raises:
        RedactionError: Unknown schema version or malformed content.
    """
    if not isinstance(doc, Mapping):
        raise RedactionError("invalid_document", "document must be an object")
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise RedactionError("unsupported_schema",
                             "unsupported schema_version %r" % (doc.get("schema_version"),))
    return sanitize_spans(doc.get("spans"), duration)


def legacy_offsets(spans: Sequence[Span]) -> List[Dict[str, Any]]:
    """Returns spans in the shape of the old ``diff_relevant_offsets``."""
    return [{"start": s.start, "end": s.end, "word": s.label.split()} for s in spans]
