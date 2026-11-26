"""File-based redaction workflow for the command line (no ML stack required).

Takes the artefacts produced by ``process.py`` and masks the audio::

    transcripts/<base>.words.json   word-level timeline (written during transcription)
    anonym/<base>_anon.txt          anonymized text containing [TAG] placeholders
    audios/<base>.wav               source audio
          |
          v
    anonym/<base>_redactions.json   spans (schema v1; original words omitted by default)
    audios_redacted/<base>_redacted.wav

Usage::

    python -m pipeline.redaction_cli --all
    python -m pipeline.redaction_cli --base interview_01 --format flac --pad 0.1
    python -m pipeline.redaction_cli --base interview_01 --redactions-from my_edits.json

``process.py --redact-audio`` calls :func:`redact_batch` after anonymization.
Hand-edited span files use the same JSON schema as the web API.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR.parent))
sys.path.insert(0, str(PIPELINE_DIR.parent / "interactive_app"))

from pipeline import redaction as rd  # noqa: E402
from pipeline.secure_store import ensure_private_dir, write_private_file  # noqa: E402
import audio_utils as au  # noqa: E402

logger = logging.getLogger(__name__)

WORDS_SUFFIX = ".words.json"


def write_words_sidecar(path: os.PathLike, segments: List[Dict[str, Any]]) -> None:
    """Writes the word timeline next to a transcript (mode 0600).

    The file contains the transcript words, so it has the same sensitivity as the
    raw transcript.
    """
    keep = [{"speaker": s.get("speaker"), "start": s.get("start"), "end": s.get("end"),
             "words": [{"word": w.get("word"), "start": w.get("start"), "end": w.get("end")}
                       for w in (s.get("words") or [])]} for s in segments]
    doc = {"schema_version": rd.SCHEMA_VERSION, "segments": keep}
    write_private_file(path, json.dumps(doc, ensure_ascii=False).encode("utf-8"))


def load_words_sidecar(path: os.PathLike) -> List[Dict[str, Any]]:
    """Loads segments from a sidecar written by :func:`write_words_sidecar`."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get("schema_version") != rd.SCHEMA_VERSION:
        raise rd.RedactionError("unsupported_schema", "unsupported words file: %s" % path)
    return doc.get("segments") or []


def redact_one(base: str, *, transcripts: Path, anonym: Path, audios: Path, out_audio: Path,
               options: rd.RedactionOptions, fmt: str = "wav",
               spans_file: Optional[Path] = None, include_words: bool = False,
               beep_gain_db: float = -30.0) -> Dict[str, Any]:
    """Redacts the audio of one transcript.

    Args:
        base: File stem shared by transcript, anonymized text and audio.
        transcripts, anonym, audios, out_audio: Folders (see module docstring).
        options: Span derivation options.
        fmt: Output format (wav/mp3/flac).
        spans_file: Use hand-edited spans instead of deriving them.
        include_words: Keep original words in the JSON file (off by default: they
            would re-identify the masked content).
        beep_gain_db: Beep level in dBFS.

    Returns:
        Summary with output paths and counts.

    Raises:
        FileNotFoundError, rd.RedactionError, au.AudioError: on bad inputs.
    """
    audio_path = audios / (base + ".wav")
    if not audio_path.exists():
        raise FileNotFoundError("audio not found: %s" % audio_path)
    duration = au.probe_media(str(audio_path)).duration
    warnings: List[str] = []
    if spans_file:
        with open(spans_file, "r", encoding="utf-8") as fh:
            spans = rd.spans_from_document(json.load(fh), duration)
    else:
        words_path = transcripts / (base + WORDS_SUFFIX)
        anon_path = anonym / (base + "_anon.txt")
        for needed in (words_path, anon_path):
            if not needed.exists():
                raise FileNotFoundError("missing input: %s" % needed)
        words, warnings = rd.words_from_segments(load_words_sidecar(words_path))
        text = anon_path.read_text(encoding="utf-8")
        if not rd.has_placeholders(text) and options.mode == "placeholders":
            warnings.append("no_placeholders_in_text")
        derived = rd.derive_spans(words, text, options, duration)
        spans, warnings = derived.spans, warnings + derived.warnings
    doc = rd.build_document(spans, duration, options, warnings, include_words)
    json_path = anonym / (base + "_redactions.json")
    write_private_file(json_path, json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8"))
    ensure_private_dir(out_audio)
    out_path = out_audio / ("%s_redacted.%s" % (base, fmt))
    result = au.render_redacted_audio(str(audio_path), str(out_path), spans, fmt=fmt,
                                      beep_gain_db=beep_gain_db)
    return {"base": base, "spans": len(spans), "warnings": warnings,
            "redactions_json": str(json_path), "audio": result["path"]}


def redact_batch(bases: Iterable[str], **kwargs: Any) -> Dict[str, int]:
    """Runs :func:`redact_one` for many files, continuing after individual failures."""
    done = failed = 0
    for base in bases:
        try:
            info = redact_one(base, **kwargs)
            logger.info("Redacted %s: %d span(s) -> %s", base, info["spans"], info["audio"])
            for warning in info["warnings"]:
                logger.warning("  %s: %s", base, warning)
            done += 1
        except (OSError, ValueError, au.AudioError) as exc:   # RedactionError is a ValueError
            logger.error("Redaction failed for %s: %s", base, getattr(exc, "message", exc))
            failed += 1
    return {"redacted": done, "failed": failed}


def discover_bases(transcripts: Path) -> List[str]:
    """Base names that have a words sidecar."""
    return sorted(p.name[:-len(WORDS_SUFFIX)] for p in transcripts.glob("*" + WORDS_SUFFIX))


def main(argv: Optional[List[str]] = None) -> int:
    """Command-line entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Mask anonymized spans in audio with beeps.")
    parser.add_argument("--base", action="append", help="File stem to process (repeatable)")
    parser.add_argument("--all", action="store_true", help="Process every transcript with a words file")
    parser.add_argument("--transcripts", type=Path, default=PIPELINE_DIR / "transcripts")
    parser.add_argument("--anonym", type=Path, default=PIPELINE_DIR / "anonym")
    parser.add_argument("--audios", type=Path, default=PIPELINE_DIR / "audios")
    parser.add_argument("--out", type=Path, default=PIPELINE_DIR / "audios_redacted")
    parser.add_argument("--mode", choices=rd.VALID_MODES, default="placeholders")
    parser.add_argument("--pad", type=float, default=rd.DEFAULT_PAD_S, help="Padding seconds per side")
    parser.add_argument("--merge-gap", type=float, default=rd.DEFAULT_MERGE_GAP_S)
    parser.add_argument("--format", choices=au.SUPPORTED_OUTPUT_FORMATS, default="wav")
    parser.add_argument("--redactions-from", type=Path, help="Hand-edited spans JSON (schema v1)")
    parser.add_argument("--include-words", action="store_true",
                        help="Keep original words in the JSON (not recommended)")
    parser.add_argument("--beep-gain-db", type=float, default=-30.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        options = rd.RedactionOptions.from_mapping(
            {"mode": args.mode, "pad_s": args.pad, "merge_gap_s": args.merge_gap})
    except rd.RedactionError as exc:
        parser.error(exc.message)
    bases = args.base or (discover_bases(args.transcripts) if args.all else [])
    if not bases:
        parser.error("nothing to do: pass --base NAME or --all")
    if args.redactions_from and len(bases) != 1:
        parser.error("--redactions-from needs exactly one --base")
    summary = redact_batch(bases, transcripts=args.transcripts, anonym=args.anonym,
                           audios=args.audios, out_audio=args.out, options=options,
                           fmt=args.format, spans_file=args.redactions_from,
                           include_words=args.include_words, beep_gain_db=args.beep_gain_db)
    logger.info("Done: %(redacted)d redacted, %(failed)d failed", summary)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
