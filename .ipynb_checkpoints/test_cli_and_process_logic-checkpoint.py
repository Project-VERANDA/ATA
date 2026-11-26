"""Tests for the redaction CLI workflow and the ML-free logic inside pipeline/process.py.

``process.py`` imports torch/whisperx at module level, so its pure functions are
extracted with ``ast`` and executed in a small namespace instead of importing the file.
"""
import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import types
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "interactive_app"))

import audio_utils as au  # noqa: E402
from pipeline import redaction as rd, redaction_cli as cli, surrogates  # noqa: E402

PROCESS_SRC = os.path.join(ROOT, "pipeline", "process.py")
WANTED = {"merge_consecutive_speaker_segments", "WHISPER_LANG_MAP", "WHISPER_MODEL_DIRS", "_MODEL_ALIASES",
          "normalize_model_name", "resolve_language", "TAG_PLACEHOLDERS", "AnonymizationEngine",
          "apply_surrogate_substitution", "TranscriptionResult", "TranscriptionError"}


def load_process_symbols():
    """Executes selected top-level definitions of process.py in an isolated namespace."""
    with open(PROCESS_SRC, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    nodes = []
    for node in tree.body:
        names = ([node.name] if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                 [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)])
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if WANTED.intersection(names):
            nodes.append(node)
    from dataclasses import dataclass, field
    from typing import Any, Dict, List, Optional
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    ns = {"re": re, "os": os, "logger": logging.getLogger("t"), "threading": threading, "torch": torch,
          "dataclass": dataclass, "field": field, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional,
          "ANONYMIZATION_ENABLED": False, "MODEL_FOLDER": None,
          "apply_surrogates": surrogates.apply_surrogates, "lang_from_locales": surrogates.lang_from_locales,
          "_ANON_LOCK": threading.Lock()}
    exec(compile(ast.Module(nodes, []), PROCESS_SRC, "exec"), ns)
    return ns


P = load_process_symbols()


class ProcessLogic(unittest.TestCase):
    def test_merge_preserves_words(self):
        segs = [{"speaker": "A", "start": 0, "end": 1, "text": "hi", "words": [{"word": "hi", "start": 0, "end": 1}]},
                {"speaker": "A", "start": 1.5, "end": 2, "text": "there", "words": [{"word": "there", "start": 1.5, "end": 2}]},
                {"speaker": "B", "start": 2.1, "end": 3, "text": "yo"},
                {"speaker": "A", "start": 9, "end": 10, "text": "late", "words": [{"word": "late"}]}]
        out = P["merge_consecutive_speaker_segments"](segs, max_gap_seconds=2.0)
        self.assertEqual([len(s["words"]) for s in out], [2, 0, 1])
        self.assertEqual(out[0]["text"], "hi there")
        self.assertEqual(len(P["merge_consecutive_speaker_segments"](segs, max_gap_seconds=None)), 3)
        self.assertEqual(P["merge_consecutive_speaker_segments"]([]), [])

    def test_resolve_language_accepts_any_case(self):
        f = P["resolve_language"]
        self.assertEqual([f("de"), f("DE"), f("Sp"), f("es")], ["de", "de", "es", "es"])
        self.assertEqual([f(None), f(""), f("auto"), f("AUTO")], [None] * 4)
        with self.assertRaises(ValueError):
            f("klingon")

    def test_model_names(self):
        f = P["normalize_model_name"]
        self.assertEqual((f("Large-V3"), f("tiny"), f("large-turbo")), ("large", "tiny", "large-turbo"))
        with self.assertRaises(ValueError):
            f("huge")

    def test_transcription_result_is_independent_of_text(self):
        res = P["TranscriptionResult"](ok=True, text="SPEAKER_00: The Error: it failed.")
        self.assertTrue(res.ok)
        self.assertIsNone(res.error_code)

    def test_tag_filtering_actually_filters(self):
        Engine = P["AnonymizationEngine"]
        eng = Engine.__new__(Engine)
        eng.original_id2label = {"0": "O", "1": "B-PERSON", "2": "I-PERSON", "3": "B-LOC_CITY", "4": "B-ORG"}
        full = eng._build_safe_label_mapping()
        self.assertEqual(full, {"0": "", "1": "[PERSON]", "2": "[PERSON]", "3": "[CITY]", "4": "[ORGANISATION]"})
        only = eng._build_safe_label_mapping(["PERSON"], None)
        self.assertEqual((only["1"], only["3"], only["4"]), ("[PERSON]", "", ""))
        excl = eng._build_safe_label_mapping(None, ["person"])
        self.assertEqual((excl["1"], excl["3"]), ("", "[CITY]"))
        none = eng._build_safe_label_mapping([], None)            # Clear All -> nothing anonymized
        self.assertEqual(set(none.values()), {""})

    def test_anonymize_always_returns_five_values(self):
        Engine = P["AnonymizationEngine"]
        eng = Engine.__new__(Engine)
        eng.method = eng.model = eng.tokenizer = None
        result = eng.anonymize("SPEAKER_00: x")
        self.assertEqual(len(result), 5)
        self.assertFalse(result[1])

    def test_surrogate_wrapper(self):
        f = P["apply_surrogate_substitution"]
        self.assertEqual(f("[PERSON]", {}, use_surrogates=False), ("[PERSON]", {}))
        text, registry = f("SPEAKER_00: [PERSON] in [CITY]", {"[PERSON]": ["Max"], "[CITY]": ["Berlin"]},
                           use_surrogates=True, seed=1, locales="de_DE,en_US")
        self.assertNotRegex(text, r"\[")
        self.assertTrue(text.startswith("SPEAKER_00: "))
        self.assertEqual(set(registry), {"PERSON", "CITY"})
        self.assertNotIn("Max", str(registry))


@unittest.skipUnless(au.ffmpeg_available(), "ffmpeg not available")
class CliWorkflow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        d = self.tmp.name
        self.dirs = {k: os.path.join(d, k) for k in ("transcripts", "anonym", "audios", "out")}
        for p in self.dirs.values():
            os.makedirs(p)
        subprocess.run([au.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=4", os.path.join(self.dirs["audios"], "talk.wav")], check=True)
        segments = [{"speaker": "SPEAKER_00", "start": 0.4, "end": 3.1, "words": [
            {"word": "Max", "start": 0.5, "end": 0.9, "score": 0.9}, {"word": "Mustermann", "start": 1.0, "end": 1.6},
            {"word": "lives", "start": 1.8, "end": 2.1}, {"word": "in", "start": 2.2, "end": 2.3},
            {"word": "Berlin.", "start": 2.4, "end": 3.0}]}]
        cli.write_words_sidecar(os.path.join(self.dirs["transcripts"], "talk.words.json"), segments)
        with open(os.path.join(self.dirs["anonym"], "talk_anon.txt"), "w") as fh:
            fh.write("SPEAKER_00: [PERSON] lives in [CITY].\n")
        self.kw = dict(transcripts=self.d("transcripts"), anonym=self.d("anonym"), audios=self.d("audios"),
                       out_audio=self.d("out"), options=rd.RedactionOptions())

    def d(self, name):
        from pathlib import Path
        return Path(self.dirs[name])

    def test_redact_one_writes_private_json_without_original_words(self):
        info = cli.redact_one("talk", **self.kw)
        self.assertEqual(info["spans"], 2)
        self.assertTrue(os.path.exists(info["audio"]))
        with open(info["redactions_json"], encoding="utf-8") as fh:
            raw = fh.read()
        self.assertNotIn("Mustermann", raw)
        self.assertNotIn("Berlin", raw)
        doc = json.loads(raw)
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual([s["word"] for s in doc["spans"]], ["[PERSON]", "[CITY]"])
        self.assertEqual(oct(os.stat(info["redactions_json"]).st_mode & 0o777), oct(0o600))

    def test_hand_edited_spans_are_rendered(self):
        edited = os.path.join(self.tmp.name, "edit.json")
        with open(edited, "w") as fh:
            json.dump({"schema_version": 1, "spans": [{"start": 3.2, "end": 3.6}]}, fh)
        from pathlib import Path
        info = cli.redact_one("talk", spans_file=Path(edited), **self.kw)
        self.assertEqual(info["spans"], 1)

    def test_batch_continues_after_failures_and_main_entrypoint(self):
        self.assertEqual(cli.redact_batch(["missing", "talk"], **self.kw), {"redacted": 1, "failed": 1})
        argv = ["--all", "--transcripts", self.dirs["transcripts"], "--anonym", self.dirs["anonym"],
                "--audios", self.dirs["audios"], "--out", self.dirs["out"], "--format", "flac", "--pad", "0.1"]
        self.assertEqual(cli.main(argv), 0)
        self.assertTrue(os.path.exists(os.path.join(self.dirs["out"], "talk_redacted.flac")))
        self.assertEqual(cli.main(["--base", "nope"] + argv[1:]), 1)

    def test_words_sidecar_roundtrip_and_schema_check(self):
        path = os.path.join(self.dirs["transcripts"], "talk.words.json")
        self.assertEqual(len(cli.load_words_sidecar(path)[0]["words"]), 5)
        with open(path, "w") as fh:
            json.dump({"schema_version": 7}, fh)
        with self.assertRaises(rd.RedactionError):
            cli.load_words_sidecar(path)
        self.assertEqual(cli.discover_bases(self.d("transcripts")), ["talk"])


if __name__ == "__main__":
    unittest.main()
