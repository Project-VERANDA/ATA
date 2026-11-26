"""End-to-end tests of interactive_app/app.py with the ML parts stubbed.

The real Flask routes, job store, encryption, redaction logic and ffmpeg renderer
run; only transcription/anonymization/TTS models are replaced by small stubs.
The project tree is copied to a temp directory first so no files are created in
the repository.
"""
import dataclasses
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ATA"))
sys.path.insert(0, ROOT)
from interactive_app import audio_utils as au  # noqa: E402

HAVE_FFMPEG = au.ffmpeg_available()
TEXT = "SPEAKER_00: Max Mustermann lives in Berlin."
SEGMENTS = [{"speaker": "SPEAKER_00", "start": 0.4, "end": 3.1, "text": "Max Mustermann lives in Berlin.",
             "words": [{"word": "Max", "start": 0.5, "end": 0.9}, {"word": "Mustermann", "start": 1.0, "end": 1.6},
                       {"word": "lives", "start": 1.8, "end": 2.1}, {"word": "in", "start": 2.2, "end": 2.3},
                       {"word": "Berlin.", "start": 2.4, "end": 3.0}]}]


@dataclasses.dataclass
class FakeResult:
    ok: bool
    text: str = ""
    segments: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    language: Optional[str] = "en"
    duration: Optional[float] = 4.0
    error_code: Optional[str] = None
    error: Optional[str] = None
    warnings: List[str] = dataclasses.field(default_factory=list)


class FakeEngine:
    """Stand-in for AnonymizationEngine with the real 5-tuple contract."""

    method = "local_mmbert"

    def __init__(self, *a, **k):
        self.calls = []

    def anonymize(self, text, use_surrogates=False, surrogate_seed=None, surrogate_locales="",
                  include_tags=None, exclude_tags=None):
        self.calls.append((include_tags, exclude_tags))
        out = text.replace("Max Mustermann", "[PERSON]").replace("Berlin", "[CITY]")
        return out, True, "Success", {"[PERSON]": ["Max Mustermann"], "[CITY]": ["Berlin"]}, {}


def install_stubs(workdir):
    """Registers stub modules for torch/process/openai/dotenv/tts before app import."""
    ns = types.SimpleNamespace
    torch = types.ModuleType("torch")
    torch.backends = ns(cudnn=ns(enabled=True, benchmark=True, allow_tf32=True),
                        cuda=ns(matmul=ns(allow_tf32=True)))
    torch.cuda = ns(is_available=lambda: False)
    sys.modules["torch"] = torch

    def transcribe_audio_result(path, language=None, whisper_model_name=None, max_duration=None):
        if os.path.getsize(path) < 1000:
            return FakeResult(False, error_code="unreadable_media", error="The file is not readable.")
        text = "SPEAKER_00: The first test failed. Error: Max Mustermann lives in Berlin."
        return FakeResult(True, text, SEGMENTS)

    process = types.ModuleType("process")
    process.transcribe_audio_result = transcribe_audio_result
    process.transcribe_audio_locally = lambda *a, **k: ("", [])
    process.load_models = lambda *a, **k: (object(), None)
    process.call_llm_rewriter = lambda *a, **k: ("x", "Success")
    process.generate_paraphrase = lambda *a, **k: "x"
    process.AVAILABLE_LLM_MODELS = {"medgemma": "medgemma"}
    process.CHAT_AI_API_KEY = None
    process.CHAT_AI_ENDPOINT = "https://example.invalid/v1"
    process.MODEL_FOLDER = types.SimpleNamespace  # replaced below with a Path
    from pathlib import Path
    process.MODEL_FOLDER = Path(workdir) / "model"
    process.LLM_ANONYM_FOLDER = Path(workdir) / "llm"
    process.ANONYM_FOLDER = Path(workdir) / "anonym"
    process.BASE_PATH = Path(workdir)
    process.AnonymizationEngine = FakeEngine
    process.apply_surrogate_substitution = None
    process.SURROGATES = {}
    process.TAG_TO_SURROGATE_CATEGORY = {}
    process.AVAILABLE_TAGS = ["PERSON", "ORG"]
    sys.modules["process"] = process

    wav = os.path.join(workdir, "tts_out.wav")
    subprocess.run([au.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=300:duration=1", wav], check=True)
    tts = types.ModuleType("pipeline.tts.tts_engine")
    tts.generate_speech = lambda *a, **k: None
    tts.synthesize_segment = lambda *a, **k: None
    tts.get_available_tts_voices = lambda language="en": [{"id": "en_US-amy-medium", "language": language}]
    tts.get_tts_status = lambda: {"enabled": True, "backend": "piper", "available_voices": ["a"]}
    tts.TTS_BACKEND, tts.TTS_ENABLED = "piper", True
    seen = {}

    def generate_multi_speaker_tts(transcript_text, word_offsets=None, audio_source_path=None):
        seen["text"] = transcript_text
        return wav, {"SPEAKER_00": "en_US-amy-medium"}

    tts.generate_multi_speaker_tts = generate_multi_speaker_tts
    pkg = types.ModuleType("pipeline.tts")
    pkg.tts_engine = tts
    sys.modules["pipeline.tts"], sys.modules["pipeline.tts.tts_engine"] = pkg, tts
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    sys.modules["openai"] = openai
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *a, **k: None
    sys.modules["dotenv"] = dotenv
    return seen


def slurp(path):
    """Reads a whole file, closing the handle."""
    with open(path, "rb") as fh:
        return fh.read()


def read_wav(blob):
    with wave.open(io.BytesIO(blob), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"), w.getframerate()


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not available")
class AppFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        root = os.path.join(cls.tmp, "proj")
        for name in ("pipeline", "interactive_app"):
            shutil.copytree(os.path.join(ROOT, name), os.path.join(root, name),
                            ignore=shutil.ignore_patterns("__pycache__", "process.py"))
        cls.work = os.path.join(cls.tmp, "work")
        cls.saved_env = dict(os.environ)
        os.environ.update({
            "ATA_WORK_DIR": cls.work, "ATA_JOBS_DIR": os.path.join(cls.tmp, "jobs"),
            "ATA_UPLOAD_DIR": os.path.join(cls.tmp, "uploads"), "ATA_CORS_ORIGINS": "http://localhost:5173",
            "ATA_MAX_JSON_BYTES": "20000", "ATA_PROJECT_ROOT": root,
        })
        for name in ("ATA_MASTER_KEY_FILE", "ATA_REQUIRE_MASTER_KEY", "ATA_DEBUG_TOKEN",
                     "ATA_ENABLE_BULK", "ATA_ENABLE_FILE_DOWNLOADS"):
            os.environ.pop(name, None)
        cls.saved_modules = {k: sys.modules.get(k) for k in
                             ("torch", "process", "pipeline.tts", "pipeline.tts.tts_engine", "openai", "dotenv")}
        cls.tts_seen = install_stubs(cls.tmp)
        os.makedirs(os.path.join(cls.tmp, "model"), exist_ok=True)
        cls.path_backup = list(sys.path)
        sys.path[:0] = [root, os.path.join(root, "interactive_app")]
        for mod in [m for m in sys.modules if m == "pipeline" or m.startswith("pipeline.")
                    and m not in ("pipeline.tts", "pipeline.tts.tts_engine")]:
            del sys.modules[mod]
        spec = importlib.util.spec_from_file_location("ata_app", os.path.join(root, "interactive_app", "app.py"))
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)
        cls.mod.app.testing = True
        cls.client = cls.mod.app.test_client()
        cls.audio = os.path.join(cls.tmp, "in.wav")
        subprocess.run([au.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=4", cls.audio], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.mod.JOB_STORE.stop()
        sys.path[:] = cls.path_backup
        os.environ.clear()
        os.environ.update(cls.saved_env)
        for name, mod in cls.saved_modules.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # -- helpers ---------------------------------------------------------------
    def upload(self, **form):
        data = {"audio": (io.BytesIO(slurp(self.audio)), "interview 01.wav"), "language": "de",
                "model_size": "base"}
        data.update(form)
        return self.client.post("/upload", data=data, content_type="multipart/form-data")

    def session(self):
        body = self.upload().get_json()
        hdr = {"X-Job-Token": body["job_token"]}
        r = self.client.post("/anonymize", json={"transcript": TEXT, "job_id": body["job_id"],
                                                 "include_tags": []}, headers=hdr)
        self.assertEqual(r.status_code, 200, r.get_json())
        return body["job_id"], hdr

    # -- tests -----------------------------------------------------------------
    def test_upload_structured_result_not_text_sniffing(self):
        r = self.upload()
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(body["success"])
        # The transcript contains "failed" and "Error:" - a valid result, not an error.
        self.assertIn("failed", body["transcription"])
        self.assertEqual(body["audio_name"], "interview_01.wav")
        self.assertNotIn("/", body["audio_name"])
        self.assertEqual(len(body["job_id"]), 32)
        self.assertEqual(len(body["offsets"][0]["words"]), 5)
        self.assertEqual(r.headers["Cache-Control"], "no-store")

    def test_upload_failure_is_reported_by_code(self):
        r = self.client.post("/upload", data={"audio": (io.BytesIO(b"tiny"), "x.wav")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["code"], "unreadable_media")
        self.assertFalse(r.get_json()["success"])
        for bad in ({"audio": (io.BytesIO(b"x" * 2000), "x.exe")}, {}):
            r = self.client.post("/upload", data=bad, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 415 if bad else 400)

    def test_full_redaction_flow(self):
        job_id, hdr = self.session()
        # anonymize stored the placeholder text; surrogates must not influence redaction.
        r = self.client.post("/surrogate_text", json={"text": "[PERSON] wohnt in [CITY].", "lang": "de",
                                                      "session_id": job_id, "seed": 1})
        sj = r.get_json()
        self.assertTrue(sj["success"])
        self.assertNotIn("[", sj["surrogated_text"])
        self.assertNotIn("no_entity_map", sj["warnings"])
        r = self.client.post("/get_redaction_segments", json={"job_id": job_id,
                                                              "text": sj["surrogated_text"]}, headers=hdr)
        seg = r.get_json()
        self.assertEqual(seg["used_text"], "session")
        self.assertEqual(len(seg["segments"]), 2)
        first, second = seg["segments"]
        self.assertAlmostEqual(first["start"], 0.42, delta=0.1)
        self.assertGreaterEqual(first["end"], 1.6)
        self.assertEqual(second["tag"], "[CITY]")
        self.assertEqual(first["word"], "Max Mustermann")

        wf = self.client.get("/waveform?job_id=%s&bars=100" % job_id, headers=hdr).get_json()
        self.assertEqual(len(wf["levels"]), 100)
        zoom = self.client.get("/waveform?job_id=%s&bars=60&start=1&end=2" % job_id, headers=hdr).get_json()
        self.assertEqual((zoom["start"], zoom["end"]), (1.0, 2.0))

        r = self.client.post("/generate_org_audio", json={"job_id": job_id, "segments": seg["segments"]},
                             headers=hdr)
        out = r.get_json()
        self.assertTrue(out["success"], out)
        self.assertEqual(out["segments_applied"], 2)
        audio = self.client.get(out["audio_url"])
        self.assertEqual(audio.status_code, 200)
        self.assertEqual(audio.mimetype, "audio/wav")
        samples, rate = read_wav(audio.data)
        original, _ = read_wav(slurp(self.audio))
        mid = slice(int(0.6 * rate), int(1.5 * rate))
        self.assertFalse(np.array_equal(samples[mid], original[mid]))            # masked
        quiet = slice(int(3.3 * rate), int(3.9 * rate))
        self.assertTrue(np.array_equal(samples[quiet], original[quiet]))         # untouched
        self.assertEqual(os.listdir(self.work), [])                              # no plaintext left behind
        # Tampered or expired links are refused.
        base, query = out["audio_url"].split("?")
        self.assertEqual(self.client.get(base + "?" + query[:-2] + "00").status_code, 403)
        self.assertEqual(self.client.get(base).status_code, 403)

    def test_job_data_encrypted_at_rest(self):
        body = self.upload().get_json()
        blob = b"".join(slurp(os.path.join(r, f))
                        for r, _, fs in os.walk(os.environ["ATA_JOBS_DIR"]) for f in fs)
        self.assertNotIn(b"Mustermann", blob)
        self.assertNotIn(body["job_token"].encode(), blob)
        self.assertNotIn(slurp(self.audio)[100:400], blob)

    def test_authentication_failures(self):
        body = self.upload().get_json()
        job_id, token = body["job_id"], body["job_token"]
        payload = {"job_id": job_id, "segments": [{"start": 0, "end": 1}]}
        for headers in ({}, {"X-Job-Token": "wrong"}, {"X-Job-Token": token[::-1]}):
            r = self.client.post("/generate_org_audio", json=payload, headers=headers)
            self.assertEqual(r.status_code, 404)
            self.assertEqual(r.get_json()["code"], "not_found")
        for bad_id in ("../../etc/passwd", "A" * 32, "x"):
            r = self.client.post("/get_redaction_segments", json={"job_id": bad_id},
                                 headers={"X-Job-Token": token})
            self.assertEqual(r.status_code, 404)
        r = self.client.post("/generate_org_audio", json={"segments": [{"start": 0, "end": 1}],
                                                          "audioName": "/etc/passwd"})
        self.assertEqual((r.status_code, r.get_json()["code"]), (400, "job_required"))

    def test_input_validation(self):
        job_id, hdr = self.session()
        post = lambda payload: self.client.post("/generate_org_audio", json=payload, headers=hdr)  # noqa: E731
        self.assertEqual(post({"job_id": job_id, "segments": "x"}).status_code, 400)
        self.assertEqual(post({"job_id": job_id, "segments": [{"start": "nan", "end": 1}]}).status_code, 400)
        self.assertEqual(post({"job_id": job_id, "segments": []}).get_json()["code"], "no_segments")
        self.assertEqual(post({"job_id": job_id, "segments": [{"start": 0, "end": 1}],
                               "format": "exe"}).status_code, 400)
        r = self.client.post("/anonymize", data="not json", content_type="application/json")
        self.assertEqual(r.get_json()["code"], "invalid_json")
        r = self.client.post("/anonymize", json={"transcript": TEXT, "include_tags": ["bad tag!"]})
        self.assertEqual(r.get_json()["code"], "invalid_tags")
        big = {"transcript": "x" * 30000}
        self.assertEqual(self.client.post("/anonymize", json=big).status_code, 413)

    def test_anonymize_passes_tag_filters_and_session(self):
        r = self.client.post("/anonymize", json={"transcript": TEXT, "include_tags": ["PERSON"],
                                                 "exclude_tags": ["ORG"]})
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(body["session_id"]), 32)
        self.assertEqual(self.mod._ANONYMIZER.calls[-1], (["PERSON"], ["ORG"]))
        r = self.client.post("/surrogate_text", json={"text": "[PERSON]", "session_id": "0" * 32})
        self.assertIn("no_entity_map", r.get_json()["warnings"])

    def test_delete_job_destroys_data(self):
        job_id, hdr = self.session()
        self.assertEqual(self.client.delete("/job/" + job_id).status_code, 404)        # needs token
        self.assertEqual(self.client.delete("/job/" + job_id, headers=hdr).status_code, 200)
        r = self.client.post("/get_redaction_segments", json={"job_id": job_id}, headers=hdr)
        self.assertEqual(r.status_code, 404)
        self.assertFalse(os.path.exists(os.path.join(os.environ["ATA_JOBS_DIR"], job_id)))

    def test_cors(self):
        ok = self.client.options("/upload", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(ok.status_code, 204)
        self.assertEqual(ok.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
        self.assertIn("X-Job-Token", ok.headers["Access-Control-Allow-Headers"])
        bad = self.client.get("/health", headers={"Origin": "http://evil.example"})
        self.assertNotIn("Access-Control-Allow-Origin", bad.headers)

    def test_disabled_features_and_debug(self):
        self.assertEqual(self.client.post("/upload_bulk").get_json()["code"], "feature_disabled")
        self.assertEqual(self.client.get("/list_batches").status_code, 404)
        self.assertEqual(self.client.get("/download_text/original/x.txt").status_code, 404)
        self.assertEqual(self.client.get("/debug/sessions").status_code, 404)

    def test_errors_are_json_and_do_not_leak(self):
        r = self.client.get("/definitely-not-a-route")
        self.assertEqual(r.status_code, 404)
        self.assertFalse(r.get_json()["success"])

    def test_zip_extraction_limits(self):
        import zipfile
        zpath = os.path.join(self.tmp, "bomb.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("big.wav", b"\0" * (5 * 1024 * 1024))
        dest = os.path.join(self.tmp, "ex")
        os.makedirs(dest)
        with self.assertRaises(self.mod.ApiError) as ctx:
            self.mod._safe_extract_zip(zpath, dest)
        self.assertEqual(ctx.exception.code, "zip_rejected")
        nested = os.path.join(self.tmp, "nested.zip")
        with zipfile.ZipFile(nested, "w") as zf:
            zf.writestr("inner.zip", b"PK")
        with self.assertRaises(self.mod.ApiError):
            self.mod._safe_extract_zip(nested, dest)
        good = os.path.join(self.tmp, "good.zip")
        with zipfile.ZipFile(good, "w") as zf:
            zf.write(self.audio, "../../evil/name.wav")
        files = self.mod._safe_extract_zip(good, dest)
        self.assertEqual(len(files), 1)
        self.assertEqual(os.path.dirname(files[0]["filepath"]), dest)          # no path escape

    def test_tts_routes_still_work_and_receive_speaker_labels(self):
        self.assertEqual(self.client.get("/tts/status").get_json()["backend"], "piper")
        self.assertTrue(self.client.get("/available_voices").get_json()["voices"])
        text = "SPEAKER_00: Anna wohnt in Hamburg.\nSPEAKER_01: Ja."
        r = self.client.post("/surrogate_text", json={"text": "SPEAKER_00: [PERSON] wohnt in [CITY].\n"
                                                              "SPEAKER_01: Ja.", "lang": "de"})
        fake_text = r.get_json()["surrogated_text"]
        r = self.client.post("/generate_speech", json={"text": fake_text, "lang": "de"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "audio/wav")
        sent = self.tts_seen["text"].splitlines()
        self.assertTrue(sent[0].startswith("SPEAKER_00:") and sent[1].startswith("SPEAKER_01:"))
        del text

    def test_legacy_stateless_segments_endpoint(self):
        r = self.client.post("/get_redaction_segments", json={"text": "[PERSON] lives in Berlin.",
                                                              "offSet": SEGMENTS})
        self.assertEqual(len(r.get_json()["segments"]), 1)
        self.assertEqual(self.mod.diff_relevant_offsets(SEGMENTS, "Max Mustermann lives in [CITY].")[0]["word"],
                         ["Berlin."])

    def test_recording_endpoint(self):
        webm = os.path.join(self.tmp, "rec.webm")
        proc = subprocess.run([au.FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                               "sine=frequency=300:duration=2", "-c:a", "libopus", webm])
        if proc.returncode != 0:
            self.skipTest("libopus unavailable")
        r = self.client.post("/transcribe_recording", data={"audio": (io.BytesIO(slurp(webm)), "rec.webm")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200, r.get_json())
        self.assertEqual(r.get_json()["audio_name"], "recording.wav")
        self.assertEqual(os.listdir(self.work), [])
        bad = self.client.post("/transcribe_recording", data={"audio": (io.BytesIO(b"junk" * 500), "r.webm")},
                               content_type="multipart/form-data")
        self.assertEqual(bad.get_json()["code"], "unreadable_media")
        self.assertEqual(os.listdir(self.work), [])


if __name__ == "__main__":
    unittest.main()
