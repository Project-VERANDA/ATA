"""Job storage for the web interface: encrypted, token-protected and expiring.

A *job* is created when audio is uploaded and transcribed. It lets the redaction
editor refer to server-side audio without the client ever sending a file path.

On disk (``<root>/<job_id>/``)::

    meta.json     non-sensitive: timestamps, duration, SHA-256 of the access token
    key.bin       per-job data key, wrapped by the master key (delete = crypto-shred)
    audio.enc     uploaded audio, AES-256-GCM
    words.enc     word-level timeline (JSON), AES-256-GCM
    redacted.enc  rendered redacted audio, AES-256-GCM (after rendering)

Deliberately **not** stored: the uploaded file name, the transcript text, the
anonymized text and the entity map (original PII). The latter two live in a
RAM-only, time-limited memory (:meth:`JobStore.remember`).

Lifecycle: a job expires after ``idle_ttl`` without access or ``max_age`` since
creation, whichever comes first, or immediately via :meth:`JobStore.delete`.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .secure_store import (DecryptionError, MasterKey, cleanup_work_dir, decrypt_bytes,
                           decrypt_file, encrypt_bytes, encrypt_file, ensure_private_dir,
                           unwrap_key, work_path, wrap_key, write_private_file)

logger = logging.getLogger(__name__)

ID_RE = re.compile(r"[0-9a-f]{32}")
TOUCH_INTERVAL_S = 30.0
MIME_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}


class JobError(Exception):
    """Job access problem with an HTTP-style status."""

    def __init__(self, code: str, message: str, status: int = 404):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class Job:
    """Handle for an authenticated job."""

    id: str
    path: Path
    meta: Dict[str, Any]

    @property
    def duration(self) -> float:
        return float(self.meta.get("duration") or 0.0)


def _ctx(job_id: str, name: str) -> bytes:
    """Binds ciphertext to its job and role so files cannot be swapped between jobs."""
    return ("ATA:%s:%s" % (job_id, name)).encode()


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class JobStore:
    """Encrypted, expiring job store (thread-safe; multi-process safe on disk).

    Args:
        root: Directory holding job folders (created with mode 0700).
        master: Master key used to wrap per-job data keys.
        idle_ttl: Seconds without access before a job expires.
        max_age: Absolute lifetime in seconds.
        memory_ttl: Idle lifetime of RAM-only entries.
        max_memory_entries: Upper bound for RAM-only entries (oldest evicted).
    """

    def __init__(self, root: os.PathLike, master: MasterKey, idle_ttl: float = 3600,
                 max_age: float = 86400, memory_ttl: float = 3600, max_memory_entries: int = 2000):
        self.root = ensure_private_dir(root)
        self.master = master
        self.idle_ttl = idle_ttl
        self.max_age = max_age
        self.memory_ttl = memory_ttl
        self.max_memory_entries = max_memory_entries
        self._mem: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sign_key = master.derive("url-signing")

    @classmethod
    def from_env(cls, root: os.PathLike, master: MasterKey) -> "JobStore":
        """Builds a store using ``ATA_JOB_IDLE_TTL``, ``ATA_JOB_MAX_AGE``, ``ATA_MEMORY_TTL``."""
        return cls(root, master, _env_float("ATA_JOB_IDLE_TTL", 3600),
                   _env_float("ATA_JOB_MAX_AGE", 86400), _env_float("ATA_MEMORY_TTL", 3600))

    # -- internals -------------------------------------------------------------
    def _dir(self, job_id: str) -> Path:
        if not isinstance(job_id, str) or not ID_RE.fullmatch(job_id):
            raise JobError("not_found", "Job not found.")
        return self.root / job_id

    def _read_meta(self, job_id: str) -> Optional[Dict[str, Any]]:
        try:
            with open(self._dir(job_id) / "meta.json", "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else None
        except (OSError, ValueError, JobError):
            return None

    def _write_meta(self, job_id: str, meta: Dict[str, Any]) -> None:
        write_private_file(self._dir(job_id) / "meta.json", json.dumps(meta).encode("utf-8"))

    def _expired(self, meta: Dict[str, Any], now: float) -> bool:
        return (now - float(meta.get("last_access", 0)) > self.idle_ttl
                or now - float(meta.get("created", 0)) > self.max_age)

    def _data_key(self, job: Job) -> bytes:
        try:
            blob = (job.path / "key.bin").read_bytes()
            return unwrap_key(self.master, blob, _ctx(job.id, "key"))
        except (OSError, DecryptionError):
            raise JobError("job_unreadable", "This job can no longer be read.", 410) from None

    # -- lifecycle -------------------------------------------------------------
    def create(self, audio_path: os.PathLike, segments: List[Dict[str, Any]], duration: float,
               owner: Optional[str] = None) -> Tuple[Job, str]:
        """Stores a new job.

        Args:
            audio_path: Plaintext audio (the caller deletes its copy afterwards).
            segments: WhisperX-style segments including ``words``.
            duration: Audio length in seconds.
            owner: Optional opaque user id (e.g. Keycloak ``sub``) for later binding.

        Returns:
            ``(job, token)``. The token is shown once; only its hash is stored.
        """
        job_id = uuid.uuid4().hex
        path = ensure_private_dir(self.root / job_id)
        token = os.urandom(32).hex()
        try:
            data_key = os.urandom(32)
            write_private_file(path / "key.bin", wrap_key(self.master, data_key, _ctx(job_id, "key")))
            encrypt_file(audio_path, path / "audio.enc", data_key, _ctx(job_id, "audio"))
            write_private_file(path / "words.enc", encrypt_bytes(
                json.dumps(segments, ensure_ascii=False).encode("utf-8"), data_key,
                _ctx(job_id, "words")))
            now = time.time()
            meta = {"v": 1, "id": job_id, "created": now, "last_access": now,
                    "duration": round(float(duration), 3),
                    "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "owner_sha256": hashlib.sha256(owner.encode()).hexdigest() if owner else None,
                    "redacted_format": None}
            self._write_meta(job_id, meta)
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise
        return Job(job_id, path, meta), token

    def authenticate(self, job_id: str, token: Optional[str], owner: Optional[str] = None) -> Job:
        """Returns the job if ``job_id`` and ``token`` match and it has not expired.

        Wrong id, wrong token and wrong owner are indistinguishable (404) so ids
        cannot be probed. Expired jobs are deleted and reported as 410.
        """
        path = self._dir(job_id)
        meta = self._read_meta(job_id)
        if meta is None or not isinstance(token, str) or not token:
            raise JobError("not_found", "Job not found.")
        digest = hashlib.sha256(token.encode()).hexdigest()
        if not hmac.compare_digest(digest, str(meta.get("token_sha256", ""))):
            raise JobError("not_found", "Job not found.")
        if meta.get("owner_sha256"):
            given = hashlib.sha256(owner.encode()).hexdigest() if owner else ""
            if not hmac.compare_digest(given, meta["owner_sha256"]):
                raise JobError("not_found", "Job not found.")
        return self._activate(job_id, path, meta)

    def get_by_id(self, job_id: str) -> Job:
        """Loads a job *without* a token. Only call after verifying a signed URL."""
        path = self._dir(job_id)
        meta = self._read_meta(job_id)
        if meta is None:
            raise JobError("not_found", "Job not found.")
        return self._activate(job_id, path, meta)

    def _activate(self, job_id: str, path: Path, meta: Dict[str, Any]) -> Job:
        now = time.time()
        if self._expired(meta, now):
            self.delete(job_id)
            raise JobError("job_expired", "This job has expired and was deleted.", 410)
        if now - float(meta.get("last_access", 0)) > TOUCH_INTERVAL_S:
            meta["last_access"] = now
            with contextlib.suppress(OSError):
                self._write_meta(job_id, meta)
        return Job(job_id, path, meta)

    def delete(self, job_id: str) -> bool:
        """Deletes a job. The key goes first, so the data is unrecoverable at once."""
        path = self._dir(job_id)
        existed = path.exists()
        with contextlib.suppress(OSError):
            (path / "key.bin").unlink()
        shutil.rmtree(path, ignore_errors=True)
        self.forget(job_id)
        return existed

    def purge_expired(self, now: Optional[float] = None) -> int:
        """Deletes expired and orphaned jobs and stale RAM entries. Returns jobs removed."""
        now = time.time() if now is None else now
        removed = 0
        for entry in os.scandir(self.root):
            if not entry.is_dir(follow_symlinks=False) or not ID_RE.fullmatch(entry.name):
                continue
            meta = self._read_meta(entry.name)
            try:
                orphan = meta is None and now - entry.stat().st_mtime > 600
            except OSError:
                continue
            if orphan or (meta is not None and self._expired(meta, now)):
                self.delete(entry.name)
                removed += 1
        self._purge_memory(now)
        return removed

    def startup_sweep(self) -> int:
        """Run once at start-up. With an ephemeral key old jobs are unreadable: remove all."""
        if self.master.ephemeral:
            count = 0
            for entry in os.scandir(self.root):
                if entry.is_dir(follow_symlinks=False) and ID_RE.fullmatch(entry.name):
                    self.delete(entry.name)
                    count += 1
            return count
        return self.purge_expired()

    # -- data access -----------------------------------------------------------
    def load_segments(self, job: Job) -> List[Dict[str, Any]]:
        """Decrypts and returns the job's WhisperX segments."""
        key = self._data_key(job)
        try:
            blob = (job.path / "words.enc").read_bytes()
            return json.loads(decrypt_bytes(blob, key, _ctx(job.id, "words")).decode("utf-8"))
        except (OSError, DecryptionError, ValueError):
            raise JobError("job_unreadable", "This job can no longer be read.", 410) from None

    @contextlib.contextmanager
    def open_audio(self, job: Job) -> Iterator[str]:
        """Yields a path to a decrypted *temporary* copy of the audio (removed on exit)."""
        key = self._data_key(job)
        with work_path(".media") as tmp:
            try:
                decrypt_file(job.path / "audio.enc", tmp, key, _ctx(job.id, "audio"))
            except (OSError, DecryptionError):
                raise JobError("job_unreadable", "This job can no longer be read.", 410) from None
            yield tmp

    def store_redacted(self, job: Job, plain_path: os.PathLike, fmt: str) -> None:
        """Encrypts a rendered file into the job and records its format."""
        key = self._data_key(job)
        encrypt_file(plain_path, job.path / "redacted.enc", key, _ctx(job.id, "redacted"))
        meta = dict(job.meta, redacted_format=fmt, redacted_at=time.time())
        self._write_meta(job.id, meta)
        job.meta.update(meta)

    @contextlib.contextmanager
    def open_redacted(self, job: Job) -> Iterator[Tuple[str, str]]:
        """Yields ``(decrypted_temp_path, mimetype)`` for the rendered output."""
        fmt = job.meta.get("redacted_format")
        if not fmt or not (job.path / "redacted.enc").exists():
            raise JobError("no_redacted_audio", "No redacted audio has been generated yet.", 404)
        key = self._data_key(job)
        with work_path("." + fmt) as tmp:
            try:
                decrypt_file(job.path / "redacted.enc", tmp, key, _ctx(job.id, "redacted"))
            except (OSError, DecryptionError):
                raise JobError("job_unreadable", "This job can no longer be read.", 410) from None
            yield tmp, MIME_TYPES.get(fmt, "application/octet-stream")

    # -- RAM-only memory (entity maps, placeholder text) -----------------------
    def remember(self, key: str, value: Any) -> None:
        """Stores ``value`` in memory only (never on disk), with idle expiry."""
        with self._lock:
            self._mem[key] = (time.time(), value)
            while len(self._mem) > self.max_memory_entries:
                oldest = min(self._mem, key=lambda k: self._mem[k][0])
                del self._mem[oldest]

    def recall(self, key: str) -> Optional[Any]:
        """Returns the stored value (refreshing its lifetime) or None if absent/expired."""
        with self._lock:
            item = self._mem.get(key)
            if item is None:
                return None
            if time.time() - item[0] > self.memory_ttl:
                del self._mem[key]
                return None
            self._mem[key] = (time.time(), item[1])
            return item[1]

    def forget(self, key: str) -> None:
        """Drops a memory entry."""
        with self._lock:
            self._mem.pop(key, None)

    def _purge_memory(self, now: float) -> None:
        with self._lock:
            for key in [k for k, (ts, _) in self._mem.items() if now - ts > self.memory_ttl]:
                del self._mem[key]

    # -- signed download links -------------------------------------------------
    def sign(self, job_id: str, purpose: str, ttl: int = 300) -> Tuple[int, str]:
        """Creates ``(expires_at, signature)`` for a short-lived download link."""
        exp = int(time.time()) + int(ttl)
        return exp, self._mac(job_id, purpose, exp)

    def verify(self, job_id: str, purpose: str, exp: Any, sig: Any) -> bool:
        """Constant-time check of a signature and its expiry."""
        try:
            exp_i = int(exp)
        except (TypeError, ValueError):
            return False
        if exp_i < time.time() or not isinstance(sig, str):
            return False
        return hmac.compare_digest(self._mac(job_id, purpose, exp_i), sig)

    def _mac(self, job_id: str, purpose: str, exp: int) -> str:
        message = ("%s:%s:%d" % (job_id, purpose, exp)).encode()
        return hmac.new(self._sign_key, message, hashlib.sha256).hexdigest()

    # -- housekeeping ----------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        """Counts only (no identifiers) for a protected health endpoint."""
        jobs = sum(1 for e in os.scandir(self.root) if ID_RE.fullmatch(e.name))
        with self._lock:
            return {"jobs": jobs, "memory_entries": len(self._mem)}

    def start_purge_thread(self, interval: float = 300.0) -> threading.Thread:
        """Starts the background sweeper once (also removes stale scratch files)."""
        if self._thread and self._thread.is_alive():
            return self._thread

        def loop() -> None:
            while not self._stop.wait(interval):
                try:
                    removed = self.purge_expired()
                    cleanup_work_dir()
                    if removed:
                        logger.info("Purged %d expired job(s)", removed)
                except Exception:                         # keep the sweeper alive
                    logger.exception("Job purge failed")

        self._thread = threading.Thread(target=loop, name="ata-job-purge", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        """Stops the sweeper (tests, graceful shutdown)."""
        self._stop.set()
