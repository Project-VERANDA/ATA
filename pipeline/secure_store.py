"""Encryption at rest and private-file helpers.

Goal: job data (audio, word timelines, redacted output) written to disk must not
be readable from copied disks, snapshots or backups, and deleting a job's key
makes its data unrecoverable ("crypto-shredding") - which plain file deletion
cannot guarantee on SSDs or copy-on-write file systems.

Out of scope: a compromised server process or root (they can read keys from
memory) and the plaintext that processing tools transiently need. That lives in a
private, ideally RAM-backed work directory (:func:`get_work_dir`).

File format ("ATAE" v1, chunked AES-256-GCM, STREAM construction)::

    header (16 B): b"ATAE" | version (1) | chunk_size (u32 BE) | nonce_prefix (7)
    chunk i:       AES-GCM(key, nonce = prefix | i (u32 BE) | last (1 B),
                           plaintext_chunk, aad = header | context)

The final chunk carries ``last = 1``, so decryption fails on truncation,
reordering, duplication, header tampering or a wrong ``context``.
"""
from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)

MAGIC = b"ATAE"
VERSION = 1
HEADER_LEN = 16
DEFAULT_CHUNK = 1 << 20          # 1 MiB
MAX_CHUNK = 64 << 20
TAG_LEN = 16


class DecryptionError(Exception):
    """Wrong key/context, corrupted, truncated or tampered data."""


class MasterKeyError(RuntimeError):
    """No usable master key could be loaded."""


class StorageError(Exception):
    """Storage precondition failed (e.g. not enough free space)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# --- Streaming encryption -----------------------------------------------------
def _nonce(prefix: bytes, counter: int, last: bool) -> bytes:
    return prefix + counter.to_bytes(4, "big") + (b"\x01" if last else b"\x00")


def _read_exact(reader: BinaryIO, size: int) -> bytes:
    """Reads up to ``size`` bytes, looping over short reads; b"" only at EOF."""
    parts = []
    remaining = size
    while remaining:
        piece = reader.read(remaining)
        if not piece:
            break
        parts.append(piece)
        remaining -= len(piece)
    return b"".join(parts)


def encrypt_stream(reader: BinaryIO, writer: BinaryIO, key: bytes, context: bytes = b"",
                   chunk_size: int = DEFAULT_CHUNK) -> int:
    """Encrypts ``reader`` into ``writer``; returns plaintext bytes processed."""
    if len(key) != 32:
        raise ValueError("key must be 32 bytes")
    if not 1 <= chunk_size <= MAX_CHUNK:
        raise ValueError("invalid chunk size")
    prefix = os.urandom(7)
    header = MAGIC + bytes([VERSION]) + chunk_size.to_bytes(4, "big") + prefix
    aead = AESGCM(key)
    writer.write(header)
    buf = _read_exact(reader, chunk_size)
    counter = total = 0
    while True:
        nxt = _read_exact(reader, chunk_size) if len(buf) == chunk_size else b""
        last = not nxt
        writer.write(aead.encrypt(_nonce(prefix, counter, last), buf, header + context))
        total += len(buf)
        counter += 1
        if counter >= 2 ** 32 - 1:
            raise ValueError("file too large")
        if last:
            return total
        buf = nxt


def decrypt_iter(reader: BinaryIO, key: bytes, context: bytes = b"") -> Iterator[bytes]:
    """Yields verified plaintext chunks.

    Raises:
        DecryptionError: On any integrity failure. Chunks yielded before a failure
            were individually authenticated; callers needing all-or-nothing
            behaviour should use :func:`decrypt_file`, which discards partial output.
    """
    header = _read_exact(reader, HEADER_LEN)
    if len(header) != HEADER_LEN or header[:4] != MAGIC or header[4] != VERSION:
        raise DecryptionError("not an ATA encrypted file")
    chunk_size = int.from_bytes(header[5:9], "big")
    if not 1 <= chunk_size <= MAX_CHUNK:
        raise DecryptionError("invalid header")
    prefix = header[9:16]
    aead = AESGCM(key)
    frame = chunk_size + TAG_LEN
    buf = _read_exact(reader, frame)
    if not buf:
        raise DecryptionError("truncated file")
    counter = 0
    while True:
        if len(buf) == frame:
            nxt = _read_exact(reader, frame)
            last = not nxt
        else:
            nxt, last = b"", True
        try:
            plain = aead.decrypt(_nonce(prefix, counter, last), buf, header + context)
        except InvalidTag:
            raise DecryptionError("authentication failed") from None
        yield plain
        counter += 1
        if last:
            return
        buf = nxt


def encrypt_bytes(data: bytes, key: bytes, context: bytes = b"") -> bytes:
    """Encrypts a small in-memory payload (JSON documents)."""
    import io
    out = io.BytesIO()
    encrypt_stream(io.BytesIO(data), out, key, context)
    return out.getvalue()


def decrypt_bytes(blob: bytes, key: bytes, context: bytes = b"") -> bytes:
    """Decrypts a payload produced by :func:`encrypt_bytes`."""
    import io
    return b"".join(decrypt_iter(io.BytesIO(blob), key, context))


# --- Private files ------------------------------------------------------------
def ensure_private_dir(path: os.PathLike) -> Path:
    """Creates ``path`` with mode 0700 (and tightens it if it already exists)."""
    p = Path(path)
    p.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(p, 0o700)
    except OSError:
        pass
    return p


def _open_private(path: str):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb")


def write_private_file(path: os.PathLike, data: bytes) -> None:
    """Atomically writes ``data`` to ``path`` with mode 0600."""
    tmp = "%s.%s.tmp" % (path, uuid.uuid4().hex[:8])
    try:
        with _open_private(tmp) as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def encrypt_file(src: os.PathLike, dst: os.PathLike, key: bytes, context: bytes = b"") -> int:
    """Encrypts ``src`` into ``dst`` (mode 0600, atomic). Returns plaintext size."""
    tmp = "%s.%s.tmp" % (dst, uuid.uuid4().hex[:8])
    try:
        with open(src, "rb") as reader, _open_private(tmp) as writer:
            size = encrypt_stream(reader, writer, key, context)
        os.replace(tmp, dst)
        return size
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def decrypt_file(src: os.PathLike, dst: os.PathLike, key: bytes, context: bytes = b"") -> int:
    """Decrypts ``src`` into ``dst`` (mode 0600). Partial output is removed on failure."""
    written = 0
    try:
        with open(src, "rb") as reader, _open_private(str(dst)) as writer:
            for chunk in decrypt_iter(reader, key, context):
                writer.write(chunk)
                written += len(chunk)
        return written
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(dst)
        raise


def secure_delete(path: os.PathLike) -> None:
    """Best-effort overwrite then unlink.

    Not reliable on SSDs, journaling or copy-on-write file systems; use
    crypto-shredding (delete the key) for guarantees.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "r+b") as fh:
            remaining = size
            block = b"\x00" * min(size, 1 << 20) if size else b""
            while remaining > 0:
                fh.write(block[:remaining])
                remaining -= len(block)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass
    with contextlib.suppress(OSError):
        os.unlink(path)


# --- Work directory (plaintext during processing) -----------------------------
def get_work_dir() -> Path:
    """Private scratch directory, RAM-backed (``/dev/shm``) when available.

    Override with ``ATA_WORK_DIR``. A tmpfs mount is best: plaintext then never
    reaches persistent storage.
    """
    configured = os.environ.get("ATA_WORK_DIR")
    if configured:
        return ensure_private_dir(configured)
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        return ensure_private_dir(shm / ("ata-%d" % os.getuid()))
    return ensure_private_dir(Path(tempfile.gettempdir()) / ("ata-%d" % os.getuid()))


@contextlib.contextmanager
def work_path(suffix: str = "") -> Iterator[str]:
    """Yields a unique path in the work directory and removes it afterwards."""
    path = str(get_work_dir() / (uuid.uuid4().hex + suffix))
    try:
        yield path
    finally:
        for leftover in (path, path + ".part"):
            with contextlib.suppress(OSError):
                os.unlink(leftover)


def cleanup_work_dir(max_age_s: float = 3600.0) -> int:
    """Removes stale scratch files left behind by crashes. Returns count removed."""
    removed = 0
    cutoff = time.time() - max_age_s
    for entry in get_work_dir().iterdir():
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def ensure_free_space(path: os.PathLike, needed_bytes: int, reserve: int = 512 << 20) -> None:
    """Raises :class:`StorageError` if the file system cannot hold ``needed_bytes``."""
    if shutil.disk_usage(path).free < needed_bytes + reserve:
        raise StorageError("insufficient_storage", "The server does not have enough free space.")


# --- Keys ---------------------------------------------------------------------
@dataclass
class MasterKey:
    """Process-wide root key.

    Attributes:
        key: 32 bytes of key material.
        source: Where it came from (file path or ``"ephemeral"``); never the value.
        ephemeral: True if generated at start-up. Jobs then cannot survive a restart,
            the safest default for single-user installs.
    """

    key: bytes
    source: str
    ephemeral: bool = False

    def derive(self, purpose: str) -> bytes:
        """Derives an independent 32-byte sub-key for ``purpose``."""
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                    info=b"ata-sub-" + purpose.encode()).derive(self.key)


def _decode_key_text(raw: str) -> bytes:
    text = raw.strip()
    try:
        decoded = base64.b64decode(text, validate=True)
        if len(decoded) >= 32:
            return decoded
    except (binascii.Error, ValueError):
        pass
    if re.fullmatch(r"[0-9a-fA-F]{64,}", text):
        return bytes.fromhex(text)
    return text.encode("utf-8")


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _candidate_key_files() -> list:
    root = Path(os.environ.get("ATA_PROJECT_ROOT") or Path(__file__).resolve().parents[1])
    paths = [os.environ.get("ATA_MASTER_KEY_FILE"), "/run/secrets/ata_master_key",
             "/run/secrets/encryption_key", str(root / "secrets" / "encryption_key.txt")]
    return [p for p in paths if p]


def load_master_key(required: Optional[bool] = None) -> MasterKey:
    """Loads the master key (never from a plain environment variable by default).

    Search order: ``ATA_MASTER_KEY_FILE``, Docker secrets ``ata_master_key`` and
    ``encryption_key`` (as created by ``secrets/setup-secrets.sh``), then
    ``secrets/encryption_key.txt``. ``ATA_MASTER_KEY`` is honoured only with
    ``ATA_ALLOW_ENV_KEY=1``: environment variables leak via process listings and
    ``docker inspect``.

    Args:
        required: Refuse to continue without a persistent key. Defaults to the
            ``ATA_REQUIRE_MASTER_KEY`` environment variable.

    Raises:
        MasterKeyError: Key required but not found, or too short.
    """
    if required is None:
        required = _truthy(os.environ.get("ATA_REQUIRE_MASTER_KEY"))
    material: Optional[bytes] = None
    source = ""
    for candidate in _candidate_key_files():
        try:
            mode = os.stat(candidate).st_mode & 0o077
            with open(candidate, "r", encoding="utf-8") as fh:
                material = _decode_key_text(fh.read())
            source = candidate
            if mode and not candidate.startswith("/run/secrets/"):
                logger.warning("Key file %s is readable by other users; run chmod 600.", candidate)
            break
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError as exc:
            raise MasterKeyError("Cannot read key file %s: %s" % (candidate, exc.strerror)) from None
    if material is None and _truthy(os.environ.get("ATA_ALLOW_ENV_KEY")) \
            and os.environ.get("ATA_MASTER_KEY"):
        material = _decode_key_text(os.environ["ATA_MASTER_KEY"])
        source = "environment"
        logger.warning("Master key taken from the environment; prefer a Docker secret file.")
    if material is None:
        if required:
            raise MasterKeyError("No master key found. Run secrets/setup-secrets.sh or set "
                                 "ATA_MASTER_KEY_FILE.")
        logger.warning("No master key configured: using an ephemeral in-memory key. Stored jobs "
                       "will be unreadable after a restart (safe default for single users).")
        return MasterKey(os.urandom(32), "ephemeral", True)
    if len(material) < 32:
        raise MasterKeyError("Master key must provide at least 32 bytes of key material.")
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"ata-master-v1").derive(material)
    return MasterKey(key, source, False)


def wrap_key(master: MasterKey, data_key: bytes, context: bytes) -> bytes:
    """Encrypts a per-job data key under the master key (``nonce | ciphertext``)."""
    nonce = os.urandom(12)
    return nonce + AESGCM(master.key).encrypt(nonce, data_key, b"ATA-DEK1" + context)


def unwrap_key(master: MasterKey, blob: bytes, context: bytes) -> bytes:
    """Inverse of :func:`wrap_key`.

    Raises:
        DecryptionError: Wrong master key (e.g. after an ephemeral restart) or tampering.
    """
    if len(blob) < 12 + 32 + TAG_LEN:
        raise DecryptionError("invalid wrapped key")
    try:
        return AESGCM(master.key).decrypt(blob[:12], blob[12:], b"ATA-DEK1" + context)
    except InvalidTag:
        raise DecryptionError("cannot unwrap key") from None
