"""Tests for pipeline.secure_store and pipeline.jobs."""
import base64
import io
import os
import shutil
import stat
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import jobs, secure_store as ss  # noqa: E402

KEY = os.urandom(32)


def slurp(path):
    """Reads a whole file, closing the handle."""
    with open(path, "rb") as fh:
        return fh.read()


def spit(path, data):
    """Writes a whole file, closing the handle."""
    with open(path, "wb") as fh:
        fh.write(data)


def enc(data, chunk=16, ctx=b"", key=KEY):
    out = io.BytesIO()
    ss.encrypt_stream(io.BytesIO(data), out, key, ctx, chunk_size=chunk)
    return out.getvalue()


def dec(blob, ctx=b"", key=KEY):
    return b"".join(ss.decrypt_iter(io.BytesIO(blob), key, ctx))


class Encryption(unittest.TestCase):
    def test_roundtrip_various_sizes(self):
        for size in (0, 1, 15, 16, 17, 32, 33, 1000):
            data = os.urandom(size)
            self.assertEqual(dec(enc(data)), data, size)

    def test_hides_plaintext_and_is_randomised(self):
        data = b"Max Mustermann " * 20
        a, b = enc(data), enc(data)
        self.assertNotIn(b"Mustermann", a)
        self.assertNotEqual(a, b)

    def test_tampering_detected(self):
        blob = bytearray(enc(b"x" * 100))
        for pos in (0, 6, 12, 20, len(blob) - 1):
            bad = bytearray(blob)
            bad[pos] ^= 1
            with self.assertRaises(ss.DecryptionError):
                dec(bytes(bad))

    def test_truncation_reordering_duplication_detected(self):
        blob = enc(b"y" * 64, chunk=16)                # 4 full chunks; the 4th is flagged "last"
        frame, header, body = 16 + ss.TAG_LEN, blob[:ss.HEADER_LEN], blob[ss.HEADER_LEN:]
        for keep in range(1, 4):
            with self.assertRaises(ss.DecryptionError):
                dec(blob[:ss.HEADER_LEN + keep * frame])
        for cut in (ss.HEADER_LEN, len(blob) - 1):
            with self.assertRaises(ss.DecryptionError):
                dec(blob[:cut])
        chunks = [body[i:i + frame] for i in range(0, len(body), frame)]
        with self.assertRaises(ss.DecryptionError):
            dec(header + chunks[1] + chunks[0] + b"".join(chunks[2:]))
        with self.assertRaises(ss.DecryptionError):
            dec(header + chunks[0] + chunks[0] + b"".join(chunks[1:]))

    def test_wrong_key_context_and_format(self):
        blob = enc(b"secret", ctx=b"job1:audio")
        with self.assertRaises(ss.DecryptionError):
            dec(blob, ctx=b"job1:audio", key=os.urandom(32))
        with self.assertRaises(ss.DecryptionError):
            dec(blob, ctx=b"job2:audio")
        with self.assertRaises(ss.DecryptionError):
            dec(b"hello world, definitely not encrypted")
        self.assertEqual(dec(blob, ctx=b"job1:audio"), b"secret")

    def test_file_helpers(self):
        with tempfile.TemporaryDirectory() as d:
            src, e, out = (os.path.join(d, n) for n in ("a", "a.enc", "a.out"))
            spit(src, os.urandom(3 * ss.DEFAULT_CHUNK + 5))
            self.assertEqual(ss.encrypt_file(src, e, KEY, b"c"), 3 * ss.DEFAULT_CHUNK + 5)
            self.assertEqual(stat.S_IMODE(os.stat(e).st_mode), 0o600)
            ss.decrypt_file(e, out, KEY, b"c")
            self.assertEqual(slurp(src), slurp(out))
            with self.assertRaises(ss.DecryptionError):
                ss.decrypt_file(e, os.path.join(d, "bad"), KEY, b"other")
            self.assertFalse(os.path.exists(os.path.join(d, "bad")))          # partial output removed
            p = ss.ensure_private_dir(os.path.join(d, "x", "y"))
            self.assertEqual(stat.S_IMODE(os.stat(p).st_mode), 0o700)
            f = os.path.join(p, "f")
            ss.write_private_file(f, b"data")
            self.assertEqual(stat.S_IMODE(os.stat(f).st_mode), 0o600)
            ss.secure_delete(f)
            self.assertFalse(os.path.exists(f))
            with self.assertRaises(ss.StorageError):
                ss.ensure_free_space(d, 10 ** 18)


class MasterKeys(unittest.TestCase):
    def setUp(self):
        self.saved = dict(os.environ)
        for name in ("ATA_MASTER_KEY_FILE", "ATA_MASTER_KEY", "ATA_ALLOW_ENV_KEY", "ATA_REQUIRE_MASTER_KEY"):
            os.environ.pop(name, None)
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["ATA_PROJECT_ROOT"] = self.tmp.name              # no secrets/ dir there
        self.addCleanup(self.restore)

    def restore(self):
        os.environ.clear()
        os.environ.update(self.saved)
        self.tmp.cleanup()

    def keyfile(self, text):
        path = os.path.join(self.tmp.name, "k")
        with open(path, "w") as fh:
            fh.write(text)
        os.chmod(path, 0o600)
        os.environ["ATA_MASTER_KEY_FILE"] = path

    def test_key_formats_and_derivation(self):
        self.keyfile(base64.b64encode(os.urandom(32)).decode())      # setup-secrets.sh format
        a, b = ss.load_master_key(), ss.load_master_key()
        self.assertEqual((a.key, a.ephemeral), (b.key, False))
        self.assertNotEqual(a.derive("x"), a.derive("y"))
        self.keyfile(os.urandom(32).hex())
        self.assertEqual(len(ss.load_master_key().key), 32)
        self.keyfile("a-very-long-passphrase-with-more-than-32-chars!!")
        self.assertEqual(len(ss.load_master_key().key), 32)
        self.keyfile("short")
        with self.assertRaises(ss.MasterKeyError):
            ss.load_master_key()

    def test_env_key_ignored_unless_allowed(self):
        os.environ["ATA_MASTER_KEY"] = "x" * 40
        self.assertTrue(ss.load_master_key().ephemeral)
        os.environ["ATA_ALLOW_ENV_KEY"] = "1"
        self.assertEqual(ss.load_master_key().source, "environment")

    def test_required_vs_ephemeral(self):
        with self.assertRaises(ss.MasterKeyError):
            ss.load_master_key(required=True)
        os.environ["ATA_REQUIRE_MASTER_KEY"] = "1"
        with self.assertRaises(ss.MasterKeyError):
            ss.load_master_key()
        del os.environ["ATA_REQUIRE_MASTER_KEY"]
        a, b = ss.load_master_key(), ss.load_master_key()
        self.assertTrue(a.ephemeral)
        self.assertNotEqual(a.key, b.key)

    def test_key_wrapping(self):
        m = ss.MasterKey(os.urandom(32), "t")
        dk = os.urandom(32)
        blob = ss.wrap_key(m, dk, b"job")
        self.assertEqual(ss.unwrap_key(m, blob, b"job"), dk)
        with self.assertRaises(ss.DecryptionError):
            ss.unwrap_key(m, blob, b"other")
        with self.assertRaises(ss.DecryptionError):
            ss.unwrap_key(ss.MasterKey(os.urandom(32), "t"), blob, b"job")


SEGMENTS = [{"speaker": "SPEAKER_00", "start": 0, "end": 1,
             "words": [{"word": "Max", "start": 0, "end": 0.5}]}]


class JobStore(unittest.TestCase):
    def setUp(self):
        os.environ["ATA_WORK_DIR"] = tempfile.mkdtemp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.master = ss.MasterKey(os.urandom(32), "test")
        self.store = jobs.JobStore(os.path.join(self.tmp.name, "jobs"), self.master,
                                   idle_ttl=100, max_age=1000, memory_ttl=100)
        self.audio = os.path.join(self.tmp.name, "a.wav")
        spit(self.audio, b"RIFF....SECRETAUDIO" * 100)

    def make(self, **kw):
        return self.store.create(self.audio, SEGMENTS, 12.5, **kw)

    def walk_blob(self, root):
        return b"".join(slurp(os.path.join(r, f)) for r, _, fs in os.walk(root) for f in fs)

    def test_create_authenticate_load_and_scratch_cleanup(self):
        job, token = self.make()
        again = self.store.authenticate(job.id, token)
        self.assertEqual(again.duration, 12.5)
        self.assertEqual(self.store.load_segments(again), SEGMENTS)
        with self.store.open_audio(again) as path:
            self.assertEqual(slurp(path), slurp(self.audio))
        self.assertFalse(os.path.exists(path))

    def test_nothing_sensitive_in_plaintext_and_permissions(self):
        job, token = self.make()
        blob = self.walk_blob(job.path)
        for needle in (b"SECRETAUDIO", b"Max", token.encode()):
            self.assertNotIn(needle, blob)
        for f in os.listdir(job.path):
            self.assertEqual(stat.S_IMODE(os.stat(job.path / f).st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(os.stat(job.path).st_mode), 0o700)

    def test_bad_ids_tokens_owner(self):
        job, token = self.make(owner="alice")
        for bad in ("../etc", "", "x" * 32, job.id.upper(), None, 5):
            with self.assertRaises(jobs.JobError) as ctx:
                self.store.authenticate(bad, token)
            self.assertEqual(ctx.exception.status, 404)
        for bad in (None, "", "nope", token[:-1]):
            with self.assertRaises(jobs.JobError):
                self.store.authenticate(job.id, bad, owner="alice")
        for owner in ("mallory", None):
            with self.assertRaises(jobs.JobError):
                self.store.authenticate(job.id, token, owner=owner)
        self.store.authenticate(job.id, token, owner="alice")

    def test_expiry_idle_and_absolute(self):
        job, token = self.make()
        self.store._write_meta(job.id, dict(job.meta, last_access=time.time() - 500))
        with self.assertRaises(jobs.JobError) as ctx:
            self.store.authenticate(job.id, token)
        self.assertEqual(ctx.exception.status, 410)
        self.assertFalse(job.path.exists())
        job, token = self.make()
        self.store._write_meta(job.id, dict(job.meta, created=time.time() - 5000))
        with self.assertRaises(jobs.JobError):
            self.store.authenticate(job.id, token)

    def test_delete_is_crypto_shredding(self):
        job, _ = self.make()
        backup = os.path.join(self.tmp.name, "backup")
        shutil.copytree(job.path, backup)                  # attacker's earlier snapshot
        self.assertTrue(self.store.delete(job.id))
        self.assertFalse(job.path.exists())
        os.unlink(os.path.join(backup, "key.bin"))         # the data key existed only there
        with open(os.path.join(backup, "audio.enc"), "rb") as fh:
            with self.assertRaises(ss.DecryptionError):
                b"".join(ss.decrypt_iter(fh, os.urandom(32), b"ATA:%s:audio" % job.id.encode()))

    def test_files_bound_to_job_and_master_key(self):
        a, _ = self.make()
        b, tb = self.make()
        shutil.copy(a.path / "audio.enc", b.path / "audio.enc")             # swap attack
        with self.assertRaises(jobs.JobError):
            with self.store.open_audio(self.store.authenticate(b.id, tb)):
                pass
        other = jobs.JobStore(self.store.root, ss.MasterKey(os.urandom(32), "x"))
        with self.assertRaises(jobs.JobError) as ctx:
            other.load_segments(other.authenticate(b.id, tb))
        self.assertEqual(ctx.exception.status, 410)

    def test_startup_sweep_and_purge(self):
        keep, _ = self.make()
        old, _ = self.make()
        self.store._write_meta(old.id, dict(old.meta, last_access=time.time() - 500))
        orphan = self.store.root / ("a" * 32)
        orphan.mkdir()
        os.utime(orphan, (time.time() - 3600, time.time() - 3600))
        (self.store.root / "not-a-job").mkdir()
        self.assertEqual(self.store.purge_expired(), 2)
        self.assertTrue(keep.path.exists())
        self.assertTrue((self.store.root / "not-a-job").exists())
        fresh = jobs.JobStore(self.store.root, ss.MasterKey(os.urandom(32), "ephemeral", True))
        self.assertEqual(fresh.startup_sweep(), 1)
        self.assertFalse(keep.path.exists())

    def test_redacted_audio_roundtrip(self):
        job, token = self.make()
        out = os.path.join(self.tmp.name, "r.wav")
        spit(out, b"REDACTED" * 50)
        with self.assertRaises(jobs.JobError):
            with self.store.open_redacted(job):
                pass
        self.store.store_redacted(job, out, "wav")
        with self.store.open_redacted(self.store.authenticate(job.id, token)) as (path, mime):
            self.assertEqual((slurp(path), mime), (b"REDACTED" * 50, "audio/wav"))
        self.assertNotIn(b"REDACTED", slurp(job.path / "redacted.enc"))

    def test_memory_is_ram_only_expiring_and_capped(self):
        self.store.remember("k", "ORIGINAL-PII-VALUE")
        self.assertEqual(self.store.recall("k"), "ORIGINAL-PII-VALUE")
        self.assertNotIn(b"ORIGINAL-PII-VALUE", self.walk_blob(self.tmp.name))
        self.store._mem["k"] = (time.time() - 500, "x")
        self.assertIsNone(self.store.recall("k"))
        small = jobs.JobStore(os.path.join(self.tmp.name, "j2"), self.master, max_memory_entries=3)
        for i in range(10):
            small.remember(str(i), i)
        self.assertEqual(small.stats()["memory_entries"], 3)

    def test_signed_urls(self):
        job, _ = self.make()
        exp, sig = self.store.sign(job.id, "redacted", ttl=60)
        self.assertTrue(self.store.verify(job.id, "redacted", exp, sig))
        for args in ((job.id, "audio", exp, sig), ("0" * 32, "redacted", exp, sig),
                     (job.id, "redacted", exp + 1, sig), (job.id, "redacted", "x", sig),
                     (job.id, "redacted", exp, None)):
            self.assertFalse(self.store.verify(*args))
        old_exp, old_sig = self.store.sign(job.id, "redacted", ttl=-5)
        self.assertFalse(self.store.verify(job.id, "redacted", old_exp, old_sig))

    def test_failed_create_leaves_nothing_and_purge_thread_starts_once(self):
        with self.assertRaises(OSError):
            self.store.create(os.path.join(self.tmp.name, "missing.wav"), SEGMENTS, 1.0)
        self.assertEqual(os.listdir(self.store.root), [])
        self.assertIs(self.store.start_purge_thread(3600), self.store.start_purge_thread(3600))
        self.store.stop()


if __name__ == "__main__":
    unittest.main()
