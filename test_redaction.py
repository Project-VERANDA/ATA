"""Unit tests for pipeline.redaction (no ML or ffmpeg dependencies)."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from pipeline import redaction as r  # noqa: E402


def seg(words, speaker="SPEAKER_00"):
    """Builds a WhisperX-like segment from [(text, start, end), ...]."""
    return {"speaker": speaker, "start": words[0][1], "end": words[-1][2],
            "text": " ".join(w[0] for w in words),
            "words": [{"word": t, "start": s, "end": e} for t, s, e in words]}


SAMPLE = [seg([("Max", 0.5, 0.9), ("Mustermann", 1.0, 1.6), ("lives", 1.8, 2.1),
               ("in", 2.2, 2.3), ("Berlin.", 2.4, 3.0)])]


class WordTimeline(unittest.TestCase):
    def test_flatten(self):
        words, warn = r.words_from_segments(SAMPLE)
        self.assertEqual([w.text for w in words], ["Max", "Mustermann", "lives", "in", "Berlin."])
        self.assertEqual(warn, [])

    def test_interpolates_missing_timestamps(self):
        s = {"start": 0.0, "end": 4.0, "words": [
            {"word": "I", "start": 0.0, "end": 0.5}, {"word": "am"}, {"word": "42"},
            {"word": "years", "start": 3.0, "end": 3.5}]}
        words, _ = r.words_from_segments([s])
        self.assertAlmostEqual(words[1].start, 0.5)
        self.assertAlmostEqual(words[1].end, 1.75)
        self.assertAlmostEqual(words[2].end, 3.0)

    def test_segment_without_words_falls_back(self):
        words, warn = r.words_from_segments([{"start": 0, "end": 4, "text": "a b c d"}])
        self.assertEqual(len(words), 4)
        self.assertIn("segment_without_word_timestamps", warn)

    def test_ignores_non_numeric_times(self):
        words, _ = r.words_from_segments([{"start": 0, "end": 2, "words": [
            {"word": "x", "start": "nan", "end": float("inf")}]}])
        self.assertEqual(len(words), 1)
        self.assertTrue(words[0].end >= words[0].start)


class DeriveSpans(unittest.TestCase):
    def setUp(self):
        self.words, _ = r.words_from_segments(SAMPLE)
        self.opts = r.RedactionOptions(pad_s=0.0, merge_gap_s=0.0, min_span_s=0.02)

    def derive(self, text, **kw):
        return r.derive_spans(self.words, text, kw.pop("options", self.opts), kw.pop("duration", 3.5))

    def test_first_word_is_covered(self):
        # Regression: the old code placed the span *after* a redacted first word.
        res = self.derive("SPEAKER_00: [PERSON] Mustermann lives in Berlin.")
        self.assertEqual(len(res.spans), 1)
        self.assertAlmostEqual(res.spans[0].start, 0.5, places=2)
        self.assertAlmostEqual(res.spans[0].end, 0.9, places=2)
        self.assertEqual(res.spans[0].label, "Max")

    def test_last_word_is_covered(self):
        # Regression: the old code produced a zero-length span for the last word.
        res = self.derive("SPEAKER_00: Max Mustermann lives in [CITY].")
        self.assertAlmostEqual(res.spans[0].start, 2.4, places=2)
        self.assertAlmostEqual(res.spans[0].end, 3.0, places=2)
        self.assertEqual(res.spans[0].tag, "[CITY]")

    def test_multiword_entity_single_tag(self):
        res = self.derive("SPEAKER_00: [PERSON] lives in [CITY].")
        self.assertEqual([(round(s.start, 2), round(s.end, 2)) for s in res.spans],
                         [(0.5, 1.6), (2.4, 3.0)])

    def test_middle_word_not_padded_by_neighbours(self):
        res = self.derive("Max Mustermann [VERB] in Berlin.")
        self.assertAlmostEqual(res.spans[0].start, 1.8, places=2)
        self.assertAlmostEqual(res.spans[0].end, 2.1, places=2)

    def test_punctuation_case_and_edits_do_not_create_spans(self):
        self.assertEqual(self.derive("max mustermann lives in berlin").spans, [])
        self.assertEqual(self.derive("SPEAKER_00: Max Mustermann, lives in Berlin!").spans, [])
        self.assertEqual(self.derive("Max Mustermann resides in Berlin.").spans, [])

    def test_changes_mode_masks_every_difference(self):
        opts = r.RedactionOptions(pad_s=0.0, merge_gap_s=0.0, mode="changes")
        res = r.derive_spans(self.words, "Max Mustermann resides in Berlin.", opts, 3.5)
        self.assertEqual([s.label for s in res.spans], ["lives"])

    def test_deletions(self):
        res = self.derive("Max lives in Berlin.")
        self.assertEqual([s.label for s in res.spans], ["Mustermann"])
        opts = r.RedactionOptions(pad_s=0.0, include_deletions=False)
        self.assertEqual(r.derive_spans(self.words, "Max lives in Berlin.", opts, 3.5).spans, [])

    def test_unmapped_insertion_warns(self):
        res = self.derive("Max Mustermann lives in [CITY] Berlin. [PERSON]")
        self.assertIn("unmapped_placeholder_insertion", res.warnings)

    def test_apostrophes_do_not_misalign(self):
        words, _ = r.words_from_segments([seg([("I", 0, .2), ("don't", .3, .6),
                                               ("know", .7, 1.0), ("Max", 1.1, 1.4)])])
        res = r.derive_spans(words, "I don ' t know [PERSON]", self.opts, 2.0)
        self.assertEqual([s.label for s in res.spans], ["Max"])

    def test_default_padding_bounded_by_gap_and_duration(self):
        sp = self.derive("SPEAKER_00: Max Mustermann lives in [CITY].", options=r.RedactionOptions()).spans[0]
        self.assertGreaterEqual(sp.start, 2.3 - 1e-9)         # gap to "in" is 0.1 s -> 0.05 pad
        self.assertLessEqual(sp.end, 3.5 + 1e-9)               # clamped to duration

    def test_file_edges_get_full_padding(self):
        res = self.derive("[A] Mustermann lives in Berlin.", options=r.RedactionOptions(pad_s=1.0), duration=0.7)
        self.assertEqual(res.spans[0].start, 0.0)
        self.assertLessEqual(res.spans[0].end, 0.7)

    def test_close_spans_merge(self):
        opts = r.RedactionOptions(pad_s=0.0, merge_gap_s=0.5)
        self.assertEqual(len(r.derive_spans(self.words, "[PERSON] Mustermann lives in [CITY].", opts, 3.5).spans), 2)
        self.assertEqual(len(r.derive_spans(self.words, "[A] [B] lives in Berlin.", opts, 3.5).spans), 1)

    def test_large_repetitive_input_is_fast_and_correct(self):
        base = ("the cat sat on the mat and then the dog came to see the cat "
                "while the bird sang a song about the sun ").split()
        raw, t = [], 0.0
        for i in range(20000):
            raw.append((base[i % len(base)], t, t + 0.2))
            t += 0.25
        words, _ = r.words_from_segments([{"start": 0, "end": t, "words": [
            {"word": a, "start": b, "end": c} for a, b, c in raw]}])
        text = " ".join("[PERSON]" if i % 997 == 0 else w[0] for i, w in enumerate(raw))
        began = time.time()
        res = r.derive_spans(words, text, duration=t)
        self.assertLess(time.time() - began, 20.0)
        self.assertEqual(len(res.spans), len(range(0, 20000, 997)))
        self.assertNotIn("alignment_diverged", res.warnings)

    def test_diverged_text_warns(self):
        res = r.derive_spans(self.words, "completely different words " * 100, self.opts, 3.5)
        self.assertIn("alignment_diverged", res.warnings)


class SanitizeSpans(unittest.TestCase):
    def test_swapped_and_clamped(self):
        out = r.sanitize_spans([{"start": 2, "end": 1}, {"start": -5, "end": 0.5}, {"start": 20, "end": 30}], 10)
        self.assertEqual([(s.start, s.end) for s in out], [(0.0, 0.5), (1.0, 2.0)])

    def test_rejects_bad_values(self):
        for bad in ({"start": float("nan"), "end": 1}, {"start": 0, "end": float("inf")},
                    {"start": "x", "end": 1}, {"start": True, "end": 1}, {"start": None, "end": 1}):
            with self.assertRaises(r.RedactionError):
                r.sanitize_spans([bad], 10)
        for bad in ("nope", [1]):
            with self.assertRaises(r.RedactionError):
                r.sanitize_spans(bad, 10)

    def test_cap(self):
        with self.assertRaises(r.RedactionError) as ctx:
            r.sanitize_spans([{"start": 0, "end": 1}] * (r.MAX_SPANS + 1), 10)
        self.assertEqual(ctx.exception.code, "too_many_segments")

    def test_overlaps_merge_gaps_stay(self):
        out = r.sanitize_spans([{"start": 0, "end": 2}, {"start": 1, "end": 3}, {"start": 3.5, "end": 4}], 10)
        self.assertEqual([(s.start, s.end) for s in out], [(0.0, 3.0), (3.5, 4.0)])

    def test_ids_unique_and_labels_bounded(self):
        out = r.sanitize_spans([{"id": "x", "start": 0, "end": 1}, {"id": "x", "start": 2, "end": 3},
                                {"id": "bad id!", "start": 4, "end": 5},
                                {"start": 6, "end": 7, "word": "a\x00b" + "x" * 1000}], 10)
        self.assertEqual(len({s.id for s in out}), 4)
        self.assertTrue(all(s.id for s in out))
        self.assertLessEqual(len(out[3].label), r.MAX_LABEL_CHARS)
        self.assertNotIn("\x00", out[3].label)


class OptionsAndDocuments(unittest.TestCase):
    def test_option_validation(self):
        self.assertEqual(r.RedactionOptions.from_mapping(None).mode, "placeholders")
        for bad in ({"pad_s": -1}, {"pad_s": 5}, {"mode": "x"}, {"include_deletions": "yes"},
                    {"merge_gap_s": float("nan")}, "str"):
            with self.assertRaises(r.RedactionError):
                r.RedactionOptions.from_mapping(bad)
        self.assertEqual(r.RedactionOptions.from_mapping({"pad": 0.1}).pad_s, 0.1)

    def test_document_roundtrip_withholds_words(self):
        doc = r.build_document([r.Span(1, 2, "Max", "[PERSON]", "auto", "a1")], 10.0, include_words=False)
        self.assertEqual(doc["spans"][0]["word"], "[PERSON]")
        self.assertNotIn("Max", str(doc))
        back = r.spans_from_document(doc, 10.0)
        self.assertEqual((back[0].start, back[0].end), (1.0, 2.0))

    def test_unknown_schema_and_legacy_shape(self):
        with self.assertRaises(r.RedactionError):
            r.spans_from_document({"schema_version": 99, "spans": []})
        self.assertEqual(r.legacy_offsets([r.Span(1, 2, "Max Mustermann")]),
                         [{"start": 1, "end": 2, "word": ["Max", "Mustermann"]}])


if __name__ == "__main__":
    unittest.main()
