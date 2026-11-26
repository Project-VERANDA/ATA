"""Tests for pipeline.surrogates (pool back end; Faker wiring via a stub)."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline import surrogates as s  # noqa: E402

ALL_TAGS = sorted(s.SUPPORTED_TAGS)
TTS_LINE_RE = re.compile(r"^(SPEAKER_\d+):\s*(.*)$")   # as used by generate_multi_speaker_tts


class StubFaker:
    """Minimal Faker stand-in recording the locale and seed."""

    def __init__(self, locale):
        self.locale, self.seed, self.n = locale, None, 0

    def seed_instance(self, seed):
        self.seed = seed

    def _next(self, prefix):
        self.n += 1
        return "%s%d" % (prefix, self.n)

    first_name = lambda self: self._next("Fn")  # noqa: E731
    last_name = lambda self: self._next("Ln")  # noqa: E731
    company = lambda self: self._next("Co")  # noqa: E731
    city = lambda self: self._next("City")  # noqa: E731
    country = lambda self: self._next("Land")  # noqa: E731
    street_name = lambda self: self._next("Street")  # noqa: E731
    postcode = lambda self: "10%03d" % self.n  # noqa: E731
    phone_number = lambda self: "+00 555 %04d" % self.n  # noqa: E731
    job = lambda self: self._next("Job")  # noqa: E731
    word = lambda self: self._next("word")  # noqa: E731
    safe_domain_name = lambda self: "example.org"  # noqa: E731


class Pool(unittest.TestCase):
    def sub(self, text, entities=None, **kw):
        kw.setdefault("backend", "pool")
        kw.setdefault("lang", "EN")
        return s.apply_surrogates(text, entities, **kw)

    def test_all_19_tags_replaced_and_unknown_left(self):
        res = self.sub("SPEAKER_00: " + " ".join("[%s]" % t for t in ALL_TAGS))
        self.assertNotRegex(res.text, r"\[[A-Z_]+\]")
        self.assertEqual(res.replaced, len(ALL_TAGS))
        self.assertEqual(self.sub("see [FOOBAR] now").text, "see [FOOBAR] now")

    def test_model_tag_names_in_entity_map(self):
        res = self.sub("[CITY] and [CITY]", {"LOC_CITY": ["Berlin", "Berlin"]})
        a, b = res.text.split(" and ")
        self.assertEqual(a, b)
        self.assertNotIn("entity_count_mismatch:CITY", res.warnings)

    def test_consistency_distinctness_and_no_originals(self):
        text = "[PERSON] met [PERSON] and [PERSON] said hi to [CITY] [CITY] [CITY]"
        ents = {"[PERSON]": ["Max Mustermann", "Anna Schmidt", "max mustermann"],
                "[CITY]": ["Berlin", "Hamburg", "BERLIN"]}
        res = self.sub(text, ents, seed=1)
        people, cities = res.registry["PERSON"], res.registry["CITY"]
        self.assertEqual(people[0], people[2])
        self.assertNotEqual(people[0], people[1])
        self.assertEqual(cities[0], cities[2])
        self.assertNotEqual(cities[0], cities[1])
        for original in ("Mustermann", "Berlin", "Schmidt"):
            self.assertNotIn(original, res.text)
            self.assertNotIn(original, str(res.registry))

    def test_first_name_reuse_and_never_equals_original(self):
        res = self.sub("[PERSON] ... [PERSON]", {"PERSON": ["Max Mustermann", "Max"]}, seed=3)
        full, short = res.registry["PERSON"]
        self.assertEqual(full.split()[0], short)
        for seed in range(30):
            self.assertNotEqual(self.sub("[PERSON]", {"PERSON": ["John"]}, seed=seed).text, "John")

    def test_mismatch_and_ambiguity_degrade_with_warning(self):
        res = self.sub("[PERSON] and [PERSON]", {"PERSON": ["Max"]})
        self.assertIn("entity_count_mismatch:PERSON", res.warnings)
        self.assertNotRegex(res.text, r"\[")
        self.assertIn("entity_map_ambiguous:CITY", self.sub("[CITY]", {"CITY": ["A"], "LOC_CITY": ["B"]}).warnings)

    def test_seed_is_deterministic_and_engine_reuse_is_consistent(self):
        ents = {"PERSON": ["Max"], "CITY": ["Berlin"]}
        self.assertEqual(self.sub("[PERSON] [CITY]", ents, seed=42).text,
                         self.sub("[PERSON] [CITY]", ents, seed=42).text)
        eng = s.SurrogateEngine("EN", seed=7, backend="pool")
        self.assertEqual(eng.substitute("[PERSON]", {"PERSON": ["Max Mustermann"]}).text,
                         eng.substitute("[PERSON]", {"PERSON": ["Max Mustermann"]}).text)

    def test_shapes_preserved(self):
        res = self.sub("[CODE] [DATETIME] [DATETIME] [AGE] [EMAIL] [NAME_RELATIVE]",
                       {"CODE": ["AB-1234"], "DATETIME": ["Monday", "12.03.1980"], "AGE": ["34"],
                        "EMAIL": ["x@y.z"], "NAME_RELATIVE": ["mother"]}, seed=5)
        code, day, date, age, email, rel = res.text.split(" ")
        self.assertRegex(code, r"^[A-Z]{2}-\d{4}$")
        self.assertIn(day, s.POOLS["EN"]["weekday"])
        self.assertRegex(date, r"^\d{2}/\d{2}/\d{4}$")
        self.assertTrue(18 <= int(age) <= 90)
        self.assertRegex(email, r"^[a-z.]+@example\.(com|org|net)$")
        self.assertIn(rel, s.POOLS["EN"]["relation"])

    def test_german_pool_and_date_format(self):
        res = self.sub("[CITY] [DATETIME] [PROFESSION]", {"CITY": ["x"], "DATETIME": ["1.1.2000"],
                                                         "PROFESSION": ["y"]}, lang="de_DE")
        city, date, job = res.text.split(" ")
        self.assertIn(city, s.POOLS["DE"]["city"])
        self.assertRegex(date, r"^\d{2}\.\d{2}\.\d{4}$")
        self.assertIn(job, s.POOLS["DE"]["job"])

    def test_language_normalisation(self):
        for value, expected in (("de", "DE"), ("DE", "DE"), ("de_DE,en_US", "DE"), ("auto", "EN"),
                                (None, "EN"), ("SP", "ES"), ("xx", "EN"), ("fr-FR", "FR")):
            self.assertEqual(s.normalize_lang(value), expected)

    def test_speaker_labels_survive_for_tts(self):
        text = "SPEAKER_00: [PERSON] lebt in [CITY].\nSPEAKER_01: Ja.\nSPEAKER_02: [PERSON] auch."
        before = [TTS_LINE_RE.match(l).group(1) for l in text.splitlines()]
        res = self.sub(text, lang="DE")
        self.assertEqual(before, [TTS_LINE_RE.match(l).group(1) for l in res.text.splitlines()])

    def test_large_text(self):
        self.assertEqual(self.sub("SPEAKER_00: [PERSON] x\n" * 20000).replaced, 20000)


class FakerWiring(unittest.TestCase):
    def tearDown(self):
        s._FAKER_OVERRIDE = None

    def test_auto_uses_faker_with_locale_and_seed(self):
        s._FAKER_OVERRIDE = StubFaker
        res = s.apply_surrogates("[PERSON] [CITY]", {"PERSON": ["Max"], "CITY": ["Berlin"]}, lang="DE", seed=1)
        self.assertEqual(res.backend, "faker")
        self.assertRegex(res.text, r"^Fn\d+ City\d+$")
        eng = s.SurrogateEngine("DE", seed=1)
        self.assertEqual(eng.backend._fake.locale, "de_DE")
        self.assertIsNotNone(eng.backend._fake.seed)

    def test_provider_failure_falls_back_to_pool_value(self):
        class Flaky(StubFaker):
            def city(self):
                raise AttributeError("no provider")
        s._FAKER_OVERRIDE = Flaky
        self.assertIn(s.apply_surrogates("[CITY]", {"CITY": ["x"]}).text, s.POOLS["EN"]["city"])

    def test_missing_faker_falls_back_and_warns_only_if_requested(self):
        s._FAKER_OVERRIDE = None
        if s.faker_available():
            self.skipTest("real Faker installed")
        self.assertEqual(s.apply_surrogates("[PERSON]", None, backend="auto").backend, "pool")
        self.assertIn("faker_unavailable", s.apply_surrogates("[PERSON]", None, backend="faker").warnings)

    def test_constant_faker_still_yields_non_original(self):
        class Constant(StubFaker):
            first_name = lambda self: "Max"  # noqa: E731
            last_name = lambda self: "Max"  # noqa: E731
        s._FAKER_OVERRIDE = Constant
        self.assertNotEqual(s.apply_surrogates("[PERSON]", {"PERSON": ["Max"]}, seed=1).text, "Max")


if __name__ == "__main__":
    unittest.main()
