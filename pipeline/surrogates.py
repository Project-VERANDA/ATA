"""Surrogate substitution: replace ``[TAG]`` placeholders with realistic fakes.

Two interchangeable back ends sit behind one engine:

* **Faker** (preferred, locale-aware) - used when the ``faker`` package is installed.
* **Pool** (fallback) - static, language-specific word lists following the same
  rules, so quality degrades gracefully instead of the feature failing.

Behaviour shared by both back ends:

* All 19 anonymizer tags are supported. Model tag names (``LOC_CITY``) and
  placeholder names (``CITY``) are accepted interchangeably.
* The same original entity maps to the same fake everywhere (case-insensitive);
  different originals get different fakes; a fake never equals its original.
  "Max" later in a dialogue reuses the first name chosen for "Max Mustermann".
* Shapes are preserved: a one-word name gets a first name, codes keep their
  letter/digit pattern, weekdays stay weekdays, dates follow the language.
* Consistency uses the anonymizer's *entity map* (originals per tag in document
  order). If the placeholder count no longer matches the map (text was edited),
  consistency is skipped for that tag and a warning is returned.
* Speaker labels (``SPEAKER_00:``) are never touched, so output can go straight
  to text-to-speech.
* Originals are never logged; the engine keeps only salted hashes.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --- Tag registry -------------------------------------------------------------
#: Model tag name -> placeholder name used in the anonymized text.
TAG_ALIASES = {
    "PERSON_EMAIL": "EMAIL", "PERSON_SOCIAL_RELATION": "NAME_RELATIVE", "ORG": "ORGANISATION",
    "LOC_CITY": "CITY", "LOC_COUNTRY": "COUNTRY", "LOC_STREET": "STREET", "LOC_ZIP": "ZIP",
    "LOC_HOUSENUMBER": "HOUSENUMBER", "LOC_OTHER": "LOCATION", "DATETIME_AGE": "AGE",
    "CODE_PHONE": "PHONE", "CODE_URL": "URL",
}
SUPPORTED_TAGS = frozenset({
    "PERSON", "EMAIL", "NAME_RELATIVE", "ORGANISATION", "CITY", "COUNTRY", "STREET", "ZIP",
    "HOUSENUMBER", "LOCATION", "DATETIME", "AGE", "CODE", "PHONE", "URL", "PROFESSION",
    "PRODUCT", "QUANTITY", "MISC",
})
PLACEHOLDER_RE = re.compile(r"\[([A-Z][A-Z0-9_]*)\]")

#: UI language code -> Faker locale.
LANG_LOCALES = {
    "EN": "en_US", "DE": "de_DE", "FR": "fr_FR", "ES": "es_ES", "IT": "it_IT",
    "PL": "pl_PL", "PT": "pt_BR", "FI": "fi_FI", "TR": "tr_TR", "AR": "ar_AA", "HI": "hi_IN",
}
_LANG_ALIASES = {"SP": "ES"}
DATE_PATTERNS = {"EN": "%m/%d/%Y", "DE": "%d.%m.%Y", "PL": "%d.%m.%Y", "FI": "%d.%m.%Y",
                 "TR": "%d.%m.%Y"}
DEFAULT_DATE_PATTERN = "%d/%m/%Y"
PHONE_PREFIX = {"EN": "+1", "DE": "+49", "FR": "+33", "ES": "+34", "IT": "+39", "PL": "+48",
                "PT": "+55", "FI": "+358", "TR": "+90", "AR": "+966", "HI": "+91"}
MAX_TEXT_CHARS = 5_000_000


def canonical_tag(name: str) -> str:
    """Maps ``"[PERSON]"``, ``"person"`` or ``"LOC_CITY"`` to ``"PERSON"``/``"CITY"``."""
    name = name.strip().strip("[]").upper()
    return TAG_ALIASES.get(name, name)


def normalize_lang(value: Optional[str]) -> str:
    """Normalises ``"de"``, ``"DE"``, ``"de_DE,en_US"``, ``"auto"`` to a language key.

    Unknown values and ``auto`` fall back to ``"EN"``.
    """
    if not value:
        return "EN"
    code = re.split(r"[,\s;]", str(value).strip())[0]
    code = re.split(r"[_-]", code)[0].upper()
    code = _LANG_ALIASES.get(code, code)
    return code if code in LANG_LOCALES else "EN"


# --- Static pools (fallback back end) -----------------------------------------
# Compact on purpose: the fallback only has to be plausible and consistent.
POOLS: Dict[str, Dict[str, List[str]]] = {
    "EN": {
        "first": "John Emma Michael Sophia Daniel Olivia James Emily Benjamin Charlotte Henry Mia".split(),
        "last": "Smith Johnson Brown Miller Wilson Moore Taylor Davis Clark Walker Harris Young".split(),
        "city": "London,Boston,Seattle,Chicago,Dublin,Leeds,Bristol,Manchester,Toronto,Denver,Austin,Portland".split(","),
        "street": "Oak Avenue,Maple Road,Pine Street,Cedar Lane,Elm Street,Park Avenue,River Road,Mill Road,Station Road,Church Lane".split(","),
        "company": ["Acme Corporation", "Northbridge University", "Blue River Consulting",
                    "Pioneer Software Ltd.", "Summit Financial Partners", "Evergreen Construction",
                    "Metro Insurance Services", "Green Valley Hospital", "Digital Innovation Labs",
                    "Western Manufacturing Inc."],
        "job": "teacher engineer nurse lawyer designer accountant electrician chef journalist pharmacist architect mechanic".split(),
        "word": "window river garden table bridge lantern meadow harbour".split(),
        "country": "Canada Ireland Norway Portugal Australia Iceland Belgium Austria".split(),
        "product": "smartphone laptop camera tablet headphones printer".split(),
        "weekday": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split(),
        "relation": "mother father sister brother aunt uncle cousin grandmother grandfather wife husband daughter son".split(),
    },
    "DE": {
        "first": "Max Anna Peter Laura Thomas Julia Lukas Sarah Felix Leonie Jonas Marie".split(),
        "last": "Müller Schmidt Weber Fischer Wagner Becker Hoffmann Koch Bauer Richter Klein Wolf".split(),
        "city": "Hamburg,München,Köln,Frankfurt,Stuttgart,Dresden,Leipzig,Bremen,Hannover,Düsseldorf,Dortmund,Nürnberg".split(","),
        "street": "Gartenweg,Schillerstraße,Goethestraße,Bergstraße,Dorfstraße,Mühlenweg,Kirchstraße,Lindenweg,Parkstraße,Wiesenweg".split(","),
        "company": ["Muster GmbH", "Beispiel AG", "Klinikum Nordstadt", "Norddeutsche Logistik GmbH",
                    "Rhein-Main Consulting", "Innovationslabor Dresden", "Bildungswerk Süd",
                    "Energieversorgung Mitte"],
        "job": "Lehrer Ingenieur Krankenpfleger Anwalt Designer Buchhalter Elektriker Koch Journalist Apotheker Architekt Mechaniker".split(),
        "word": "Fenster Fluss Garten Tisch Brücke Laterne Wiese Hafen".split(),
        "country": "Kanada Irland Norwegen Portugal Australien Island Belgien Österreich".split(),
        "product": "Smartphone Laptop Kamera Tablet Kopfhörer Drucker".split(),
        "weekday": "Montag Dienstag Mittwoch Donnerstag Freitag Samstag Sonntag".split(),
        "relation": "Mutter Vater Schwester Bruder Tante Onkel Cousine Oma Opa Frau Mann Tochter Sohn".split(),
    },
    "FR": {
        "first": "Lucas Léa Hugo Chloé Louis Manon Jules Camille Arthur Inès Paul Sarah".split(),
        "last": "Martin Bernard Dubois Thomas Robert Richard Petit Durand Leroy Moreau Simon Laurent".split(),
        "city": "Lyon,Marseille,Toulouse,Nantes,Bordeaux,Lille,Rennes,Strasbourg,Nice,Dijon".split(","),
        "street": ["rue des Lilas", "rue Victor Hugo", "avenue de la Gare", "rue du Moulin",
                   "rue des Écoles", "boulevard Pasteur"],
        "company": ["Société Dupont SA", "Clinique du Parc", "Groupe Lumière",
                    "Atelier Moderne SARL", "Banque du Nord"],
        "job": "enseignant ingénieur infirmier avocat designer comptable électricien cuisinier journaliste pharmacien".split(),
        "word": "fenêtre rivière jardin table pont lanterne prairie port".split(),
        "country": "Canada Irlande Norvège Portugal Australie Islande Belgique Autriche".split(),
        "product": ["smartphone", "ordinateur portable", "appareil photo", "tablette", "casque", "imprimante"],
        "weekday": "lundi mardi mercredi jeudi vendredi samedi dimanche".split(),
        "relation": ["mère", "père", "sœur", "frère", "tante", "oncle", "cousin", "grand-mère",
                     "grand-père", "femme", "mari", "fille", "fils"],
    },
    "ES": {
        "first": "Pablo Lucía Javier Carmen Sergio Marta Diego Elena Álvaro Paula Andrés Sofía".split(),
        "last": "García Fernández González Rodríguez López Martínez Sánchez Pérez Gómez Ruiz Díaz Moreno".split(),
        "city": "Sevilla,Valencia,Bilbao,Zaragoza,Málaga,Granada,Valladolid,Alicante,Córdoba,Murcia".split(","),
        "street": ["Calle Mayor", "Calle del Sol", "Avenida de la Paz", "Calle Real", "Calle Nueva",
                   "Plaza del Carmen"],
        "company": ["Grupo Solana S.L.", "Clínica del Mar", "Constructora Levante",
                    "Banco del Norte", "Consultora Atlas"],
        "job": "profesor ingeniero enfermero abogado diseñador contable electricista cocinero periodista farmacéutico".split(),
        "word": "ventana río jardín mesa puente linterna prado puerto".split(),
        "country": "Canadá Irlanda Noruega Portugal Australia Islandia Bélgica Austria".split(),
        "product": ["smartphone", "portátil", "cámara", "tableta", "auriculares", "impresora"],
        "weekday": "lunes martes miércoles jueves viernes sábado domingo".split(),
        "relation": "madre padre hermana hermano tía tío primo abuela abuelo esposa esposo hija hijo".split(),
    },
}
_ALL_WEEKDAYS = {d.casefold() for lang in POOLS.values() for d in lang["weekday"]}


def _pool(lang: str, kind: str) -> List[str]:
    """Pool for ``lang`` with English fallback for unsupported languages."""
    return (POOLS.get(lang) or POOLS["EN"])[kind]


# --- Back ends ----------------------------------------------------------------
def _faker_class() -> Optional[type]:
    """Returns the ``Faker`` class or None (isolated so tests can override it)."""
    override = globals().get("_FAKER_OVERRIDE")
    if override is not None:
        return override
    try:
        from faker import Faker  # type: ignore
        return Faker
    except Exception:                                   # ImportError or broken install
        return None


def faker_available() -> bool:
    """True if the Faker package can be imported."""
    return _faker_class() is not None


class PoolBackend:
    """Static-list back end (always available)."""

    name = "pool"

    def __init__(self, lang: str, rng: random.Random):
        self.lang = lang
        self.rng = rng

    def _pick(self, kind: str) -> str:
        return self.rng.choice(_pool(self.lang, kind))

    def first_name(self) -> str:
        return self._pick("first")

    def last_name(self) -> str:
        return self._pick("last")

    def company(self) -> str:
        return self._pick("company")

    def city(self) -> str:
        return self._pick("city")

    def country(self) -> str:
        return self._pick("country")

    def street(self) -> str:
        return self._pick("street")

    def job(self) -> str:
        return self._pick("job")

    def word(self) -> str:
        return self._pick("word")

    def postcode(self) -> str:
        pattern = {"PL": "##-###", "PT": "####-###", "HI": "######"}.get(self.lang, "#####")
        return "".join(str(self.rng.randrange(10)) if c == "#" else c for c in pattern)

    def phone(self) -> str:
        digits = "".join(str(self.rng.randrange(10)) for _ in range(9))
        return "%s %s %s %s" % (PHONE_PREFIX.get(self.lang, "+1"), digits[:3], digits[3:6], digits[6:])

    def safe_domain(self) -> str:
        return self.rng.choice(["example.com", "example.org", "example.net"])


class FakerBackend(PoolBackend):
    """Locale-aware back end; any Faker failure falls back to the pool value."""

    name = "faker"

    def __init__(self, lang: str, rng: random.Random, seeded: bool):
        super().__init__(lang, rng)
        factory = _faker_class()
        if factory is None:
            raise RuntimeError("faker is not installed")
        self._fake = factory(LANG_LOCALES.get(lang, "en_US"))
        if seeded:
            self._fake.seed_instance(rng.randrange(2 ** 32))

    def _call(self, method: str, fallback: Callable[[], str]) -> str:
        try:
            value = str(getattr(self._fake, method)()).strip()
            return value if value else fallback()
        except Exception:                               # provider missing for locale, etc.
            return fallback()

    def first_name(self) -> str:
        return self._call("first_name", super().first_name)

    def last_name(self) -> str:
        return self._call("last_name", super().last_name)

    def company(self) -> str:
        return self._call("company", super().company)

    def city(self) -> str:
        return self._call("city", super().city)

    def country(self) -> str:
        return self._call("country", super().country)

    def street(self) -> str:
        return self._call("street_name", super().street)

    def job(self) -> str:
        return self._call("job", super().job)

    def word(self) -> str:
        return self._call("word", super().word)

    def postcode(self) -> str:
        return self._call("postcode", super().postcode)

    def phone(self) -> str:
        return self._call("phone_number", super().phone)

    def safe_domain(self) -> str:
        return self._call("safe_domain_name", super().safe_domain)


# --- Engine -------------------------------------------------------------------
@dataclass
class SurrogateResult:
    """Outcome of :meth:`SurrogateEngine.substitute`.

    Attributes:
        text: Text with placeholders replaced.
        registry: ``{tag: [fake values in order]}``. Contains no originals.
        warnings: Machine-readable notes, e.g. ``entity_count_mismatch:PERSON``.
        backend: ``"faker"`` or ``"pool"``.
        replaced: Number of placeholders replaced.
    """

    text: str
    registry: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    backend: str = "pool"
    replaced: int = 0

    def as_tuple(self) -> Tuple[str, Dict[str, List[str]]]:
        """Legacy ``(text, registry)`` pair."""
        return self.text, self.registry


def _norm(value: str) -> str:
    """Case-insensitive, whitespace/punctuation-tolerant comparison key."""
    value = unicodedata.normalize("NFKC", " ".join(value.split())).casefold()
    return value.strip(" .,;:!?\"'()[]")


def shape_match(original: str, rng: random.Random) -> str:
    """Random string with the same letter/digit/punctuation shape as ``original``."""
    out = []
    for ch in original:
        if ch.isdigit():
            out.append(str(rng.randrange(10)))
        elif ch.isalpha() and ch.isupper():
            out.append(chr(rng.randrange(65, 91)))
        elif ch.isalpha() and ch.islower() and ch.isascii():
            out.append(chr(rng.randrange(97, 123)))
        else:
            out.append(ch)
    if original[:1].isdigit() and original[:1] != "0" and out and out[0] == "0":
        out[0] = str(rng.randrange(1, 10))
    return "".join(out)


class SurrogateEngine:
    """Stateful substitution engine (share one instance for consistency across files).

    Args:
        lang: Language key or locale (``"DE"``, ``"de_DE"``...); ``auto`` -> EN.
        seed: Seed for reproducible output; ``None`` for non-deterministic.
        backend: ``"auto"`` (Faker if installed, else pool), ``"faker"`` or ``"pool"``.
    """

    def __init__(self, lang: Optional[str] = "EN", seed: Optional[int] = None,
                 backend: str = "auto"):
        if backend not in ("auto", "faker", "pool"):
            raise ValueError("backend must be 'auto', 'faker' or 'pool'")
        self.lang = normalize_lang(lang)
        self.rng = random.Random(seed)
        self.warnings: List[str] = []
        self._lock = threading.Lock()
        self._salt = (seed.to_bytes(8, "big", signed=True) if isinstance(seed, int)
                      else os.urandom(16))
        self._cache: Dict[Tuple[str, str], str] = {}          # (tag, hash(original)) -> fake
        self._used: Dict[str, set] = {}                        # kind -> fakes already issued
        self._first: Dict[str, str] = {}                       # hash(first name) -> fake
        self._last: Dict[str, str] = {}                        # hash(last name) -> fake
        self.backend = self._make_backend(backend, seed is not None)

    def _make_backend(self, requested: str, seeded: bool) -> PoolBackend:
        if requested != "pool":
            try:
                return FakerBackend(self.lang, self.rng, seeded)
            except Exception as exc:
                if requested == "faker":
                    self.warnings.append("faker_unavailable")
                logger.info("Faker back end unavailable (%s); using static pools", type(exc).__name__)
        return PoolBackend(self.lang, self.rng)

    # -- helpers ---------------------------------------------------------------
    def _key(self, text: str) -> str:
        return hashlib.blake2b(_norm(text).encode("utf-8"), key=self._salt[:16],
                               digest_size=12).hexdigest()

    def _unique(self, kind: str, make: Callable[[], str], original: str) -> str:
        """Generates a value that differs from the original and from earlier fakes."""
        used = self._used.setdefault(kind, set())
        orig = _norm(original)
        value = make()
        for _ in range(12):
            if _norm(value) != orig and value not in used:
                break
            value = make()
        if _norm(value) == orig:                       # pool exhausted: force a difference
            value = value + "-" + str(self.rng.randrange(10, 99))
        used.add(value)
        return value

    @staticmethod
    def _match_case(original: str, value: str) -> str:
        return value.upper() if len(original) > 1 and original.isupper() else value

    def _person(self, original: str) -> str:
        """Fake for a person mention, consistent for first/last-name parts."""
        tokens = original.split()
        if not tokens:
            return self._unique("first", self.backend.first_name, original)

        def first(tok: str) -> str:
            k = self._key(tok)
            if k not in self._first:
                self._first[k] = self._last.get(k) or self._unique("first", self.backend.first_name, tok)
            return self._first[k]

        def last(tok: str) -> str:
            k = self._key(tok)
            if k not in self._last:
                self._last[k] = self._first.get(k) or self._unique("last", self.backend.last_name, tok)
            return self._last[k]

        if len(tokens) == 1:
            return self._match_case(original, first(tokens[0]))
        parts = [first(t) for t in tokens[:-1]] + [last(tokens[-1])]
        return self._match_case(original, " ".join(parts))

    def _date(self, original: str) -> str:
        norm = _norm(original)
        if norm in _ALL_WEEKDAYS:
            days = _pool(self.lang, "weekday")
            return self._unique("weekday", lambda: self.rng.choice(days), original)
        if re.fullmatch(r"\d{4}", norm):                # a bare year
            return str(self.rng.randrange(1950, 2026))
        base = date(1950, 1, 1) + timedelta(days=self.rng.randrange(0, 365 * 75))
        return base.strftime(DATE_PATTERNS.get(self.lang, DEFAULT_DATE_PATTERN))

    def _generate(self, tag: str, original: str) -> str:
        """Creates a fresh fake for one entity (no caching)."""
        b, rng = self.backend, self.rng
        if tag == "PERSON":
            return self._person(original)
        if tag == "EMAIL":
            local = ("%s.%s" % (b.first_name(), b.last_name())).lower()
            local = unicodedata.normalize("NFKD", local).encode("ascii", "ignore").decode()
            return "%s@%s" % (re.sub(r"[^a-z0-9._-]", "", local) or "user", b.safe_domain())
        if tag == "NAME_RELATIVE":
            words = _pool(self.lang, "relation")
            return self._unique("relation", lambda: rng.choice(words), original)
        if tag == "ORGANISATION":
            return self._unique("org", b.company, original)
        if tag in ("CITY", "LOCATION"):
            return self._unique("city", b.city, original)
        if tag == "COUNTRY":
            return self._unique("country", b.country, original)
        if tag == "STREET":
            return self._unique("street", b.street, original)
        if tag == "ZIP":
            return self._unique("zip", b.postcode, original)
        if tag == "HOUSENUMBER":
            return str(rng.randrange(1, 200))
        if tag == "DATETIME":
            return self._date(original)
        if tag == "AGE":
            return str(rng.randrange(18, 91))
        if tag == "PHONE":
            return b.phone()
        if tag == "URL":
            return "https://www.%s" % b.safe_domain()
        if tag == "PROFESSION":
            return self._unique("job", b.job, original)
        if tag == "PRODUCT":
            words = _pool(self.lang, "product")
            return self._unique("product", lambda: rng.choice(words), original)
        if tag in ("CODE", "QUANTITY"):
            return shape_match(original, rng) if original else str(rng.randrange(1, 1000))
        return self._unique("word", b.word, original)  # MISC

    def _value(self, tag: str, original: Optional[str]) -> str:
        """Returns the fake for an entity, reusing earlier choices when known."""
        if original is None or not original.strip():
            return self._generate(tag, original or "")
        key = (tag, self._key(original))
        if key not in self._cache:
            self._cache[key] = self._generate(tag, original)
        return self._cache[key]

    # -- public API ------------------------------------------------------------
    def substitute(self, text: str, entity_map: Optional[Mapping[str, Sequence[str]]] = None
                   ) -> SurrogateResult:
        """Replaces every supported ``[TAG]`` in ``text``.

        Args:
            text: Anonymized text (may contain ``SPEAKER_nn:`` prefixes).
            entity_map: ``{tag: [original strings in document order]}`` as returned
                by the anonymizer; keys may be ``"[PERSON]"``, ``"PERSON"`` or model
                names. Omit for fakes without cross-mention consistency.

        Returns:
            :class:`SurrogateResult`.

        Raises:
            ValueError: Text is excessively large.
        """
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError("text too large")
        with self._lock:
            warnings = list(self.warnings)
            originals = self._originals_by_tag(entity_map, warnings)
            counts: Dict[str, int] = {}
            for match in PLACEHOLDER_RE.finditer(text):
                tag = canonical_tag(match.group(1))
                counts[tag] = counts.get(tag, 0) + 1
            reliable: Dict[str, List[str]] = {}
            for tag, count in counts.items():
                known = originals.get(tag)
                if known is not None and len(known) == count:
                    reliable[tag] = known
                elif known is not None:
                    warnings.append("entity_count_mismatch:%s" % tag)
            seen: Dict[str, int] = {}
            registry: Dict[str, List[str]] = {}
            replaced = 0

            def repl(match: "re.Match[str]") -> str:
                nonlocal replaced
                raw = match.group(1)
                tag = canonical_tag(raw)
                if tag not in SUPPORTED_TAGS:
                    return match.group(0)
                n = seen.get(tag, 0)
                seen[tag] = n + 1
                known = reliable.get(tag)
                original = known[n] if known is not None else None
                value = self._value(tag, original) if original is not None else self._generate(tag, "")
                registry.setdefault(raw, []).append(value)
                replaced += 1
                return value

            result = PLACEHOLDER_RE.sub(repl, text)
        return SurrogateResult(result, registry, sorted(set(warnings)), self.backend.name, replaced)

    @staticmethod
    def _originals_by_tag(entity_map: Optional[Mapping[str, Sequence[str]]],
                          warnings: List[str]) -> Dict[str, List[str]]:
        """Normalises entity-map keys; ambiguous duplicates disable consistency."""
        out: Dict[str, List[str]] = {}
        if not entity_map:
            return out
        clashes = set()
        for key, values in entity_map.items():
            tag = canonical_tag(str(key))
            if not isinstance(values, (list, tuple)):
                continue
            if tag in out:
                clashes.add(tag)
            out[tag] = [str(v) for v in values]
        for tag in clashes:
            out.pop(tag, None)
            warnings.append("entity_map_ambiguous:%s" % tag)
        return out


# --- Convenience wrappers -----------------------------------------------------
def apply_surrogates(text: str, entity_map: Optional[Mapping[str, Sequence[str]]] = None,
                     lang: Optional[str] = "EN", seed: Optional[int] = None,
                     backend: str = "auto",
                     engine: Optional[SurrogateEngine] = None) -> SurrogateResult:
    """One-call substitution. Pass ``engine`` to keep consistency across calls."""
    return (engine or SurrogateEngine(lang, seed, backend)).substitute(text, entity_map)


def lang_from_locales(locales: Optional[str]) -> str:
    """Derives a language key from a Faker locale list like ``"de_DE,en_US"``."""
    return normalize_lang(locales)
