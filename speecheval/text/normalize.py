"""
Text normalisation, applied identically to reference and hypothesis.

    NFC -> lowercase -> expand digits -> strip punctuation -> collapse whitespace

Two things are kept on purpose, against what `whisper.normalizers` would do:

* **apostrophes**, because they are word-internal (don't, 's-Gravenhage);
* **diacritics**, because año/ano and sé/se are different words and folding them
  would hide real errors.

Digit expansion exists for the hypothesis side. The benchmark texts contain no
digits at all, but recognisers write "20" where the reference says "twenty", and
without expansion that scores as a substitution the system never made.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

from ..config import TextNormalizationConfig

logger = logging.getLogger(__name__)

# Typographic apostrophes that mean the same thing as U+0027.
TYPOGRAPHIC_APOSTROPHES = "’ʼ՚＇"
ASCII_APOSTROPHE = "'"

# num2words language codes, keyed by benchmark language.
NUM2WORDS_LANGUAGES = {
    "en-US": "en",
    "es-ES": "es",
    "es-MX": "es",
    "nl-NL": "nl",
    "pt-BR": "pt_BR",
    "ky": None,          # unsupported by num2words; digits are left alone
}

_THOUSANDS = re.compile(r"(?<!\d)(\d{1,3})(?:([.,])(\d{3}))+(?!\d)")
_DECIMAL = re.compile(r"(?<!\d)(\d+)[.,](\d+)(?!\d)")
_INTEGER = re.compile(r"(?<!\d)(\d+)(?!\d)")
_ANY_DIGIT = re.compile(r"\d")


class TextNormalizer:
    """Normalises one language's text according to the configured protocol."""

    def __init__(self, config: TextNormalizationConfig, language: str):
        self.config = config
        self.language = language

        self._num2words_language = self._resolve_num2words_language(language)
        self._strip_table = self._build_strip_table()

        # Diagnostics, surfaced in the report rather than silently discarded.
        self.n_apostrophes_folded = 0
        self.n_digits_expanded = 0
        self.n_digits_left = 0

    # -- setup --------------------------------------------------------------

    def _resolve_num2words_language(self, language: str) -> Optional[str]:
        if not self.config.expand_digits:
            return None
        if language in self.config.expand_digits_skip_languages:
            return None
        code = NUM2WORDS_LANGUAGES.get(language)
        if code is None:
            logger.info(
                "%s: no num2words locale — digits in hypotheses are left as written",
                language,
            )
        return code

    def _build_strip_table(self) -> dict[int, str]:
        """
        Punctuation maps to a **space**, not to nothing.

        This is what the benchmark's own normaliser does, and it is visible in the
        shipped `text_norm`: `pre-instalado` becomes `pre instalado`, two words.
        Deleting the character instead would join them into one and change
        `N_ref`, the denominator of every corpus-level rate — so our numbers would
        no longer share a scale with the published human anchor.
        """
        punctuation = set(self.config.punctuation)
        if self.config.keep_apostrophes:
            punctuation.discard(ASCII_APOSTROPHE)
            if self.config.fold_typographic_apostrophes:
                # Already folded to U+0027 above; they only reach here when
                # folding is off, and then the protocol turns them into a space.
                for char in TYPOGRAPHIC_APOSTROPHES:
                    punctuation.discard(char)
        return {ord(char): " " for char in punctuation}

    # -- steps --------------------------------------------------------------

    def _fold_apostrophes(self, text: str) -> str:
        if not self.config.fold_typographic_apostrophes:
            return text
        folded = text
        for char in TYPOGRAPHIC_APOSTROPHES:
            if char in folded:
                self.n_apostrophes_folded += folded.count(char)
                folded = folded.replace(char, ASCII_APOSTROPHE)
        return folded

    def _expand_digits(self, text: str) -> str:
        if not _ANY_DIGIT.search(text):
            return text
        if self._num2words_language is None:
            self.n_digits_left += len(_ANY_DIGIT.findall(text))
            return text

        from num2words import num2words

        language = self._num2words_language

        def _say(value: float | int) -> str:
            try:
                return num2words(value, lang=language)
            except (NotImplementedError, OverflowError, TypeError) as exc:
                logger.debug("num2words failed for %r in %s: %s", value, language, exc)
                return str(value)

        # 1,000 / 1.000 — a grouped integer, not a decimal.
        def _grouped(match: re.Match) -> str:
            digits = re.sub(r"[.,]", "", match.group(0))
            self.n_digits_expanded += 1
            return _say(int(digits))

        text = _THOUSANDS.sub(_grouped, text)

        def _decimal(match: re.Match) -> str:
            whole, frac = match.group(1), match.group(2)
            self.n_digits_expanded += 1
            return f"{_say(int(whole))} {_say(float('0.' + frac))}"

        text = _DECIMAL.sub(_decimal, text)

        def _integer(match: re.Match) -> str:
            self.n_digits_expanded += 1
            return _say(int(match.group(1)))

        return _INTEGER.sub(_integer, text)

    # -- entry point --------------------------------------------------------

    def __call__(self, text: Optional[str]) -> str:
        if not text:
            return ""

        result = unicodedata.normalize(self.config.unicode_form, text)
        result = self._fold_apostrophes(result)
        if self.config.lowercase:
            result = result.lower()
        result = self._expand_digits(result)
        result = result.translate(self._strip_table)
        return " ".join(result.split())

    def diagnostics(self) -> dict[str, int]:
        return {
            "apostrophes_folded": self.n_apostrophes_folded,
            "digits_expanded": self.n_digits_expanded,
            "digits_left_unexpanded": self.n_digits_left,
        }
