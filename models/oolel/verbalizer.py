import re


class WolofTTSVerbalizer:
    """
    Rule-based text verbalizer for Wolof TTS pre-processing.

    Converts non-lexical tokens (numbers, dates, times, currencies,
    phone numbers, ordinals, percentages) into spoken Wolof words, and
    strips or replaces special characters that a TTS engine cannot
    pronounce.  Orthographic correction is intentionally out of scope
    and should be handled by a dedicated model (e.g. WolofCorrector)
    applied before this class.

    The output is designed to feed directly into WolofTTSVerbalizer:

        raw  →  WolofTTSVerbalizer.verbalize()  →  WolofTTSVerbalizer.verbalize()  →  TTS

    All verbalization steps run in a fixed, dependency-safe order inside
    ``verbalize()``.  Each step is a dedicated private method so rules
    can be read, tested, and extended independently.
    """

    # ------------------------------------------------------------------
    # Wolof number lexicon
    # ------------------------------------------------------------------

    _UNITS = {
        0: "tus",        1: "benn",          2: "ñaar",
        3: "ñett",       4: "ñeent",         5: "juróom",
        6: "juróom-benn", 7: "juróom-ñaar",  8: "juróom-ñett",
        9: "juróom-ñeent", 10: "fukk",       11: "fukk ak benn",
        12: "fukk ak ñaar", 13: "fukk ak ñett", 14: "fukk ak ñeent",
        15: "fukk ak juróom", 16: "fukk ak juróom-benn",
        17: "fukk ak juróom-ñaar", 18: "fukk ak juróom-ñett",
        19: "fukk ak juróom-ñeent",
    }

    _TENS = {
        20: "ñaar-fukk",      30: "ñett-fukk",
        40: "ñeent-fukk",     50: "juróom-fukk",
        60: "juróom-benn-fukk", 70: "juróom-ñaar-fukk",
        80: "juróom-ñett-fukk", 90: "juróom-ñeent-fukk",
    }

    # French number lexicon — used for dates, currencies, and phone numbers
    # (matches WolofTTSVerbalizer behaviour)
    _FR_UNITS = {
        0: "zéro",  1: "un",      2: "deux",   3: "trois",
        4: "quatre", 5: "cinq",   6: "six",    7: "sept",
        8: "huit",   9: "neuf",  10: "dix",   11: "onze",
        12: "douze", 13: "treize", 14: "quatorze", 15: "quinze",
        16: "seize",
    }

    _FR_TENS = {
        10: "dix",   20: "vingt",    30: "trente",
        40: "quarante", 50: "cinquante", 60: "soixante",
    }

    # ------------------------------------------------------------------
    # Static lookup tables
    # ------------------------------------------------------------------

    # Wolof month names (matches WolofTTSVerbalizer)
    _MONTHS = {
        1: "Samwiye",  2: "Fewriye",   3: "Mars",
        4: "Awril",    5: "Me",         6: "Suwe",
        7: "Sulet",    8: "Ut",         9: "Sàttumbar",
        10: "Oktoobar", 11: "Nowembar", 12: "Desàmbar",
    }

    # Currency codes → Wolof/French spoken name
    _CURRENCIES = {
        "XOF":  "franc CFA",
        "FCFA": "franc CFA",
        "EUR":  "euro",
        "USD":  "dollar",
        "GBP":  "pound",
    }

    # Currency glyph → ISO code (for glyph-prefixed amounts like $50, €9.99)
    _GLYPH_TO_CODE = {
        "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₣": "XOF",
    }

    # Ordinals with dedicated Wolof words
    _ORDINALS = {
        "1er": "wërëkk", "1ère": "wërëkk",
        "2e":  "ñaareelu", "3e": "ñetteelu",
    }

    # ------------------------------------------------------------------
    # Pre-compiled regular expressions
    # ------------------------------------------------------------------

    # Thousands separators: 1 000 000 or 1,000,000 (but NOT decimals)
    _RE_THOUSANDS_SPACE  = re.compile(r'(\d)[ \u00a0](?=\d{3}\b)')
    _RE_THOUSANDS_COMMA  = re.compile(r'(\d),(?=\d{3}\b)')

    # Currency glyph immediately before a number: $50, €9,99
    _RE_GLYPH_CURRENCY   = re.compile(
        r'([$\u20ac\xa3\xa5\u20a3])(\d+(?:[.,]\d+)?)'
    )
    # Amount followed by a currency code: 5000 FCFA, 9.99 EUR
    _RE_CODE_CURRENCY    = re.compile(
        r'(\d+(?:[.,]\d+)?)\s*(XOF|FCFA|EUR|USD|GBP)\b',
        re.IGNORECASE,
    )

    # Percentages: 25%, 3.5%, 3,5%
    _RE_PERCENTAGE       = re.compile(r'(\d+(?:[.,]\d+)?)\s*%')

    # Dates: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    _RE_DATE             = re.compile(
        r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b'
    )

    # Times: H:MM or HH:MM (24-h or 12-h; AM/PM stripped later)
    _RE_TIME             = re.compile(r'\b(\d{1,2}):(\d{2})\b')

    # Phone — international with + or 00 prefix, then 9 Senegalese digits
    _RE_PHONE_INTL_PLUS  = re.compile(r'\+221\d{9}')
    _RE_PHONE_INTL_00    = re.compile(r'\b00221\d{9}\b')
    # Local 9-digit Senegalese numbers (start with 7 or 3)
    _RE_PHONE_LOCAL      = re.compile(r'\b[73]\d{8}\b')

    # Ordinals: 1er, 1ère, 2e, 3e … 99e
    _RE_ORDINAL          = re.compile(r'\b(\d+)(ère|er|e)\b')

    # Decimal numbers (after thousands separators have been removed)
    _RE_DECIMAL          = re.compile(r'\b(\d+)[.,](\d+)\b')

    # Bare integers
    _RE_INTEGER          = re.compile(r'\b(\d+)\b')

    # AM/PM suffixes (case-insensitive)
    _RE_AMPM             = re.compile(r'\b([aApP][mM])\b')

    # Special characters
    _RE_GROUPING         = re.compile(r'[(){}\[\]\u00ab\u00bb\u201c\u201d\u201e\u201f\'\`\u2018\u2019]')
    _RE_ELLIPSIS         = re.compile(r'\.{2,}|\u2026')
    _RE_PAUSE_PUNCT      = re.compile(r'[;]')
    _RE_DASH             = re.compile(r'\s*[\u2014\u2013]\s*')
    # Orphan colon (not preceded by a digit:digit pattern — those are already gone)
    _RE_ORPHAN_COLON     = re.compile(r':')
    # Residual currency glyphs (any that were not attached to a number)
    _RE_RESIDUAL_GLYPHS  = re.compile(r'[$\u20ac\xa3\xa5\u20a3]')
    # Symbols with no spoken value
    _RE_SYMBOLS          = re.compile(r'[\\*#@^~|_=<>+]')
    # Orphan slash (dates are already replaced, so any remaining / has no value)
    _RE_SLASH            = re.compile(r'/')
    # Orphan % (not preceded by a digit — those were already verbalised)
    _RE_ORPHAN_PERCENT   = re.compile(r'(?<!\w)%')
    # Collapse whitespace
    _RE_WHITESPACE       = re.compile(r'[ \t]+')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verbalize(self, text: str) -> str:
        """
        Verbalize ``text`` for Wolof TTS consumption.

        Steps run in this order so that longer / higher-priority patterns
        are resolved before the bare-integer pass can break them:

        1.  Thousands-separator removal
        2.  Currency glyph + amount  (e.g. $50)
        3.  Currency code + amount   (e.g. 5000 FCFA)
        4.  Percentages
        5.  Dates
        6.  Times  (+ AM/PM strip)
        7.  Phone numbers
        8.  Ordinals
        9.  Decimal numbers
        10. Bare integers
        11. Special-character cleanup
        12. Whitespace normalisation
        13. Lowercasing  (last, so all pattern matches above see original casing)

        Args:
            text: Raw Wolof text.

        Returns:
            Text with all non-lexical tokens replaced by spoken Wolof
            words and all TTS-unsafe characters removed.
        """
        text = self._remove_thousands_separators(text)
        text = self._normalize_glyph_currencies(text)
        text = self._normalize_code_currencies(text)
        text = self._normalize_percentages(text)
        text = self._normalize_dates(text)
        text = self._normalize_times(text)
        text = self._normalize_phones(text)
        text = self._normalize_ordinals(text)
        text = self._normalize_decimals(text)
        text = self._normalize_integers(text)
        text = self._remove_special_characters(text)
        text = self._normalize_whitespace(text)
        text = self._lowercase_text(text)
        return text

    # ------------------------------------------------------------------
    # Number helpers
    # ------------------------------------------------------------------

    def _int_to_wolof(self, num: int) -> str:
        """Convert a non-negative integer to its Wolof word form."""
        if num < 0:
            return f"mënus {self._int_to_wolof(-num)}"
        if num <= 19:
            return self._UNITS[num]
        if num < 100:
            tens, units = (num // 10) * 10, num % 10
            return (self._TENS[tens] if units == 0
                    else f"{self._TENS[tens]} ak {self._UNITS[units]}")
        if num < 1000:
            hundreds, rem = num // 100, num % 100
            head = "téeméer" if hundreds == 1 else f"{self._UNITS[hundreds]} téeméer"
            return head if rem == 0 else f"{head} {self._int_to_wolof(rem)}"
        if num < 1_000_000:
            thousands, rem = num // 1000, num % 1000
            head = "junni" if thousands == 1 else f"{self._int_to_wolof(thousands)} junni"
            return head if rem == 0 else f"{head} {self._int_to_wolof(rem)}"
        if num < 1_000_000_000:
            millions, rem = num // 1_000_000, num % 1_000_000
            head = ("tamndareet" if millions == 1
                    else f"{self._int_to_wolof(millions)} tamndareet")
            return head if rem == 0 else f"{head} {self._int_to_wolof(rem)}"
        return str(num)  # fallback for very large numbers

    def _int_to_french(self, num: int) -> str:
        """Convert a non-negative integer to French words.

        Used for dates, currency amounts, and phone number segments,
        matching the behaviour of WolofTTSVerbalizer.french_number_to_words().
        """
        if num < 0:
            return f"moins {self._int_to_french(-num)}"
        if num <= 16:
            return self._FR_UNITS[num]
        if num < 20:
            return f"dix-{self._FR_UNITS[num - 10]}"
        if num < 70:
            tens, units = (num // 10) * 10, num % 10
            if units == 0:
                return self._FR_TENS[tens]
            if units == 1 and tens != 10:
                return f"{self._FR_TENS[tens]} et un"
            return f"{self._FR_TENS[tens]}-{self._FR_UNITS[units]}"
        if num < 80:
            rem = num - 60
            if rem == 10:  return "soixante-dix"
            if rem == 11:  return "soixante et onze"
            if rem <= 16:  return f"soixante-{self._FR_UNITS[rem]}"
            return f"soixante-dix-{self._FR_UNITS[rem - 10]}"
        if num < 100:
            units = num % 80
            return "quatre-vingts" if units == 0 else f"quatre-vingt-{self._int_to_french(units)}"
        if num < 1000:
            hundreds, rem = num // 100, num % 100
            if hundreds == 1:
                head = "cent"
            else:
                head = f"{self._FR_UNITS[hundreds]} cent"
                if rem == 0:
                    head += "s"
            return head if rem == 0 else f"{head} {self._int_to_french(rem)}"
        if num < 1_000_000:
            thousands, rem = num // 1000, num % 1000
            head = "mille" if thousands == 1 else f"{self._int_to_french(thousands)} mille"
            return head if rem == 0 else f"{head} {self._int_to_french(rem)}"
        if num < 1_000_000_000:
            millions, rem = num // 1_000_000, num % 1_000_000
            head = ("un million" if millions == 1
                    else f"{self._int_to_french(millions)} millions")
            return head if rem == 0 else f"{head} {self._int_to_french(rem)}"
        return str(num)

    def _decimal_to_wolof(self, integer_str: str, decimal_str: str) -> str:
        """Verbalize a decimal number in Wolof.

        The integer part is read as a Wolof number word, then "puwaan"
        ("point"), then each decimal digit individually.
        """
        integer_part = self._int_to_wolof(int(integer_str))
        digit_words = " ".join(self._UNITS[int(d)] for d in decimal_str)
        return f"{integer_part} puwaan {digit_words}"

    # ------------------------------------------------------------------
    # Normalisation steps
    # ------------------------------------------------------------------

    def _remove_thousands_separators(self, text: str) -> str:
        """Strip space/comma thousands separators from integers.

        E.g. "1 000 000" -> "1000000", "1,000" -> "1000".
        Commas that are decimal separators (followed by fewer than three
        digits) are left untouched here; the decimal step handles them.
        """
        text = self._RE_THOUSANDS_SPACE.sub(r'\1', text)
        text = self._RE_THOUSANDS_COMMA.sub(r'\1', text)
        return text

    def _normalize_glyph_currencies(self, text: str) -> str:
        """Rewrite glyph-prefixed amounts to spoken Wolof.

        E.g. "$50" -> "cinquante dollar",
             "euro9.99" -> "neuf euro ak quatre-vingt-dix-neuf centime".
        """
        def _replace(m: re.Match) -> str:
            glyph, amount_str = m.group(1), m.group(2)
            code = self._GLYPH_TO_CODE.get(glyph, "")
            currency_name = self._CURRENCIES.get(code, code)
            return self._verbalize_currency_amount(amount_str, currency_name)

        return self._RE_GLYPH_CURRENCY.sub(_replace, text)

    def _normalize_code_currencies(self, text: str) -> str:
        """Rewrite "<amount> <CODE>" patterns to spoken Wolof.

        E.g. "5000 FCFA" -> "cinq mille franc CFA".
        """
        def _replace(m: re.Match) -> str:
            amount_str = m.group(1)
            code = m.group(2).upper()
            currency_name = self._CURRENCIES.get(code, code)
            return self._verbalize_currency_amount(amount_str, currency_name)

        return self._RE_CODE_CURRENCY.sub(_replace, text)

    def _verbalize_currency_amount(self, amount_str: str, currency_name: str) -> str:
        """Shared helper: turn an amount string + currency name into words."""
        amount_str = amount_str.replace(",", ".")
        if "." in amount_str:
            major_str, minor_str = amount_str.split(".", 1)
            major = self._int_to_wolof(int(major_str))
            result = f"{major} {currency_name}"
            minor_val = int(minor_str) if minor_str else 0
            if minor_val:
                result += f" ak {self._int_to_wolof(minor_val)} centime"
            return result
        return f"{self._int_to_wolof(int(amount_str))} {currency_name}"

    def _normalize_percentages(self, text: str) -> str:
        """Rewrite "N%" to "N poorso" (Wolof for "percent").

        Decimal percentages use the Wolof decimal reading.
        E.g. "25%"  -> "ñaar-fukk ak juróom poorso",
             "3.5%" -> "ñett puwaan juróom poorso".
        """
        def _replace(m: re.Match) -> str:
            val = m.group(1).replace(",", ".")
            if "." in val:
                int_part, dec_part = val.split(".", 1)
                num_word = self._decimal_to_wolof(int_part, dec_part)
            else:
                num_word = self._int_to_wolof(int(val))
            return f"{num_word} poorso"

        return self._RE_PERCENTAGE.sub(_replace, text)

    def _normalize_dates(self, text: str) -> str:
        """Rewrite DD/MM/YYYY (or DD-MM-YYYY, DD.MM.YYYY) to spoken Wolof.

        Day and year are read in French (matching verbalizer behaviour);
        the month name comes from the Wolof month table.
        E.g. "03/04/2025" -> "trois Awril deux mille vingt-cinq".
        """
        def _replace(m: re.Match) -> str:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            day_word   = self._int_to_wolof(day)
            month_word = self._MONTHS.get(month, str(month))
            year_word  = self._int_to_wolof(year)
            return f"{day_word} {month_word} {year_word}"

        return self._RE_DATE.sub(_replace, text)

    def _normalize_times(self, text: str) -> str:
        """Rewrite HH:MM to spoken Wolof.

        Hours are read in Wolof; minutes use "simili" (Wolof for "minutes").
        A zero-minute time uses "waxtu" ("o'clock") with no minute phrase.
        AM/PM suffixes are stripped beforehand.
        E.g. "08:30" -> "juróom-ñett ak ñett-fukk waxtu ak ñett-fukk simili"
             "14:00" -> "fukk ak ñeent waxtu"
        """
        # Strip AM/PM first so the time regex can match the bare H:MM
        text = self._RE_AMPM.sub("", text)

        def _replace(m: re.Match) -> str:
            hours, minutes = int(m.group(1)), int(m.group(2))
            hour_word = self._int_to_wolof(hours)
            if minutes == 0:
                return f"{hour_word} waxtu"
            minute_word = self._int_to_wolof(minutes)
            return f"{hour_word} waxtu ak {minute_word} simili"

        return self._RE_TIME.sub(_replace, text)

    def _normalize_phones(self, text: str) -> str:
        """Rewrite Senegalese phone numbers to spoken French words.

        Supports:
        - +221XXXXXXXXX  (international with + prefix)
        - 00221XXXXXXXXX (international with 00 prefix)
        - XXXXXXXXX      (local 9-digit, starting with 7 or 3)

        The 9-digit local number is split XX-XXX-XX-XX and each group
        is read as a French number, matching verbalizer behaviour.
        """
        def _verbalize_9digits(digits: str) -> str:
            parts = [
                self._int_to_french(int(digits[0:2])),
                self._int_to_french(int(digits[2:5])),
                self._int_to_french(int(digits[5:7])),
                self._int_to_french(int(digits[7:9])),
            ]
            return " ".join(parts)

        def _replace_intl_plus(m: re.Match) -> str:
            raw = m.group(0).replace(" ", "").replace("-", "")
            words = ["plus", self._int_to_french(221)]
            words.append(_verbalize_9digits(raw[4:]))
            return " ".join(words)

        def _replace_intl_00(m: re.Match) -> str:
            raw = m.group(0).replace(" ", "").replace("-", "")
            words = [self._int_to_french(0), self._int_to_french(0),
                     self._int_to_french(221)]
            words.append(_verbalize_9digits(raw[5:]))
            return " ".join(words)

        def _replace_local(m: re.Match) -> str:
            raw = m.group(0).replace(" ", "").replace("-", "")
            return _verbalize_9digits(raw)

        text = self._RE_PHONE_INTL_PLUS.sub(_replace_intl_plus, text)
        text = self._RE_PHONE_INTL_00.sub(_replace_intl_00, text)
        text = self._RE_PHONE_LOCAL.sub(_replace_local, text)
        return text

    def _normalize_ordinals(self, text: str) -> str:
        """Rewrite ordinal numbers to spoken Wolof.

        The first three ordinals have dedicated Wolof words; all others
        use the Wolof cardinal + "eelu" suffix.
        E.g. "1er" -> "wërëkk", "2e" -> "ñaareelu", "5e" -> "juróomeelu".
        """
        def _replace(m: re.Match) -> str:
            full = m.group(0)
            if full in self._ORDINALS:
                return self._ORDINALS[full]
            num = int(m.group(1))
            return f"{self._int_to_wolof(num)}eelu"

        return self._RE_ORDINAL.sub(_replace, text)

    def _normalize_decimals(self, text: str) -> str:
        """Rewrite decimal numbers to spoken Wolof.

        E.g. "3.14" -> "ñett puwaan benn ñeent".
        """
        def _replace(m: re.Match) -> str:
            return self._decimal_to_wolof(m.group(1), m.group(2))

        return self._RE_DECIMAL.sub(_replace, text)

    def _normalize_integers(self, text: str) -> str:
        """Rewrite all remaining bare integers to Wolof words.

        E.g. "2025" -> "ñaar junni ñaar-fukk ak juróom".
        """
        return self._RE_INTEGER.sub(
            lambda m: self._int_to_wolof(int(m.group(1))), text
        )

    def _remove_special_characters(self, text: str) -> str:
        """Remove or replace characters that a TTS engine cannot pronounce.

        Grouping / quoting characters  -> removed (words flow together).
        Ellipsis                        -> single space (pause).
        Semicolon                       -> comma (pause marker).
        Em/en dash                      -> space (pause).
        Orphan colon                    -> comma.
        Remaining currency glyphs       -> removed.
        Symbols with no spoken value    -> removed.
        Orphan slash                    -> removed.
        Orphan percent sign             -> removed.
        """
        text = self._RE_GROUPING.sub("", text)
        text = self._RE_ELLIPSIS.sub(" ", text)
        text = self._RE_PAUSE_PUNCT.sub(",", text)
        text = self._RE_DASH.sub(" ", text)
        text = self._RE_ORPHAN_COLON.sub(",", text)
        text = self._RE_RESIDUAL_GLYPHS.sub("", text)
        text = self._RE_SYMBOLS.sub("", text)
        text = self._RE_SLASH.sub("", text)
        text = self._RE_ORPHAN_PERCENT.sub("", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse runs of spaces/tabs to a single space and strip ends."""
        return self._RE_WHITESPACE.sub(" ", text).strip()

    def _lowercase_text(self, text: str) -> str:
        """Convert text to lowercase."""
        return text.lower()


# ---------------------------------------------------------------------------
# Quick smoke-test  (run with:  python verbalizer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verbalizer = WolofTTSVerbalizer()

    examples = [
        # integers & thousands separators
        ("integers",      "Dafa am 1 000 000 FCFA ci këram."),
        # currency glyph
        ("currency glyph","Prix bi mooy $50 wala €9,99."),
        # percentage
        ("percentage",    "Dafa yore 25% ci suñu xam-xam."),
        # phone
        ("phone intl",    "Téléphone bi mooy +221772903390."),
        # ordinals
        ("ordinals",      "Jëm na ci 1er rang, te topp na 3e yoon."),
        # decimals
        ("decimals",      "Tëmperatur bi dafa 36.6 degrés."),
        # special chars
        ("special chars", "Dafa naan (attaya) ak «café» — waaw, dëkk bi!"),
        # mixed
        ("mixed",         "Nit ki dem na Dakar ci 14/07/2024 ci 09:15, ak 3e klas, ci 45000 FCFA."),
    ]

    for label, raw in examples:
        result = verbalizer.verbalize(raw)
        print(f"[{label}]")
        print(f"  IN : {raw}")
        print(f"  OUT: {result}")
        print()