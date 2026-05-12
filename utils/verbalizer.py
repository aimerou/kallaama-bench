import re
from typing import Dict, List, Tuple


class WolofTTSVerbalizer:
    """
    A verbalizer for converting Wolof text into speech-ready format.
    Handles numbers, dates, times, currencies, and other special formats.
    """

    def __init__(self):
        # Wolof number words (0-20, tens, hundreds, thousands)
        self.units = {
            0: "tus",
            1: "benn",
            2: "ñaar",
            3: "ñett",
            4: "ñeent",
            5: "juróom",
            6: "juróom-benn",
            7: "juróom-ñaar",
            8: "juróom-ñett",
            9: "juróom-ñeent",
            10: "fukk",
            11: "fukk ak benn",
            12: "fukk ak ñaar",
            13: "fukk ak ñett",
            14: "fukk ak ñeent",
            15: "fukk ak juróom",
            16: "fukk ak juróom-benn",
            17: "fukk ak juróom-ñaar",
            18: "fukk ak juróom-ñett",
            19: "fukk ak juróom-ñeent",
        }

        self.tens = {
            20: "ñaar-fukk",
            30: "ñett-fukk",
            40: "ñeent-fukk",
            50: "juróom-fukk",
            60: "juróom-benn-fukk",
            70: "juróom-ñaar-fukk",
            80: "juróom-ñett-fukk",
            90: "juróom-ñeent-fukk",
        }

        # Wolof month names
        self.months = {
            1: "Samwiye", 2: "Fewriye", 3: "Mars",
            4: "Awril", 5: "Me", 6: "Suwe",
            7: "Sulet", 8: "Ut", 9: "Sàttumbar",
            10: "Oktoobar", 11: "Nowembar", 12: "Desàmbar"
        }

        # Wolof day names
        self.days = {
            1: "Altine", 2: "Talaata", 3: "Àllarba",
            4: "Alxames", 5: "Àjjuma", 6: "Gaawu", 7: "Dibéer"
        }

        # Currency codes and their Wolof equivalents
        self.currencies = {
            "XOF": "franc CFA",  # West African CFA franc
            "FCFA": "franc CFA",
            "EUR": "euro",
            "USD": "dollar",
            "GBP": "pound",
        }

        # Ordinal indicators
        self.ordinals = {
            "1er": "wërëkk",
            "1ère": "wërëkk",
            "2e": "ñaareelu",
            "3e": "ñetteelu",
        }

        # French number words
        self.french_units = {
            0: "zéro",
            1: "un",
            2: "deux",
            3: "trois",
            4: "quatre",
            5: "cinq",
            6: "six",
            7: "sept",
            8: "huit",
            9: "neuf",
            10: "dix",
            11: "onze",
            12: "douze",
            13: "treize",
            14: "quatorze",
            15: "quinze",
            16: "seize",
        }

        self.french_tens = {
            10: "dix",
            20: "vingt",
            30: "trente",
            40: "quarante",
            50: "cinquante",
            60: "soixante",
        }

    def number_to_words(self, num: int) -> str:
        """
        Convert a number to its Wolof word representation.

        Args:
            num: Integer to convert

        Returns:
            Wolof word representation of the number
        """
        if num == 0:
            return self.units[0]

        if num < 0:
            return f"mënus {self.number_to_words(abs(num))}"

        if num < 20:
            return self.units[num]

        if num < 100:
            tens_digit = (num // 10) * 10
            units_digit = num % 10
            if units_digit == 0:
                return self.tens[tens_digit]
            else:
                return f"{self.tens[tens_digit]} ak {self.units[units_digit]}"

        if num < 1000:
            hundreds = num // 100
            remainder = num % 100

            if hundreds == 1:
                hundred_word = "téeméer"
            else:
                hundred_word = f"{self.units[hundreds]} téeméer"

            if remainder == 0:
                return hundred_word
            else:
                return f"{hundred_word} {self.number_to_words(remainder)}"

        if num < 1000000:
            thousands = num // 1000
            remainder = num % 1000

            if thousands == 1:
                thousand_word = "junni"
            else:
                thousand_word = f"{self.number_to_words(thousands)} junni"

            if remainder == 0:
                return thousand_word
            else:
                return f"{thousand_word} {self.number_to_words(remainder)}"

        if num < 1000000000:
            millions = num // 1000000
            remainder = num % 1000000

            if millions == 1:
                million_word = "tamndareet"
            else:
                million_word = f"{self.number_to_words(millions)} tamndareet"

            if remainder == 0:
                return million_word
            else:
                return f"{million_word} {self.number_to_words(remainder)}"

        return str(num)  # Fallback for very large numbers

    def french_number_to_words(self, num: int) -> str:
        """
        Convert a number to its French word representation.
        Used primarily for currency amounts in French contexts.

        Args:
            num: Integer to convert

        Returns:
            French word representation of the number
        """
        if num == 0:
            return self.french_units[0]

        if num < 0:
            return f"moins {self.french_number_to_words(abs(num))}"

        if num <= 16:
            return self.french_units[num]

        if num < 20:
            # 17-19: dix-sept, dix-huit, dix-neuf
            return f"dix-{self.french_units[num - 10]}"

        if num < 70:
            tens_digit = (num // 10) * 10
            units_digit = num % 10

            if units_digit == 0:
                return self.french_tens[tens_digit]
            elif units_digit == 1 and tens_digit != 10:
                # 21, 31, 41, 51, 61: vingt et un, trente et un, etc.
                return f"{self.french_tens[tens_digit]} et un"
            else:
                return f"{self.french_tens[tens_digit]}-{self.french_units[units_digit]}"

        if num < 80:
            # 70-79: soixante-dix, soixante et onze, etc.
            remainder = num - 60
            if remainder <= 16:
                if remainder == 10:
                    return "soixante-dix"
                elif remainder == 11:
                    return "soixante et onze"
                elif remainder <= 16:
                    return f"soixante-{self.french_units[remainder]}"
            else:
                # 77-79
                return f"soixante-dix-{self.french_units[remainder - 10]}"

        if num < 100:
            # 80-99: quatre-vingts, quatre-vingt-un, etc.
            units_digit = num % 80
            if units_digit == 0:
                return "quatre-vingts"
            else:
                return f"quatre-vingt-{self.french_number_to_words(units_digit)}"

        if num < 1000:
            hundreds = num // 100
            remainder = num % 100

            if hundreds == 1:
                hundred_word = "cent"
            else:
                hundred_word = f"{self.french_units[hundreds]} cent"
                if remainder == 0:
                    hundred_word += "s"  # cents when plural and no remainder

            if remainder == 0:
                return hundred_word
            else:
                return f"{hundred_word} {self.french_number_to_words(remainder)}"

        if num < 1000000:
            thousands = num // 1000
            remainder = num % 1000

            if thousands == 1:
                thousand_word = "mille"
            else:
                thousand_word = f"{self.french_number_to_words(thousands)} mille"

            if remainder == 0:
                return thousand_word
            else:
                return f"{thousand_word} {self.french_number_to_words(remainder)}"

        if num < 1000000000:
            millions = num // 1000000
            remainder = num % 1000000

            if millions == 1:
                million_word = "un million"
            else:
                million_word = f"{self.french_number_to_words(millions)} millions"

            if remainder == 0:
                return million_word
            else:
                return f"{million_word} {self.french_number_to_words(remainder)}"

        return str(num)  # Fallback for very large numbers

    def verbalize_decimal(self, text: str) -> str:
        """
        Verbalize decimal numbers.

        Args:
            text: Decimal number as string (e.g., "3.14")

        Returns:
            Verbalized decimal number
        """
        parts = text.split('.')

        if len(parts) != 2:
            return text

        integer_part = int(parts[0]) if parts[0] else 0
        decimal_part = parts[1]

        result = self.number_to_words(integer_part)
        result += " puwaan"  # "point" in Wolof

        # Read each decimal digit individually
        for digit in decimal_part:
            result += f" {self.units[int(digit)]}"

        return result

    def verbalize_currency(self, amount: str, currency: str) -> str:
        """
        Verbalize currency amounts.

        Args:
            amount: Amount as string
            currency: Currency code

        Returns:
            Verbalized currency amount
        """
        # Handle decimal amounts
        if '.' in amount or ',' in amount:
            amount = amount.replace(',', '.')
            parts = amount.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 and parts[1] else 0

            result = self.french_number_to_words(major)
            currency_name = self.currencies.get(currency.upper(), currency)
            result += f" {currency_name}"

            if minor > 0:
                result += f" ak {self.french_number_to_words(minor)} centime"

            return result
        else:
            amount_num = int(amount)
            currency_name = self.currencies.get(currency.upper(), currency)
            return f"{self.french_number_to_words(amount_num)} {currency_name}"

    def verbalize_date(self, day: int, month: int, year: int) -> str:
        """
        Verbalize dates in Wolof.

        Args:
            day: Day of month
            month: Month number (1-12)
            year: Year

        Returns:
            Verbalized date
        """
        day_word = self.french_number_to_words(day)
        month_word = self.months.get(month, str(month))
        year_word = self.french_number_to_words(year)

        return f"{day_word} {month_word} {year_word}"

    def verbalize_time(self, hours: int, minutes: int) -> str:
        """
        Verbalize time in Wolof.

        Args:
            hours: Hour (0-23)
            minutes: Minutes (0-59)

        Returns:
            Verbalized time
        """
        hour_word = self.number_to_words(hours)

        if minutes == 0:
            return f"{hour_word} waxtu"  # "o'clock"
        else:
            minute_word = self.number_to_words(minutes)
            return f"{hour_word} waxtu ak {minute_word} simili"  # hours and minutes

    def verbalize_phone_number(self, phone: str) -> str:
        """
        Verbalize phone numbers using French numbers.
        Supports formats:
        - +221772903390 (international format with country code)
        - 772903390 (local Senegalese format)

        Grouping pattern for Senegalese numbers (9 digits):
        - XX XXX XX XX (e.g., 77 290 33 90)
        - First group: 2 digits
        - Second group: 3 digits (read as hundreds)
        - Third group: 2 digits
        - Fourth group: 2 digits

        Args:
            phone: Phone number string

        Returns:
            Verbalized phone number using French numbers
        """
        # Remove spaces, dashes, and other separators
        phone = phone.replace(' ', '').replace('-', '').replace('.', '')

        result_parts = []

        # Handle international format with +221 (Senegal country code)
        if phone.startswith('+221'):
            result_parts.append("plus")
            result_parts.append(self.french_number_to_words(221))
            phone = phone[4:]  # Remove +221
        elif phone.startswith('00221'):
            result_parts.append(self.french_number_to_words(0))
            result_parts.append(self.french_number_to_words(0))
            result_parts.append(self.french_number_to_words(221))
            phone = phone[5:]  # Remove 00221

        # Process Senegalese 9-digit number in pattern: XX XXX XX XX
        if len(phone) == 9:
            # First group: 2 digits (77)
            result_parts.append(self.french_number_to_words(int(phone[0:2])))
            # Second group: 3 digits (290)
            result_parts.append(self.french_number_to_words(int(phone[2:5])))
            # Third group: 2 digits (33)
            result_parts.append(self.french_number_to_words(int(phone[5:7])))
            # Fourth group: 2 digits (90)
            result_parts.append(self.french_number_to_words(int(phone[7:9])))
        else:
            # Fallback: process in pairs for non-standard lengths
            i = 0
            while i < len(phone):
                if i + 2 <= len(phone):
                    two_digits = phone[i:i+2]
                    num = int(two_digits)
                    result_parts.append(self.french_number_to_words(num))
                    i += 2
                else:
                    digit = int(phone[i])
                    result_parts.append(self.french_number_to_words(digit))
                    i += 1

        return ' '.join(result_parts)

    def verbalize_percentage(self, value: str) -> str:
        """
        Verbalize percentages.

        Args:
            value: Percentage value

        Returns:
            Verbalized percentage
        """
        if '.' in value or ',' in value:
            verbalized_num = self.verbalize_decimal(value.replace(',', '.'))
        else:
            verbalized_num = self.number_to_words(int(value))

        return f"{verbalized_num} poorso"  # "percent" in Wolof

    def verbalize_ordinal(self, text: str) -> str:
        """
        Verbalize ordinal numbers.

        Args:
            text: Ordinal text (e.g., "1er", "2e")

        Returns:
            Verbalized ordinal
        """
        if text in self.ordinals:
            return self.ordinals[text]

        # Extract number from ordinal
        num_match = re.match(r'(\d+)', text)
        if num_match:
            num = int(num_match.group(1))
            return f"{self.number_to_words(num)}eelu"

        return text

    def normalize_text(self, text: str) -> str:
        """
        Main method to normalize text for TTS.
        Converts numbers, dates, currencies, etc. to verbal form.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text ready for speech synthesis
        """
        # Make a copy to work with
        result = text

        # Verbalize percentages (e.g., "25%")
        result = re.sub(
            r'(\d+(?:[.,]\d+)?)\s*%',
            lambda m: self.verbalize_percentage(m.group(1)),
            result
        )

        # Verbalize currency amounts (e.g., "5000 FCFA", "50 EUR")
        result = re.sub(
            r'(\d+(?:[.,]\d+)?)\s*([A-Z]{3,4})',
            lambda m: self.verbalize_currency(m.group(1), m.group(2)),
            result
        )

        # Verbalize dates in format DD/MM/YYYY or DD-MM-YYYY
        result = re.sub(
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            lambda m: self.verbalize_date(int(m.group(1)), int(m.group(2)), int(m.group(3))),
            result
        )

        # Verbalize time in format HH:MM
        result = re.sub(
            r'(\d{1,2}):(\d{2})',
            lambda m: self.verbalize_time(int(m.group(1)), int(m.group(2))),
            result
        )

        # Verbalize phone numbers (must come before other number patterns)
        # Match international format: +221... or 00221...
        result = re.sub(
            r'\+221\d{9}',
            lambda m: self.verbalize_phone_number(m.group(0)),
            result
        )
        result = re.sub(
            r'\b00221\d{9}\b',
            lambda m: self.verbalize_phone_number(m.group(0)),
            result
        )
        # Match local Senegalese format: 9 digits starting with 7 or 3
        result = re.sub(
            r'\b[73]\d{8}\b',
            lambda m: self.verbalize_phone_number(m.group(0)),
            result
        )

        # Verbalize ordinals (1er, 2e, etc.)
        result = re.sub(
            r'\d+(?:er|ère|e)',
            lambda m: self.verbalize_ordinal(m.group(0)),
            result
        )

        # Verbalize decimal numbers
        result = re.sub(
            r'\b(\d+[.,]\d+)\b',
            lambda m: self.verbalize_decimal(m.group(1).replace(',', '.')),
            result
        )

        # Verbalize standalone numbers
        result = re.sub(
            r'\b(\d+)\b',
            lambda m: self.number_to_words(int(m.group(1))),
            result
        )

        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def verbalize(self, text: str) -> str:
        """
        Public method to verbalize text. Alias for normalize_text.

        Args:
            text: Input text

        Returns:
            Verbalized text
        """
        return self.normalize_text(text)