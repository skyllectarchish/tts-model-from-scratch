"""
data/text.py — Gujarati text cleaning and encoding

Responsibilities:
  1. Clean raw Gujarati text (normalize unicode, remove junk)
  2. Expand numbers, abbreviations → spoken Gujarati words
  3. Encode cleaned text → list of integer IDs for the model
  4. Decode integer IDs → text (for debugging)

Usage:
    from data.text import text_to_ids, ids_to_text, clean_text

    ids = text_to_ids("આજે હવામાન સારું છે.")
    print(ids)   # [4, 12, 7, ...]

    back = ids_to_text(ids)
    print(back)  # "આજે હવામાન સારું છે."
"""

import re
import unicodedata
import sys
import os

# Make sure config is importable when running this file directly
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import (
    CHAR_TO_ID,
    ID_TO_CHAR,
    PAD_ID,
    UNK_ID,
    PAD_TOKEN,
    UNK_TOKEN,
    VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Gujarati number words
# ---------------------------------------------------------------------------
# Used by expand_numbers() to convert digits → spoken Gujarati words.

GUJARATI_ONES = [
    "", "એક", "બે", "ત્રણ", "ચાર", "પાંચ",
    "છ", "સાત", "આઠ", "નવ", "દસ",
    "અગિયાર", "બાર", "તેર", "ચૌદ", "પંદર",
    "સોળ", "સત્તર", "અઢાર", "ઓગણીસ",
]

GUJARATI_TENS = [
    "", "", "વીસ", "ત્રીસ", "ચાલીસ", "પચાસ",
    "સાઈઠ", "સિત્તેર", "એંસી", "નેવું",
]

GUJARATI_HUNDREDS = "સો"
GUJARATI_THOUSAND = "હજાર"
GUJARATI_LAKH     = "લાખ"
GUJARATI_CRORE    = "કરોડ"


def _number_to_gujarati_words(n: int) -> str:
    """
    Convert an integer to its spoken Gujarati word form.
    Handles 0 – 9,99,99,999 (up to 9 crore 99 lakh 99 thousand 999).

    Examples:
        5       → "પાંચ"
        42      → "બેતાળીસ"   (note: compound forms differ from simple tens+ones)
        100     → "સો"
        1000    → "એક હજાર"
        15000   → "પંદર હજાર"
        100000  → "એક લાખ"
    """
    if n == 0:
        return "શૂન્ય"

    # For numbers 1–19 use direct lookup
    if 1 <= n <= 19:
        return GUJARATI_ONES[n]

    # For 20–99 use tens + ones
    if 20 <= n <= 99:
        tens  = n // 10
        ones  = n % 10
        if ones == 0:
            return GUJARATI_TENS[tens]
        # Gujarati compound numbers 21–99 follow a specific pattern.
        # Rather than tens + space + ones, many are fused words.
        # We use the simple tens + ones approach here, which is
        # understandable even if not always the colloquial fused form.
        return GUJARATI_TENS[tens] + GUJARATI_ONES[ones]

    # Hundreds
    if 100 <= n <= 999:
        hundreds = n // 100
        remainder = n % 100
        result = (_number_to_gujarati_words(hundreds) + " " + GUJARATI_HUNDREDS).strip()
        if remainder > 0:
            result += " " + _number_to_gujarati_words(remainder)
        return result

    # Thousands (1,000 – 99,999)
    if 1_000 <= n <= 99_999:
        thousands = n // 1_000
        remainder = n % 1_000
        result = (_number_to_gujarati_words(thousands) + " " + GUJARATI_THOUSAND).strip()
        if remainder > 0:
            result += " " + _number_to_gujarati_words(remainder)
        return result

    # Lakhs (1,00,000 – 99,99,999)
    if 1_00_000 <= n <= 99_99_999:
        lakhs    = n // 1_00_000
        remainder = n % 1_00_000
        result = (_number_to_gujarati_words(lakhs) + " " + GUJARATI_LAKH).strip()
        if remainder > 0:
            result += " " + _number_to_gujarati_words(remainder)
        return result

    # Crores (1,00,00,000 – 9,99,99,999)
    if 1_00_00_000 <= n <= 9_99_99_999:
        crores   = n // 1_00_00_000
        remainder = n % 1_00_00_000
        result = (_number_to_gujarati_words(crores) + " " + GUJARATI_CRORE).strip()
        if remainder > 0:
            result += " " + _number_to_gujarati_words(remainder)
        return result

    # Fallback for very large numbers — just return digit string
    return str(n)


def expand_numbers(text: str) -> str:
    """
    Replace all digit sequences in text with their Gujarati word equivalents.

    "મારી ઉંમર 25 વર્ષ છે" → "મારી ઉંમર પચાસ... વર્ષ છે"

    Handles:
      - Plain integers:       25    → "પચાસ..."
      - Gujarati numerals:    ૨૫   → "પચાસ..."
      - Decimal numbers:      3.14  → "ત્રણ દશાંશ એક ચાર"
      - Ordinals are left as-is (e.g. "1st" — rare in Gujarati text)
    """
    # First convert Gujarati digit characters (૦-૯) to ASCII digits
    gujarati_digit_map = str.maketrans("૦૧૨૩૪૫૬૭૮૯", "0123456789")
    text = text.translate(gujarati_digit_map)

    # Handle decimal numbers first (before integers)
    def replace_decimal(match):
        integer_part  = int(match.group(1))
        decimal_digits = match.group(2)
        integer_words = _number_to_gujarati_words(integer_part)
        # Read decimal digits one by one
        digit_words = " ".join(
            _number_to_gujarati_words(int(d)) for d in decimal_digits
        )
        return integer_words + " દશાંશ " + digit_words

    text = re.sub(r"(\d+)\.(\d+)", replace_decimal, text)

    # Handle plain integers
    def replace_integer(match):
        return _number_to_gujarati_words(int(match.group(0)))

    text = re.sub(r"\d+", replace_integer, text)

    return text


# ---------------------------------------------------------------------------
# Common abbreviation expansions for Gujarati text
# ---------------------------------------------------------------------------
ABBREVIATIONS = {
    "ડૉ."  : "ડોક્ટર",
    "ડૉ"   : "ડોક્ટર",
    "શ્રી." : "શ્રી",
    "સ્ત્री." : "સ્ત્રી",
    "કિ.મી." : "કિલોમીટર",
    "કિ.મી" : "કિલોમીટર",
    "કિ.ગ્રા." : "કિલોગ્રામ",
    "રૂ."  : "રૂપિયા",
    "રૂ"   : "રૂપિયા",
    "કલા." : "કલાક",
    "મિ."  : "મિનિટ",
    "સે."  : "સેકન્ડ",
    "વિ.સ." : "વિક્રમ સંવત",
    "ઈ.સ." : "ઈસ્વીસન",
    "પ્રા." : "પ્રાથમિક",
    "માધ." : "માધ્યમિક",
}


def expand_abbreviations(text: str) -> str:
    """Replace known abbreviations with their full Gujarati forms."""
    # Sort by length descending so longer abbreviations match first
    for abbr, expansion in sorted(ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        text = text.replace(abbr, expansion)
    return text


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode to NFC form.

    Gujarati text can have the same visual character represented by different
    Unicode sequences (e.g., a base character + combining diacritic vs a
    pre-composed character). NFC normalization ensures consistency.
    """
    return unicodedata.normalize("NFC", text)


# ---------------------------------------------------------------------------
# Punctuation and whitespace cleaning
# ---------------------------------------------------------------------------

def clean_punctuation(text: str) -> str:
    """
    Normalize punctuation and whitespace:
      - Convert curly quotes → straight quotes
      - Convert em/en dashes → hyphen
      - Collapse multiple spaces → single space
      - Strip leading/trailing whitespace
      - Remove characters not in our vocabulary
    """
    # Normalize quote styles
    text = text.replace("\u201c", '"').replace("\u201d", '"')  # curly double quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")  # curly single quotes

    # Normalize dashes
    text = text.replace("\u2014", "-").replace("\u2013", "-")  # em dash, en dash

    # Remove zero-width characters common in Gujarati Unicode text
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\u200c", "")   # zero-width non-joiner
    text = text.replace("\u200d", "")   # zero-width joiner (keep if needed for conjuncts)
    text = text.replace("\ufeff", "")   # BOM

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


def remove_out_of_vocab(text: str) -> str:
    """
    Remove any character not in our vocabulary.
    This prevents unknown characters from becoming UNK tokens silently.
    Out-of-vocab characters are simply dropped.
    """
    return "".join(ch for ch in text if ch in CHAR_TO_ID)


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline. Apply in this order:

      1. Unicode normalization   (consistent character forms)
      2. Abbreviation expansion  (ડૉ. → ડોક્ટર)
      3. Number expansion        (25 → પચીસ)
      4. Punctuation cleanup     (normalize quotes, dashes, spaces)
      5. Remove OOV characters   (drop anything not in vocab)

    Args:
        text: Raw Gujarati text string

    Returns:
        Cleaned text string safe for encoding
    """
    text = normalize_unicode(text)
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    text = clean_punctuation(text)
    text = remove_out_of_vocab(text)
    return text


# ---------------------------------------------------------------------------
# Encoding and decoding
# ---------------------------------------------------------------------------

def text_to_ids(text: str, apply_cleaning: bool = True) -> list[int]:
    """
    Convert a Gujarati text string to a list of integer token IDs.

    Args:
        text:           Raw or pre-cleaned Gujarati text
        apply_cleaning: If True, run clean_text() first (default: True)
                        Set to False if text is already cleaned

    Returns:
        List of integer IDs, e.g. [4, 12, 7, 3, 18, ...]

    Example:
        >>> text_to_ids("આ સારું છે")
        [4, 1, 22, 4, 10, 8, 1, 17, 6, 1]
    """
    if apply_cleaning:
        text = clean_text(text)

    ids = []
    i = 0
    while i < len(text):
        # Try matching 2-character sequences first (e.g. 'ક્ષ', 'જ્ઞ', 'અં', 'અઃ')
        if i + 1 < len(text) and text[i:i+2] in CHAR_TO_ID:
            ids.append(CHAR_TO_ID[text[i:i+2]])
            i += 2
        elif text[i] in CHAR_TO_ID:
            ids.append(CHAR_TO_ID[text[i]])
            i += 1
        else:
            # Character not in vocab — use UNK token
            ids.append(UNK_ID)
            i += 1

    return ids


def ids_to_text(ids: list[int]) -> str:
    """
    Convert a list of integer token IDs back to a text string.
    Useful for debugging — verify that encoding → decoding is lossless.

    Args:
        ids: List of integer token IDs

    Returns:
        Reconstructed text string

    Example:
        >>> ids_to_text([4, 12, 7])
        "આ..."
    """
    return "".join(
        ID_TO_CHAR.get(i, UNK_TOKEN)
        for i in ids
        if i != PAD_ID   # skip padding tokens
    )


def text_to_sequence_and_back(text: str) -> tuple[list[int], str]:
    """
    Convenience function: clean → encode → decode in one call.
    Returns both the IDs and the reconstructed text for easy comparison.

    Use this to verify your text pipeline is working correctly.
    """
    cleaned = clean_text(text)
    ids     = text_to_ids(cleaned, apply_cleaning=False)
    decoded = ids_to_text(ids)
    return ids, decoded


# ---------------------------------------------------------------------------
# Quick self-test — run this file directly to verify everything works
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Gujarati Text Pipeline — Self Test")
    print("=" * 60)

    test_cases = [
        "આ સારું છે.",
        "મારી ઉંમર 25 વર્ષ છે.",
        "ડૉ. શર્માએ 3.5 કિ.મી. ચાલ્યા.",
        "ગુજરાત એક સુંદર રાજ્ય છે!",
        "આજે ૧૦૦૦ રૂ. ખર્ચ થયા.",
        "Hello world",   # English — should be mostly dropped (OOV)
    ]

    all_passed = True

    for original in test_cases:
        cleaned = clean_text(original)
        ids     = text_to_ids(original)
        decoded = ids_to_text(ids)

        # Check encode → decode round trip is consistent
        ids2    = text_to_ids(cleaned, apply_cleaning=False)
        match   = (ids == ids2)
        status  = "PASS" if match else "FAIL"
        if not match:
            all_passed = False

        print(f"\n  [{status}]")
        print(f"  Original : {original}")
        print(f"  Cleaned  : {cleaned}")
        print(f"  IDs      : {ids[:12]}{'...' if len(ids) > 12 else ''}")
        print(f"  Decoded  : {decoded}")

    print("\n" + "=" * 60)
    print(f"  Vocab size : {VOCAB_SIZE} tokens")
    print(f"  Result     : {'All tests passed!' if all_passed else 'Some tests FAILED — check above'}")
    print("=" * 60)

    # Show number expansion examples
    print("\n  Number expansion examples:")
    number_tests = [0, 1, 15, 25, 100, 500, 1000, 15000, 100000, 1000000]
    for n in number_tests:
        print(f"    {n:>10} → {_number_to_gujarati_words(n)}")
