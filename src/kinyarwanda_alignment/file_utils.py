"""Utility functions for extracting filename metadata and normalizing word labels.

This module supports the dataset-building and matching stages.
It extracts recording metadata from filenames and standardizes
word labels before lexical comparison.
"""

import re
import unicodedata
from pathlib import Path


# Different apostrophe symbols that may occur in TextGrid labels.
APOSTROPHE_VARIANTS = ("’", "‘", "ʼ", "ʹ", "`", "´")


def extract_file_id(filename: str | Path) -> str | None:
    """Extract the three-digit recording ID from a project filename.

    The ID is expected to appear between underscores, for example:
    "_051_".

    Returns the ID as a string so leading zeros are preserved.
    Returns None if no matching ID is found.
    """

    name = Path(filename).name

    match = re.search(r"_([0-9]{3})_", name)

    if match:
        return match.group(1)

    return None


def extract_gender(filename: str | Path) -> str | None:
    """Extract the speaker gender from the filename.

    The filename is expected to contain F or M before the
    three-digit recording ID.

    Returns "Female", "Male", or None.
    """

    name = Path(filename).name

    match = re.search(r"_([FM])_[0-9]{3}_", name)

    if not match:
        return None

    gender_code = match.group(1)

    if gender_code == "F":
        return "Female"

    if gender_code == "M":
        return "Male"

    return None


def extract_recording_location(filename: str | Path) -> str | None:
    """Extract the recording location from the project filename.

    The known location labels in this dataset are
    "Kigali" and "UR-CE".
    """

    name = Path(filename).name

    if "_Kigali_" in name:
        return "Kigali"

    if "_UR-CE_" in name:
        return "UR-CE"

    return None


def normalize_word(text: object) -> str:
    """Standardize a word label before lexical matching.

    Normalization removes irrelevant formatting differences between
    alignment systems while preserving the actual lexical content.

    It:
    - normalizes Unicode representation,
    - standardizes apostrophe symbols,
    - removes extra whitespace,
    - ignores capitalization differences.

    Accents and diacritics are preserved.
    """

    word = str(text)

    # Standardize Unicode representation without removing accents.
    word = unicodedata.normalize("NFC", word)

    # Standardize different apostrophe characters.
    for apostrophe in APOSTROPHE_VARIANTS:
        word = word.replace(apostrophe, "'")

    # Collapse repeated or unnecessary whitespace.
    word = " ".join(word.split())

    # Make capitalization irrelevant for matching.
    word = word.casefold()

    return word