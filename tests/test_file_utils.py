"""Tests for filename metadata extraction and conservative word normalization.

These tests check that file_utils.py:
- extracts recording IDs correctly,
- preserves leading zeros,
- handles filenames without a valid project ID,
- extracts speaker/location metadata,
- normalizes labels without removing lexical information.
"""

from pathlib import Path
import sys


# Add src/ to Python's search path so this test file
# can import the project package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from kinyarwanda_alignment.file_utils import (
    extract_file_id,
    extract_gender,
    extract_recording_location,
    normalize_word,
)


def test_extract_file_id_preserves_leading_zero():
    """Check that the three-digit recording ID is returned as a string.

    This is important because an ID such as 051 must remain "051"
    rather than being converted to the number 51.
    """

    filename = "2024-10-26_F_051_CookSickFootballParty_UR-CE_muted.TextGrid"

    assert extract_file_id(filename) == "051"


def test_extract_file_id_returns_none_without_project_id():
    """Check that filenames without the expected ID pattern return None.

    The function should not guess or invent a recording ID.
    """

    assert extract_file_id("example.TextGrid") is None


def test_filename_metadata():
    """Check that gender and recording location are extracted correctly.

    The controlled filename contains:
    - F -> Female
    - UR-CE -> recording location
    """

    filename = "2024-10-26_F_051_CookSickFootballParty_UR-CE_muted.TextGrid"

    assert extract_gender(filename) == "Female"
    assert extract_recording_location(filename) == "UR-CE"


def test_normalize_word_is_conservative():
    """Check that normalization removes only irrelevant differences.

    The test confirms that normalization:
    - removes extra spaces,
    - standardizes the apostrophe,
    - ignores capitalization,
    - preserves the accented character á.
    """

    assert normalize_word("  Á’B  ") == "á'b"