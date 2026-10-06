"""Tests for lexical sequence matching.

These tests check that matching.py:
- finds the correct starting position of a contiguous word sequence,
- rejects ambiguous cases where the same sequence occurs more than once.
"""

from pathlib import Path
import sys


# Add src/ to Python's search path so this test file
# can import the project package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pytest

from kinyarwanda_alignment.matching import (
    SequenceMatchError,
    find_contiguous_sequence,
    find_unique_sequence_start,
)


def test_find_contiguous_sequence():
    """Check that the correct start position is found.

    The target sequence ["b", "c"] occurs once inside
    ["a", "b", "c", "d"], starting at index 1.
    """

    assert find_contiguous_sequence(
        ["b", "c"],
        ["a", "b", "c", "d"],
    ) == [1]


def test_unique_match_rejects_ambiguous_sequence():
    """Check that ambiguous sequence matching is rejected.

    The target ["a"] appears twice in ["a", "a"].
    Because there is not exactly one unique match,
    find_unique_sequence_start() must raise SequenceMatchError.
    """

    with pytest.raises(SequenceMatchError):
        find_unique_sequence_start(
            ["a"],
            ["a", "a"],
            source_label="example",
        )