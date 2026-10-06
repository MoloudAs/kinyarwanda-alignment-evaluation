"""Tests for recording-level aggregation.

This test checks that statistics.py correctly:
- groups word-level errors by recording,
- counts the number of words,
- calculates mean MFA and WebMAUS error per recording,
- calculates the difference WebMAUS - MFA.
"""

from pathlib import Path
import sys

import pandas as pd


# Add src/ to Python's search path so this test file
# can import the project package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from kinyarwanda_alignment.statistics import (
    recording_level_summary,
)


def test_recording_level_summary_calculates_recording_means():
    """Check recording-level means using a small controlled dataset.

    The artificial dataset contains two recordings,
    with two words in each recording.

    For recording 001:
    - MFA mean = (10 + 30) / 2 = 20 ms
    - WebMAUS mean = (30 + 50) / 2 = 40 ms
    - WebMAUS - MFA = 20 ms

    The test verifies that recording_level_summary()
    produces these expected values.
    """

    # Create a simple word-level dataset with known values.
    df = pd.DataFrame(
        {
            "file_id": [
                "001",
                "001",
                "002",
                "002",
            ],
            "word_index": [
                1,
                2,
                1,
                2,
            ],
            "mfa_word_mae_ms": [
                10.0,
                30.0,
                20.0,
                40.0,
            ],
            "webmaus_word_mae_ms": [
                30.0,
                50.0,
                40.0,
                60.0,
            ],
        }
    )


    # Aggregate the word-level values to recording level.
    result = recording_level_summary(df)


    # Select recording 001 from the result.
    first_recording = result[
        result["file_id"] == "001"
    ].iloc[0]


    # Check the number of words and the expected recording-level means.
    assert first_recording["words"] == 2
    assert first_recording["MFA_MAE_ms"] == 20.0
    assert first_recording["WebMAUS_MAE_ms"] == 40.0
    assert first_recording["WebMAUS_minus_MFA_ms"] == 20.0