"""Tests for calculation of boundary-error measures.

This test checks that metrics.py:
- computes signed boundary errors correctly,
- converts seconds to milliseconds,
- calculates word-level MAE from start and end errors.
"""

from pathlib import Path
import sys


# Add src/ to Python's search path so this test file
# can import the project package.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


import pandas as pd

from kinyarwanda_alignment.metrics import add_error_columns


def test_add_error_columns_converts_seconds_to_milliseconds():
    """Check the main error calculations on a controlled example.

    The test creates one artificial word with known Manual, MFA,
    and WebMAUS boundaries.

    For MFA:
    - start: 1.010 - 1.000 = +0.010 s = +10 ms
    - end:   1.980 - 2.000 = -0.020 s = -20 ms
    - word MAE = (10 + 20) / 2 = 15 ms

    The assertions verify that add_error_columns() produces
    exactly these expected values.
    """

    df = pd.DataFrame(
        {
            "manual_start": [1.0],
            "manual_end": [2.0],
            "mfa_start": [1.010],
            "mfa_end": [1.980],
            "webmaus_start": [0.950],
            "webmaus_end": [2.100],
        }
    )

    result = add_error_columns(df)

    # Positive start error means MFA placed the start boundary later
    # than the manual reference.
    assert round(result.loc[0, "mfa_start_error_ms"], 8) == 10.0

    # Negative end error means MFA placed the end boundary earlier
    # than the manual reference.
    assert round(result.loc[0, "mfa_end_error_ms"], 8) == -20.0

    # Word MAE uses the absolute start and end errors:
    # (10 + 20) / 2 = 15 ms.
    assert round(result.loc[0, "mfa_word_mae_ms"], 8) == 15.0