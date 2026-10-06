"""Compares MFA and WebMAUS at the recording level.

This module moves from word-level errors to recording-level summaries.
It gives each recording one mean error value per aligner and then
compares the paired MFA and WebMAUS recording errors statistically.
"""

import pandas as pd
from scipy.stats import wilcoxon


def recording_level_summary(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate one mean error value per recording and aligner.

    The word-level MAEs are grouped by file_id.
    For each recording, the function calculates:
    - number of words,
    - mean MFA word MAE,
    - mean WebMAUS word MAE,
    - the difference WebMAUS - MFA.
    """

    # Group all word rows by recording ID.
    grouped = analysis_df.groupby(
        "file_id",
        as_index=False,
    )

    # Reduce many word-level rows to one summary row per recording.
    per_file = grouped.agg(
        words=("word_index", "count"),
        MFA_MAE_ms=("mfa_word_mae_ms", "mean"),
        WebMAUS_MAE_ms=("webmaus_word_mae_ms", "mean"),
    )

    # Keep recordings in a stable, readable order.
    per_file = per_file.sort_values(
        "file_id"
    )

    # Direct recording-level difference between the two aligners.
    # Positive values mean WebMAUS has the larger error,
    # so MFA performs better for that recording.
    per_file["WebMAUS_minus_MFA_ms"] = (
        per_file["WebMAUS_MAE_ms"]
        - per_file["MFA_MAE_ms"]
    )

    return per_file


def paired_recording_test(
    per_file_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Compare paired recording-level errors with a Wilcoxon test.

    The same recordings are evaluated with both MFA and WebMAUS,
    so the observations are paired.

    The function performs a two-sided Wilcoxon signed-rank test
    and also summarizes the direction and size of the paired differences.
    """

    # Extract the two paired recording-level error series.
    mfa_errors = per_file_df["MFA_MAE_ms"]
    webmaus_errors = per_file_df["WebMAUS_MAE_ms"]

    # Perform the paired non-parametric statistical comparison.
    # The two-sided test checks whether the paired differences
    # are systematically different from zero.
    test_result = wilcoxon(
        mfa_errors,
        webmaus_errors,
        alternative="two-sided",
        zero_method="wilcox",
    )

    differences = per_file_df["WebMAUS_minus_MFA_ms"]

    # Count how often each aligner has the lower recording-level MAE.
    mfa_lower = (differences > 0).sum()
    webmaus_lower = (differences < 0).sum()
    ties = (differences == 0).sum()

    # Collect the statistical result and descriptive paired comparison
    # into one dictionary for saving and reporting.
    results = {
        "N_paired_recordings": int(len(per_file_df)),
        "wilcoxon_statistic": float(test_result.statistic),
        "wilcoxon_p": float(test_result.pvalue),
        "mean_WebMAUS_minus_MFA_ms": float(
            differences.mean()
        ),
        "median_WebMAUS_minus_MFA_ms": float(
            differences.median()
        ),
        "MFA_lower_MAE_recordings": int(mfa_lower),
        "WebMAUS_lower_MAE_recordings": int(webmaus_lower),
        "ties": int(ties),
    }

    return results