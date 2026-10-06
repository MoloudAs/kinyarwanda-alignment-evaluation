"""Provides simple helpers for reviewing large alignment errors.

This module is used for diagnostics after the main error measures
have already been calculated.

It helps answer two practical questions:
- How common are very large boundary errors?
- Which individual words have the largest alignment errors?
"""

import pandas as pd


def large_error_summary(
    edge_df: pd.DataFrame,
    thresholds: tuple[int, ...] = (100, 250, 500),
) -> pd.DataFrame:
    """Count large boundary errors separately for MFA and WebMAUS.

    For each aligner, the function checks how many individual
    word boundaries have an absolute error greater than each threshold.

    The default thresholds are 100, 250, and 500 milliseconds.
    Both the count and percentage are reported.
    """

    rows = []

    # Create one diagnostic summary for MFA and one for WebMAUS.
    for aligner in ("MFA", "WebMAUS"):
        aligner_edges = edge_df[
            edge_df["aligner"] == aligner
        ]

        result = {
            "aligner": aligner,
            "word_edges": len(aligner_edges),
        }

        # Check the frequency of large errors at each threshold.
        # The comparison is strictly greater than the threshold.
        for threshold in thresholds:
            large_errors = (
                aligner_edges["absolute_error_ms"] > threshold
            )

            count = int(large_errors.sum())

            percentage = float(
                100 * count / len(aligner_edges)
            )

            result[f"n_gt_{threshold}ms"] = count
            result[f"pct_gt_{threshold}ms"] = percentage

        rows.append(result)

    return pd.DataFrame(rows)


def top_error_words(
    analysis_df: pd.DataFrame,
    top_n_per_aligner: int = 10,
) -> pd.DataFrame:
    """Return the words with the largest word-level MAE for each aligner.

    The function selects the highest-error words separately for MFA
    and WebMAUS, then combines them into one diagnostic table.

    By default, the 10 largest-error words are returned per aligner.
    """

    frames = []

    # Use the relevant word-level MAE column for each aligner.
    for aligner, error_column in (
        ("MFA", "mfa_word_mae_ms"),
        ("WebMAUS", "webmaus_word_mae_ms"),
    ):
        selected_columns = [
            "file_id",
            "word_index",
            "word",
            error_column,
        ]

        # Select the rows with the largest word-level errors.
        largest_errors = analysis_df[
            selected_columns
        ].nlargest(
            top_n_per_aligner,
            error_column,
        )

        # Rename the aligner-specific error column so both
        # aligners can be stored in one common output table.
        largest_errors = largest_errors.rename(
            columns={
                error_column: "word_MAE_ms"
            }
        ).copy()

        largest_errors["aligner"] = aligner

        # Keep a consistent column order for the final diagnostic table.
        largest_errors = largest_errors[
            [
                "aligner",
                "file_id",
                "word_index",
                "word",
                "word_MAE_ms",
            ]
        ]

        frames.append(largest_errors)

    # Combine the MFA and WebMAUS top-error tables.
    return pd.concat(
        frames,
        ignore_index=True,
    )