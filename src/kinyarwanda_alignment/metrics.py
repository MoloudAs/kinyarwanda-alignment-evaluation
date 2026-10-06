"""Calculates word-boundary error measures for MFA and WebMAUS.

This module takes the canonical matched-word DataFrame and calculates
timing errors for both automatic aligners.

It creates:
- word-level error measures,
- boundary-level data,
- overall descriptive summaries,
- whole-word tolerance summaries.
"""

import numpy as np
import pandas as pd


def add_error_columns(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Add signed and absolute boundary-error measures to the word-level data.

    For each aligner, the function compares automatic start/end times
    with the manual reference.

    Errors are converted from seconds to milliseconds.

    It also calculates one word-level MAE by averaging the absolute
    start and end errors.
    """

    # Work on a copy so the original evaluation DataFrame is unchanged.
    df = evaluation_df.copy()

    for aligner in ("mfa", "webmaus"):

        # Signed error = automatic boundary - manual boundary.
        # Positive means the automatic boundary is later.
        # Negative means it is earlier.
        start_error = (
            df[f"{aligner}_start"] - df["manual_start"]
        ) * 1000

        end_error = (
            df[f"{aligner}_end"] - df["manual_end"]
        ) * 1000

        df[f"{aligner}_start_error_ms"] = start_error
        df[f"{aligner}_end_error_ms"] = end_error

        # Absolute errors measure the size of the difference
        # regardless of whether the automatic boundary is early or late.
        df[f"{aligner}_abs_start_error_ms"] = start_error.abs()
        df[f"{aligner}_abs_end_error_ms"] = end_error.abs()

        # Word MAE is the average absolute error of the
        # start and end boundaries of the same word.
        df[f"{aligner}_word_mae_ms"] = (
            df[f"{aligner}_abs_start_error_ms"]
            + df[f"{aligner}_abs_end_error_ms"]
        ) / 2


    # Direct word-level comparison between the two aligners.
    # Positive values mean WebMAUS has the larger error,
    # so MFA is closer to the manual reference for that word.
    df["webmaus_minus_mfa_mae_ms"] = (
        df["webmaus_word_mae_ms"]
        - df["mfa_word_mae_ms"]
    )

    return df


def build_edge_dataframe(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert word-level rows into individual boundary-level rows.

    Each word contributes four rows:
    - MFA start
    - MFA end
    - WebMAUS start
    - WebMAUS end

    This structure makes it easier to calculate summaries
    over all individual word boundaries.
    """

    rows = []

    # Process every matched word.
    for row in analysis_df.itertuples(index=False):

        # Process both automatic alignment systems.
        for aligner, prefix in (
            ("MFA", "mfa"),
            ("WebMAUS", "webmaus"),
        ):

            # Separate the start and end boundary of each word.
            for edge_type in ("start", "end"):
                signed_error = getattr(
                    row,
                    f"{prefix}_{edge_type}_error_ms",
                )

                absolute_error = getattr(
                    row,
                    f"{prefix}_abs_{edge_type}_error_ms",
                )

                rows.append(
                    {
                        "file_id": row.file_id,
                        "word_index": row.word_index,
                        "word": row.word,
                        "aligner": aligner,
                        "edge_type": edge_type,
                        "signed_error_ms": signed_error,
                        "absolute_error_ms": absolute_error,
                    }
                )

    return pd.DataFrame(rows)


def overall_edge_summary(
    edge_df: pd.DataFrame,
    n_words: int,
) -> pd.DataFrame:
    """Calculate overall descriptive boundary-error measures.

    For each aligner, the function reports:
    - number of words and boundaries,
    - mean absolute boundary error,
    - median absolute boundary error,
    - separate start and end MAE,
    - percentage of boundaries within 20, 50, and 100 ms.
    """

    rows = []

    # Produce one summary row for MFA and one for WebMAUS.
    for aligner in ("MFA", "WebMAUS"):
        aligner_edges = edge_df[
            edge_df["aligner"] == aligner
        ]

        absolute_errors = (
            aligner_edges["absolute_error_ms"]
            .to_numpy()
        )

        # Separate start and end boundaries
        # so their mean errors can also be reported individually.
        start_errors = aligner_edges.loc[
            aligner_edges["edge_type"] == "start",
            "absolute_error_ms",
        ]

        end_errors = aligner_edges.loc[
            aligner_edges["edge_type"] == "end",
            "absolute_error_ms",
        ]


        # Boolean arrays identify which individual boundaries
        # fall within the selected tolerance thresholds.
        within_20 = absolute_errors <= 20
        within_50 = absolute_errors <= 50
        within_100 = absolute_errors <= 100


        # Build the complete descriptive summary for this aligner.
        summary = {
            "aligner": aligner,
            "N_words": n_words,
            "N_word_edges": len(absolute_errors),
            "MAE_ms": float(np.mean(absolute_errors)),
            "median_absolute_error_ms": float(
                np.median(absolute_errors)
            ),
            "start_MAE_ms": float(start_errors.mean()),
            "end_MAE_ms": float(end_errors.mean()),
            "edges_within_20ms_pct": float(
                100 * np.mean(within_20)
            ),
            "edges_within_50ms_pct": float(
                100 * np.mean(within_50)
            ),
            "edges_within_100ms_pct": float(
                100 * np.mean(within_100)
            ),
        }

        rows.append(summary)

    return pd.DataFrame(rows)


def complete_word_tolerance_summary(
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the percentage of complete words within each tolerance.

    A word is counted as within a threshold only when BOTH its
    start and end boundaries are within that threshold.

    Thresholds used are 20, 50, and 100 milliseconds.
    """

    rows = []

    # Calculate the same whole-word tolerance measures
    # separately for MFA and WebMAUS.
    for aligner, prefix in (
        ("MFA", "mfa"),
        ("WebMAUS", "webmaus"),
    ):
        result = {
            "aligner": aligner,
        }

        for threshold in (20, 50, 100):

            # Check start and end boundaries separately.
            start_within = (
                analysis_df[f"{prefix}_abs_start_error_ms"]
                <= threshold
            )

            end_within = (
                analysis_df[f"{prefix}_abs_end_error_ms"]
                <= threshold
            )

            # A complete word passes only if both conditions are True.
            both_edges_within = start_within & end_within

            percentage = float(
                100 * both_edges_within.mean()
            )

            result[
                f"words_both_edges_within_{threshold}ms_pct"
            ] = percentage

        rows.append(result)

    return pd.DataFrame(rows)