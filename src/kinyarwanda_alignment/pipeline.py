"""Runs the complete Kinyarwanda alignment evaluation pipeline.

This module coordinates all stages of the analysis.

It connects the dataset, metrics, statistics, diagnostics, and plotting
modules and saves the generated results to the configured output folders.
"""

import json
from pathlib import Path

import pandas as pd

from .dataset import build_evaluation_dataframe
from .diagnostics import large_error_summary, top_error_words
from .metrics import (
    add_error_columns,
    build_edge_dataframe,
    complete_word_tolerance_summary,
    overall_edge_summary,
)
from .plots import (
    save_cumulative_edge_error,
    save_paired_recording_mae,
)
from .statistics import (
    paired_recording_test,
    recording_level_summary,
)


def _save_dataframe(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Save a pandas DataFrame as a UTF-8 CSV file.

    The parent directory is created automatically if it does not exist.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


def run_pipeline(config: dict) -> dict:
    """Run the complete alignment-evaluation workflow.

    The function performs the stages in this order:

    1. Build and validate the canonical word-level dataset.
    2. Calculate boundary errors.
    3. Calculate descriptive summaries.
    4. Compare MFA and WebMAUS at recording level.
    5. Inspect large errors.
    6. Prepare output directories.
    7. Save result tables.
    8. Save the statistical test result.
    9. Create figures.

    It returns the important DataFrames, statistical result,
    and output paths.
    """

    # Read the relevant sections of the configuration.
    inputs = config["inputs"]
    outputs = config["outputs"]

    expected = config.get(
        "expected",
        {},
    )

    analysis_config = config.get(
        "analysis",
        {},
    )

    expected_annotators = set(
        expected["annotators"]
    )


    # 1. Build and validate the canonical matched-word dataset.
    evaluation_df = build_evaluation_dataframe(
        inputs["manual_reference_dir"],
        inputs["webmaus_dir"],
        expected_files=expected.get(
            "evaluation_files",
            16,
        ),
        expected_words=expected.get(
            "manual_words",
            3676,
        ),
        expected_annotators=expected_annotators,
    )

    # Save the canonical dataset before calculating additional measures.
    canonical_csv = Path(
        outputs["canonical_csv"]
    )

    _save_dataframe(
        evaluation_df,
        canonical_csv,
    )


    # 2. Calculate word-level and boundary-level errors.
    analysis_df = add_error_columns(
        evaluation_df
    )

    edge_df = build_edge_dataframe(
        analysis_df
    )


    # 3. Calculate descriptive accuracy summaries.
    overall_df = overall_edge_summary(
        edge_df,
        n_words=len(analysis_df),
    )

    tolerance_df = complete_word_tolerance_summary(
        analysis_df
    )


    # 4. Aggregate to recording level and perform the paired comparison.
    recording_df = recording_level_summary(
        analysis_df
    )

    recording_test = paired_recording_test(
        recording_df
    )


    # 5. Create diagnostic summaries for unusually large errors.
    large_errors_df = large_error_summary(
        edge_df
    )

    top_errors_df = top_error_words(
        analysis_df,
        top_n_per_aligner=analysis_config.get(
            "top_error_words_per_aligner",
            10,
        ),
    )


    # 6. Prepare the output folders.
    tables_dir = Path(
        outputs["tables_dir"]
    )

    figures_dir = Path(
        outputs["figures_dir"]
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # 7. Save the descriptive and diagnostic result tables.
    _save_dataframe(
        overall_df,
        tables_dir / "overall_edge_summary.csv",
    )

    _save_dataframe(
        tolerance_df,
        tables_dir / "word_tolerance_summary.csv",
    )

    _save_dataframe(
        recording_df,
        tables_dir / "recording_level_summary.csv",
    )

    _save_dataframe(
        large_errors_df,
        tables_dir / "large_error_summary.csv",
    )

    _save_dataframe(
        top_errors_df,
        tables_dir / "top_error_words.csv",
    )


    # 8. Save the recording-level statistical result as JSON.
    test_path = (
        tables_dir
        / "recording_level_test.json"
    )

    test_path.write_text(
        json.dumps(
            recording_test,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # 9. Create and save the evaluation figures.
    save_cumulative_edge_error(
        edge_df,
        figures_dir
        / "cumulative_word_edge_error.pdf",
    )

    save_paired_recording_mae(
        recording_df,
        figures_dir
        / "paired_recording_mae.pdf",
    )


    # Return the main intermediate results and output locations.
    return {
        "evaluation_df": evaluation_df,
        "analysis_df": analysis_df,
        "edge_df": edge_df,
        "overall_summary": overall_df,
        "tolerance_summary": tolerance_df,
        "recording_summary": recording_df,
        "recording_test": recording_test,
        "large_error_summary": large_errors_df,
        "top_error_words": top_errors_df,
        "canonical_csv": canonical_csv,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
    }