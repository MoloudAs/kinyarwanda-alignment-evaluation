"""Builds the Manual-MFA-WebMAUS word-level evaluation dataset.

This module is responsible for constructing the canonical evaluation
DataFrame. It finds corresponding TextGrid files, loads the required
tiers, matches the same lexical sequence across Manual, MFA, and
WebMAUS, creates one row per matched word, and validates the final data.
"""

from pathlib import Path

import pandas as pd

from .file_utils import (
    extract_file_id,
    extract_gender,
    extract_recording_location,
)
from .matching import find_unique_sequence_start, normalized_labels
from .textgrid_io import (
    get_required_tier,
    get_single_manual_tier,
    load_textgrid,
    nonempty_intervals,
)


# The anonymized annotator folders expected in the evaluation data.
DEFAULT_ANNOTATORS = {
    "Annotator_1",
    "Annotator_2",
    "Annotator_3",
    "Annotator_4",
}


class DatasetConstructionError(ValueError):
    """Error raised when the evaluation dataset cannot be built safely."""


def build_id_lookup(paths: list[Path]) -> dict[str, Path]:
    """Map each recording ID to its corresponding TextGrid path.

    The recording ID is extracted from each filename.
    Every file must have a valid ID, and each ID must occur only once.
    """

    lookup = {}

    for path in paths:
        file_id = extract_file_id(path.name)

        # Stop if the filename does not contain the expected recording ID.
        if file_id is None:
            raise DatasetConstructionError(
                f"Could not extract recording ID from: {path}"
            )

        # Recording IDs must be unique within each file collection.
        if file_id in lookup:
            raise DatasetConstructionError(
                f"Duplicate recording ID {file_id}: "
                f"{lookup[file_id]} and {path}"
            )

        lookup[file_id] = path

    return lookup


def collect_evaluation_files(
    manual_reference_dir: str | Path,
    webmaus_dir: str | Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Collect and pair Manual/MFA and WebMAUS TextGrid files.

    The function finds the TextGrids in both input locations,
    builds recording-ID lookups, and checks that both sources
    contain exactly the same recording IDs.
    """

    manual_dir = Path(manual_reference_dir)
    web_dir = Path(webmaus_dir)

    # Manual-reference files may be inside annotator subfolders,
    # so rglob() searches recursively.
    # WebMAUS files are expected directly inside its directory.
    manual_paths = sorted(manual_dir.rglob("*.TextGrid"))
    webmaus_paths = sorted(web_dir.glob("*.TextGrid"))

    manual_lookup = build_id_lookup(manual_paths)
    webmaus_lookup = build_id_lookup(webmaus_paths)

    manual_ids = set(manual_lookup)
    webmaus_ids = set(webmaus_lookup)

    # Both alignment sources must represent exactly the same recordings.
    if manual_ids != webmaus_ids:
        missing_in_webmaus = sorted(manual_ids - webmaus_ids)
        missing_in_manual = sorted(webmaus_ids - manual_ids)

        raise DatasetConstructionError(
            "Manual and WebMAUS recording IDs do not match. "
            f"Missing in WebMAUS: {missing_in_webmaus}. "
            f"Missing in Manual: {missing_in_manual}."
        )

    return manual_lookup, webmaus_lookup


def _build_recording_rows(
    file_id: str,
    manual_path: Path,
    webmaus_path: Path,
    expected_annotators: set[str],
) -> list[dict]:
    """Build all word-level evaluation rows for one recording.

    For one recording, this function:
    - validates the annotator,
    - loads Manual, MFA, and WebMAUS tiers,
    - keeps labelled word intervals,
    - normalizes and matches the lexical sequences,
    - verifies word correspondence,
    - stores timing and metadata for every matched word.
    """

    # The annotator identity is taken from the parent folder
    # containing the manual-reference TextGrid.
    annotator = manual_path.parent.name

    if annotator not in expected_annotators:
        raise DatasetConstructionError(
            f"Unexpected annotator folder {annotator!r} "
            f"for {manual_path.name}."
        )


    # Load the merged reference TextGrid.
    # It contains both the manual word tier and the MFA word tier.
    merged_tg = load_textgrid(manual_path)

    manual_tier, manual_tier_name = get_single_manual_tier(
        merged_tg,
        expected_annotator=annotator,
    )

    mfa_tier = get_required_tier(merged_tg, "wordsMFA")


    # Load the separate WebMAUS TextGrid and its word tier.
    webmaus_tg = load_textgrid(webmaus_path)
    webmaus_tier = get_required_tier(webmaus_tg, "ORT-MAU")


    # Remove empty TextGrid intervals because only labelled words
    # should participate in lexical matching and evaluation.
    manual_words = nonempty_intervals(manual_tier)
    mfa_words = nonempty_intervals(mfa_tier)
    webmaus_words = nonempty_intervals(webmaus_tier)

    if not manual_words:
        raise DatasetConstructionError(
            f"{manual_path.name}: "
            f"{manual_tier_name} contains no labelled intervals."
        )


    # Normalize labels before matching so harmless differences in
    # capitalization, spacing, or apostrophe style do not block a match.
    manual_labels = normalized_labels(manual_words)
    mfa_labels = normalized_labels(mfa_words)
    webmaus_labels = normalized_labels(webmaus_words)


    # Find exactly where the complete manual lexical sequence starts
    # inside the MFA and WebMAUS word sequences.
    # Timing is not used to establish this correspondence.
    mfa_start = find_unique_sequence_start(
        manual_labels,
        mfa_labels,
        source_label="wordsMFA",
    )

    webmaus_start = find_unique_sequence_start(
        manual_labels,
        webmaus_labels,
        source_label="ORT-MAU",
    )


    # Build one dictionary for each matched manual word.
    # The corresponding MFA and WebMAUS intervals are located
    # using the matched sequence start plus the local word index.
    rows = []

    for local_index, manual_interval in enumerate(manual_words):
        mfa_index = mfa_start + local_index
        webmaus_index = webmaus_start + local_index

        mfa_interval = mfa_words[mfa_index]
        webmaus_interval = webmaus_words[webmaus_index]

        manual_label = manual_labels[local_index]
        mfa_label = mfa_labels[mfa_index]
        webmaus_label = webmaus_labels[webmaus_index]

        # Final safety check: all three intervals must refer
        # to exactly the same normalized word.
        if not (manual_label == mfa_label == webmaus_label):
            raise DatasetConstructionError(
                f"{manual_path.name}, word {local_index + 1}: "
                "lexical mismatch after sequence matching."
            )


        # Store recording metadata, lexical information,
        # and start/end times from all three annotation sources.
        row = {
            "filename": manual_path.name,
            "file_id": file_id,
            "annotator": annotator,
            "gender": extract_gender(manual_path.name),
            "recording_location": extract_recording_location(
                manual_path.name
            ),
            "word_index": local_index + 1,
            "word": str(manual_interval.mark).strip(),
            "manual_start": float(manual_interval.minTime),
            "manual_end": float(manual_interval.maxTime),
            "mfa_start": float(mfa_interval.minTime),
            "mfa_end": float(mfa_interval.maxTime),
            "webmaus_start": float(webmaus_interval.minTime),
            "webmaus_end": float(webmaus_interval.maxTime),
        }

        rows.append(row)

    return rows


def build_evaluation_dataframe(
    manual_reference_dir: str | Path,
    webmaus_dir: str | Path,
    expected_files: int | None = 16,
    expected_words: int | None = 3676,
    expected_annotators: set[str] | None = None,
) -> pd.DataFrame:
    """Build the complete canonical word-level evaluation DataFrame.

    The function coordinates the dataset-construction stage:
    it pairs the files, processes each recording, combines all
    word-level rows, sorts them, validates the final structure,
    and returns the completed DataFrame.
    """

    # Find the corresponding Manual/MFA and WebMAUS files.
    manual_lookup, webmaus_lookup = collect_evaluation_files(
        manual_reference_dir,
        webmaus_dir,
    )


    # Check that the expected number of evaluation recordings is present.
    if expected_files is not None:
        if len(manual_lookup) != expected_files:
            raise DatasetConstructionError(
                f"Expected {expected_files} evaluation recordings; "
                f"found {len(manual_lookup)}."
            )


    # Use the project's four default anonymized annotators
    # unless another expected set is explicitly supplied.
    if expected_annotators is None:
        expected_annotators = DEFAULT_ANNOTATORS


    # Process each recording in ID order and combine its word rows.
    rows = []

    for file_id in sorted(manual_lookup):
        manual_path = manual_lookup[file_id]
        webmaus_path = webmaus_lookup[file_id]

        recording_rows = _build_recording_rows(
            file_id,
            manual_path,
            webmaus_path,
            expected_annotators,
        )

        rows.extend(recording_rows)


    # Convert all word dictionaries into one pandas DataFrame
    # and give it a deterministic recording/word order.
    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["file_id", "word_index"]
    ).reset_index(drop=True)


    # Perform final structural and content checks before the
    # DataFrame is accepted as the canonical evaluation dataset.
    validate_canonical_dataframe(
        df,
        expected_files=expected_files,
        expected_words=expected_words,
        expected_annotators=len(expected_annotators),
    )

    return df


def validate_canonical_dataframe(
    df: pd.DataFrame,
    expected_files: int | None = 16,
    expected_words: int | None = 3676,
    expected_annotators: int | None = 4,
) -> None:
    """Validate the structure and contents of the canonical DataFrame.

    The validation checks:
    - required columns,
    - expected word and recording counts,
    - expected annotator count,
    - missing boundary values,
    - invalid start/end times,
    - duplicate word rows,
    - continuous word indices,
    - required filename-derived metadata.
    """


    # Check that every column required by later analysis exists.
    required_columns = [
        "filename",
        "file_id",
        "annotator",
        "gender",
        "recording_location",
        "word_index",
        "word",
        "manual_start",
        "manual_end",
        "mfa_start",
        "mfa_end",
        "webmaus_start",
        "webmaus_end",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise DatasetConstructionError(
            f"Canonical dataframe missing columns: {missing_columns}"
        )


    # Check the expected size of the evaluation dataset.
    if expected_words is not None:
        if len(df) != expected_words:
            raise DatasetConstructionError(
                f"Expected {expected_words} rows; found {len(df)}."
            )

    recording_count = df["file_id"].nunique()

    if expected_files is not None:
        if recording_count != expected_files:
            raise DatasetConstructionError(
                f"Expected {expected_files} recordings; "
                f"found {recording_count}."
            )

    annotator_count = df["annotator"].nunique()

    if expected_annotators is not None:
        if annotator_count != expected_annotators:
            raise DatasetConstructionError(
                f"Expected {expected_annotators} annotators; "
                f"found {annotator_count}."
            )


    # All Manual, MFA, and WebMAUS start/end boundaries must be present.
    boundary_columns = [
        "manual_start",
        "manual_end",
        "mfa_start",
        "mfa_end",
        "webmaus_start",
        "webmaus_end",
    ]

    missing_boundaries = (
        df[boundary_columns]
        .isna()
        .any()
        .any()
    )

    if missing_boundaries:
        raise DatasetConstructionError(
            "Missing boundary values detected."
        )


    # For every alignment source, a word interval must start
    # before or at its end time.
    for source in ["manual", "mfa", "webmaus"]:
        invalid_intervals = (
            df[f"{source}_start"] > df[f"{source}_end"]
        ).sum()

        if invalid_intervals:
            raise DatasetConstructionError(
                f"Found {invalid_intervals} {source} intervals "
                "with start > end."
            )


    # Each recording + word-index combination must identify
    # exactly one row in the canonical dataset.
    duplicate_rows = df.duplicated(
        ["file_id", "word_index"]
    ).any()

    if duplicate_rows:
        raise DatasetConstructionError(
            "Duplicate file_id + word_index rows detected."
        )


    # Within each recording, word indices must form a continuous
    # sequence: 1, 2, 3, ... with no gaps or unexpected numbering.
    for file_id, group in df.groupby("file_id"):
        observed_indices = (
            group
            .sort_values("word_index")["word_index"]
            .tolist()
        )

        expected_indices = list(
            range(1, len(group) + 1)
        )

        if observed_indices != expected_indices:
            raise DatasetConstructionError(
                f"Non-continuous word indices "
                f"in recording {file_id}."
            )


    # Filename-derived metadata must be available for all rows.
    metadata_columns = [
        "gender",
        "recording_location",
    ]

    missing_metadata = (
        df[metadata_columns]
        .isna()
        .any()
        .any()
    )

    if missing_metadata:
        raise DatasetConstructionError(
            "Filename-derived metadata are incomplete."
        )