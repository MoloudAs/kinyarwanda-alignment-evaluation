"""Functions for matching word sequences across alignment systems.

This module prepares word labels for comparison and finds where
the manual word sequence occurs inside an automatic alignment.

The matching is based only on lexical identity and word order,
not on timing.
"""

from collections.abc import Sequence

from .file_utils import normalize_word


class SequenceMatchError(ValueError):
    """Error raised when a word sequence cannot be matched exactly once."""


def find_contiguous_sequence(
    target_labels: Sequence[str],
    candidate_labels: Sequence[str],
) -> list[int]:
    """Find all starting positions where the target sequence occurs.

    The function compares the target and candidate sequences
    by word labels and order.

    It returns a list because the same sequence may occur
    zero, one, or multiple times.
    """

    target = list(target_labels)
    candidate = list(candidate_labels)

    n_target = len(target)
    n_candidate = len(candidate)

    # No match is possible if the target is empty
    # or longer than the candidate sequence.
    if n_target == 0 or n_target > n_candidate:
        return []

    starts = []

    # Slide across the candidate sequence and compare
    # each possible contiguous section with the target.
    # Timing is deliberately not used here.
    for start in range(n_candidate - n_target + 1):
        candidate_part = candidate[start : start + n_target]

        if candidate_part == target:
            starts.append(start)

    return starts


def normalized_labels(intervals: Sequence) -> list[str]:
    """Return normalized word labels from TextGrid intervals.

    Each interval's label is passed through normalize_word()
    so small formatting differences do not prevent lexical matching.
    """

    return [
        normalize_word(interval.mark)
        for interval in intervals
    ]


def find_unique_sequence_start(
    target_labels: Sequence[str],
    candidate_labels: Sequence[str],
    source_label: str,
) -> int:
    """Return the start position of one unique sequence match.

    The function first finds all possible matches.

    Exactly one match is required:
    - zero matches means the sequence was not found;
    - more than one match means the correspondence is ambiguous.

    In either case, SequenceMatchError is raised.
    """

    starts = find_contiguous_sequence(
        target_labels,
        candidate_labels,
    )

    if len(starts) != 1:
        raise SequenceMatchError(
            f"Expected exactly one complete manual sequence in "
            f"{source_label}; found {len(starts)}."
        )

    return starts[0]