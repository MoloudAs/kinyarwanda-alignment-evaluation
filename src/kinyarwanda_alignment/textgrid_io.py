"""Functions for loading TextGrid files and accessing their tiers.

This module handles the TextGrid input structure used by the project.
It loads TextGrid files, retrieves required tiers, validates the
manual-reference tier, and removes empty intervals before matching.
"""

from pathlib import Path

import textgrid


class TextGridStructureError(ValueError):
    """Error raised when a TextGrid does not have the expected tier structure."""


def load_textgrid(path: str | Path) -> textgrid.TextGrid:
    """Load a TextGrid file from disk.

    The function first checks that the file exists.
    If the file is missing, it raises FileNotFoundError.
    Otherwise, it returns the TextGrid as a Python object.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TextGrid not found: {path}")

    return textgrid.TextGrid.fromFile(str(path))


def get_required_tier(
    tg: textgrid.TextGrid,
    tier_name: str,
):
    """Return a required tier from a TextGrid.

    The requested tier must exist by name.
    If it is missing, the function raises TextGridStructureError
    and reports the available tier names.
    """

    if tier_name not in tg.getNames():
        raise TextGridStructureError(
            f"Required tier {tier_name!r} not found. "
            f"Available tiers: {tg.getNames()}"
        )

    return tg.getFirst(tier_name)


def get_single_manual_tier(
    tg: textgrid.TextGrid,
    expected_annotator: str | None = None,
):
    """Find and return the single manual-reference tier.

    The function searches for tier names beginning with "manual".
    Exactly one such tier must exist.

    If an expected annotator is provided, the manual tier name is
    also checked against that annotator.

    Returns both the tier object and its tier name.
    """

    manual_names = [
        name
        for name in tg.getNames()
        if name.lower().startswith("manual")
    ]

    # The reference TextGrid must contain exactly one manual tier.
    if len(manual_names) != 1:
        raise TextGridStructureError(
            f"Expected exactly one manual tier, found: {manual_names}"
        )

    manual_name = manual_names[0]

    # If the annotator is known, verify that the tier belongs
    # to the expected anonymized annotator.
    if expected_annotator is not None:
        expected_name = f"manual{expected_annotator}"

        if manual_name.casefold() != expected_name.casefold():
            raise TextGridStructureError(
                f"Expected manual tier {expected_name!r}, "
                f"found {manual_name!r}."
            )

    return tg.getFirst(manual_name), manual_name


def nonempty_intervals(tier) -> list:
    """Return only intervals that contain a non-empty label.

    Blank intervals are ignored because they do not represent
    lexical items that should enter the word-matching stage.
    """

    # If the object is not an interval tier, return an empty list.
    if not hasattr(tier, "intervals"):
        return []

    return [
        interval
        for interval in tier.intervals
        if str(interval.mark).strip()
    ]