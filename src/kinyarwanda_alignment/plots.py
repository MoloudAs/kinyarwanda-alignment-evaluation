"""Creates figures for the alignment evaluation.

This module visualizes results that have already been calculated
by the metrics and statistics modules.

It creates:
- a cumulative boundary-error figure,
- a paired recording-level MAE figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_cumulative_edge_error(
    edge_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Create and save the cumulative boundary-error figure.

    For each aligner, the absolute boundary errors are sorted
    from smallest to largest and converted into a cumulative proportion.

    The figure shows what proportion of boundaries have an error
    up to a given number of milliseconds.
    """

    # Prepare the output path and create its parent folder if needed.
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    # Build one cumulative error curve for MFA
    # and one for WebMAUS.
    for aligner in ("MFA", "WebMAUS"):
        aligner_edges = edge_df[
            edge_df["aligner"] == aligner
        ]

        errors = (
            aligner_edges["absolute_error_ms"]
            .to_numpy()
        )

        # Sort errors so the cumulative proportion
        # can be calculated from smallest to largest.
        errors = np.sort(errors)

        ranks = np.arange(
            1,
            len(errors) + 1,
        )

        cumulative = ranks / len(errors)

        ax.step(
            errors,
            cumulative,
            where="post",
            label=aligner,
        )


    # Add labels, title, axis limits, and legend.
    ax.set_xlabel(
        "Absolute word-edge displacement (ms)"
    )

    ax.set_ylabel(
        "Cumulative proportion"
    )

    ax.set_title(
        "Automatic word edges relative to manual annotation"
    )

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.legend()


    # Adjust the layout, save the figure, and close it
    # so the plotting object does not remain open in memory.
    fig.tight_layout()
    fig.savefig(output_path)

    plt.close(fig)


def save_paired_recording_mae(
    per_file_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Create and save the paired recording-level MAE figure.

    Each recording contributes one MFA value and one WebMAUS value.

    The two values are connected by a line so the direction
    of the difference can be seen for every recording.
    """

    # Prepare the output path and create its parent folder if needed.
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(4.2, 3.6)
    )


    # Connect the MFA and WebMAUS values for each recording.
    # Green solid lines mean MFA has the lower error.
    # Pink dashed lines mean WebMAUS has the lower error.
    for row in per_file_df.itertuples(
        index=False
    ):
        if row.MFA_MAE_ms < row.WebMAUS_MAE_ms:
            color = "#009E73"
            linestyle = "-"
        else:
            color = "#CC79A7"
            linestyle = "--"

        ax.plot(
            [0, 1],
            [
                row.MFA_MAE_ms,
                row.WebMAUS_MAE_ms,
            ],
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=0.75,
            zorder=1,
        )


    # Plot the MFA recording-level MAE values on the left.
    ax.scatter(
        np.zeros(len(per_file_df)),
        per_file_df["MFA_MAE_ms"],
        color="#0072B2",
        s=28,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )


    # Plot the WebMAUS recording-level MAE values on the right.
    ax.scatter(
        np.ones(len(per_file_df)),
        per_file_df["WebMAUS_MAE_ms"],
        color="#E69F00",
        s=28,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )


    # Label the two x-axis positions and the y-axis.
    ax.set_xticks(
        [0, 1],
        ["MFA", "WebMAUS"],
    )

    ax.set_ylabel(
        "Recording MAE (ms)"
    )

    ax.set_ylim(bottom=0)


    # Add a light horizontal grid and remove unnecessary frame lines
    # to make the figure easier to read.
    ax.grid(
        axis="y",
        linestyle=":",
        linewidth=0.6,
        color="0.82",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    # Add legend entries that summarize which aligner
    # had the lower error across the 16 recordings.
    ax.plot(
        [],
        [],
        color="#009E73",
        linestyle="-",
        label="MFA lower error (13)",
    )

    ax.plot(
        [],
        [],
        color="#CC79A7",
        linestyle="--",
        label="WebMAUS lower error (3)",
    )

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=7,
        ncol=2,
    )


    # Adjust spacing, save the figure, and close it.
    fig.tight_layout()

    fig.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)