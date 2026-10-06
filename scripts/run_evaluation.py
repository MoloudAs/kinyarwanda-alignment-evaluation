#!/usr/bin/env python3

"""Command-line entry point for the Kinyarwanda alignment AP project.

This script prepares access to the project package, reads the command-line
configuration option, loads the YAML configuration, runs the complete
evaluation pipeline, and prints a short summary of the generated outputs.
"""

import argparse
import sys
from pathlib import Path


# Define the repository root and add src/ to Python's module search path
# so the kinyarwanda_alignment package can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from kinyarwanda_alignment.config import load_config
from kinyarwanda_alignment.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Read command-line arguments.

    The script accepts an optional --config argument.
    If no path is supplied, the default project configuration
    at config/config.yaml is used.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build and evaluate the canonical Manual–MFA–WebMAUS "
            "Kinyarwanda word-alignment dataset."
        )
    )

    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to the YAML configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the complete evaluation workflow.

    The function:
    1. reads the command-line arguments,
    2. loads and prepares the YAML configuration,
    3. passes the configuration to the main pipeline,
    4. receives the generated results,
    5. prints a short completion summary.
    """

    args = parse_args()

    config = load_config(args.config)

    result = run_pipeline(config)


    # Report the main characteristics of the completed analysis
    # and the locations of the generated outputs.
    print("=" * 72)
    print("KINYARWANDA ALIGNMENT EVALUATION: COMPLETE")
    print("=" * 72)

    print(f"Words: {len(result['evaluation_df']):,}")

    print(
        "Recordings:",
        result["evaluation_df"]["file_id"].nunique(),
    )

    print(f"Canonical CSV: {result['canonical_csv']}")
    print(f"Tables: {result['tables_dir']}")
    print(f"Figures: {result['figures_dir']}")

    print(
        "\nInterpretation: in-domain validation of the final "
        "Storyboard-adapted MFA system."
    )

    print("=" * 72)


# Execute main() only when this script is run directly.
# If the module is imported elsewhere, the pipeline will not start automatically.
if __name__ == "__main__":
    main()