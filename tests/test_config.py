"""Tests for YAML configuration loading.

These tests check two important behaviors of config.py:
- relative project paths are resolved correctly,
- the required project_root setting is enforced.
"""

from pathlib import Path
import sys

import pytest


# Add the project's src/ directory to Python's search path
# so the test file can import the package being tested.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from kinyarwanda_alignment.config import (
    ConfigurationError,
    load_config,
)


def test_load_config_resolves_relative_paths(tmp_path):
    """Check that relative paths are resolved from the project root.

    The test creates a temporary YAML configuration with relative paths,
    loads it with load_config(), and verifies that the returned paths
    are the expected absolute Path objects.
    """

    # Create a temporary project-like folder structure.
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    config_file = config_dir / "config.yaml"

    # Write a small controlled YAML configuration for the test.
    config_file.write_text(
        """
project_root: .

inputs:
  manual_reference_dir: data/manual_reference

outputs:
  tables_dir: results/tables
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    expected_root = tmp_path.resolve()

    # Check that project_root and the relative input/output paths
    # were resolved to the correct absolute locations.
    assert config["project_root"] == expected_root

    assert config["inputs"]["manual_reference_dir"] == (
        expected_root / "data" / "manual_reference"
    )

    assert config["outputs"]["tables_dir"] == (
        expected_root / "results" / "tables"
    )


def test_load_config_requires_project_root(tmp_path):
    """Check that a configuration without project_root is rejected.

    The test deliberately creates an incomplete YAML file and verifies
    that load_config() raises the expected ConfigurationError.
    """

    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
inputs:
  manual_reference_dir: data/manual_reference
""",
        encoding="utf-8",
    )

    # The test passes only if ConfigurationError is raised.
    with pytest.raises(ConfigurationError):
        load_config(config_file)