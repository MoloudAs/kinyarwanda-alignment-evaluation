"""Loads and prepares the YAML configuration for the evaluation pipeline.

This module reads the project configuration, checks required settings,
and converts project-relative input and output paths into absolute paths.
"""

from pathlib import Path

import yaml


class ConfigurationError(ValueError):
    """Error raised when the project configuration is incomplete or invalid."""


def load_config(path: str | Path) -> dict:
    """Load and prepare the YAML configuration.

    The function:
    - checks that the YAML file exists,
    - reads it into a Python dictionary,
    - requires a project_root setting,
    - resolves the project root,
    - converts input and output paths to absolute Path objects,
    - returns the prepared configuration dictionary.
    """

    # Resolve the location of the configuration file
    # and stop if the file cannot be found.
    config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )


    # Read the YAML file and convert it into Python objects.
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)


    # An empty YAML file produces None, so replace it
    # with an empty dictionary before further checks.
    if config is None:
        config = {}


    # project_root is required because the other relative paths
    # are resolved from this location.
    if "project_root" not in config:
        raise ConfigurationError(
            "Configuration must define 'project_root'."
        )


    # Determine the absolute project root.
    # A relative root is interpreted relative to the repository.
    configured_root = Path(
        config["project_root"]
    )

    if configured_root.is_absolute():
        project_root = configured_root.resolve()
    else:
        project_root = (
            config_path.parent.parent
            / configured_root
        ).resolve()

    config["project_root"] = project_root


    # Resolve every path stored under the input and output sections.
    # This allows the YAML file to contain portable relative paths
    # such as data/manual_reference or results/tables.
    for section_name in ("inputs", "outputs"):
        section = config.get(
            section_name,
            {},
        )

        for key, value in section.items():
            path_value = Path(value)

            if path_value.is_absolute():
                resolved_path = path_value.resolve()
            else:
                resolved_path = (
                    project_root / path_value
                ).resolve()

            section[key] = resolved_path

        config[section_name] = section


    # The resulting dictionary is ready to be passed to run_pipeline().
    return config