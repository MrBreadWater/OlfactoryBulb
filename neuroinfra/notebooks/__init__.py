"""Reusable notebook-facing run metadata helpers."""

from .runs import (
    RunRecord,
    list_run_dirs,
    load_run_config,
    load_run_record,
    read_json_if_present,
    resolve_run_dir,
)
from .config_store import (
    json_ready,
    list_json_configs,
    load_json_config,
    save_json_config,
)
from .reporting import (
    diff_values,
    flatten_for_diff,
    format_diff_value,
    print_diff_section,
    save_figure,
)

__all__ = [
    "diff_values",
    "flatten_for_diff",
    "format_diff_value",
    "RunRecord",
    "json_ready",
    "list_json_configs",
    "list_run_dirs",
    "load_json_config",
    "load_run_config",
    "load_run_record",
    "print_diff_section",
    "read_json_if_present",
    "resolve_run_dir",
    "save_json_config",
    "save_figure",
]
