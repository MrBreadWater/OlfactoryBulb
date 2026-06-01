"""Generic campaign filesystem helpers for batched optimization workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable


def safe_campaign_slug(text: str) -> str:
    """Return a filesystem-safe campaign slug."""
    cleaned = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "campaign"


def ensure_campaign_dir(
    campaign_name: str,
    *,
    base_dir: str | Path,
    batch_dir_name: str = "batches",
) -> Path:
    """Create and return one campaign workspace root with its batch subdir."""
    campaign_dir = Path(base_dir) / safe_campaign_slug(campaign_name)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / str(batch_dir_name)).mkdir(exist_ok=True)
    return campaign_dir


def state_path(campaign_dir: str | Path, *, filename: str = "state.json") -> Path:
    """Return the JSON state path for one campaign."""
    return Path(campaign_dir) / str(filename)


def archive_path(campaign_dir: str | Path, *, kind: str) -> Path:
    """Return one JSONL archive path under a campaign workspace."""
    return Path(campaign_dir) / f"{kind}_archive.jsonl"


def batch_artifact_path(
    campaign_dir: str | Path,
    batch_name: str,
    artifact_kind: str,
    *,
    batch_dir_name: str = "batches",
    suffix: str = ".json",
) -> Path:
    """Return the path for one batch-scoped artifact file."""
    return Path(campaign_dir) / str(batch_dir_name) / f"{batch_name}_{artifact_kind}{suffix}"


def batch_index_from_name(batch_name: Any) -> int | None:
    """Parse the numeric tail from names like ``batch_0012``."""
    text = str(batch_name or "")
    if "_" not in text:
        return None
    tail = text.rsplit("_", 1)[-1]
    if not tail.isdigit():
        return None
    return int(tail)


def read_json(path: str | Path, default: Any) -> Any:
    """Read one JSON file or return ``default`` when missing."""
    json_path = Path(path)
    if not json_path.exists():
        return default
    return json.loads(json_path.read_text())


def write_json(path: str | Path, payload: Any) -> Path:
    """Write one JSON payload and return the target path."""
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return json_path


def append_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    prepare_row: Callable[[dict[str, Any]], Any] | None = None,
) -> Path:
    """Append rows to one JSONL archive."""
    jsonl_path = Path(path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as handle:
        for row in rows:
            payload = prepare_row(row) if prepare_row is not None else row
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return jsonl_path
