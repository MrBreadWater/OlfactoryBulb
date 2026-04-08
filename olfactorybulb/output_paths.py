import json
import os
import re
from datetime import datetime
from pathlib import Path


TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
TIMESTAMP_SUFFIX_RE = re.compile(r"(^|_)\d{8}_\d{6}$")


def make_timestamp():
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def sync_timestamp(comm=None):
    timestamp = os.environ.get("OB_RUN_TIMESTAMP")
    if comm is not None:
        if timestamp is None and comm.Get_rank() == 0:
            timestamp = make_timestamp()
        timestamp = comm.bcast(timestamp, root=0)
    elif timestamp is None:
        timestamp = make_timestamp()

    os.environ["OB_RUN_TIMESTAMP"] = timestamp
    return timestamp


def label_has_timestamp(label):
    return bool(TIMESTAMP_SUFFIX_RE.search(str(label or "")))


def label_with_timestamp(label, timestamp=None):
    label = str(label or "run")
    if label_has_timestamp(label):
        return label
    timestamp = timestamp or sync_timestamp()
    return f"{label}_{timestamp}"


def configure_output_env(default_label, comm=None, results_base=None):
    timestamp = sync_timestamp(comm=comm)
    requested_label = os.environ.get("OB_RESULT_LABEL", default_label)
    final_label = label_with_timestamp(requested_label, timestamp=timestamp)
    os.environ["OB_RESULT_LABEL"] = final_label
    if results_base is not None:
        os.environ.setdefault("OB_RESULTS_BASE", str(results_base))
    return final_label, timestamp


def get_results_dir(default_label, base_dir=None):
    base_dir = Path(base_dir or os.environ.get("OB_RESULTS_BASE", "results"))
    label = os.environ.get("OB_RESULT_LABEL", default_label)
    return base_dir / label_with_timestamp(label)


def write_run_info(results_dir, payload):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_info_path = results_dir / "run_info.json"
    run_info_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return run_info_path
