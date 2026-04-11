"""Run ``benchmark_ob.py`` while forcing the target repository to the front of ``sys.path``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--script", default=str(Path(__file__).resolve().with_name("benchmark_ob.py")))
    args, rest = parser.parse_known_args()

    repo_root = str(Path(args.repo_root).resolve())
    script_path = str(Path(args.script).resolve())
    local_checkout_root = str(Path(__file__).resolve().parents[2])

    cleaned = []
    for entry in sys.path:
        if not entry:
            continue
        resolved = str(Path(entry).resolve())
        if resolved in {local_checkout_root, repo_root}:
            continue
        cleaned.append(entry)

    sys.path[:] = [repo_root] + cleaned
    sys.argv = [script_path] + rest

    code = Path(script_path).read_text()
    glb = {"__name__": "__main__", "__file__": script_path}
    exec(compile(code, script_path, "exec"), glb)


if __name__ == "__main__":
    main()
