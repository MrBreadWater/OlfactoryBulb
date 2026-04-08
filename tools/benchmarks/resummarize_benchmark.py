import argparse
import json
from pathlib import Path

from benchmark_ob import summarize_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_dir")
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    summary_path = benchmark_dir / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    file_summaries = {}
    for filename in ["soma_vs.pkl", "input_times.pkl", "lfp.pkl"]:
        path = benchmark_dir / filename
        if path.exists():
            file_summaries[filename] = summarize_pickle(path)

    summary["files"] = file_summaries

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
