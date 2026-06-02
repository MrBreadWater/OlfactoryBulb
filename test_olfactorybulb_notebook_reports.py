"""Focused tests for olfactory-bulb notebook run-summary helpers."""

from __future__ import annotations

from types import SimpleNamespace

from olfactorybulb.notebook_reports import NotebookReportHooks, print_run_summary


def main() -> None:
    writes: list[str] = []
    diff_calls: list[tuple[str, list[dict[str, object]], int | None]] = []

    def _print_diff_section(title, changes, max_items):
        diff_calls.append((title, changes, max_items))
        writes.append(f"diff:{title}:{len(changes)}")

    hooks = NotebookReportHooks(
        result_overview_fn=lambda result: {
            "label": "demo",
            "result_dir": result["result_dir"],
        },
        build_run_config_fn=lambda **config: {"mode": "local", **config},
        resolve_effective_params_fn=lambda config: {
            "input_odors_source": "paramset",
            "n_odor_presentations": 1,
            "odor_names": ["Apple"],
            "input_odors": {0: {"name": "Apple", "rel_conc": 0.1}},
            "max_firing_rate_hz": 50.0,
            "inhale_duration_ms": 125.0,
            "mc_input_weight": 0.4,
            "tc_input_weight": 0.2,
            "full_param_snapshot": {
                "paramset": config["paramset"],
                "gaba_tau2_ms": config["gaba_tau2_ms"],
            },
        },
        resolve_paramset_defaults_fn=lambda paramset_name: {
            "paramset": paramset_name,
            "gaba_tau2_ms": 36.0,
        },
        diff_values_fn=lambda before, after: (
            []
            if before == after
            else [{"path": "gaba_tau2_ms", "before": before["gaba_tau2_ms"], "after": after["gaba_tau2_ms"]}]
        ),
        extract_runtime_control_snapshot_fn=lambda config: {
            "mode": config["mode"],
            "paramset": config["paramset"],
        },
        print_diff_section_fn=_print_diff_section,
        write_fn=writes.append,
    )

    run = SimpleNamespace(
        config={"paramset": "GammaSignature", "gaba_tau2_ms": 50.0},
        result_dir="/tmp/demo-run",
        command=["python", "demo.py", "--flag"],
    )
    result = {
        "result_dir": "/tmp/demo-run",
        "run_info": {
            "remote": {"host": "sol"},
        },
    }

    print_run_summary(hooks, run, result)

    text = "\n".join(writes)
    assert '"label": "demo"' in text
    assert '"input_odors_source": "paramset"' in text
    assert '"mode": "local"' in text
    assert '"host": "sol"' in text
    assert "Result directory: /tmp/demo-run" in text
    assert "Command: python demo.py --flag" in text
    assert diff_calls == [
        (
            "Requested/effective param changes vs clean paramset",
            [{"path": "gaba_tau2_ms", "before": 36.0, "after": 50.0}],
            None,
        )
    ]

    print("olfactorybulb notebook reports: OK")


if __name__ == "__main__":
    main()
