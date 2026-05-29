"""Regression checks for HFO campaign resume helpers.

Run with:
    /opt/miniconda3/envs/OBGPU/bin/python test_run_hfo_campaign.py
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import tools.run_hfo_campaign as runner


with TemporaryDirectory() as tmp:
    campaign = Path(tmp)
    batch_dir = campaign / "batches"
    batch_dir.mkdir(parents=True)
    (batch_dir / "batch_0254_plan.json").write_text(
        json.dumps(
            {
                "batch_name": "batch_0254",
                "strategy": "elite_truncated_gaussian_plus_lhs",
                "stage": "elite_refine",
                "candidates": [{} for _ in range(72)],
            }
        )
    )

    captured: dict[str, object] = {}

    def _fake_elite_batch(campaign_dir, search_space, n_candidates, seed, elite_frac, explore_frac, stage):
        captured.update(
            {
                "campaign_dir": Path(campaign_dir),
                "search_space": search_space,
                "n_candidates": n_candidates,
                "seed": seed,
                "elite_frac": elite_frac,
                "explore_frac": explore_frac,
                "stage": stage,
            }
        )
        return {"batch_name": "batch_0255", "strategy": "elite_truncated_gaussian_plus_lhs", "candidates": []}

    with patch.object(runner, "propose_elite_batch", side_effect=_fake_elite_batch):
        plan = runner._pick_batch(campaign, search_space=["new-search-space"], batch_index=255)

    assert plan["batch_name"] == "batch_0255"
    assert captured["campaign_dir"] == campaign
    assert captured["search_space"] == ["new-search-space"]
    assert captured["n_candidates"] == 72
    assert captured["seed"] == 20260782
    assert captured["stage"] == "elite_refine"
    print("recent batch shape inference: OK")


with TemporaryDirectory() as tmp:
    campaign = Path(tmp)
    batch_dir = campaign / "batches"
    batch_dir.mkdir(parents=True)
    (batch_dir / "batch_0254_plan.json").write_text(
        json.dumps(
            {
                "batch_name": "batch_0254",
                "strategy": "elite_truncated_gaussian_plus_lhs",
                "stage": "elite_refine",
                "candidates": [{} for _ in range(72)],
            }
        )
    )

    initial_completed = [f"batch_{idx:04d}" for idx in range(254)]
    state_responses = [
        {
            "next_batch_index": 255,
            "next_candidate_index": 12648,
            "completed_batches": initial_completed,
        },
        {
            "next_batch_index": 255,
            "next_candidate_index": 12648,
            "completed_batches": initial_completed + ["batch_0254"],
        },
    ]
    run_calls: list[str] = []
    pick_calls: list[int] = []

    def _fake_load_campaign_state(_campaign_dir):
        if len(run_calls) == 0:
            return deepcopy(state_responses[0])
        return deepcopy(state_responses[1])

    def _fake_run_hfo_batch(campaign_dir, base_config, batch_plan, ketamine_block_values):
        run_calls.append(batch_plan["batch_name"])
        return {"sweep_dir": str(campaign_dir / "sweeps" / batch_plan["batch_name"]), "items": []}

    def _fake_score_hfo_batch(campaign_dir, batch_plan, sweep):
        return {
            "candidate_rows": [
                {
                    "candidate_id": f"{batch_plan['batch_name']}_best",
                    "pair_score": 1.0,
                    "target_contrast_log10": 0.0,
                    "control_metrics": {
                        "condition_score": 0.0,
                        "relative_band_power": {"target_hfo": 0.0},
                    },
                    "ketamine_metrics": {
                        "condition_score": 0.0,
                        "peak_ratio": 0.0,
                        "relative_band_power": {"target_hfo": 0.0},
                    },
                }
            ]
        }

    def _fake_pick_batch(campaign_dir, search_space, batch_index):
        pick_calls.append(batch_index)
        return {
            "batch_name": "batch_0255",
            "strategy": "elite_truncated_gaussian_plus_lhs",
            "stage": "elite_refine",
            "candidates": [],
        }

    with (
        patch.object(runner, "_build_configs", return_value=({"remote_host": "jmpaniag@localhost"}, {"cfg": 1})),
        patch.object(runner, "_load_or_init_campaign", return_value=campaign),
        patch.object(runner, "_require_live_paramiko_session", return_value=True),
        patch.object(runner, "load_campaign_state", side_effect=_fake_load_campaign_state),
        patch.object(runner, "run_hfo_batch", side_effect=_fake_run_hfo_batch),
        patch.object(runner, "score_hfo_batch", side_effect=_fake_score_hfo_batch),
        patch.object(runner, "_pick_batch", side_effect=_fake_pick_batch),
        patch.object(runner, "top_candidate_rows", return_value=[]),
    ):
        out = runner.run_campaign(
            allocation="14537854",
            campaign_name=campaign.name,
            max_batches=256,
            total_tasks=120,
            nranks=5,
            tstop_ms=2000.0,
            cell_permute=0,
            early_stop_score=float("inf"),
            min_ketamine_target=0.0,
            max_control_target=1.0,
            min_ketamine_peak_ratio=0.0,
            min_target_contrast_log10=float("-inf"),
            max_control_score=float("inf"),
            require_criteria_for_early_stop=False,
            require_live_paramiko_session=True,
            verify_auth=False,
        )

    assert out == campaign
    assert run_calls == ["batch_0254", "batch_0255"]
    assert pick_calls == [255]
    print("pending batch resume before new proposal: OK")


print("All tests passed.")
