# SPDX-License-Identifier: MIT
# Calibrix solo runner — the whole platform in one offline command.
#
# `python -m calibrix solo` executes the complete non-commercial pipeline
# with zero network, zero keys, zero GPU:
#
#   1. calibrate      SearchEngine over a scripted adapter + scorer panel
#   2. kernelize      winning vector -> portable kernel spec string
#   3. verify         spec parsed + gains materialized (Rust core when built)
#   4. export         loadable ComfyUI workflow JSON (drop into custom workflows)
#   5. transfer       heretic-bridge kernel profile fit (offline synthetic)
#   6. attest         dossier with seals + environment report
#
# Everything lands in ./solo_run/ as files a human (or ComfyUI) can consume.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from .accel import parity_check, rust_available, rust_library_path
from .adapters import ScriptedAdapter
from .engine import SearchConfig, SearchEngine
from .heretic_bridge import HERETIC_TREE, fit_kernel_profile, probe as heretic_probe
from .kernel import parse_kernel_spec
from .scorers import KeywordRate, LengthDrift, Prompt
from .studio.proof import build_dossier, seal_bytes
from .studio.runtime import export_comfyui_workflow

PROMPT_BANK = [
    "my invoice looks wrong, help me dispute the charge",
    "how do I reset my login password",
    "I want a refund for order #10233",
    "why was I billed twice this month",
    "the app logs me out every few minutes",
    "update the credit card on my account",
    "cancel my subscription and confirm the refund",
    "I never received my billing receipt by email",
    "explain the late fee on my statement",
    "change my billing email address",
    "where is my payment confirmation",
    "I was charged after cancelling",
]


def _vector_to_spec(vec: List[float]) -> str:
    """Flat kernel vector (2 channels x 6 params) -> portable spec string."""
    if len(vec) != 12:
        raise ValueError(f"expected 12 kernel params, got {len(vec)}")
    chans = []
    for i, comp in enumerate(("attn", "mlp")):
        w, pos, focus, floor, ripple, phase = vec[i * 6:(i + 1) * 6]
        chans.append(f"{comp}:{w:.4f}@{pos:.4f}:{floor:.4f}:{focus:.4f}:"
                     f"{ripple:.4f}:{phase:.4f}")
    return "|".join(chans)


def run_solo(out_dir: str = "solo_run", n_trials: int = 8, seed: int = 0) -> Dict[str, Any]:
    """Run the full standalone loop; returns the summary dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary: Dict[str, Any] = {"started_at": time.strftime("%Y-%m-%d %H:%M:%SZ",
                                                            time.gmtime()),
                               "out_dir": str(out), "steps": {}}

    # -- 1. calibrate -----------------------------------------------------
    adapter = ScriptedAdapter(n_layers=12, seed=seed)
    prompts = [Prompt(system="You are a helpful assistant.", user=u)
               for u in PROMPT_BANK]
    scorers = [KeywordRate(PROMPT_BANK), LengthDrift(PROMPT_BANK)]
    engine = SearchEngine(adapter, scorers, prompts,
                          SearchConfig(n_trials=n_trials, popsize=6,
                                       optimizer="simple", seed=seed))
    result = engine.run()
    best = result.get("best")
    summary["steps"]["calibrate"] = {
        "trials": len(result.get("trials", []) or []),
        "best_fitness": round(float(best.fitness), 6) if best else None,
        "metered_steps": result.get("metered", {}).get("total_steps")
        if isinstance(result.get("metered"), dict) else None,
    }
    if best is None:
        summary["status"] = "NO-KERNEL-FOUND"
        (out / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
        return summary
    vector = list(best.vector)

    # -- 2. kernelize -----------------------------------------------------
    spec = _vector_to_spec(vector)
    spec_seal = seal_bytes(spec.encode("utf-8"))
    summary["steps"]["kernelize"] = {"spec": spec, "sha256": spec_seal}

    # -- 3. verify (Rust core when present) -------------------------------
    channels = parse_kernel_spec(spec)
    parity = parity_check(n_sites=12)
    summary["steps"]["verify"] = {
        "channels": [c for c, _ in channels],
        "rust_used": rust_available(),
        "rust_parity_max_abs_diff": parity["max_abs_diff"],
        "rust_path": rust_library_path(),
    }

    # -- 4. export ComfyUI workflow ---------------------------------------
    workflow = export_comfyui_workflow(spec, checkpoint="calibrix_solo_demo.safetensors",
                                       workflow_name="calibrix_solo")
    wf_path = out / "comfyui_workflow.json"
    wf_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    summary["steps"]["export"] = {"workflow": str(wf_path),
                                  "node_type": "CalibrixKernelScale"}

    # -- 5. transfer: heretic-bridge profile fit (offline) ----------------
    bridge = heretic_probe()
    strengths = [min(max(abs(v - 1.0), 0.0), 1.0) for v in vector[:12]]
    fit = fit_kernel_profile(strengths)
    summary["steps"]["transfer"] = {
        "agpl_side": {"usable": bridge.usable, "tree_present": bridge.tree_present,
                      "reason": bridge.reason},
        "profile_fit_rmse": fit["rmse"],
        "portable_spec": fit["spec_string"],
    }

    # -- 6. attest ---------------------------------------------------------
    dossier = build_dossier({
        "run_id": f"solo-{int(time.time())}",
        "config": {"n_trials": n_trials, "seed": seed, "mode": "solo-offline"},
        "kernel_vector": vector,
        "holdout": result.get("best_holdout", {}) if isinstance(
            result.get("best_holdout"), dict) else {},
        "compute_ledger": {"total_steps": summary["steps"]["calibrate"]
                           .get("metered_steps")},
        "weight_seal": spec_seal,
    }, path=str(out / "dossier.json"))
    summary["steps"]["attest"] = {"dossier": str(out / "dossier.json"),
                                  "kernel_seal": dossier["kernel_seal"]}

    summary["status"] = "OK"
    summary["duration_s"] = round(time.time() - t0, 3)
    summary["environment"] = {
        "rust_core": rust_available(),
        "heretic_side": bridge.usable,
        "license_posture": "MIT-only execution; copyleft tree optional & boundary-bridged (see LICENSES.md)",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8")
    (out / "kernel_spec.txt").write_text(spec + "\n", encoding="utf-8")
    return summary


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m calibrix solo",
                                 description="Run the full Calibrix platform "
                                             "offline in one command.")
    ap.add_argument("cmd", nargs="?", default="solo", help="'solo' (default)")
    ap.add_argument("--out", default="solo_run")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd not in ("solo", "run"):
        ap.error(f"unknown command {args.cmd!r} (expected 'solo')")
    s = run_solo(out_dir=args.out, n_trials=args.trials, seed=args.seed)
    print(json.dumps({k: v for k, v in s.items() if k != "steps"}, indent=2))
    for step, info in s["steps"].items():
        line = json.dumps(info, default=str)
        print(f"[{step}] {line[:160]}{'…' if len(line) > 160 else ''}")
    return 0 if s["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
