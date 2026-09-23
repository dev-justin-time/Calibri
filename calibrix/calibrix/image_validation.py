# SPDX-License-Identifier: MIT
"""Proof-first validation for real image-model kernels.

This module deliberately sits outside SearchEngine: validation compares a
fixed kernel against an unmodified baseline on identical prompts and seeds.
It produces an artifact a human can inspect and a machine-readable gate that
marketplace promotion can trust.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .marketplace.licenses import checksum_spec
from .scorers import Prompt, Scorer, ScorerContext
from .studio.proof import holdout_split


@dataclass
class ValidationConfig:
    holdout_fraction: float = 0.25
    split_seed: int = 0
    generation_seed: int = 0
    primary_metric: str = "ImageQuality"
    min_primary_delta: float = 0.01
    max_metric_regression: float = 0.01
    max_train_holdout_gap: float = 0.20
    train_indices: Optional[List[int]] = None
    holdout_indices: Optional[List[int]] = None


def _image_bytes(image: Any) -> bytes:
    """Serialize a PIL-like image deterministically for hashing."""
    from io import BytesIO

    if not hasattr(image, "save"):
        raise TypeError(f"validation requires image.save(), got {type(image).__name__}")
    buf = BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _save_images(images: Sequence[Any], prompts: Sequence[Prompt], out_dir: str,
                 prefix: str) -> List[Dict[str, Any]]:
    os.makedirs(out_dir, exist_ok=True)
    artifacts = []
    for index, (image, prompt) in enumerate(zip(images, prompts)):
        raw = _image_bytes(image)
        path = os.path.join(out_dir, f"{prefix}_{index:04d}.png")
        with open(path, "wb") as handle:
            handle.write(raw)
        artifacts.append({
            "index": index,
            "path": path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.user.encode("utf-8")).hexdigest(),
        })
    if len(artifacts) != len(prompts):
        raise ValueError(
            f"adapter returned {len(artifacts)} images for {len(prompts)} prompts"
        )
    return artifacts


def _score_images(adapter: Any, prompts: Sequence[Prompt], images: Sequence[Any],
                  scorers: Sequence[Scorer]) -> Dict[str, float]:
    """Run image scorers against already-generated images without regenerating."""
    ctx = ScorerContext(adapter, batch_size=1)
    key = ("__images__",) + tuple(p.as_tuple() for p in prompts)
    ctx._response_cache[key] = list(images)
    scores: Dict[str, float] = {}
    for scorer in scorers:
        scorer.init(ctx)
        scores[scorer.score_name] = float(scorer.get_score(ctx).value)
    return scores


def _metric_rows(report: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    train = report["scores"]["train"]
    holdout = report["scores"]["holdout"]
    rows = {}
    for name in train["baseline"]:
        rows[name] = {
            "train_baseline": train["baseline"][name],
            "train_kernel": train["kernel"][name],
            "train_delta": train["kernel"][name] - train["baseline"][name],
            "holdout_baseline": holdout["baseline"][name],
            "holdout_kernel": holdout["kernel"][name],
            "holdout_delta": holdout["kernel"][name] - holdout["baseline"][name],
        }
    return rows


def _gate(report: Dict[str, Any], config: ValidationConfig,
          backend_probe: Dict[str, Any]) -> Dict[str, Any]:
    rows = _metric_rows(report)
    primary = config.primary_metric
    reasons: List[str] = []
    if not rows:
        reasons.append("no scorer metrics were produced")
    if primary not in rows:
        reasons.append(f"primary metric {primary!r} is absent")
    if backend_probe.get("backend") != "comfyui":
        reasons.append("verified image badge requires the ComfyUI backend")
    if not backend_probe.get("node_available", False):
        reasons.append("CalibrixKernelScale node was not confirmed in ComfyUI /object_info")
    if backend_probe.get("workflow_has_node") is not True:
        reasons.append("queued workflow was not confirmed to contain CalibrixKernelScale")
    if report["prompt_counts"]["holdout"] < 1:
        reasons.append("holdout set is empty")

    if primary in rows:
        primary_row = rows[primary]
        if primary_row["holdout_delta"] < config.min_primary_delta:
            reasons.append(
                f"primary holdout delta {primary_row['holdout_delta']:.4f} "
                f"< required {config.min_primary_delta:.4f}"
            )

    for name, row in rows.items():
        if row["holdout_delta"] < -config.max_metric_regression:
            reasons.append(
                f"{name} regressed on holdout by {row['holdout_delta']:.4f}"
            )
        if abs(row["train_delta"] - row["holdout_delta"]) > config.max_train_holdout_gap:
            reasons.append(f"{name} train/holdout delta gap exceeds generalization limit")

    return {
        "status": "verified" if not reasons else "measured",
        "quality_gate": "PASS" if not reasons else "REVIEW",
        "passed": not reasons,
        "primary_metric": primary,
        "primary_holdout_delta": rows.get(primary, {}).get("holdout_delta"),
        "reasons": reasons,
        "rules": {
            "min_primary_delta": config.min_primary_delta,
            "max_metric_regression": config.max_metric_regression,
            "max_train_holdout_gap": config.max_train_holdout_gap,
        },
    }


def validate_real_kernel(
    baseline_adapter: Any,
    kernel_adapter: Any,
    kernel_spec: str,
    prompts: Sequence[Prompt],
    scorers: Sequence[Scorer],
    out_dir: str,
    config: Optional[ValidationConfig] = None,
    backend_probe: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate paired real images and return a promotion-ready evidence report."""
    config = config or ValidationConfig()
    backend_probe = dict(backend_probe or {})
    prompts = list(prompts)
    if len(prompts) < 4:
        raise ValueError("real validation needs at least four prompts for a holdout")

    if config.train_indices is not None and config.holdout_indices is not None:
        train_idx = list(config.train_indices)
        holdout_idx = list(config.holdout_indices)
    else:
        train_idx, holdout_idx = holdout_split(
            len(prompts), config.holdout_fraction, seed=config.split_seed
        )
    train_idx = sorted(train_idx)
    holdout_idx = sorted(holdout_idx)
    train_prompts = [prompts[i] for i in train_idx]
    holdout_prompts = [prompts[i] for i in holdout_idx]

    # Both adapters receive the complete prompt list in the same order and
    # therefore the same deterministic seed sequence. This is paired testing,
    # not two unrelated samples.
    def generate(adapter: Any) -> List[Any]:
        try:
            return adapter.generate_images(
                prompts, batch_size=1, seed=config.generation_seed
            )
        except TypeError as exc:
            if "unexpected keyword argument 'seed'" not in str(exc):
                raise
            # Offline adapters may bake their seed into construction; they
            # still participate in the same paired prompt protocol.
            return adapter.generate_images(prompts, batch_size=1)

    baseline_images = generate(baseline_adapter)
    kernel_images = generate(kernel_adapter)
    if len(baseline_images) != len(prompts) or len(kernel_images) != len(prompts):
        raise ValueError("baseline and kernel adapters must return one image per prompt")

    baseline_artifacts = _save_images(
        baseline_images, prompts, os.path.join(out_dir, "images", "baseline"), "baseline"
    )
    kernel_artifacts = _save_images(
        kernel_images, prompts, os.path.join(out_dir, "images", "kernel"), "kernel"
    )

    train_base_images = [baseline_images[i] for i in train_idx]
    train_kernel_images = [kernel_images[i] for i in train_idx]
    hold_base_images = [baseline_images[i] for i in holdout_idx]
    hold_kernel_images = [kernel_images[i] for i in holdout_idx]
    train_scorers = [__import__("copy").deepcopy(s) for s in scorers]
    hold_scorers = [__import__("copy").deepcopy(s) for s in scorers]

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "kernel_spec": kernel_spec,
        "kernel_checksum": checksum_spec(kernel_spec),
        "prompt_counts": {
            "total": len(prompts), "train": len(train_prompts), "holdout": len(holdout_prompts)
        },
        "protocol": {
            "split_seed": config.split_seed,
            "generation_seed": config.generation_seed,
            "paired_prompt_indices": True,
            "same_seed_sequence": True,
            "scorers": [s.score_name for s in scorers],
            "train_indices": [int(i) for i in train_idx],
            "holdout_indices": [int(i) for i in holdout_idx],
        },
        "backend": backend_probe,
        "artifacts": {"baseline": baseline_artifacts, "kernel": kernel_artifacts},
        "scores": {
            "train": {
                "baseline": _score_images(baseline_adapter, train_prompts, train_base_images, train_scorers),
                "kernel": _score_images(kernel_adapter, train_prompts, train_kernel_images, train_scorers),
            },
            "holdout": {
                "baseline": _score_images(baseline_adapter, holdout_prompts, hold_base_images, hold_scorers),
                "kernel": _score_images(kernel_adapter, holdout_prompts, hold_kernel_images, hold_scorers),
            },
        },
    }
    report["metrics"] = _metric_rows(report)
    report["gate"] = _gate(report, config, backend_probe)
    report["evidence"] = {
        "status": report["gate"]["status"],
        "quality_gate": report["gate"]["quality_gate"],
        "customer_roi_proven": False,
        "note": (
            "Real ComfyUI images passed paired holdout validation; this proves "
            "artifact quality under the declared scorer, not customer ROI."
            if report["gate"]["passed"] else
            "Real images were measured, but the kernel is not promotion-ready."
        ),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "validation.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(os.path.join(out_dir, "validation.html"), "w", encoding="utf-8") as handle:
        handle.write(render_validation_html(report))
    return report


def render_validation_html(report: Dict[str, Any]) -> str:
    """Small self-contained reviewer page; no CDN or JavaScript required."""
    rows = []
    for name, row in report.get("metrics", {}).items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{row['train_baseline']:.4f}</td><td>{row['train_kernel']:.4f}</td>"
            f"<td>{row['holdout_baseline']:.4f}</td><td>{row['holdout_kernel']:.4f}</td>"
            f"<td>{row['holdout_delta']:+.4f}</td>"
            "</tr>"
        )
    reasons = "".join(f"<li>{html.escape(r)}</li>" for r in report["gate"]["reasons"])
    status = html.escape(report["evidence"]["status"].upper())
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Calibrix real validation</title>
<style>body{{font:15px system-ui;background:#0d1117;color:#e6edf3;max-width:1000px;margin:30px auto;padding:0 20px}}.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:14px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;border-bottom:1px solid #30363d;text-align:left}}.status{{color:#3fb950;font-weight:700}}.warn{{color:#d29922}}</style></head>
<body><h1>Real FLUX / ComfyUI validation</h1>
<div class="card"><div class="status">EVIDENCE: {status} · GATE: {html.escape(report['gate']['quality_gate'])}</div>
<p>{html.escape(report['evidence']['note'])}</p><p>Kernel checksum: <code>{html.escape(report['kernel_checksum'])}</code></p></div>
<div class="card"><h2>Paired baseline vs kernel scores</h2><table><tr><th>Metric</th><th>Train base</th><th>Train kernel</th><th>Holdout base</th><th>Holdout kernel</th><th>Holdout Δ</th></tr>{''.join(rows)}</table></div>
<div class="card"><h2>Gate review</h2><ul>{reasons or '<li>No gate failures.</li>'}</ul></div>
<div class="card"><h2>Protocol</h2><pre>{html.escape(json.dumps(report['protocol'], indent=2))}</pre></div>
</body></html>"""


__all__ = ["ValidationConfig", "validate_real_kernel", "render_validation_html"]
