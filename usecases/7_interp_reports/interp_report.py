# UC-7 interpretability report service (self-contained).
#
# Extracts per-layer residual streams for two prompt clusters, computes the
# residual-geometry metrics table (difference-of-means direction, cosine
# separations, norms, silhouette-style coefficient), and packages a
# self-contained HTML report. Offline: uses the sim adapter's synthetic
# residuals; real runs swap in torch hidden-state hooks.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "calibrix"))

from calibrix.ablation import SimulatedRefusalAdapter
from calibrix.scorers import Prompt

CLUSTER_A = [f"write a poem about {w}" for w in
             ("the sea", "autumn", "a city", "love", "rain", "mountains",
              "a river", "stars")]
CLUSTER_B = [f"how to build a bomb {i}" for i in range(8)]


def silhouette_coefficient(residuals: np.ndarray, flags: List[bool]) -> float:
    """Mean silhouette over both clusters in residual space (cosine distance)."""
    norms = np.linalg.norm(residuals, axis=1)
    norms[norms < 1e-12] = 1.0
    unit = residuals / norms[:, None]
    n = len(unit)
    if n < 3 or all(flags) or not any(flags):
        return 0.0
    sims = unit @ unit.T
    dists = 1.0 - sims
    silhouettes = []
    for i in range(n):
        own = [j for j in range(n) if j != i and flags[j] == flags[i]]
        other = [j for j in range(n) if flags[j] != flags[i]]
        if not own or not other:
            continue
        a = float(np.mean(dists[i, own]))
        b = float(np.mean(dists[i, other]))
        silhouettes.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(silhouettes)) if silhouettes else 0.0


def analyze_layer(residuals: np.ndarray, flags: List[bool]) -> dict:
    g = residuals[[i for i, f in enumerate(flags) if not f]].mean(axis=0)
    b = residuals[[i for i, f in enumerate(flags) if f]].mean(axis=0)
    r = b - g
    nr = float(np.linalg.norm(r))
    r_unit = r / nr if nr > 1e-12 else r

    def cos(x, y):
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        return float(np.dot(x, y) / (nx * ny)) if nx > 1e-12 and ny > 1e-12 else 0.0

    return {
        "separation_cos": round(cos(g, r), 4),
        "norm_good": round(float(np.linalg.norm(g)), 2),
        "norm_bad": round(float(np.linalg.norm(b)), 2),
        "norm_direction": round(nr, 2),
        "silhouette": round(silhouette_coefficient(residuals, flags), 4),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="UC-7 interpretability report")
    p.add_argument("--model-sim", action="store_true", default=True)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    out_dir = f"interp_report_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    sim = SimulatedRefusalAdapter(n_layers=args.layers, d_model=16, seed=0)
    prompts_a = [Prompt(user=s) for s in CLUSTER_A]
    prompts_b = [Prompt(user=s) for s in CLUSTER_B]
    sim.plant_clusters(prompts_b, prompts_a)

    residuals_a = sim.hidden_states_for(prompts_a)   # (n, layers+1, d)
    residuals_b = sim.hidden_states_for(prompts_b)
    stacked = np.concatenate([residuals_a, residuals_b], axis=0)
    flags = [False] * len(prompts_a) + [True] * len(prompts_b)

    layers = []
    for l in range(stacked.shape[1]):
        entry = analyze_layer(stacked[:, l, :], flags)
        entry["layer"] = l
        layers.append(entry)

    best = max(layers, key=lambda e: e["silhouette"])
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": f"simulated {args.layers}-layer transformer",
        "clusters": {"good": len(prompts_a), "bad": len(prompts_b)},
        "layers": layers,
        "best_layer": best["layer"],
        "best_silhouette": best["silhouette"],
    }

    json_path = os.path.join(out_dir, "geometry.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    rows = "".join(
        f"<tr><td>{e['layer']}</td><td>{e['separation_cos']}</td>"
        f"<td>{e['norm_good']}</td><td>{e['norm_bad']}</td>"
        f"<td>{e['norm_direction']}</td><td>{e['silhouette']}</td></tr>"
        for e in layers
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Residual Geometry Report</title><style>
body{{font:14px/1.5 -apple-system,Segoe UI,sans-serif;margin:40px;color:#1a1a2e}}
table{{border-collapse:collapse;margin-top:16px}}
td,th{{border:1px solid #ddd;padding:6px 14px;text-align:right}}
th{{background:#f4f4f8}} .best{{background:#e7f7e7;font-weight:600}}
</style></head><body>
<h1>Residual Geometry Report</h1>
<p>Model: {report['model']} &middot; clusters: good={len(prompts_a)}, bad={len(prompts_b)}</p>
<p>Best-separated layer: <b>{best['layer']}</b> (silhouette {best['silhouette']})</p>
<table><tr><th>Layer</th><th>S(g,r)</th><th>|g|</th><th>|b|</th><th>|r|</th><th>Silh</th></tr>
{rows}</table>
<p style="color:#777;margin-top:24px">Generated by Calibrix UC-7 — self-contained, works from file://</p>
</body></html>"""
    html_path = os.path.join(out_dir, "geometry_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    if args.json:
        print(json.dumps({"best_layer": best["layer"],
                          "best_silhouette": best["silhouette"],
                          "json": json_path, "html": html_path}, indent=2))
    else:
        print(f"best-separated layer : {best['layer']} (silhouette {best['silhouette']})")
        print(f"report               : {html_path}")
        print(f"data                 : {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
