# Calibrix research probes.
#
# Repurposed from Heretic's --print-residual-geometry: quantitative analysis
# of how a "target" prompt cluster separates from a "neutral" cluster across
# transformer depth. Heretic fixes the semantics (harmful vs harmless);
# Calibrix generalizes it to ARBITRARY contrast pairs, enabling research on
# refusal, style, verbosity, language, safety-tax, persona — any behavior
# that can be described with two prompt sets.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .adapters import Adapter
from .scorers import Prompt


@dataclass
class LayerGeometry:
    """Per-layer separation statistics between two prompt clusters."""

    layer: int
    cosine_similarity: float      # alignment of cluster means
    direction_norm: float         # ||mean_a - mean_b|| (effect strength)
    silhouette: float             # cluster separation quality (-1..1)
    centroid_distance: float      # euclidean distance of centroids


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / max(na * nb, 1e-12)


def _silhouette(pool: List[List[float]], labels: List[int]) -> float:
    """Mean silhouette coefficient with cosine distance (small pools OK)."""
    if len(set(labels)) < 2 or len(pool) < 3:
        return 0.0

    def dist(u: List[float], v: List[float]) -> float:
        return 1.0 - _cosine(u, v)

    scores = []
    for i, u in enumerate(pool):
        same = [dist(u, pool[j]) for j in range(len(pool))
                if j != i and labels[j] == labels[i]]
        other = [dist(u, pool[j]) for j in range(len(pool))
                 if labels[j] != labels[i]]
        if not same or not other:
            continue
        a = sum(same) / len(same)
        b = min(sum(o) / len(o) for o in [other]) if len(set(labels)) == 2 \
            else sum(other) / len(other)
        s = (b - a) / max(a, b)
        scores.append(s)
    return sum(scores) / len(scores) if scores else 0.0


class DirectionProbe:
    """Two-set contrast probe over adapter hidden states.

    Requires the adapter to expose `hidden_states(prompts) -> array shaped
    (n_prompts, n_layers+1, d)`. The ScriptedAdapter provides a synthetic
    version so the probe is testable offline; real adapters read residual
    streams (repurposing Heretic's get_residuals machinery).
    """

    def __init__(self, adapter: Adapter, set_a: List[Prompt], set_b: List[Prompt]) -> None:
        if not hasattr(adapter, "hidden_states"):
            raise TypeError(
                f"{type(adapter).__name__} does not expose hidden_states(); "
                "probes need an adapter with residual-stream access."
            )
        self.adapter = adapter
        self.set_a = set_a
        self.set_b = set_b

    def run(self) -> List[LayerGeometry]:
        hs_a = self.adapter.hidden_states(self.set_a)   # (nA, L+1, d)
        hs_b = self.adapter.hidden_states(self.set_b)   # (nB, L+1, d)
        n_layers = hs_a.shape[1]

        geoms: List[LayerGeometry] = []
        for l in range(n_layers):
            a = hs_a[:, l, :]
            b = hs_b[:, l, :]
            ca = a.mean(axis=0)
            cb = b.mean(axis=0)
            cs = float(_cosine(list(ca), list(cb)))
            dnorm = float(sum((x - y) ** 2 for x, y in zip(ca, cb)) ** 0.5)
            cdist = dnorm  # same as direction norm for means
            pool = [list(row) for row in list(a) + list(b)]
            labels = [0] * len(a) + [1] * len(b)
            sil = _silhouette(pool, labels)
            geoms.append(LayerGeometry(
                layer=l, cosine_similarity=cs, direction_norm=dnorm,
                silhouette=sil, centroid_distance=cdist,
            ))
        return geoms

    def best_layers(self, top_k: int = 3) -> List[LayerGeometry]:
        """Layers where the contrast separates best — candidate kernel peaks."""
        geoms = self.run()
        return sorted(geoms, key=lambda g: g.silhouette, reverse=True)[:top_k]

    def summary(self) -> Dict[str, Any]:
        geoms = self.run()
        best = sorted(geoms, key=lambda g: g.silhouette, reverse=True)[:3]
        return {
            "n_layers": len(geoms),
            "best_layers": [
                {"layer": g.layer, "silhouette": round(g.silhouette, 4),
                 "direction_norm": round(g.direction_norm, 4)}
                for g in best
            ],
            "table": [
                {"layer": g.layer, "cos": round(g.cosine_similarity, 4),
                 "dir_norm": round(g.direction_norm, 4),
                 "silhouette": round(g.silhouette, 4)}
                for g in geoms
            ],
        }
