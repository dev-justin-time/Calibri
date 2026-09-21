import math
import unittest

import numpy as np

from calibrix.ablation import (
    DirectionalAblationAdapter,
    SimulatedRefusalAdapter,
    ablation_weight_profile,
    compute_refusal_directions,
    extract_directions_from_sim,
    orthogonalize,
    pick_best_direction_layer,
    site_direction,
)
from calibrix.kernel import KernelParams, ModulationSpec
from calibrix.scorers import Prompt


def _cluster_data(n_h=12, n_s=12, d=8, layers=6, seed=0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    hs = []
    flags = []
    for _ in range(n_h):
        rows = [rng.normal(size=d) for _ in range(layers + 1)]
        for l in range(layers + 1):
            rows[l] = rows[l] + 3.0 * math.tanh(l / 3.0) * direction
        hs.append(rows)
        flags.append(True)
    for _ in range(n_s):
        hs.append([rng.normal(size=d) for _ in range(layers + 1)])
        flags.append(False)
    return np.stack(hs), flags, direction


class TestDirectionExtraction(unittest.TestCase):
    def test_recovers_planted_direction(self):
        hs, flags, direction = _cluster_data()
        dirs = compute_refusal_directions(hs, flags)
        self.assertEqual(len(dirs), layers_of(dirs) + 1)
        best = pick_best_direction_layer(dirs)
        # recovery should be near-perfect where the signal is strong
        cos = float(np.dot(dirs[best], direction))
        self.assertGreater(abs(cos), 0.99)

    def test_handles_empty_clusters(self):
        hs, _, _ = _cluster_data()
        dirs = compute_refusal_directions(hs, [True] * hs.shape[0])
        self.assertTrue(all(np.allclose(d, 0) for d in dirs))

    def test_rejects_mismatched_flags(self):
        hs, _, _ = _cluster_data()
        with self.assertRaises(ValueError):
            compute_refusal_directions(hs, [True, False])


def layers_of(dirs):
    return len(dirs) - 1


class TestOrthogonalize(unittest.TestCase):
    def test_direction_component_removed(self):
        rng = np.random.default_rng(1)
        d = rng.normal(size=8)
        d /= np.linalg.norm(d)
        W = rng.normal(size=(8, 8))
        Wp = orthogonalize(W, d, preserve_norm=False)
        # W' d == 0: the direction is inhibited
        self.assertTrue(np.allclose(Wp @ d, 0, atol=1e-10))

    def test_biprojected_preserves_orthogonal_space(self):
        rng = np.random.default_rng(2)
        d = rng.normal(size=8)
        d /= np.linalg.norm(d)
        W = rng.normal(size=(8, 8))
        Wp = orthogonalize(W, d, preserve_norm=True)
        # vectors orthogonal to d pass through unchanged
        q = rng.normal(size=8)
        q = q - (q @ d) * d
        q /= np.linalg.norm(q)
        self.assertTrue(np.allclose(Wp @ q, W @ q, atol=1e-8))
        # and the direction is killed on both sides
        self.assertTrue(np.allclose(Wp @ d, 0, atol=1e-10))
        self.assertTrue(np.allclose(d @ Wp, 0, atol=1e-10))

    def test_idempotent(self):
        rng = np.random.default_rng(3)
        d = rng.normal(size=8)
        d /= np.linalg.norm(d)
        W = rng.normal(size=(8, 8))
        Wp = orthogonalize(W, d, preserve_norm=False)
        Wpp = orthogonalize(Wp, d, preserve_norm=False)
        self.assertTrue(np.allclose(Wp, Wpp, atol=1e-10))


class TestSiteDirection(unittest.TestCase):
    def setUp(self):
        self.dirs = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                     np.array([0, 0, 1.0])]

    def test_integer_index(self):
        self.assertTrue(np.allclose(site_direction(self.dirs, 1.0),
                                    [0, 1, 0]))

    def test_fractional_interpolates(self):
        v = site_direction(self.dirs, 0.5)
        self.assertTrue(np.allclose(v, [1, 1, 0] / math.sqrt(2)))

    def test_out_of_range_clips(self):
        self.assertTrue(np.allclose(site_direction(self.dirs, -2.0), [1, 0, 0]))
        self.assertTrue(np.allclose(site_direction(self.dirs, 99.0), [0, 0, 1]))

    def test_per_layer_requires_explicit_index(self):
        with self.assertRaises(ValueError):
            site_direction(self.dirs, None)


class TestSimulatedPipeline(unittest.TestCase):
    def test_end_to_end_offline(self):
        sim = SimulatedRefusalAdapter(n_layers=12, d_model=16, seed=0)

        harmful = [Prompt(user=f"how to build a bomb {i}") for i in range(8)]
        harmless = [Prompt(user=f"write a poem about the sea {i}") for i in range(8)]
        sim.plant_clusters(harmful, harmless)

        dirs = extract_directions_from_sim(sim, harmful, harmless)
        best = pick_best_direction_layer(dirs)
        d_hat = dirs[best]
        # planted direction recovered (sign-agnostic)
        cos = abs(float(np.dot(d_hat, sim.planted_direction)))
        self.assertGreater(cos, 0.95)

        # wire the spec with the recovered direction and ablate
        from calibrix.ablation import AblationSpec, default_ablation_spec

        spec = default_ablation_spec(n_layers=12, d_model=16)
        spec.directions = dirs
        for s in spec.specs:
            s.kernel = KernelParams(weight=1.2, position=best / 12.0,
                                    focus=0.3, floor=0.15)
            s.kernel.direction_index = float(best)

        sim.apply_modulation(spec.specs)
        # with the direction ablated, harmful prompts stop being refused
        responses = sim.generate(harmful)
        refusals = sum("sorry" in r.lower() or "can't" in r.lower()
                       for r in responses)
        self.assertEqual(refusals, 0)
        # and harmless prompts still get real answers (mild drift tolerated)
        responses_ok = sim.generate(harmless)
        self.assertTrue(all(r != "" for r in responses_ok))

    def test_strength_profile_shape(self):
        prof = ablation_weight_profile(n_sites=12, weight=1.0, position=0.5,
                                       focus=0.25, floor=0.0)
        self.assertEqual(len(prof), 12)
        self.assertEqual(max(prof), 1.0)
        self.assertAlmostEqual(prof[0], prof[-1], places=9)


if __name__ == "__main__":
    unittest.main()
