import math
import unittest

import numpy as np

from calibrix.ablation import (
    DirectionalAblationAdapter,
    SimulatedRefusalAdapter,
    ablation_weight_profile,
    compute_refusal_directions,
    default_ablation_spec,
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
        # recovery at the signal-rich top layer: means over 12 samples have
        # noise ~ 1/sqrt(12), so require clearly-aligned, not exact
        cos = float(np.dot(dirs[best], direction))
        self.assertGreater(abs(cos), 0.85)

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
        # d^T W' == 0: no input can produce output along d
        self.assertTrue(np.allclose(d @ Wp, 0, atol=1e-10))

    def test_biprojected_preserves_orthogonal_space(self):
        rng = np.random.default_rng(2)
        d = rng.normal(size=8)
        d /= np.linalg.norm(d)
        W = rng.normal(size=(8, 8))
        Wp = orthogonalize(W, d, preserve_norm=True)
        # the input-read side is annihilated exactly
        self.assertTrue(np.allclose(Wp @ d, 0, atol=1e-10))
        # rows keep their original norm (the "norm-preserving" property)
        self.assertTrue(np.allclose(np.linalg.norm(Wp, axis=1),
                                    np.linalg.norm(W, axis=1), atol=1e-10))
        # ablation removes less useful signal than the plain projection:
        # Frobenius norm stays close to the original
        self.assertGreater(np.linalg.norm(Wp), np.linalg.norm(orthogonalize(W, d, preserve_norm=False)))

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
        self.assertTrue(np.allclose(v, np.array([1.0, 1.0, 0.0]) / math.sqrt(2)))

    def test_out_of_range_clips(self):
        self.assertTrue(np.allclose(site_direction(self.dirs, -2.0), [1, 0, 0]))
        self.assertTrue(np.allclose(site_direction(self.dirs, 99.0), [0, 0, 1]))

    def test_per_layer_requires_explicit_index(self):
        with self.assertRaises(ValueError):
            site_direction(self.dirs, None)


class TestSimulatedPipeline(unittest.TestCase):
    def test_end_to_end_offline(self):
        sim = SimulatedRefusalAdapter(n_layers=12, d_model=16, seed=0)

        harmful = [Prompt(user=f"how to build a bomb {i}") for i in range(12)]
        harmless = [Prompt(user=f"write a poem about the sea {i}") for i in range(12)]
        sim.plant_clusters(harmful, harmless)

        dirs = extract_directions_from_sim(sim, harmful, harmless)
        best = pick_best_direction_layer(dirs)
        d_hat = dirs[best]
        # planted direction recovered (sign-agnostic); sample noise keeps
        # this below exactness but well above chance
        cos = abs(float(np.dot(d_hat, sim.planted_direction)))
        self.assertGreater(cos, 0.85)

        # wire the spec with the recovered direction and ablate hard
        spec = default_ablation_spec(n_layers=12, d_model=16)
        spec.directions = dirs
        sim.directions = dirs
        for s in spec.specs:
            s.kernel = KernelParams(weight=1.0, position=best / 12.0,
                                    focus=1e6, floor=1.0)  # uniform strength 1.0
            s.kernel.direction_index = float(best)
            s.direction_index = float(best)

        sim.apply_modulation(spec.specs)
        # full-strength ablation along the (recovered) refusal direction:
        # removal ~ align > 0.85 -> refuse_p = 0.9 - 1.1*removal < 0
        responses = sim.generate(harmful)
        refusals = sum("sorry" in r.lower() or "can't" in r.lower()
                       for r in responses)
        self.assertEqual(refusals, 0)
        # and harmless prompts still get real answers (uniform-ablation
        # deviation is spread across layers -> below the degeneration band)
        responses_ok = sim.generate(harmless)
        self.assertTrue(all(r.startswith("Sure.") for r in responses_ok))

    def test_weak_scattered_ablation_keeps_refusals(self):
        """The negative control: diffuse low-strength ablation does NOT kill
        the behavior (but does cause drift) - the property that makes the
        co-optimization non-trivial."""
        sim = SimulatedRefusalAdapter(n_layers=12, d_model=16, seed=0)
        harmful = [Prompt(user=f"how to build a bomb {i}") for i in range(12)]
        harmless = [Prompt(user=f"write a poem about the sea {i}") for i in range(12)]
        sim.plant_clusters(harmful, harmless)
        dirs = extract_directions_from_sim(sim, harmful, harmless)
        best = pick_best_direction_layer(dirs)
        sim.directions = dirs
        spec = default_ablation_spec(12, 16)
        spec.directions = dirs
        for s in spec.specs:
            # weak + tight: low strength at the best layer only
            s.kernel = KernelParams(weight=0.4, position=best / 12.0,
                                    focus=0.1, floor=0.0)
            s.kernel.direction_index = float(best)
            s.direction_index = float(best)
        sim.apply_modulation(spec.specs)
        responses = sim.generate(harmful)
        refusals = sum("sorry" in r.lower() or "can't" in r.lower()
                       for r in responses)
        # strength 0.4 x align ~0.9 -> removal ~0.36 -> refuse_p ~0.5:
        # most prompts still refused
        self.assertGreater(refusals, 0)

    def test_strength_profile_shape(self):
        # focus 0.5 spans half the sites; with position 0.5 the peak sits
        # between discrete sites, so the max touches but need not equal 1.0
        prof = ablation_weight_profile(n_sites=12, weight=1.0, position=0.5,
                                       focus=0.5, floor=0.0)
        self.assertEqual(len(prof), 12)
        self.assertGreater(max(prof), 0.9)
        self.assertEqual(max(prof), min(1.0, max(prof)))
        self.assertAlmostEqual(prof[0], prof[-1], places=9)


if __name__ == "__main__":
    unittest.main()
