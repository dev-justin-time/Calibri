# SPDX-License-Identifier: MIT
"""Tests for the standalone non-commercial platform: license boundary,
Rust accel, ablation upgrades, and the solo runner."""

import ast
import json
import unittest
from pathlib import Path

from calibrix.ablation import (biprojection_ablate_matrix,
                               compute_refusal_directions_robust,
                               multi_direction_strength)
from calibrix.accel import parity_check, rust_available, spec_gains
from calibrix.heretic_bridge import (HERETIC_TREE, fit_kernel_profile, probe,
                                     run as heretic_run)
from calibrix.kernel import ModulationSpec, parse_kernel_spec
from calibrix.solo import run_solo


class TestHereticBridge(unittest.TestCase):
    def test_probe_never_imports_and_reports_honestly(self):
        s = probe()
        self.assertIsInstance(s.usable, bool)
        self.assertTrue(s.reason)
        # tree is optional; MIT side must work regardless
        self.assertIsInstance(HERETIC_TREE.exists(), bool)

    def test_run_is_noop_when_unavailable(self):
        s = probe()
        if s.usable:
            self.skipTest("heretic CLI installed; subprocess path tested manually")
        rec = heretic_run("google/gemma-3-12b-it")
        self.assertFalse(rec.invoked)
        self.assertIsNone(rec.returncode)
        self.assertIn("unavailable", rec.stderr_tail)

    def test_kernel_fit_recovers_bump_and_roundtrips_comfyui_parser(self):
        rng_rng = __import__("numpy").random.default_rng(3)
        n = 12
        layers = __import__("numpy").arange(n, dtype=float)
        true = 0.9 * __import__("numpy").exp(
            -((layers - n * 0.4) ** 2) / (2 * (n / 5) ** 2)) + 0.15
        strengths = (true + rng_rng.normal(0, 0.02, n)).clip(0, 1).tolist()
        fit = fit_kernel_profile(strengths)
        self.assertLess(fit["rmse"], 0.08)
        # spec string must parse with the canonical parser
        chans = dict(parse_kernel_spec(fit["spec_string"]))
        self.assertIn("ablate", chans)
        w, pos, focus, floor, ripple, phase = fit["kernel"]
        self.assertAlmostEqual(chans["ablate"].weight, w, places=6)

    def test_fit_grid_recovers_exact_profile(self):
        # a pure linear-decay profile (no noise, no ripple) must fit ~exactly
        n = 20
        exact = ModulationSpec(
            component="ablate", n_sites=n,
            kernel=parse_kernel_spec("ablate:1.0@0.3:0.2:0.4")[0][1]).gains()
        fit = fit_kernel_profile(exact)
        self.assertLess(fit["rmse"], 0.01)


class TestAblationUpgrades(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        rng = np.random.default_rng(0)
        d = 8
        cls.d = d
        H = rng.normal(size=(8, 3, d))
        H[:, :, 0] += 2.5                      # planted cluster separation
        S = rng.normal(size=(8, 3, d))
        cls.hs = np.concatenate([H, S], axis=0)
        cls.flags = [True] * 8 + [False] * 8
        cls.H, cls.S = H, S

    def test_multi_direction_extraction_and_mass_separation(self):
        np = self.np
        dirs = compute_refusal_directions_robust(self.hs, self.flags,
                                                 n_directions=3, estimator="median")
        self.assertEqual(len(dirs), self.hs.shape[1])
        l = 2
        self.assertGreaterEqual(len(dirs[l]), 1)
        # orthonormal directions
        if len(dirs[l]) > 1:
            g = abs(float(np.dot(dirs[l][0], dirs[l][1])))
            self.assertLess(g, 1e-8)
        # harmful-side activations carry far more mass than harmless-side
        h_mass = multi_direction_strength(dirs[l][:1], self.H[l, 0] + np.array([3.0] + [0.0] * (self.d - 1)))
        s_mass = multi_direction_strength(dirs[l][:1], self.S[l, 0])
        self.assertGreater(h_mass, 5.0 * s_mass)

    def test_median_estimator_resists_outliers(self):
        np = self.np
        hs = self.hs.copy()
        # corrupt one harmful prompt with a wild activation
        hs[0, 2, :] += 500.0
        dirs_med = compute_refusal_directions_robust(hs, self.flags, 1, "median")[2][0]
        dirs_mean = compute_refusal_directions_robust(hs, self.flags, 1, "mean")[2][0]
        clean = compute_refusal_directions_robust(self.hs, self.flags, 1, "mean")[2][0]
        med_drift = 1.0 - float(np.dot(dirs_med, clean))
        mean_drift = 1.0 - float(np.dot(dirs_mean, clean))
        self.assertLess(med_drift, mean_drift + 1e-9)

    def test_biprojection_annihilates_all_directions(self):
        np = self.np
        rng = np.random.default_rng(1)
        d = 8
        W = rng.normal(size=(d, d))
        ds = [np.eye(d)[k] for k in range(3)]
        W2 = biprojection_ablate_matrix(W, ds, [1.0, 1.0, 1.0])
        for k in range(3):
            self.assertLess(float(np.linalg.norm(ds[k] @ W2)), 1e-10)
        # partial strength leaves partial mass
        W3 = biprojection_ablate_matrix(W, ds[:1], [0.5])
        self.assertGreater(float(np.linalg.norm(ds[0] @ W3)), 1e-6)


class TestAccel(unittest.TestCase):
    def test_parity_rust_or_python(self):
        rep = parity_check(n_sites=21)
        self.assertTrue(rep["parity"])
        self.assertEqual(rep["channels"], 2)
        if rust_available():
            self.assertLessEqual(rep["max_abs_diff"], 1e-9)

    def test_spec_gains_matches_python_engine(self):
        spec = "attn:1.2@0.5:0.9:0.3:0.05:0.25|mlp:0.8@0.25:1.0:4.0"
        gains = spec_gains(spec, n_sites=12)
        self.assertEqual(set(gains), {"attn", "mlp"})
        for comp, p in parse_kernel_spec(spec):
            expected = ModulationSpec(component=comp, n_sites=12, kernel=p).gains()
            self.assertEqual(len(gains[comp]), 12)
            for a, b in zip(gains[comp], expected):
                self.assertAlmostEqual(a, b, places=12)


class TestSoloRunner(unittest.TestCase):
    def test_end_to_end_offline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            s = run_solo(out_dir=str(Path(td) / "run"), n_trials=6, seed=1)
            self.assertEqual(s["status"], "OK")
            self.assertTrue(s["steps"]["calibrate"]["trials"] > 0)
            self.assertTrue(s["steps"]["verify"]["rust_parity_max_abs_diff"] <= 1e-9)
            # artifacts exist and are valid
            out = Path(s["out_dir"])
            wf = json.loads((out / "comfyui_workflow.json").read_text(encoding="utf-8"))
            node_types = {n["type"] for n in wf["nodes"]}
            self.assertIn("CalibrixKernelScale", node_types)
            dossier = json.loads((out / "dossier.json").read_text(encoding="utf-8"))
            self.assertTrue(dossier["kernel_seal"])
            spec = (out / "kernel_spec.txt").read_text(encoding="utf-8").strip()
            self.assertEqual(len(parse_kernel_spec(spec)), 2)


class TestLicenseFirewall(unittest.TestCase):
    """The AGPL boundary, machine-checked: no calibrix module may import
    (or exec/importlib-load) anything from the heretic tree."""

    def test_no_heretic_imports_in_calibrix(self):
        pkg = Path(__file__).resolve().parents[1] / "calibrix"
        offenders = []
        for py in pkg.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.split(".")[0].lower() == "heretic":
                        offenders.append(f"{py.name}: {name}")
        self.assertEqual(offenders, [])

    def test_bridge_uses_subprocess_only(self):
        src = (Path(__file__).resolve().parents[1] / "calibrix"
               / "heretic_bridge.py").read_text(encoding="utf-8")
        self.assertIn("subprocess", src)
        self.assertIn("shutil.which", src)
        # and the import-firewall scan above proves no `import heretic`


if __name__ == "__main__":
    unittest.main()
