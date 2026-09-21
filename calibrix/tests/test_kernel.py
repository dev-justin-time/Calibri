import math
import unittest

from calibrix.kernel import (
    KernelParams,
    ModulationSpec,
    pack_spec_vector,
    unpack_spec_vector,
    total_param_count,
)


class TestKernelParams(unittest.TestCase):
    def test_from_heretic_roundtrip(self):
        k = KernelParams.from_heretic(
            max_weight=1.2,
            max_weight_position=9,
            min_weight=0.1,
            min_weight_distance=6,
            n_sites=18,
        )
        h = k.to_heretic()
        self.assertAlmostEqual(h["max_weight"], 1.2)
        self.assertAlmostEqual(h["max_weight_position"], 9)
        self.assertAlmostEqual(h["min_weight"], 0.1)
        self.assertAlmostEqual(h["min_weight_distance"], 6)

    def test_validate_rejects_bad_params(self):
        with self.assertRaises(ValueError):
            KernelParams(weight=-1).validate()
        with self.assertRaises(ValueError):
            KernelParams(focus=0).validate()
        with self.assertRaises(ValueError):
            KernelParams(floor=-0.5).validate()
        KernelParams(weight=0, floor=0, focus=1).validate()  # ok

    def test_identity_kernel_yields_flat_gains(self):
        # weight=1, floor=1 -> decayed or not, gain is always 1.0
        k = KernelParams(weight=1.0, floor=1.0, focus=1e6)
        spec = ModulationSpec(component="attn", n_sites=12, kernel=k)
        self.assertTrue(all(abs(g - 1.0) < 1e-12 for g in spec.gains()))


class TestModulationSpec(unittest.TestCase):
    def test_gains_peak_at_position(self):
        # Odd n_sites: position 0.5 maps to an exact site (p*(n-1)
        # convention), so the kernel peaks exactly there.
        spec = ModulationSpec(
            component="attn", n_sites=21,
            kernel=KernelParams(weight=1.5, position=0.5, focus=0.1, floor=0.0),
        )
        gains = spec.gains()
        peak_idx = max(range(21), key=lambda i: gains[i])
        self.assertEqual(peak_idx, 10)  # middle layer
        self.assertAlmostEqual(gains[10], 1.5)

    def test_gains_symmetric_between_center_sites_when_even(self):
        # Even n_sites: position 0.5 sits between the two center sites, so
        # the kernel is mirror-symmetric and the peak is a two-way tie.
        spec = ModulationSpec(
            component="attn", n_sites=20,
            kernel=KernelParams(weight=1.5, position=0.5, focus=0.1, floor=0.0),
        )
        gains = spec.gains()
        self.assertAlmostEqual(gains[9], gains[10])
        peak_idx = max(range(20), key=lambda i: gains[i])
        self.assertIn(peak_idx, (9, 10))

    def test_ripple_changes_gains_and_phase_shifts_them(self):
        base = ModulationSpec(
            component="mlp", n_sites=16,
            kernel=KernelParams(weight=1.0, floor=1.0, ripple=0.2, phase=0.0),
        )
        shifted = ModulationSpec(
            component="mlp", n_sites=16,
            kernel=KernelParams(weight=1.0, floor=1.0, ripple=0.2, phase=0.25),
        )
        self.assertNotEqual(base.gains(), shifted.gains())

    def test_explicit_gains_multiply_kernel(self):
        spec = ModulationSpec(
            component="attn", n_sites=4,
            kernel=KernelParams(weight=1.0, floor=1.0),
            explicit=[2.0, 2.0, 2.0, 2.0],
        )
        self.assertTrue(all(abs(g - 2.0) < 1e-12 for g in spec.gains()))

    def test_explicit_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ModulationSpec(component="attn", n_sites=4, explicit=[1.0, 2.0])

    def test_vector_roundtrip(self):
        spec = ModulationSpec(
            component="attn", n_sites=8,
            kernel=KernelParams(weight=1.3, position=0.4, focus=0.2,
                                floor=0.5, ripple=0.1, phase=0.3),
        )
        vec = spec.to_vector()
        spec2 = ModulationSpec(component="attn", n_sites=8)
        spec2.from_vector(vec)
        self.assertEqual(spec.kernel, spec2.kernel)

    def test_pack_unpack_multiple_specs(self):
        specs = [
            ModulationSpec("attn", 10),
            ModulationSpec("mlp", 10),
            ModulationSpec("single", 5),
        ]
        vec = pack_spec_vector(specs)
        self.assertEqual(len(vec), total_param_count(specs))
        specs2 = [ModulationSpec("attn", 10), ModulationSpec("mlp", 10),
                  ModulationSpec("single", 5)]
        unpack_spec_vector(specs2, vec)
        for a, b in zip(specs, specs2):
            self.assertEqual(a.kernel, b.kernel)

    def test_unpack_wrong_length_raises(self):
        specs = [ModulationSpec("attn", 4)]
        with self.assertRaises(ValueError):
            unpack_spec_vector(specs, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
