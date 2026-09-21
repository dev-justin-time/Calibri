import unittest

from calibrix.adapters import ScriptedAdapter
from calibrix.kernel import KernelParams, ModulationSpec
from calibrix.scorers import (
    DiversityDrop,
    EmptyRate,
    KLDrift,
    KeywordRate,
    LengthDrift,
    NFETrap,
    Prompt,
    ScorerContext,
    seed_prompts,
)


def _identity(adapter):
    specs = adapter.spec_sites()
    for s in specs:
        s.kernel = KernelParams(weight=1.0, position=0.5, focus=1e6, floor=1.0)
    adapter.apply_modulation(specs)


def _damaged(adapter, weight=1.6, floor=1.6):
    """Uniform over-gain -> large deviation -> refusal/empty behavior."""
    specs = adapter.spec_sites()
    for s in specs:
        s.kernel = KernelParams(weight=weight, position=0.5, focus=1e6, floor=floor)
    adapter.apply_modulation(specs)


class TestSeedPrompts(unittest.TestCase):
    def test_from_strings(self):
        ps = seed_prompts(["hello", "world"], system="sys")
        self.assertEqual(len(ps), 2)
        self.assertEqual(ps[0].system, "sys")
        self.assertEqual(ps[1].user, "world")

    def test_from_dicts(self):
        ps = seed_prompts([{"system": "s", "text": "t"}])
        self.assertEqual(ps[0].system, "s")
        self.assertEqual(ps[0].user, "t")

    def test_from_file(self):
        import os
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("line one\n\nline two\n")
            path = f.name
        try:
            ps = seed_prompts(path)
            self.assertEqual([p.user for p in ps], ["line one", "line two"])
        finally:
            os.unlink(path)


class TestKeywordRate(unittest.TestCase):
    def test_refusal_detection(self):
        adapter = ScriptedAdapter(n_layers=6)
        prompts = [Prompt(user="Do X"), Prompt(user="Do Y")]
        scorer = KeywordRate(prompts)

        ctx = ScorerContext(adapter)
        scorer.init(ctx)
        _identity(adapter)
        ctx2 = ScorerContext(adapter)
        scorer_ok = KeywordRate(prompts)
        scorer_ok.init(ctx2)
        score_ok = scorer_ok.get_score(ctx2)

        _damaged(adapter)
        ctx3 = ScorerContext(adapter)
        scorer_bad = KeywordRate(prompts)
        scorer_bad.init(ctx3)
        score_bad = scorer_bad.get_score(ctx3)

        self.assertEqual(score_ok.value, 0.0)
        self.assertEqual(score_bad.value, 1.0)


class TestEmptyRate(unittest.TestCase):
    def test_extreme_modulation_yields_empties(self):
        adapter = ScriptedAdapter(n_layers=6)
        prompts = [Prompt(user="a"), Prompt(user="b"), Prompt(user="c")]
        _damaged(adapter, weight=1.9, floor=1.9)
        scorer = EmptyRate(prompts)
        ctx = ScorerContext(adapter)
        scorer.init(ctx)
        score = scorer.get_score(ctx)
        self.assertGreater(score.value, 0.5)


class TestLengthDrift(unittest.TestCase):
    def test_drift_is_zero_at_baseline_and_grows_when_damaged(self):
        adapter = ScriptedAdapter(n_layers=6)
        prompts = [Prompt(user="question one"), Prompt(user="question two")]

        _identity(adapter)
        s1 = LengthDrift(prompts)
        c1 = ScorerContext(adapter)
        s1.init(c1)
        base_score = s1.get_score(c1)

        _damaged(adapter, weight=1.25, floor=1.25)
        s2 = LengthDrift(prompts)
        c2 = ScorerContext(adapter)
        s2.init(c2)
        drift_score = s2.get_score(c2)

        self.assertAlmostEqual(base_score.value, 0.0, places=6)
        self.assertGreater(drift_score.value, 0.0)


class TestDiversityDrop(unittest.TestCase):
    def test_diversity_ratio_at_baseline_is_one(self):
        adapter = ScriptedAdapter(n_layers=6)
        prompts = [Prompt(user=f"unique prompt {i}") for i in range(6)]
        _identity(adapter)
        s = DiversityDrop(prompts)
        c = ScorerContext(adapter)
        s.init(c)
        score = s.get_score(c)
        self.assertAlmostEqual(score.value, 1.0, places=6)


class TestKLDrift(unittest.TestCase):
    def test_kl_zero_at_baseline_positive_when_damaged(self):
        adapter = ScriptedAdapter(n_layers=6)
        prompts = [Prompt(user="p1"), Prompt(user="p2")]

        _identity(adapter)
        s = KLDrift(prompts)
        c = ScorerContext(adapter)
        s.init(c)
        self.assertAlmostEqual(s.get_score(c).value, 0.0, places=9)

        _damaged(adapter, weight=1.3, floor=1.3)
        s2 = KLDrift(prompts)
        c2 = ScorerContext(adapter)
        s2.init(c2)
        self.assertGreater(s2.get_score(c2).value, 0.0)


class TestNFETrap(unittest.TestCase):
    def test_flags_step_overrun(self):
        adapter = ScriptedAdapter(n_layers=4)
        prompts = [Prompt(user="x")]
        scorer = NFETrap(steps_budget=1)
        ctx = ScorerContext(adapter)
        scorer.init(ctx)
        ctx.get_responses(prompts)  # consumes emulated steps
        score = scorer.get_score(ctx)
        self.assertGreater(score.value, 1.0)


if __name__ == "__main__":
    unittest.main()
