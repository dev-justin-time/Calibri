"""Budget-guard integration tests: studio.billing wired into SearchEngine.

The engine must halt (not crash) when metered spend hits the configured
ceiling, skip trials it cannot afford, still validate/report whatever it
found before halting, and behave exactly as before when no budget is set.
"""

import unittest

from calibrix.adapters import ScriptedAdapter
from calibrix.engine import SearchConfig, SearchEngine
from calibrix.scorers import DiversityDrop, EmptyRate, KeywordRate, LengthDrift, Prompt

PRICE = 1.0  # USD per 1k steps -> eval cost = steps / 1000


def make_scorers(prompts):
    return [
        KeywordRate(prompts, score_name="Refusals"),
        LengthDrift(prompts),
        EmptyRate(prompts),
        DiversityDrop(prompts),
    ]


def probe_eval_cost():
    """Measure the actual metered step cost of one train evaluation."""
    prompts = [Prompt(user=f"probe prompt {i} about the sea") for i in range(6)]
    engine = SearchEngine(
        ScriptedAdapter(n_layers=6), make_scorers(prompts), prompts,
        config=SearchConfig(n_trials=2, popsize=2, optimizer="simple"),
    )
    engine.run()
    steps = [r.steps for r in engine.meter.records if r.phase == "search" and r.steps > 0]
    assert steps, "probe run recorded no search-phase step usage"
    return steps[0]


class TestEngineBudget(unittest.TestCase):
    def setUp(self):
        self.eval_steps = probe_eval_cost()
        self.eval_cost = self.eval_steps / 1000.0 * PRICE
        self.prompts = [Prompt(user=f"prompt number {i} about the sea and sky")
                        for i in range(6)]

    def _engine(self, n_trials=3, popsize=4, **cfg):
        cfg.setdefault("price_per_1k_steps", PRICE)
        cfg.setdefault("optimizer", "simple")
        config = SearchConfig(n_trials=n_trials, popsize=popsize, **cfg)
        return SearchEngine(
            ScriptedAdapter(n_layers=6), make_scorers(self.prompts),
            self.prompts, config=config,
        )

    # -- legacy behavior: no budget configured --------------------------
    def test_unbounded_run_runs_all_trials(self):
        engine = self._engine(n_trials=3, popsize=4)
        result = engine.run()
        self.assertEqual(result["n_evals"], 3 * 4)
        budget = result["budget"]
        self.assertFalse(budget["halted"])
        self.assertEqual(budget["trials_skipped_budget"], 0)
        self.assertIsNone(budget["max_budget_usd"])
        self.assertFalse(budget["breaker_tripped"])
        self.assertGreater(budget["spent_usd"], 0.0)

    # -- BudgetGuard: halt at the ceiling --------------------------------
    def test_run_halts_when_budget_exhausted(self):
        engine = self._engine(n_trials=3, popsize=4,
                              max_budget_usd=self.eval_cost)
        result = engine.run()
        # Exactly one evaluation was affordable; the run stopped after it.
        self.assertEqual(result["n_evals"], 1)
        self.assertTrue(result["budget"]["halted"])
        self.assertAlmostEqual(result["budget"]["spent_usd"],
                               self.eval_cost, places=9)
        self.assertAlmostEqual(result["budget"]["remaining_usd"], 0.0, places=9)
        # Results found before the halt are still reported.
        self.assertIsNotNone(result["best"])
        self.assertEqual(len(result["trials"]), 1)

    def test_unaffordable_trials_are_skipped_then_run_halts(self):
        # Budget covers 2 evaluations plus a sliver: trial 3+ must be
        # skipped, not charged, and the run must halt without hanging.
        engine = self._engine(n_trials=3, popsize=4,
                              max_budget_usd=2.5 * self.eval_cost)
        result = engine.run()
        self.assertEqual(result["n_evals"], 2)
        budget = result["budget"]
        self.assertTrue(budget["halted"])
        # Gen 0: 2 run + 2 skipped; gen 1: all 4 skipped -> halt; gen 2
        # is never entered, so its candidates are not counted as skipped.
        self.assertEqual(budget["trials_skipped_budget"], 6)
        # Skipped trials consumed nothing: spend == 2 evaluations.
        self.assertAlmostEqual(budget["spent_usd"], 2 * self.eval_cost, places=9)

    def test_preflight_rejects_budget_below_one_evaluation(self):
        engine = self._engine(n_trials=3, popsize=4,
                              max_budget_usd=self.eval_cost / 10.0)
        with self.assertRaises(ValueError):
            engine.run()

    # -- CostBreaker: fail-closed killswitch ------------------------------
    def test_breaker_trips_and_blocks_validation(self):
        # Quote == one evaluation, zero tolerance: the second evaluation
        # crosses the limit, the breaker trips, and holdout validation
        # (paid work) is refused outright.
        engine = self._engine(n_trials=3, popsize=4,
                              max_budget_usd=100.0,
                              quoted_usd=self.eval_cost,
                              breaker_tolerance=0.0)
        result = engine.run()
        budget = result["budget"]
        self.assertTrue(budget["breaker_tripped"])
        self.assertTrue(budget["halted"])
        self.assertGreaterEqual(result["n_evals"], 2)
        self.assertLessEqual(result["n_evals"], 2 * 4)
        # Breaker forbids further paid work -> holdout never ran.
        self.assertTrue(all(not t.holdout for t in result["trials"]))
        self.assertFalse(result["overfit"]["flagged"])

    def test_breaker_requires_price(self):
        with self.assertRaises(ValueError):
            self._engine(quoted_usd=5.0, price_per_1k_steps=0.0)


if __name__ == "__main__":
    unittest.main()
