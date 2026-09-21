# SPDX-License-Identifier: MIT
"""Tests for the architecture ledger: coverage claims, test-count parity,
cleanroom scan verdict, and digest reproducibility."""

import unittest

from calibrix.studio import ledger


class TestLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = ledger.build_ledger()

    def test_full_taxonomy_covered(self):
        cov = self.ledger["coverage"]
        self.assertEqual(cov["implemented"], cov["expected"])
        self.assertEqual(cov["missing"], [])
        # every composed mapping is verified against the live core
        for fid, spec in cov["composed"].items():
            self.assertTrue(spec["verified"], f"stale composed mapping: {fid}")

    def test_feature_ids_unique_across_domains(self):
        seen = set()
        for dom in self.ledger["domains"]:
            for f in dom["features"]:
                self.assertNotIn(f["id"], seen)
                seen.add(f["id"])
        self.assertEqual(len(self.ledger["domains"]), 8)

    def test_services_s1_s8_parsed(self):
        ids = [f["id"] for f in self.ledger["services"]["entries"]]
        self.assertEqual(ids, [f"S{i}" for i in range(1, 9)])

    def test_test_counts_match_real_suite(self):
        # per-module attribution must sum to the suite total
        per_file = self.ledger["tests"]["per_file"]
        self.assertEqual(sum(per_file.values()), self.ledger["tests"]["total"])
        studio = self.ledger["tests"]["per_file"]["test_studio.py"]
        self.assertEqual(studio, 70 + 4 + 1)  # 70 baseline + 4 steering + 1 optim
        self.assertGreater(self.ledger["tests"]["total"], 130)

    def test_cleanroom_verdict_in_ledger(self):
        cr = self.ledger["provenance"]["cleanroom"]
        self.assertTrue(cr["clean"])
        self.assertEqual(cr["verdict"], "CLEANROOM-VERIFIED")
        # scanner-definition + license-boundary modules are recorded
        # exemptions, never hidden. This pins the exact set: adding an
        # exemption must be an explicit, reviewed decision in
        # governance.DEFAULT_SCAN_ALLOWLIST.
        self.assertEqual(set(cr["allowlisted"]),
                         {"calibrix/studio/governance.py",
                          "calibrix/heretic_bridge.py"})

    def test_spdx_grants_recorded(self):
        self.assertIn("MIT", self.ledger["provenance"]["spdx"])

    def test_build_digest_is_deterministic(self):
        again = ledger.build_ledger()
        self.assertEqual(again["build_digest"],
                         self.ledger["build_digest"])

    def test_domain_module_integrity(self):
        for dom in self.ledger["domains"]:
            self.assertGreater(dom["loc"], 100)
            self.assertGreater(dom["symbols"]["functions"], 2)
            self.assertGreater(dom["tests"], 0)


if __name__ == "__main__":
    unittest.main()
