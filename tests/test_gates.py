"""Gate behaviour on missing, corrupt, stale, mismatched and failing evidence (python -m unittest discover tests)."""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gates  # noqa: E402

NOW = datetime(2026, 10, 22, 22, 0, tzinfo=timezone.utc)
OK_BLOCKING = {"truncation_invariance": True, "label_alignment": True, "duplicates_timestamps": True}


def audit(**kw):
    d = {"generated_at_utc": (NOW - timedelta(hours=1)).isoformat(), "dataset_version": "ds-abc", "blocking": dict(OK_BLOCKING)}
    d.update(kw)
    return d


def gate(**kw):
    d = {"generated_at_utc": (NOW - timedelta(hours=20)).isoformat(), "model_run_ids": {"broad": "run-1"},
         "profiles": {"broad": {"20": {"status": "PAPER_TEST_ELIGIBLE", "failed_checks": [], "forward_ic": 0.03, "forward_ic_se": 0.01}}}}
    d.update(kw)
    return d


class AuditTests(unittest.TestCase):
    def test_fresh_matching_passing_audit_passes(self):
        self.assertEqual(gates.check_audit(audit(), "ds-abc", NOW), (True, None))

    def test_missing_audit_blocks(self):
        self.assertFalse(gates.check_audit(None, "ds-abc", NOW)[0])

    def test_stale_audit_blocks(self):
        ok, why = gates.check_audit(audit(generated_at_utc=(NOW - timedelta(hours=30)).isoformat()), "ds-abc", NOW)
        self.assertFalse(ok)
        self.assertIn("old", why)

    def test_audit_of_other_dataset_blocks(self):
        ok, why = gates.check_audit(audit(), "ds-new", NOW)
        self.assertFalse(ok)
        self.assertIn("ds-new", why)

    def test_failed_test_blocks(self):
        self.assertFalse(gates.check_audit(audit(blocking={**OK_BLOCKING, "label_alignment": False}), "ds-abc", NOW)[0])

    def test_incomplete_or_corrupt_audit_blocks(self):
        self.assertFalse(gates.check_audit(audit(blocking={"label_alignment": True}), "ds-abc", NOW)[0])
        self.assertFalse(gates.check_audit(audit(blocking={**OK_BLOCKING, "label_alignment": "yes"}), "ds-abc", NOW)[0])
        self.assertFalse(gates.check_audit(audit(generated_at_utc="not a date"), "ds-abc", NOW)[0])
        self.assertFalse(gates.check_audit({"blocking": OK_BLOCKING}, "ds-abc", NOW)[0])


class ForwardGateTests(unittest.TestCase):
    def test_fresh_gate_for_current_model_is_used(self):
        self.assertEqual(gates.check_forward_gate(gate(), "broad", 20, "run-1", NOW)[0], "PAPER_TEST_ELIGIBLE")

    def test_missing_or_corrupt_gate_is_unvalidated(self):
        self.assertEqual(gates.check_forward_gate(None, "broad", 20, "run-1", NOW)[0], "UNVALIDATED")
        self.assertEqual(gates.check_forward_gate({"profiles": {}}, "broad", 20, "run-1", NOW)[0], "UNVALIDATED")

    def test_stale_gate_is_unvalidated(self):
        g = gate(generated_at_utc=(NOW - timedelta(days=5)).isoformat())
        self.assertEqual(gates.check_forward_gate(g, "broad", 20, "run-1", NOW)[0], "UNVALIDATED")

    def test_gate_for_other_model_is_unvalidated(self):
        self.assertEqual(gates.check_forward_gate(gate(), "broad", 20, "run-2", NOW)[0], "UNVALIDATED")

    def test_unscored_horizon_or_unknown_status_is_unvalidated(self):
        self.assertEqual(gates.check_forward_gate(gate(), "broad", 5, "run-1", NOW)[0], "UNVALIDATED")
        g = gate(profiles={"broad": {"20": {"status": "LIVE", "failed_checks": []}}})
        self.assertEqual(gates.check_forward_gate(g, "broad", 20, "run-1", NOW)[0], "UNVALIDATED")


if __name__ == "__main__":
    unittest.main()
