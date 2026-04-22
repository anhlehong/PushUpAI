from __future__ import annotations

import unittest
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from hybrid_aqa_video_benchmark import (
    CASE_DEFINITIONS,
    EXPERT_VIDEO_FILENAME,
    run_benchmark,
)


class HybridAQAVideoBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        missing = []
        expert_path = root / EXPERT_VIDEO_FILENAME
        if not expert_path.exists():
            missing.append(str(expert_path))
        for case in CASE_DEFINITIONS.values():
            path = root / case["student"]
            if not path.exists():
                missing.append(str(path))

        if missing:
            raise unittest.SkipTest(
                "Missing benchmark videos: " + ", ".join(sorted(set(missing))))

        cls.report = run_benchmark(write_logs=True)

    def test_all_cases_status_ok(self) -> None:
        for case_id in ["self_baseline", "neck_drop_case", "core_brace_case"]:
            case = self.report["cases"].get(case_id, {})
            self.assertEqual(
                case.get("status"),
                "ok",
                msg=f"Case {case_id} is not ready: {case}",
            )

    def test_checklist_passed(self) -> None:
        failed_checks = [
            check for check in self.report["checks"] if not check["passed"]]
        self.assertFalse(failed_checks, msg=f"Failed checks: {failed_checks}")

    def test_non_pushup_case_is_rejected(self) -> None:
        case = self.report["cases"].get("non_pushup_case", {})
        self.assertIn(
            case.get("status"),
            {"quality_gate_failed", "no_student_rep"},
            msg=f"Non-pushup case should be rejected: {case}",
        )

    def test_core_brace_case_counts_at_least_five_reps(self) -> None:
        case = self.report["cases"].get("core_brace_case", {})
        self.assertGreaterEqual(
            int(case.get("student_rep_count", 0)),
            5,
            msg=f"Expected khong gong bung.mp4 to yield at least 5 reps: {case}",
        )


if __name__ == "__main__":
    unittest.main()
