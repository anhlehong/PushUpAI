from __future__ import annotations
from hybrid_aqa_video_benchmark import CASE_DEFINITIONS, run_benchmark

import unittest
from pathlib import Path
import sys

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


class HybridAQAVideoBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        missing = []
        for case in CASE_DEFINITIONS.values():
            for key in ["expert", "student"]:
                path = root / case[key]
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


if __name__ == "__main__":
    unittest.main()
