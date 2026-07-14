import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.scan_dataset import (
    EstimateConfig,
    _member_index,
    _year_label,
    build_report,
    discover_archives,
    scan_index,
)


def _game(year="2023", rule=None):
    data = {
        "year": year,
        "log": [[
            {"qipai": {"shoupai": ["", "", "", ""]}},
            {"zimo": {"l": 0, "p": "m1"}},
            {"dapai": {"l": 0, "p": "m1*"}},
            {"fulou": {"l": 1, "m": "m1-23"}},
            {"gang": {"l": 2, "m": "p5555"}},
            {"hule": {"l": 3}},
        ]],
    }
    if rule is not None:
        data["rule"] = rule
    return data


class DatasetScannerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.archive = self.root / "data2023.zip"
        with zipfile.ZipFile(self.archive, "w") as zf:
            zf.writestr("game-1.txt", json.dumps(_game(rule={"aka": True})))
            zf.writestr("game-2.json", json.dumps(_game()))
            zf.writestr("broken.txt", "{not json")
            zf.writestr("ignored.csv", "not,scanned")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_counts_events_tasks_malformed_and_dimensions(self):
        archives = discover_archives([str(self.root)])
        counts, elapsed = scan_index(_member_index(archives, None), workers=1)

        self.assertEqual(counts["archive_members"], 3)
        self.assertEqual(counts["files_scanned"], 3)
        self.assertEqual(counts["files_valid_json"], 2)
        self.assertEqual(counts["games"], 2)
        self.assertEqual(counts["rounds"], 2)
        self.assertEqual(counts["events"], 12)
        self.assertEqual(counts["malformed_json"], 1)
        self.assertEqual(counts["event_counts"]["dapai"], 2)
        self.assertEqual(counts["task_candidates"]["riichi"], 2)
        self.assertEqual(counts["task_positives"]["riichi"], 2)
        self.assertEqual(counts["years"]["2023"], 2)
        self.assertGreaterEqual(elapsed, 0.0)

    def test_archive_year_wins_over_unrelated_member_or_date_year(self):
        game = {"date": "2099-01-01"}
        self.assertEqual(
            _year_label("/raw/data2014.zip", "logs/2050-game.json", game),
            "2014",
        )

    def test_explicit_game_year_wins_over_archive_year(self):
        self.assertEqual(
            _year_label("/raw/data2014.zip", "game.json", {"year": "2013"}),
            "2013",
        )

    def test_parallel_scan_matches_single_process(self):
        indexed = _member_index([self.archive], None)
        single, _ = scan_index(indexed, workers=1)
        try:
            parallel, _ = scan_index(indexed, workers=2)
        except PermissionError as exc:
            self.skipTest(f"process semaphores unavailable in this sandbox: {exc}")
        for key in (
            "archive_members", "files_scanned", "files_valid_json", "games", "rounds", "events",
            "malformed_json", "event_counts", "task_candidates", "years", "rules",
        ):
            self.assertEqual(single[key], parallel[key])

    def test_report_estimates_prepared_size_and_ten_year_scale(self):
        counts, elapsed = scan_index(_member_index([self.archive], None), workers=1)
        report = build_report(
            [self.archive], counts, elapsed, 1, [], EstimateConfig(), 10
        )
        expected = counts["events"] * 8 + counts["rounds"] * 264 + counts["games"] * 64 + 8
        self.assertEqual(report["prepared_size_estimate"]["bytes"], expected)
        self.assertEqual(report["projection"]["scale_factor"], 10.0)
        self.assertEqual(report["projection"]["estimated_events"], counts["events"] * 10)

    def test_cli_writes_json_report(self):
        output = self.root / "reports" / "scan.json"
        result = subprocess.run(
            [
                sys.executable, "scripts/scan_dataset.py", str(self.archive),
                "--workers", "1", "--output", str(output),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], "dataset-scan-v1")
        self.assertEqual(report["counts"]["games"], 2)


if __name__ == "__main__":
    unittest.main()
