"""Verify that the replacement experiment contains twenty entirely NEW trials."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from src.contract import OUTPUT_DIR, REPO_ROOT

RUN = Path(
    os.environ.get(
        "LRUREC_FRESH20_DIR", str(OUTPUT_DIR / "lrurec_fresh20_cuda_20261109")
    )
).resolve()
MANIFEST = RUN / "metrics" / "lrurec_fresh20_manifest.json"
COMPLETE = RUN / "metrics" / "lrurec_fresh20_complete.json"


@unittest.skipUnless(MANIFEST.exists(), "Fresh experiment not started")
class Fresh20Tests(unittest.TestCase):
    def test_new_twenty_trial_manifest(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["trials"], 20)
        self.assertEqual(manifest["old_trials_counted"], 0)
        self.assertEqual(manifest["old_checkpoints_loaded"], 0)
        self.assertEqual([p["trial"] for p in manifest["plans"]], list(range(1, 21)))
        configs = {tuple(sorted(p["params"].items())) for p in manifest["plans"]}
        self.assertEqual(len(configs), 20)
        for key in ["epochs", "patience", "batch_size", "seed", "threads", "device"]:
            self.assertEqual(len({p["params"][key] for p in manifest["plans"]}), 1)

    def test_existing_run_cannot_be_reused(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.adapters.lrurec_fresh20",
                "--run-dir",
                str(RUN),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileExistsError", result.stderr)

    @unittest.skipUnless(COMPLETE.exists(), "Fresh experiment still training")
    def test_complete_new_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.adapters.lrurec_fresh20",
                "--run-dir",
                str(RUN),
                "--check",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
        self.assertIn("FRESH20 CHECK PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
