"""Change 3: a single selected-candidate set must drive ``calibrate``'s
``group_specs`` construction *and* ``config``'s per-command tier validation
together -- filtering ``calibrate``'s input alone would leave
``_full_tier_missing`` still requiring every declared group's calibration
before ``generate-psd``/``metrics``/``analyze`` become resolvable. Without an
explicit ``--candidates-file``, behavior is unchanged: every declared group
is required/used, exactly as before Change 3.
"""
import tempfile
import unittest
from pathlib import Path

from pc_specific_psd import cli
from pc_specific_psd.compat_generation import write_json
from pc_specific_psd.tests._runner_test_support import load_test_config
from pc_specific_psd.tests.test_cli import _run


class CalibrateCandidatesFilterTests(unittest.TestCase):
    """Fixture config declares groups B5 and B6 (see ``test_calibrate_dry_run``
    in test_cli.py); a candidates file naming only B5 must narrow both the
    dry-run report and the real ``group_specs``/registry to B5 alone.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        loaded = load_test_config(Path(self._tmp.name))
        self.config_path = str(loaded.config_path)
        self.tmp_path = Path(self._tmp.name)
        self.candidates_path = self.tmp_path / "candidates.json"
        write_json(self.candidates_path, {"candidates": ["B5"], "reserve": []})

    def test_dry_run_group_ids_filtered_by_candidates_file(self):
        code, payload, _, _ = _run([
            "calibrate", "--config", self.config_path, "--dry-run",
            "--candidates-file", str(self.candidates_path),
        ])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")
        self.assertEqual(payload["group_ids"], ["B5"])

    def test_dry_run_without_candidates_file_lists_all_declared_groups(self):
        code, payload, _, _ = _run(["calibrate", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(sorted(payload["group_ids"]), ["B5", "B6"])

    def test_real_run_only_calibrates_selected_group(self):
        code, payload, _, _ = _run([
            "calibrate", "--config", self.config_path, "--allow-synthetic-basis",
            "--candidates-file", str(self.candidates_path),
        ])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "SELECTED")
        self.assertEqual(set(payload["group_taus"]), {"B5"})

    def test_generate_psd_dry_run_resolves_once_only_selected_group_is_calibrated(self):
        code, _, _, _ = _run([
            "calibrate", "--config", self.config_path, "--allow-synthetic-basis",
            "--candidates-file", str(self.candidates_path),
        ])
        self.assertEqual(code, 0)

        code, payload, stdout_text, _ = _run([
            "generate-psd", "--config", self.config_path, "--dry-run", "--stage", "preview",
            "--candidates-file", str(self.candidates_path),
        ])
        self.assertEqual(code, 0)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["status"], "dry-run")
        self.assertEqual(payload["candidates"], ["B5"])

    def test_generate_psd_dry_run_without_candidates_file_still_requires_every_declared_group(self):
        code, _, _, _ = _run([
            "calibrate", "--config", self.config_path, "--allow-synthetic-basis",
            "--candidates-file", str(self.candidates_path),
        ])
        self.assertEqual(code, 0)

        code, payload, stdout_text, _ = _run([
            "generate-psd", "--config", self.config_path, "--dry-run", "--stage", "preview",
        ])
        self.assertEqual(code, 0)
        self.assertIsNone(payload)
        self.assertIn("missing:", stdout_text)


if __name__ == "__main__":
    unittest.main()
