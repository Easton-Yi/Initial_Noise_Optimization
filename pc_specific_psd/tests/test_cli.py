"""End-to-end smoke tests for ``cli.main`` against the CPU/synthetic fixture
harness (``_runner_test_support``) -- never touching diffusers/HF. Covers:
argparse wiring and JSON output shape for every command's ``--dry-run`` path,
per-command config-tier refusal (``generate-psd`` etc. must refuse before
calibration is SELECTED; ``probe`` must succeed with no calibration at all),
the zero-candidates refusal chain, and one real (non-dry-run) generation
smoke test with a patched-in fake adapter.
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pc_specific_psd import cli, manifests
from pc_specific_psd.compat_generation import read_json, write_json
from pc_specific_psd.tests._runner_test_support import (
    FakeAdapterPCA,
    freeze_selected_calibration,
    load_test_config,
)
from pc_specific_psd.tests.test_review import _fill_probe_row


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = cli.main(argv)
        except SystemExit as exc:
            code = exc.code
    stdout_text = out.getvalue()
    try:
        payload = json.loads(stdout_text)
    except json.JSONDecodeError:
        payload = None
    return code, payload, stdout_text, err.getvalue()


class DryRunSmokeTests(unittest.TestCase):
    """Every command must accept --dry-run and return exit code 0, reporting
    planned actions/required inputs rather than raising -- even before any
    calibration/candidate artifacts exist on disk.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        loaded = load_test_config(Path(self._tmp.name))
        self.config_path = str(loaded.config_path)

    def test_validate_config(self):
        code, payload, _, _ = _run(["validate-config", "--config", self.config_path])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "ok")

    def test_legacy_workflow_explicitly_rejects_effect_v2(self):
        config_path = Path(__file__).parents[1] / "configs" / "sdxl_turbo_pca_v2.yaml"
        code, _, _, err = _run(["workflow", "--config", str(config_path), "--dry-run"])
        self.assertEqual(code, 1)
        self.assertIn("workflow does not support effect_size_v2", err)
        self.assertIn("basis-stability", err)

    def test_build_basis_dry_run_reports_missing_dataset_manifest(self):
        code, payload, _, _ = _run(["build-basis", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry_run")
        self.assertFalse(payload["ready"])

    def test_inspect_basis_dry_run_reports_existing_synthetic_basis(self):
        code, payload, _, _ = _run(["inspect-basis", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry_run")
        self.assertTrue(payload["ready"])

    def test_inspect_basis_real_run_rejects_synthetic_without_flag(self):
        code, _, _, err = _run(["inspect-basis", "--config", self.config_path])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_inspect_basis_real_run_with_allow_synthetic_flag(self):
        code, payload, _, _ = _run(["inspect-basis", "--config", self.config_path, "--allow-synthetic-basis"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "ok")

    def test_probe_dry_run_succeeds_with_needs_calibration_fields_unset(self):
        """The plan's verification step 4: probe --dry-run must succeed
        despite the calibration registry never having been resolved.
        """
        code, payload, _, _ = _run(["probe", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")
        self.assertGreater(payload["image_count"], 0)

    def test_probe_rho020_followup_dry_run_reports_required_inputs(self):
        code, payload, _, _ = _run(["probe", "--config", self.config_path, "--rho020-followup", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")
        self.assertIn("--review-file", payload["requires"])

    def test_probe_candidate_recheck_dry_run_reports_required_inputs(self):
        code, payload, _, _ = _run(["probe", "--config", self.config_path, "--candidate-recheck", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")
        self.assertIn("--group-ids", payload["requires"])

    def test_probe_mutually_exclusive_followup_flags_rejected(self):
        code, _, _, err = _run([
            "probe", "--config", self.config_path, "--rho020-followup", "--candidate-recheck", "--dry-run",
        ])
        self.assertEqual(code, 2)
        self.assertIn("not allowed", err)

    def test_export_review_dry_run(self):
        code, payload, _, _ = _run(["export-review", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")
        self.assertGreater(payload["pair_count"], 0)

    def test_select_candidates_dry_run(self):
        code, payload, _, _ = _run(["select-candidates", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")

    def test_calibrate_dry_run(self):
        code, payload, _, _ = _run(["calibrate", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")
        self.assertEqual(payload["gate_candidate_count"], 1)
        self.assertEqual(sorted(payload["group_ids"]), ["B5", "B6"])

    def test_validate_noise_dry_run(self):
        code, payload, _, _ = _run(["validate-noise", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "dry-run")

    def test_generate_psd_dry_run_before_calibration_reports_unresolved_fields(self):
        """The plan's verification step 5: generate-psd --dry-run must refuse
        and list unresolved needs_calibration fields (as an exit-0 dry-run
        report, not a crash) before any calibration has been run.
        """
        code, payload, stdout_text, _ = _run([
            "generate-psd", "--config", self.config_path, "--dry-run", "--stage", "preview",
        ])
        self.assertEqual(code, 0)
        self.assertIsNone(payload)
        self.assertIn("dry-run:", stdout_text)
        self.assertIn("missing:", stdout_text)

    def test_generate_psd_real_run_before_calibration_refuses(self):
        code, _, _, err = _run(["generate-psd", "--config", self.config_path, "--stage", "preview"])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_export_gallery_review_dry_run_before_calibration_reports_unresolved_fields(self):
        code, payload, stdout_text, _ = _run(["export-gallery-review", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIsNone(payload)
        self.assertIn("missing:", stdout_text)

    def test_metrics_dry_run_before_calibration_reports_unresolved_fields(self):
        """metrics is a "full"-tier command (config.COMMAND_TIERS), gated the
        same way as generate-psd: --dry-run reports unresolved fields rather
        than crashing, before ever reaching _cmd_metrics's own dry-run branch.
        """
        code, payload, stdout_text, _ = _run(["metrics", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIsNone(payload)
        self.assertIn("missing:", stdout_text)

    def test_analyze_dry_run_before_calibration_reports_unresolved_fields(self):
        code, payload, stdout_text, _ = _run(["analyze", "--config", self.config_path, "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIsNone(payload)
        self.assertIn("missing:", stdout_text)


class ZeroCandidateRefusalChainTests(unittest.TestCase):
    """Drives probe -> export-review -> select-candidates with an all-absent
    annotation fixture (nominates nothing) through the real CLI, then checks
    downstream commands treat "0 candidates" as a real branch: generate-psd
    returns no run, and export-preview-review/export-gallery-review refuse
    outright rather than building an empty package.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        loaded = load_test_config(Path(self._tmp.name))
        self.config_path = str(loaded.config_path)
        self.tmp_path = Path(self._tmp.name)

    def test_zero_candidates_refuses_downstream_review_exports(self):
        with mock.patch("pc_specific_psd.probing.SDXLTurboAdapterPCA", FakeAdapterPCA):
            code, _, _, _ = _run(["probe", "--config", self.config_path, "--allow-synthetic-basis"])
            self.assertEqual(code, 0)

        code, payload, _, _ = _run(["export-review", "--config", self.config_path])
        self.assertEqual(code, 0)
        review_path = Path(payload["review_template_path"])
        mapping_path = Path(payload["mapping_path"])

        raw_rows = read_json(review_path)
        filled_rows = [_fill_probe_row(row) for row in raw_rows]
        filled_path = self.tmp_path / "filled_probe_review.json"
        write_json(filled_path, filled_rows)

        code, payload, _, _ = _run([
            "select-candidates", "--config", self.config_path,
            "--review-file", str(filled_path), "--mapping-file", str(mapping_path),
        ])
        self.assertEqual(code, 0)
        self.assertEqual(payload["candidates"], [])
        candidates_path = Path(payload["output_path"])

        code, payload, _, _ = _run([
            "calibrate", "--config", self.config_path, "--allow-synthetic-basis",
        ])
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "SELECTED")

        code, payload, _, _ = _run([
            "generate-psd", "--config", self.config_path, "--stage", "preview",
            "--allow-synthetic-basis", "--candidates-file", str(candidates_path),
        ])
        self.assertEqual(code, 0)
        self.assertIsNone(payload["run_dir"])

        code, _, _, err = _run([
            "export-preview-review", "--config", self.config_path, "--candidates-file", str(candidates_path),
        ])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

        code, _, _, err = _run([
            "export-gallery-review", "--config", self.config_path, "--candidates-file", str(candidates_path),
        ])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)


class SelectCandidatesReviewFileIngestionRefusalTests(unittest.TestCase):
    """``select-candidates`` and ``probe --rho020-followup`` both route their
    own ``--review-file``/``--mapping-file`` input through the same shared
    ``cli._load_and_ingest_probe_review`` helper (which itself calls
    ``review.ingest_probe_review()``) -- this checks both call sites refuse
    identically, before ever running the nomination rule or the trigger
    check, when that file is missing or fails schema validation.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        loaded = load_test_config(Path(self._tmp.name))
        self.config_path = str(loaded.config_path)
        self.tmp_path = Path(self._tmp.name)

        with mock.patch("pc_specific_psd.probing.SDXLTurboAdapterPCA", FakeAdapterPCA):
            code, _, _, _ = _run(["probe", "--config", self.config_path, "--allow-synthetic-basis"])
            self.assertEqual(code, 0)
        code, payload, _, _ = _run(["export-review", "--config", self.config_path])
        self.assertEqual(code, 0)
        self.review_path = Path(payload["review_template_path"])
        self.mapping_path = Path(payload["mapping_path"])

    def _write_malformed_review(self) -> Path:
        raw_rows = read_json(self.review_path)
        filled_rows = [_fill_probe_row(row) for row in raw_rows]
        del filled_rows[0]["evidence"]  # required field, per test_review.py::IngestProbeReviewTests
        malformed_path = self.tmp_path / "malformed_probe_review.json"
        write_json(malformed_path, filled_rows)
        return malformed_path

    def test_select_candidates_refuses_on_missing_review_file(self):
        missing_path = self.tmp_path / "does_not_exist.json"
        code, payload, _, err = _run([
            "select-candidates", "--config", self.config_path,
            "--review-file", str(missing_path), "--mapping-file", str(self.mapping_path),
        ])
        self.assertEqual(code, 1)
        self.assertIsNone(payload)
        self.assertIn("error:", err)
        self.assertIn("not found", err)

    def test_select_candidates_refuses_on_malformed_review_file(self):
        malformed_path = self._write_malformed_review()
        code, payload, _, err = _run([
            "select-candidates", "--config", self.config_path,
            "--review-file", str(malformed_path), "--mapping-file", str(self.mapping_path),
        ])
        self.assertEqual(code, 1)
        self.assertIsNone(payload)
        self.assertIn("error:", err)

    def test_rho020_followup_refuses_on_missing_review_file(self):
        missing_path = self.tmp_path / "does_not_exist.json"
        code, payload, _, err = _run([
            "probe", "--config", self.config_path, "--rho020-followup",
            "--review-file", str(missing_path), "--mapping-file", str(self.mapping_path),
        ])
        self.assertEqual(code, 1)
        self.assertIsNone(payload)
        self.assertIn("error:", err)
        self.assertIn("not found", err)

    def test_rho020_followup_refuses_on_malformed_review_file(self):
        malformed_path = self._write_malformed_review()
        code, payload, _, err = _run([
            "probe", "--config", self.config_path, "--rho020-followup",
            "--review-file", str(malformed_path), "--mapping-file", str(self.mapping_path),
        ])
        self.assertEqual(code, 1)
        self.assertIsNone(payload)
        self.assertIn("error:", err)


class UnverifiedAdapterRefusalTests(unittest.TestCase):
    """Plan §2.3/§10/§11: an adapter other than the one verified basis/PC
    group partition/codec was built and tested against (SDXL-Turbo) must be
    refused as `not_verified` at the CLI, not silently run with a mismatched
    basis. This must hold for every command tier and under --dry-run too --
    dry-run reports planned actions for a resolvable config, it must not
    paper over an unsupported model entirely.
    """

    def _write_flux_config(self, tmp_path: Path) -> str:
        import yaml

        from pc_specific_psd.tests._runner_test_support import RAW_CONFIG

        raw = {**RAW_CONFIG, "model": {**RAW_CONFIG["model"], "adapter": "flux2klein"}}
        path = tmp_path / "flux_config.yaml"
        path.write_text(yaml.safe_dump(raw))
        return str(path)

    def test_validate_config_refuses_flux_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_flux_config(Path(tmp))
            code, payload, _, err = _run(["validate-config", "--config", config_path])
            self.assertEqual(code, 1)
            self.assertIsNone(payload)
            self.assertIn("not_verified", err)

    def test_build_basis_dry_run_still_refuses_flux_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_flux_config(Path(tmp))
            code, payload, _, err = _run(["build-basis", "--config", config_path, "--dry-run"])
            self.assertEqual(code, 1)
            self.assertIsNone(payload)
            self.assertIn("not_verified", err)

    def test_probe_dry_run_still_refuses_flux_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_flux_config(Path(tmp))
            code, payload, _, err = _run(["probe", "--config", config_path, "--dry-run"])
            self.assertEqual(code, 1)
            self.assertIsNone(payload)
            self.assertIn("not_verified", err)

    def test_generate_psd_refuses_flux_adapter_before_any_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = self._write_flux_config(Path(tmp))
            code, payload, _, err = _run([
                "generate-psd", "--config", config_path, "--stage", "preview",
                "--candidates-file", str(Path(tmp) / "candidates.json"),
            ])
            self.assertEqual(code, 1)
            self.assertIsNone(payload)
            self.assertIn("not_verified", err)


class RealPreviewGenerationSmokeTest(unittest.TestCase):
    """One real (non-dry-run) generation path end-to-end through the CLI,
    with a manually-written candidates.json (nomination itself is covered by
    test_review.py::SelectCandidatesTests) and a frozen calibration registry
    (calibration selection itself is covered by test_calibration*.py), so
    this test is purely about generate-psd's own CLI wiring: it patches
    runner.SDXLTurboAdapterPCA so no diffusers/HF dependency is touched.
    """

    def test_generate_psd_preview_real_run_with_one_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loaded = load_test_config(tmp_path)
            # The "full" config tier (config._full_tier_missing) requires a
            # frozen tau selection for every configured PC group, not just
            # the one being generated -- unlike _build_scored_run in
            # test_analysis.py, this goes through the real CLI's
            # config.resolve_config, which enforces that tier gate.
            freeze_selected_calibration(loaded, group_ids=[group.group_id for group in loaded.psd.groups])
            config_path = str(loaded.config_path)

            candidates_path = tmp_path / "candidates.json"
            write_json(candidates_path, {"candidates": ["B5"], "reserve": []})

            with mock.patch("pc_specific_psd.runner.SDXLTurboAdapterPCA", FakeAdapterPCA):
                code, payload, _, _ = _run([
                    "generate-psd", "--config", config_path, "--stage", "preview",
                    "--allow-synthetic-basis", "--candidates-file", str(candidates_path),
                    "--run-id", "preview_smoke",
                ])

            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "ok")
            run_dir = Path(payload["run_dir"])
            self.assertTrue(run_dir.exists())
            sample_dirs = list(run_dir.rglob("sample.jsonl"))
            self.assertEqual(len(sample_dirs), manifests.preview_image_count(1))


if __name__ == "__main__":
    unittest.main()
