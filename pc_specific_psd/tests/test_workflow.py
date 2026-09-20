"""Change 1: ``workflow.run_workflow`` chains the existing 15-command CLI's
stages into one resumable state machine that STOPs exactly twice for human
review. Covers: stage sequencing/resumability, both STOP points and their
three existence/incomplete/complete review states, the review-changed-after
-consumption conflict check, smoke-check caching (pass cached, failure never
cached), the venv-dispatch decision, the two no-candidates/no-approved
-conditions terminal states, the reference-excluded hard stop, adapter reuse
across the whole state machine, and a full successful CPU end-to-end run.
"""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pc_specific_psd import manifests, runner, workflow
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, load_test_config
from pc_specific_psd.tests.test_analysis import FakeMetricRunner
from pc_specific_psd.tests.test_review_group_consolidation import _make_group_review_fixture

ALL_GROUP_IDS = tuple(g.group_id for g in manifests.PC_GROUPS)


def _write_probe_group_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_id", "prompt_id", "draws_with_change", "main_change_type", "artifact_recurring"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "group_id": row.group_id, "prompt_id": row.prompt_id,
                "draws_with_change": row.draws_with_change, "main_change_type": row.main_change_type,
                "artifact_recurring": "true" if row.artifact_recurring else "false",
            })


def _nominate_b5_only_rows():
    all_prompts = {p.prompt_id for p in manifests.PROMPTS}
    return _make_group_review_fixture(pass_spec={("B5", "layout"): all_prompts}, groups=ALL_GROUP_IDS)


def _write_preview_exclusion_csv(path: Path, decisions: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition_id", "excluded", "change_type"])
        writer.writeheader()
        for condition_id, (excluded, change_type) in decisions.items():
            writer.writerow({"condition_id": condition_id, "excluded": "true" if excluded else "false", "change_type": change_type or ""})


class WorkflowTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.cfg = load_test_config(self.tmp_path)
        self.config_path = str(self.cfg.config_path)
        self.probe_review_path = self.tmp_path / "probe_group_review.csv"
        self.preview_review_path = self.tmp_path / "preview_exclusion_review.csv"

    def _fake_adapter(self):
        return FakeAdapterPCA(self.cfg.model.as_model_config_dict())

    def _run(self, **kwargs):
        kwargs.setdefault("allow_synthetic_basis", True)
        kwargs.setdefault("adapter", self._fake_adapter())
        return workflow.run_workflow(self.config_path, **kwargs)


class DryRunTests(WorkflowTestCase):
    def test_dry_run_never_touches_a_model_and_reports_all_stages(self):
        result = workflow.run_workflow(self.config_path, dry_run=True)
        self.assertEqual(result["status"], "dry-run")
        stage_names = [s["stage"] for s in result["stages"]]
        self.assertEqual(len(stage_names), 14)
        # setUp's load_test_config already writes a synthetic basis file, so
        # the resume point skips straight past build_basis to probe.
        self.assertEqual(result["next_stage"], "probe")


class ReviewOneTriStateTests(WorkflowTestCase):
    def test_missing_file_exports_template_and_stops(self):
        result = self._run()
        self.assertEqual(result["status"], "awaiting_review_1")
        self.assertEqual(result["review_status"], "missing")
        self.assertTrue(Path(result["csv_path"]).exists())
        self.assertEqual(Path(result["csv_path"]).name, "probe_group_review.csv")
        self.assertTrue(self.probe_review_path.exists())

    def test_untouched_template_is_incomplete_on_resume(self):
        self._run()
        result = self._run()
        self.assertEqual(result["status"], "awaiting_review_1")
        self.assertEqual(result["review_status"], "incomplete")
        self.assertGreater(len(result["incomplete_keys"]), 0)

    def test_filled_template_proceeds_past_review_one(self):
        self._run()
        _write_probe_group_csv(self.probe_review_path, _nominate_b5_only_rows())
        result = self._run()
        self.assertNotEqual(result["status"], "awaiting_review_1")


class NoCandidatesTerminalTests(WorkflowTestCase):
    def test_all_no_signal_rows_stop_before_calibration(self):
        self._run()
        no_signal_rows = _make_group_review_fixture(pass_spec={}, groups=ALL_GROUP_IDS)
        _write_probe_group_csv(self.probe_review_path, no_signal_rows)

        with patch.object(workflow, "_stage_calibrate") as mock_calibrate, \
             patch.object(runner, "generate_preview") as mock_preview, \
             patch.object(runner, "generate_full_pilot") as mock_full:
            result = self._run()

        self.assertEqual(result["status"], "no_candidates")
        self.assertEqual(result["message"], workflow.NO_CANDIDATES_MESSAGE)
        mock_calibrate.assert_not_called()
        mock_preview.assert_not_called()
        mock_full.assert_not_called()


class ReviewConflictTests(WorkflowTestCase):
    def test_editing_probe_review_after_candidates_derived_raises(self):
        self._run()
        _write_probe_group_csv(self.probe_review_path, _nominate_b5_only_rows())
        self._run()  # derives candidates.json, records the review file's hash

        # Mutate the already-consumed review file (still a valid, complete file).
        _write_probe_group_csv(self.probe_review_path, _nominate_b5_only_rows())
        import time
        time.sleep(0.01)
        with self.probe_review_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")

        with self.assertRaises(workflow.WorkflowError):
            self._run()


class ReviewTwoTriStateAndTerminalTests(WorkflowTestCase):
    def _advance_to_review_two(self):
        self._run()
        _write_probe_group_csv(self.probe_review_path, _nominate_b5_only_rows())
        return self._run()

    def test_missing_preview_review_exports_template_and_stops(self):
        result = self._advance_to_review_two()
        self.assertEqual(result["status"], "awaiting_review_2")
        self.assertEqual(result["review_status"], "missing")
        self.assertTrue(Path(result["csv_path"]).exists())
        self.assertEqual(Path(result["csv_path"]).name, "preview_exclusion_review.csv")

    def test_untouched_preview_template_is_incomplete_on_resume(self):
        self._advance_to_review_two()
        result = self._run()
        self.assertEqual(result["status"], "awaiting_review_2")
        self.assertEqual(result["review_status"], "incomplete")

    def test_reference_excluded_is_a_hard_stop(self):
        self._advance_to_review_two()
        _write_preview_exclusion_csv(self.preview_review_path, {
            "reference": (True, "extra_object"), "B5_plus": (False, None), "B5_minus": (False, None),
        })
        with self.assertRaises(workflow.WorkflowError):
            self._run()

    def test_all_edited_conditions_excluded_stops_before_full_pilot(self):
        self._advance_to_review_two()
        _write_preview_exclusion_csv(self.preview_review_path, {
            "reference": (False, None), "B5_plus": (True, "extra_object"), "B5_minus": (True, "extra_object"),
        })
        with patch.object(runner, "generate_full_pilot") as mock_full:
            result = self._run()
        self.assertEqual(result["status"], "no_approved_conditions")
        self.assertEqual(result["message"], workflow.NO_APPROVED_CONDITIONS_MESSAGE)
        mock_full.assert_not_called()


class SmokeCheckCachingTests(WorkflowTestCase):
    def test_passing_result_is_cached_and_reused(self):
        cfg = self.cfg
        state: dict = {}
        adapter = self._fake_adapter()

        with patch("pc_specific_psd.workflow.smoke_check", wraps=workflow.smoke_check) as spy:
            first = workflow._run_smoke_check(cfg, state, adapter, allow_synthetic_basis=True)
            self.assertEqual(first["status"], "ok")
            self.assertEqual(spy.call_count, 1)

            second = workflow._run_smoke_check(cfg, state, adapter, allow_synthetic_basis=True)
            self.assertEqual(second["status"], "ok")
            self.assertEqual(second["smoke_check"], "cached")
            self.assertEqual(spy.call_count, 1, "a cached pass must not re-invoke smoke_check")

    def test_failure_is_never_cached_and_always_retried(self):
        cfg = self.cfg
        state: dict = {}
        adapter = self._fake_adapter()
        fake_result = type("R", (), {"passed": False, "reason": "boom"})()

        with patch("pc_specific_psd.workflow.smoke_check", return_value=fake_result) as spy:
            first = workflow._run_smoke_check(cfg, state, adapter, allow_synthetic_basis=True)
            self.assertEqual(first["status"], "smoke_check_failed")
            self.assertNotIn("smoke_check", state)

            second = workflow._run_smoke_check(cfg, state, adapter, allow_synthetic_basis=True)
            self.assertEqual(second["status"], "smoke_check_failed")
            self.assertEqual(spy.call_count, 2, "a failed check must always be retried, never cached")


class VenvDispatchTests(WorkflowTestCase):
    def _stub_subprocess(self, mock_run):
        mock_run.return_value = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def test_same_interpreter_never_dispatches(self):
        import sys
        with patch("pc_specific_psd.workflow.subprocess.run") as mock_run:
            result = workflow._maybe_subprocess(sys.executable, ["probe", "--config", "x"])
        self.assertIsNone(result)
        mock_run.assert_not_called()

    def test_different_interpreter_dispatches_with_expected_argv(self):
        with patch("pc_specific_psd.workflow.subprocess.run") as mock_run:
            self._stub_subprocess(mock_run)
            result = workflow._maybe_subprocess("/opt/other/bin/python", ["probe", "--config", "x", "--force"])
        self.assertIsNotNone(result)
        mock_run.assert_called_once()
        argv = mock_run.call_args[0][0]
        self.assertEqual(argv, ["/opt/other/bin/python", "-m", "pc_specific_psd", "probe", "--config", "x", "--force"])

    def test_full_pilot_dispatch_argv_includes_approved_conditions_file(self):
        self._run()
        _write_probe_group_csv(self.probe_review_path, _nominate_b5_only_rows())
        self._run()
        _write_preview_exclusion_csv(self.preview_review_path, {
            "reference": (False, None), "B5_plus": (False, None), "B5_minus": (True, "extra_object"),
        })

        seen_argvs = []

        def _record(python_path, argv):
            seen_argvs.append(argv)
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        # The stubbed subprocess.run reports success without actually writing
        # a metrics/per_group.csv, so the subsequent in-process analyze stage
        # is expected to fail -- irrelevant here, since the full-pilot argv
        # we care about is already captured by the time that happens.
        from pc_specific_psd.analysis import AnalysisError
        with patch("pc_specific_psd.workflow.subprocess.run", side_effect=lambda cmd, **kw: _record(cmd[0], cmd)):
            with self.assertRaises(AnalysisError):
                self._run(generation_python="/opt/other/bin/python", metrics_python="/opt/other/bin/python",
                           metric_runner=FakeMetricRunner())

        full_pilot_argv = next(a for a in seen_argvs if "--stage" in a and "full" in a)
        self.assertIn("--approved-conditions-file", full_pilot_argv)
        approved_path = Path(full_pilot_argv[full_pilot_argv.index("--approved-conditions-file") + 1])
        self.assertTrue(approved_path.exists())
        import json
        approved = json.loads(approved_path.read_text())["approved_condition_ids"]
        self.assertNotIn("B5_minus", approved)
        self.assertIn("B5_plus", approved)
        self.assertIn("reference", approved)


class FullEndToEndTests(WorkflowTestCase):
    def test_full_run_reuses_one_adapter_instance_across_every_real_stage(self):
        # Pre-author both review files up front so a single run_workflow call
        # sails straight through every stage without stopping -- this is what
        # actually proves adapter reuse: any call site that silently
        # constructed a *real* SDXLTurboAdapterPCA instead of reusing the
        # injected fake would hang/crash trying to load "org/sdxl-turbo-fake".
        _write_probe_group_csv(self.probe_review_path, _nominate_b5_only_rows())
        _write_preview_exclusion_csv(self.preview_review_path, {
            "reference": (False, None), "B5_plus": (False, None), "B5_minus": (False, None),
        })

        adapter = self._fake_adapter()
        result = self._run(adapter=adapter, metric_runner=FakeMetricRunner())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["candidates"], ["B5"])
        self.assertEqual(set(result["approved_condition_ids"]), {"reference", "B5_plus", "B5_minus"})
        self.assertEqual(result["executed_image_count"], 144)
        self.assertEqual(result["pre_exclusion_image_count_ceiling"], 144)

        # 1 (pre-probe smoke) + 84 (probe) + 36 (preview) + 144 (full pilot);
        # the pre-full-pilot smoke check reuses the cached pass (same config
        # + basis), so it contributes zero extra calls.
        self.assertEqual(adapter.call_count, 1 + 84 + 36 + 144)

    def test_resuming_after_full_success_is_a_no_op_status_ok(self):
        _write_probe_group_csv(self.probe_review_path, _nominate_b5_only_rows())
        _write_preview_exclusion_csv(self.preview_review_path, {
            "reference": (False, None), "B5_plus": (False, None), "B5_minus": (False, None),
        })
        self._run(metric_runner=FakeMetricRunner())
        result = self._run(metric_runner=FakeMetricRunner())
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
