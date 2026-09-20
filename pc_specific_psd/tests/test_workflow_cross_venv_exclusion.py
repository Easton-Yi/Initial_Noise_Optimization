"""Change 4b's cross-venv correctness requirement: when a full-pilot stage is
dispatched to a *different* Python interpreter via ``--generation-python``,
the child process only knows the group-level ``candidates.json`` unless the
parent also hands it ``--approved-conditions-file`` on the command line.
Without that flag, the child's own ``generate-psd --stage full`` would
regenerate every sign of every selected group -- silently reviving a sign
the human explicitly excluded at review 2 (e.g. keeping ``B5_minus`` while
dropping ``B5_plus``). This file isolates that one cross-venv argv contract,
separate from ``test_workflow.py``'s broader state-machine coverage.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pc_specific_psd import manifests, metrics as metrics_module, workflow
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, load_test_config
from pc_specific_psd.tests.test_analysis import FakeMetricRunner
from pc_specific_psd.tests.test_review_group_consolidation import _make_group_review_fixture

ALL_GROUP_IDS = tuple(g.group_id for g in manifests.PC_GROUPS)


def _write_csv(path: Path, fieldnames, rows_as_dicts) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_as_dicts:
            writer.writerow(row)


def _nominate_b5_only_rows():
    all_prompts = {p.prompt_id for p in manifests.PROMPTS}
    return _make_group_review_fixture(pass_spec={("B5", "layout"): all_prompts}, groups=ALL_GROUP_IDS)


class CrossVenvExclusionSurvivesDispatchTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.cfg = load_test_config(self.tmp_path)
        self.config_path = str(self.cfg.config_path)
        self.probe_review_path = self.tmp_path / "probe_group_review.csv"
        self.preview_review_path = self.tmp_path / "preview_exclusion_review.csv"

    def _run(self, **kwargs):
        kwargs.setdefault("allow_synthetic_basis", True)
        kwargs.setdefault("adapter", FakeAdapterPCA(self.cfg.model.as_model_config_dict()))
        return workflow.run_workflow(self.config_path, **kwargs)

    def _advance_to_approved_conditions(self, *, exclude_b5_plus: bool):
        self._run()  # export probe-group-review template
        _write_csv(
            self.probe_review_path,
            ["group_id", "prompt_id", "draws_with_change", "main_change_type", "artifact_recurring"],
            [
                {
                    "group_id": r.group_id, "prompt_id": r.prompt_id,
                    "draws_with_change": r.draws_with_change, "main_change_type": r.main_change_type,
                    "artifact_recurring": "true" if r.artifact_recurring else "false",
                }
                for r in _nominate_b5_only_rows()
            ],
        )
        self._run()  # derive candidates.json, export preview-exclusion template

        decisions = {
            "reference": (False, None),
            "B5_plus": (exclude_b5_plus, "extra_object" if exclude_b5_plus else None),
            "B5_minus": (False, None),
        }
        _write_csv(
            self.preview_review_path, ["condition_id", "excluded", "change_type"],
            [{"condition_id": cid, "excluded": "true" if excluded else "false", "change_type": change_type or ""}
             for cid, (excluded, change_type) in decisions.items()],
        )

    def test_full_pilot_dispatch_argv_carries_approved_conditions_excluding_the_dropped_sign(self):
        self._advance_to_approved_conditions(exclude_b5_plus=True)

        captured_argvs = []

        def _fake_run(cmd, **kwargs):
            captured_argvs.append(cmd)
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        # The stub reports success without writing any actual generation
        # output, so the in-process metrics stage that follows is expected to
        # fail (there is nothing under generations/ to score) -- irrelevant
        # here, since the full-pilot argv this test cares about is already
        # captured by the time that happens.
        with patch("pc_specific_psd.workflow.subprocess.run", side_effect=_fake_run):
            with self.assertRaises(metrics_module.MetricsError):
                self._run(generation_python="/opt/other/bin/python", metric_runner=FakeMetricRunner())

        full_pilot_argvs = [a for a in captured_argvs if "generate-psd" in a and "full" in a]
        self.assertEqual(len(full_pilot_argvs), 1)
        argv = full_pilot_argvs[0]

        self.assertIn("--approved-conditions-file", argv)
        approved_path = Path(argv[argv.index("--approved-conditions-file") + 1])
        self.assertTrue(approved_path.exists())

        approved = json.loads(approved_path.read_text())["approved_condition_ids"]
        self.assertIn("B5_minus", approved)
        self.assertNotIn("B5_plus", approved, "the excluded sign must not be handed to the child interpreter as approved")
        self.assertIn("reference", approved)

        # The child process still needs the group-level candidates file too
        # (both signs get calibrated/previewed together) -- only the
        # approved-conditions file is what narrows the *final* full-pilot
        # render down to the surviving signs.
        self.assertIn("--candidates-file", argv)

    def test_same_interpreter_never_shells_out_for_full_pilot(self):
        self._advance_to_approved_conditions(exclude_b5_plus=False)

        with patch("pc_specific_psd.workflow.subprocess.run") as mock_run:
            result = self._run(metric_runner=FakeMetricRunner())

        self.assertEqual(result["status"], "ok")
        # Patching ``pc_specific_psd.workflow.subprocess.run`` patches the
        # ``subprocess`` module globally (``workflow.subprocess`` is the same
        # module object), so unrelated calls from elsewhere in the process
        # (e.g. torch's own CPU-capability probing via ``lscpu``) can also
        # land on this mock; only our own dispatch calls pass an argv list
        # naming this package, so filter to those specifically.
        our_dispatch_calls = [
            call for call in mock_run.call_args_list
            if isinstance(call.args[0], list) and "pc_specific_psd" in call.args[0]
        ]
        self.assertEqual(our_dispatch_calls, [])


if __name__ == "__main__":
    unittest.main()
