"""Change 2, touchpoint 1: the probe-group review artifact is a group-level
*consolidation* view of the same 84 probe images ``export_probe_review``
already uses -- ``export_probe_group_review``/``ingest_probe_group_review``
must produce/consume it with the tri-state existence/incomplete/complete
contract, and ``select_candidates_from_group_review`` must agree with
``select_candidates()`` on equivalent evidence, since both funnel through the
same shared ``_rank_and_select_candidates`` ranking tail.
"""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pc_specific_psd import manifests, probing, review
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, load_test_config


def _make_group_review_fixture(pass_spec, groups=("B1", "B2", "B3")):
    """One ``ProbeGroupReviewRow`` per (group, prompt) -- mirrors
    ``test_review._make_probe_fixture``'s pass_spec shape at group
    granularity: ``pass_spec`` maps ``(group_id, role_field) -> set of
    prompt_ids`` that should nominate; every other row is a non-qualifying
    "no change seen" answer.
    """
    rows = []
    field_by_group_prompt = {}
    for (group_id, field), prompt_ids in pass_spec.items():
        for prompt_id in prompt_ids:
            field_by_group_prompt[(group_id, prompt_id)] = field

    for prompt in manifests.PROMPTS:
        for group_id in groups:
            field = field_by_group_prompt.get((group_id, prompt.prompt_id))
            if field is not None:
                rows.append(review.ProbeGroupReviewRow(
                    group_id=group_id, prompt_id=prompt.prompt_id,
                    draws_with_change=review.LOCAL_PASS_MIN_BASE_DRAWS,
                    main_change_type=field, artifact_recurring=False,
                ))
            else:
                rows.append(review.ProbeGroupReviewRow(
                    group_id=group_id, prompt_id=prompt.prompt_id,
                    draws_with_change=0, main_change_type="none", artifact_recurring=False,
                ))
    return tuple(rows)


class SelectCandidatesFromGroupReviewAgreementTests(unittest.TestCase):
    """Each case here has a direct analogue in test_review.py's
    ``SelectCandidatesTests`` (same pass_spec shape, same expected outcome)
    -- proving the two entry points never silently diverge on ranking.
    """

    def test_empty_pool_returns_no_candidates(self):
        rows = _make_group_review_fixture(pass_spec={})
        result = review.select_candidates_from_group_review(rows, groups=manifests.PC_GROUPS, prompts=manifests.PROMPTS)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.reserve, ())

    def test_single_nominated_group_is_selected(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows = _make_group_review_fixture(pass_spec={("B1", "layout"): all_prompts})
        result = review.select_candidates_from_group_review(rows)
        self.assertEqual(result.candidates, ("B1",))
        self.assertEqual(result.evidence_by_group["B1"].coverage, 4)

    def test_structural_candidate_prioritized_over_appearance(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        two_prompts = {"p000", "p001"}
        rows = _make_group_review_fixture(pass_spec={
            ("B1", "pose"): two_prompts,
            ("B2", "color"): all_prompts,
        })
        result = review.select_candidates_from_group_review(rows)
        self.assertEqual(result.candidates[0], "B1")
        self.assertIn("B2", result.candidates)

    def test_max_two_candidates(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows = _make_group_review_fixture(pass_spec={
            ("B1", "layout"): all_prompts,
            ("B2", "pose"): all_prompts,
            ("B3", "shape"): all_prompts,
        }, groups=("B1", "B2", "B3"))
        result = review.select_candidates_from_group_review(rows)
        self.assertEqual(len(result.candidates), 2)

    def test_tie_broken_by_declared_pc_groups_order(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows = _make_group_review_fixture(pass_spec={
            ("B2", "layout"): all_prompts,
            ("B3", "layout"): all_prompts,
        }, groups=("B2", "B3"))
        result = review.select_candidates_from_group_review(
            rows, groups=(manifests.pc_group_by_id("B2"), manifests.pc_group_by_id("B3")),
        )
        self.assertEqual(result.candidates, ("B2", "B3"))

    def test_recurrence_below_threshold_is_not_nominated_but_reserved(self):
        one_prompt = {"p000"}
        rows = _make_group_review_fixture(pass_spec={("B1", "layout"): one_prompt})
        result = review.select_candidates_from_group_review(rows)
        self.assertEqual(result.candidates, ())
        self.assertIn("B1", result.reserve)

    def test_reserve_excludes_selected_candidates(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        one_prompt = {"p000"}
        rows = _make_group_review_fixture(pass_spec={
            ("B1", "layout"): all_prompts,
            ("B2", "pose"): one_prompt,
        }, groups=("B1", "B2"))
        result = review.select_candidates_from_group_review(rows)
        self.assertIn("B1", result.candidates)
        self.assertNotIn("B1", result.reserve)
        self.assertIn("B2", result.reserve)


class ArtifactRecurringDisqualifiesTests(unittest.TestCase):
    def test_recurring_artifact_disqualifies_an_otherwise_nominating_row(self):
        import dataclasses

        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows = _make_group_review_fixture(pass_spec={("B1", "layout"): all_prompts})
        rows = tuple(dataclasses.replace(r, artifact_recurring=True) if r.group_id == "B1" else r for r in rows)
        result = review.select_candidates_from_group_review(rows)
        self.assertNotIn("B1", result.candidates)
        self.assertNotIn("B1", result.reserve)


def _build_probe_run(tmp_path: Path) -> tuple:
    loaded = load_test_config(tmp_path)
    entries = probing.build_probing_manifest(
        channels=loaded.basis.channels, height=loaded.generation.config.height,
        width=loaded.generation.config.width, patch_size=loaded.basis.patch_size,
    )
    adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
    run_dir = probing.generate_probes(loaded, entries, run_id="probe", adapter=adapter, allow_synthetic_basis=True)
    return loaded, run_dir


class ExportProbeGroupReviewTests(unittest.TestCase):
    def test_writes_csv_with_one_row_per_group_and_prompt_all_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, run_dir = _build_probe_run(Path(tmp))
            result = review.export_probe_group_review(loaded, run_dir, output_dir=Path(tmp) / "review_out")

            with result["csv_path"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(manifests.PC_GROUPS) * len(manifests.PROMPTS))
            self.assertTrue(all(row["draws_with_change"] == "" for row in rows))
            self.assertTrue(all(row["main_change_type"] == "" for row in rows))
            self.assertTrue(all(row["artifact_recurring"] == "" for row in rows))

    def test_writes_one_contact_sheet_per_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, run_dir = _build_probe_run(Path(tmp))
            result = review.export_probe_group_review(loaded, run_dir, output_dir=Path(tmp) / "review_out")

            self.assertEqual(set(result["contact_sheet_paths"]), {g.group_id for g in manifests.PC_GROUPS})
            for path in result["contact_sheet_paths"].values():
                self.assertTrue(path.exists())
                with Image.open(path) as img:
                    img.verify()

    def test_default_output_dir_is_alongside_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, run_dir = _build_probe_run(Path(tmp))
            result = review.export_probe_group_review(loaded, run_dir)
            self.assertEqual(result["csv_path"].parent, Path(loaded.config_path).resolve().parent)


class IngestProbeGroupReviewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.loaded, self.run_dir = _build_probe_run(self.tmp_path)
        self.result = review.export_probe_group_review(self.loaded, self.run_dir, output_dir=self.tmp_path / "review_out")
        self.csv_path = self.result["csv_path"]

    def _rewrite(self, mutate):
        fieldnames = ["group_id", "prompt_id", "draws_with_change", "main_change_type", "artifact_recurring"]
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = mutate(rows)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_file_reports_missing(self):
        result = review.ingest_probe_group_review(self.tmp_path / "does_not_exist.csv")
        self.assertEqual(result.status, "missing")

    def test_untouched_template_is_incomplete_not_all_false(self):
        result = review.ingest_probe_group_review(self.csv_path)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(len(result.incomplete_keys), len(manifests.PC_GROUPS) * len(manifests.PROMPTS))

    def test_partially_filled_reports_incomplete_with_exact_remaining_keys(self):
        def mutate(rows):
            rows[0]["draws_with_change"] = "2"
            rows[0]["main_change_type"] = "layout"
            rows[0]["artifact_recurring"] = "false"
            return rows
        self._rewrite(mutate)
        result = review.ingest_probe_group_review(self.csv_path)
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(len(result.incomplete_keys), len(manifests.PC_GROUPS) * len(manifests.PROMPTS) - 1)

    def test_fully_filled_is_complete_and_round_trips_types(self):
        def mutate(rows):
            for row in rows:
                row["draws_with_change"] = "2"
                row["main_change_type"] = "layout"
                row["artifact_recurring"] = "true"
            return rows
        self._rewrite(mutate)
        result = review.ingest_probe_group_review(self.csv_path)
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.rows), len(manifests.PC_GROUPS) * len(manifests.PROMPTS))
        self.assertTrue(all(r.draws_with_change == 2 and r.main_change_type == "layout" and r.artifact_recurring is True for r in result.rows))

    def test_invalid_draws_with_change_raises(self):
        def mutate(rows):
            for row in rows:
                row["draws_with_change"] = "9"
                row["main_change_type"] = "layout"
                row["artifact_recurring"] = "false"
            return rows
        self._rewrite(mutate)
        with self.assertRaises(ValueError):
            review.ingest_probe_group_review(self.csv_path)

    def test_invalid_main_change_type_raises(self):
        def mutate(rows):
            for row in rows:
                row["draws_with_change"] = "2"
                row["main_change_type"] = "not_a_real_field"
                row["artifact_recurring"] = "false"
            return rows
        self._rewrite(mutate)
        with self.assertRaises(ValueError):
            review.ingest_probe_group_review(self.csv_path)


if __name__ == "__main__":
    unittest.main()
