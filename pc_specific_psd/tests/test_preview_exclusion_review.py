"""Change 2, touchpoint 2: preview-exclusion review is a per-condition
(group x sign) roll-up over the actual preview-run manifest, asking only a
yes/no structural-damage question -- ``export_preview_exclusion_review``/
``ingest_preview_exclusion_review``/``approved_condition_ids`` must honor the
tri-state existence/incomplete/complete contract and the "reference is always
approved unless explicitly caught by the caller" rule from the plan.
"""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pc_specific_psd import manifests, review, runner
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, freeze_selected_calibration, load_test_config


def _build_preview_run(tmp_path: Path, group_ids=("B5",)) -> tuple:
    loaded = load_test_config(tmp_path)
    freeze_selected_calibration(loaded, group_ids=group_ids)
    entries = manifests.build_preview_manifest_entries(list(group_ids))
    adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
    run_dir = runner.generate_manifest(loaded, entries, run_id="preview", force=False, adapter=adapter, allow_synthetic_basis=True)
    return loaded, run_dir


class ExportPreviewExclusionReviewTests(unittest.TestCase):
    def test_writes_csv_with_one_row_per_condition_all_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, run_dir = _build_preview_run(Path(tmp))
            result = review.export_preview_exclusion_review(loaded, run_dir, ["B5"], output_dir=Path(tmp) / "review_out")

            with result["csv_path"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["condition_id"] for row in rows], ["reference", "B5_plus", "B5_minus"])
            self.assertTrue(all(row["excluded"] == "" and row["change_type"] == "" for row in rows))

    def test_writes_one_contact_sheet_per_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, run_dir = _build_preview_run(Path(tmp))
            result = review.export_preview_exclusion_review(loaded, run_dir, ["B5"], output_dir=Path(tmp) / "review_out")

            self.assertEqual(set(result["contact_sheet_paths"]), {"reference", "B5_plus", "B5_minus"})
            for path in result["contact_sheet_paths"].values():
                self.assertTrue(path.exists())
                with Image.open(path) as img:
                    img.verify()

    def test_two_candidates_produce_five_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, run_dir = _build_preview_run(Path(tmp), group_ids=("B5", "B6"))
            result = review.export_preview_exclusion_review(loaded, run_dir, ["B5", "B6"], output_dir=Path(tmp) / "review_out")
            self.assertEqual(set(result["contact_sheet_paths"]), {"reference", "B5_plus", "B5_minus", "B6_plus", "B6_minus"})

    def test_zero_candidates_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            with self.assertRaises(ValueError):
                review.export_preview_exclusion_review(loaded, Path(tmp) / "nonexistent_run", [], output_dir=Path(tmp) / "out")


class IngestPreviewExclusionReviewTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.loaded, self.run_dir = _build_preview_run(self.tmp_path)
        self.result = review.export_preview_exclusion_review(self.loaded, self.run_dir, ["B5"], output_dir=self.tmp_path / "review_out")
        self.csv_path = self.result["csv_path"]

    def _rewrite(self, mutate):
        fieldnames = ["condition_id", "excluded", "change_type"]
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows = mutate(rows)
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_file_reports_missing(self):
        result = review.ingest_preview_exclusion_review(self.tmp_path / "does_not_exist.csv", ["B5"])
        self.assertEqual(result.status, "missing")

    def test_untouched_template_is_incomplete(self):
        result = review.ingest_preview_exclusion_review(self.csv_path, ["B5"])
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(set(result.incomplete_condition_ids), {"reference", "B5_plus", "B5_minus"})

    def test_excluded_true_without_change_type_is_incomplete(self):
        def mutate(rows):
            for row in rows:
                row["excluded"] = "true" if row["condition_id"] == "B5_plus" else "false"
            return rows
        self._rewrite(mutate)
        result = review.ingest_preview_exclusion_review(self.csv_path, ["B5"])
        self.assertEqual(result.status, "incomplete")
        self.assertIn("B5_plus", result.incomplete_condition_ids)

    def test_fully_filled_is_complete(self):
        def mutate(rows):
            for row in rows:
                if row["condition_id"] == "B5_plus":
                    row["excluded"] = "true"
                    row["change_type"] = "extra_object"
                else:
                    row["excluded"] = "false"
            return rows
        self._rewrite(mutate)
        result = review.ingest_preview_exclusion_review(self.csv_path, ["B5"])
        self.assertEqual(result.status, "complete")
        by_id = {r.condition_id: r for r in result.rows}
        self.assertTrue(by_id["B5_plus"].excluded)
        self.assertEqual(by_id["B5_plus"].change_type, "extra_object")
        self.assertFalse(by_id["B5_minus"].excluded)
        self.assertIsNone(by_id["B5_minus"].change_type)

    def test_invalid_excluded_value_raises(self):
        def mutate(rows):
            for row in rows:
                row["excluded"] = "maybe"
            return rows
        self._rewrite(mutate)
        with self.assertRaises(ValueError):
            review.ingest_preview_exclusion_review(self.csv_path, ["B5"])

    def test_unexpected_condition_id_raises(self):
        def mutate(rows):
            rows[0]["condition_id"] = "not_a_real_condition"
            rows[0]["excluded"] = "false"
            return rows
        self._rewrite(mutate)
        with self.assertRaises(ValueError):
            review.ingest_preview_exclusion_review(self.csv_path, ["B5"])


class ApprovedConditionIdsTests(unittest.TestCase):
    def test_excludes_rows_marked_excluded(self):
        rows = (
            review.PreviewExclusionRow(condition_id="reference", excluded=False, change_type=None),
            review.PreviewExclusionRow(condition_id="B5_plus", excluded=True, change_type="extra_object"),
            review.PreviewExclusionRow(condition_id="B5_minus", excluded=False, change_type=None),
        )
        approved = review.approved_condition_ids(rows)
        self.assertEqual(approved, ("reference", "B5_minus"))

    def test_reference_always_included_even_if_row_marks_it_excluded(self):
        """``approved_condition_ids`` itself never drops reference -- the plan
        assigns the hard-stop-on-broken-reference check to the caller
        (``workflow.py``), not to this function.
        """
        rows = (
            review.PreviewExclusionRow(condition_id="reference", excluded=True, change_type="extra_object"),
            review.PreviewExclusionRow(condition_id="B5_plus", excluded=False, change_type=None),
        )
        approved = review.approved_condition_ids(rows)
        self.assertIn("reference", approved)

    def test_reference_included_even_if_missing_from_rows_entirely(self):
        rows = (review.PreviewExclusionRow(condition_id="B5_plus", excluded=False, change_type=None),)
        approved = review.approved_condition_ids(rows)
        self.assertIn("reference", approved)

    def test_all_excluded_reduces_to_reference_only(self):
        rows = (
            review.PreviewExclusionRow(condition_id="reference", excluded=False, change_type=None),
            review.PreviewExclusionRow(condition_id="B5_plus", excluded=True, change_type="extra_object"),
            review.PreviewExclusionRow(condition_id="B5_minus", excluded=True, change_type="extra_object"),
        )
        approved = review.approved_condition_ids(rows)
        self.assertEqual(approved, ("reference",))


if __name__ == "__main__":
    unittest.main()
