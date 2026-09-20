"""analyze_run must explicitly mark the gallery-review human-rating axis as
"incomplete" -- never silently omit it -- whenever gallery-review annotations
haven't been ingested for the run under analysis, and must flip it to "ok",
correctly keyed by real condition_id (not blind label), once a real
export_gallery_review/ingest_gallery_review round trip is supplied.
"""
import tempfile
import unittest
from pathlib import Path

from pc_specific_psd import analysis, manifests, metrics, review, runner
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, freeze_selected_calibration, load_test_config
from pc_specific_psd.tests.test_metrics import FakeMetricRunner


def _build_scored_run(tmp: Path, group_ids: list[str]) -> Path:
    loaded = load_test_config(tmp)
    freeze_selected_calibration(loaded, group_ids=group_ids)
    adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
    entries = manifests.build_full_pilot_manifest_entries(group_ids)
    run_dir = runner.generate_manifest(loaded, entries, run_id="scored", force=False, adapter=adapter, allow_synthetic_basis=True)
    metrics.run_metrics(run_dir, metric_runner=FakeMetricRunner())
    return run_dir


def _fill_gallery_review_template(blind_rows, mapping) -> list[dict]:
    """A filled-in blind template picking the first blind label as the
    reviewer's preference and marking every artifact field absent -- content
    doesn't matter for this test, only that every required field is present
    and valid so ``ingest_gallery_review`` accepts it.
    """
    filled = []
    for row in blind_rows:
        ratings = {}
        for label in row["blind_labels"]:
            ratings[label] = {
                "per_image_artifacts": [
                    {field: "absent" for field in review.ARTIFACT_FIELDS}
                    for _ in range(manifests.NUM_BASE_INDICES_PER_BLOCK)
                ],
                "plausibility": "yes",
                "diversity_impression": "medium",
            }
        filled.append({
            "group_key": row["group_key"],
            "blind_labels": row["blind_labels"],
            "ratings": ratings,
            "preference": row["blind_labels"][0],
            "evidence": "fixture row, content not under test",
            "confidence": "clear",
        })
    return filled


class GalleryReviewAxisFlipsFromIncompleteToOkTests(unittest.TestCase):
    def test_axis_is_incomplete_before_ingestion_and_ok_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])

            before = analysis.analyze_run(run_dir)
            self.assertEqual(before["gallery_review"]["status"], "incomplete")

            entries = manifests.build_full_pilot_manifest_entries(["B5"])
            blind_rows, mapping = review.export_gallery_review(entries)
            filled = _fill_gallery_review_template(blind_rows, mapping)

            expected_group_keys = [row["group_key"] for row in blind_rows]
            blind_labels_by_group = {row["group_key"]: row["blind_labels"] for row in blind_rows}
            gallery_review_rows = review.ingest_gallery_review(filled, expected_group_keys, blind_labels_by_group)

            after = analysis.analyze_run(run_dir, gallery_review_rows=gallery_review_rows, gallery_review_mapping=mapping)
            axis = after["gallery_review"]
            self.assertEqual(axis["status"], "ok")
            self.assertEqual(axis["group_count"], len(blind_rows))

            # Keyed by real condition_id, resolved via the mapping -- never a bare blind label.
            self.assertEqual(set(axis["per_condition_preference_rate"]), {"reference", "B5_plus", "B5_minus"})
            for blind_label in ("A", "B", "C", "D", "E"):
                self.assertNotIn(blind_label, axis["per_condition_preference_rate"])
            self.assertTrue(all(0.0 <= rate <= 1.0 for rate in axis["per_condition_preference_rate"].values()))

            self.assertEqual(set(axis["per_condition_artifact_rate"]), {"reference", "B5_plus", "B5_minus"})
            for rates in axis["per_condition_artifact_rate"].values():
                self.assertEqual(set(rates), set(review.ARTIFACT_FIELDS))
                # every artifact was filled in "absent" above
                self.assertTrue(all(rate == 0.0 for rate in rates.values()))

    def test_written_gallery_review_axis_json_reflects_incomplete_status(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            analysis.analyze_run(run_dir)
            payload = json.loads((run_dir / "analysis" / "gallery_review_axis.json").read_text())
            self.assertEqual(payload["status"], "incomplete")
            self.assertIn("per_condition_preference_rate", payload)
            self.assertIn("per_condition_artifact_rate", payload)


if __name__ == "__main__":
    unittest.main()
