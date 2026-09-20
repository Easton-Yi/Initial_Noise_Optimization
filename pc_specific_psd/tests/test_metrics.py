"""metrics.py must not call noise_init.metric_runner.evaluate_run (its
"image_"-prefix filename filter would discard every one of our "image.png"
samples, and its alpha/gamma _common() shape doesn't match our PC-condition
schema) and must mark every non-4-image gallery -- in particular every
preview-track single-image "gallery" -- with an explicit not_applicable/
incomplete status rather than silently skipping it.
"""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pc_specific_psd import manifests, metrics, runner
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, freeze_selected_calibration, load_test_config


class FakeMetricRunner:
    """Cheap, deterministic stand-in for compat_metrics.MetricRunner -- no
    torch model, no network, no GPU -- mirroring _runner_test_support's
    FakeAdapterPCA pattern so run_metrics can be exercised on CPU.
    """

    def __init__(self):
        self.clip_cosine_calls = 0
        self.hpsv3_calls = 0
        self.dreamsim_calls = 0
        self.lpips_calls = 0
        self.clip_embedding_calls = 0
        self.release_calls = 0

    @staticmethod
    def _digest_fraction(*parts: str) -> float:
        digest = int(hashlib.sha256("|".join(parts).encode()).hexdigest(), 16)
        return (digest % 10_000) / 10_000.0

    def metric_versions(self):
        return {"torch": "test", "transformers": "test", "lpips": "test", "dreamsim": "test", "vendi-score": "test", "hpsv3": "test"}

    def release_models(self):
        self.release_calls += 1

    def clip_embedding(self, image_path):
        self.clip_embedding_calls += 1
        seed = int(hashlib.sha256(str(image_path).encode()).hexdigest(), 16) % (2**32)
        vector = np.random.default_rng(seed).normal(size=8)
        return vector / np.linalg.norm(vector)

    def clip_cosine(self, image_path, prompt):
        self.clip_cosine_calls += 1
        return self._digest_fraction(str(image_path), prompt, "clip")

    def hpsv3(self, image_path, prompt):
        self.hpsv3_calls += 1
        return self._digest_fraction(str(image_path), prompt, "hps")

    def lpips_distance(self, a, b):
        self.lpips_calls += 1
        return self._digest_fraction(str(a), str(b), "lpips")

    def dreamsim_distance(self, a, b):
        self.dreamsim_calls += 1
        return self._digest_fraction(str(a), str(b), "dreamsim")


def _build_run(tmp: Path, group_ids: list[str], *, kind: str) -> Path:
    loaded = load_test_config(tmp)
    freeze_selected_calibration(loaded, group_ids=group_ids)
    adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
    entries = (
        manifests.build_full_pilot_manifest_entries(group_ids)
        if kind == "full"
        else manifests.build_preview_manifest_entries(group_ids)
    )
    return runner.generate_manifest(loaded, entries, run_id=kind, force=False, adapter=adapter, allow_synthetic_basis=True)


class CollectSampleRowsTests(unittest.TestCase):
    def test_collects_every_sample_record_with_pc_condition_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp), ["B5"], kind="full")
            rows = metrics.collect_sample_rows(run_dir)
            self.assertEqual(len(rows), 144)
            self.assertTrue(all("pc_group_id" in row and "tau" in row and "gate_id" in row for row in rows))
            self.assertTrue(all("alpha" not in row and "gamma" not in row and "prompt" not in row for row in rows))

    def test_missing_run_dir_yields_no_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(metrics.collect_sample_rows(Path(tmp) / "nonexistent"), [])


class GalleryStatusTests(unittest.TestCase):
    def test_four_is_ok(self):
        self.assertEqual(metrics.gallery_status(4), "ok")

    def test_one_is_not_applicable(self):
        self.assertEqual(metrics.gallery_status(1), "not_applicable")

    def test_other_sizes_are_incomplete(self):
        for size in (0, 2, 3, 5):
            self.assertEqual(metrics.gallery_status(size), "incomplete")


class GroupByGalleryTests(unittest.TestCase):
    def test_groups_by_block_and_condition_and_sorts_by_base_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp), ["B5"], kind="full")
            galleries = metrics.group_by_gallery(metrics.collect_sample_rows(run_dir))
            self.assertEqual(len(galleries), 3 * 12)
            for gallery in galleries.values():
                self.assertEqual(len(gallery), 4)
                self.assertEqual([row["base_index"] for row in gallery], [4, 5, 6, 7])


class RunMetricsFullGalleryTests(unittest.TestCase):
    def test_quality_and_diversity_metrics_computed_for_4_image_galleries(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp), ["B5"], kind="full")
            fake = FakeMetricRunner()
            summary = metrics.run_metrics(run_dir, metric_runner=fake)

            self.assertEqual(len(summary.per_image), 144 * 2)  # clip_cosine + hpsv3, one row per image per metric
            per_group_status = {(row["metric"], row["status"]) for row in summary.per_group}
            for metric_name in ("clip_cosine", "hpsv3", "dreamsim_mean_pair_distance", "lpips_alex_mean_pair_distance", "vendi_clip"):
                self.assertIn((metric_name, "ok"), per_group_status)
            self.assertEqual(fake.dreamsim_calls, 3 * 12 * 6)  # 6 unordered pairs per 4-image gallery
            self.assertEqual(fake.clip_embedding_calls, 3 * 12 * 4)  # vendi over the 4-image gallery
            self.assertTrue((run_dir / "metrics" / "per_image.csv").exists())
            self.assertTrue((run_dir / "metrics" / "per_pair.csv").exists())
            self.assertTrue((run_dir / "metrics" / "metric_manifest.json").exists())

    def test_rerun_without_force_reuses_cached_quality_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp), ["B5"], kind="full")
            metrics.run_metrics(run_dir, metric_runner=FakeMetricRunner())

            fake_second = FakeMetricRunner()
            metrics.run_metrics(run_dir, metric_runner=fake_second, force=False)
            self.assertEqual(fake_second.clip_cosine_calls, 0)
            self.assertEqual(fake_second.hpsv3_calls, 0)

    def test_force_ignores_the_stale_csv_and_rebuilds_from_scratch(self):
        """A poisoned score value in the persisted CSV must not survive a
        ``force=True`` rerun -- proving force actually discards ``existing``
        rather than reusing it, independent of whether the fixture's
        byte-identical placeholder images happen to collapse the underlying
        content-addressed score cache to a single real model call.
        """
        import csv

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp), ["B5"], kind="full")
            summary = metrics.run_metrics(run_dir, metric_runner=FakeMetricRunner())
            self.assertEqual(len(summary.per_image), 288)

            per_image_path = run_dir / "metrics" / "per_image.csv"
            with per_image_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                row["score"] = "999.0"
            with per_image_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)

            fake_second = FakeMetricRunner()
            rebuilt = metrics.run_metrics(run_dir, metric_runner=fake_second, force=True)
            self.assertEqual(len(rebuilt.per_image), 288)
            self.assertTrue(all(float(row["score"]) != 999.0 for row in rebuilt.per_image))
            self.assertGreaterEqual(fake_second.clip_cosine_calls, 1)
            self.assertGreaterEqual(fake_second.hpsv3_calls, 1)


class RunMetricsPreviewGalleryTests(unittest.TestCase):
    def test_single_image_gallery_marked_not_applicable_for_diversity_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_run(Path(tmp), ["B5"], kind="preview")
            summary = metrics.run_metrics(run_dir, metric_runner=FakeMetricRunner())

            diversity_rows = [
                row for row in summary.per_group
                if row["metric"] in ("dreamsim_mean_pair_distance", "lpips_alex_mean_pair_distance", "vendi_clip")
            ]
            self.assertTrue(diversity_rows)
            self.assertTrue(all(row["status"] == "not_applicable" for row in diversity_rows))
            self.assertTrue(all("score" not in row for row in diversity_rows))
            self.assertEqual(len(diversity_rows), 3 * 12 * 3)  # 3 diversity metrics x 3 configs x 12 pairs

            quality_rows = [row for row in summary.per_group if row["metric"] in ("clip_cosine", "hpsv3")]
            self.assertTrue(all(row["status"] == "ok" and row["n"] == 1 for row in quality_rows))


class NoSampleRecordsTests(unittest.TestCase):
    def test_raises_when_run_dir_has_no_sample_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_run = Path(tmp) / "empty_run"
            empty_run.mkdir()
            with self.assertRaises(metrics.MetricsError):
                metrics.run_metrics(empty_run, metric_runner=FakeMetricRunner())


if __name__ == "__main__":
    unittest.main()
