"""analysis.py must not reuse noise_init.analysis.analyze_run's family/alpha/
gamma-keyed, curve-interpolating machinery -- our PC-condition schema has no
continuous curve, only a small fixed set of directly-generated conditions --
but it must reuse noise_init.analysis._bootstrap's exact numeric recipe
(2000 replicates, seed 20260829, 95% CI).
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pc_specific_psd import analysis, manifests, metrics, runner
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


class NoMetricsYetTests(unittest.TestCase):
    def test_load_per_group_scores_raises_when_metrics_not_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_run = Path(tmp) / "empty_run"
            empty_run.mkdir()
            with self.assertRaises(analysis.AnalysisError):
                analysis.load_per_group_scores(empty_run)


class SummarizeConditionsTests(unittest.TestCase):
    def test_every_condition_and_metric_gets_a_12_block_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            rows = analysis.summarize_conditions(run_dir)
            by_key = {(r["condition_id"], r["metric"]): r for r in rows}
            for condition in ("reference", "B5_plus", "B5_minus"):
                for metric in analysis.QUALITY_METRICS + analysis.DIVERSITY_METRICS:
                    self.assertIn((condition, metric), by_key)
                    self.assertEqual(by_key[(condition, metric)]["block_count"], 12)
                    self.assertLessEqual(by_key[(condition, metric)]["ci_low"], by_key[(condition, metric)]["ci_high"])

    def test_no_family_alpha_gamma_keys_leak_into_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            rows = analysis.summarize_conditions(run_dir)
            for row in rows:
                self.assertNotIn("family", row)
                self.assertNotIn("alpha", row)
                self.assertNotIn("gamma", row)
                self.assertNotIn("curve_id", row)


class PairedEffectsTests(unittest.TestCase):
    def test_plus_and_minus_conditions_paired_against_reference_over_12_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            rows = analysis.paired_effects(run_dir)
            by_key = {(r["condition_id"], r["metric"]): r for r in rows}
            for condition in ("B5_plus", "B5_minus"):
                for metric in analysis.QUALITY_METRICS + analysis.DIVERSITY_METRICS:
                    row = by_key[(condition, metric)]
                    self.assertEqual(row["reference_condition_id"], "reference")
                    self.assertEqual(row["block_count"], 12)
            self.assertNotIn(("reference", "hpsv3"), by_key)  # reference is never paired against itself

    def test_two_candidate_run_pairs_each_group_against_the_single_shared_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5", "B6"])
            rows = analysis.paired_effects(run_dir)
            conditions = {r["condition_id"] for r in rows}
            self.assertEqual(conditions, {"B5_plus", "B5_minus", "B6_plus", "B6_minus"})
            self.assertTrue(all(r["reference_condition_id"] == "reference" for r in rows))


class CompareConditionsNoInterpolationTests(unittest.TestCase):
    """The plan is explicit (PC-Specific-PSD-Implementation-Prompt.md:277):
    the initial configurations give local evidence only, never linked into
    one intensity curve or extrapolated to a matched-Q/D target. This module
    has no curve-fitting function anywhere -- a request for an unmeasured
    comparison must raise, not approximate.
    """

    def test_returns_the_directly_measured_row_for_an_actual_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            row = analysis.compare_conditions(run_dir, "B5_plus", "hpsv3")
            self.assertEqual(row["condition_id"], "B5_plus")
            self.assertEqual(row["metric"], "hpsv3")

    def test_raises_for_a_condition_never_generated_in_this_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            with self.assertRaises(analysis.AnalysisError):
                analysis.compare_conditions(run_dir, "B6_plus", "hpsv3")

    def test_raises_for_a_metric_never_computed_in_this_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            with self.assertRaises(analysis.AnalysisError):
                analysis.compare_conditions(run_dir, "B5_plus", "not_a_real_metric")

    def test_module_defines_no_curve_fitting_or_interpolation_helpers(self):
        for banned in ("pareto_frontier", "interpolate_within", "matched_diversity"):
            self.assertFalse(hasattr(analysis, banned), f"analysis.py must not define {banned}")


class GalleryReviewAxisIncompleteByDefaultTests(unittest.TestCase):
    def test_missing_gallery_review_is_explicitly_incomplete_not_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            summary = analysis.analyze_run(run_dir)
            self.assertIn("gallery_review", summary)
            self.assertEqual(summary["gallery_review"]["status"], "incomplete")
            self.assertEqual(summary["gallery_review"]["group_count"], 0)

    def test_summarize_gallery_review_treats_none_and_empty_sequence_alike(self):
        self.assertEqual(analysis.summarize_gallery_review(None, None)["status"], "incomplete")
        self.assertEqual(analysis.summarize_gallery_review((), ())["status"], "incomplete")


class AnalyzeRunWritesTablesTests(unittest.TestCase):
    def test_writes_condition_summaries_paired_effects_and_gallery_axis_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = _build_scored_run(Path(tmp), ["B5"])
            analysis.analyze_run(run_dir)
            self.assertTrue((run_dir / "analysis" / "tables" / "condition_summaries.csv").exists())
            self.assertTrue((run_dir / "analysis" / "tables" / "paired_effects.csv").exists())
            self.assertTrue((run_dir / "analysis" / "gallery_review_axis.json").exists())


class BootstrapRecipeMatchesNoiseInitTests(unittest.TestCase):
    """The block-bootstrap recipe itself is reused byte-for-byte from
    noise_init.analysis._bootstrap: same seed (20260829), same replicate
    count (2000), same confidence level (0.95) -- verified by feeding both
    implementations the identical array and checking numeric agreement.
    """

    def test_bootstrap_matches_noise_init_analysis_bootstrap_exactly(self):
        from noise_init import analysis as noise_init_analysis

        values = np.linspace(0.1, 0.9, 12)
        ours = analysis._bootstrap(values)
        theirs = noise_init_analysis._bootstrap(values, {
            "bootstrap_seed": analysis.BOOTSTRAP_SEED,
            "bootstrap_replicates": analysis.BOOTSTRAP_REPLICATES,
            "confidence_level": analysis.CONFIDENCE_LEVEL,
        })
        for key in ("point_estimate", "standard_error", "ci_low", "ci_high", "block_count"):
            self.assertAlmostEqual(ours[key], theirs[key], places=12)

    def test_frozen_constants_match_the_plan_exactly(self):
        self.assertEqual(analysis.BOOTSTRAP_REPLICATES, 2000)
        self.assertEqual(analysis.BOOTSTRAP_SEED, 20260829)
        self.assertEqual(analysis.CONFIDENCE_LEVEL, 0.95)


if __name__ == "__main__":
    unittest.main()
