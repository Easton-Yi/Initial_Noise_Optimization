import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import analysis
from analysis import analyze_run
from io_utils import condition_id, read_jsonl, write_json, write_jsonl
from metric_runner import MetricRunner, _read_csv, _write_csv, evaluate_run
from result_integrity import expected_condition_ids
from run_experiment import all_conditions


def _config(*, baseline_alphas=(0.0,), same_phase=False, clip=True, hps=False, dreamsim=False):
    return {
        "experiment": {"gallery_size": 4},
        "model": {"device": "cpu"},
        "baseline": {"enabled": True, "alpha_values": list(baseline_alphas)},
        "same_phase_floor": {"enabled": same_phase, "alpha_values": [0.0], "gamma_values": [0.1]},
        "independent_white": {"enabled": False, "alpha_values": [0.5], "gamma_values": [0.1]},
        "quality_metrics": {
            "clip": {"enabled": clip, "checkpoint": "fake"},
            "hpsv3": {"enabled": hps},
        },
        "diversity_metrics": {
            "dreamsim": {"enabled": dreamsim},
            "lpips": {"enabled": False, "backbone": "alex"},
            "vendi_clip": {"enabled": False},
        },
        "analysis": {
            "bootstrap_replicates": 2,
            "bootstrap_seed": 1,
            "confidence_level": 0.95,
            "primary_metric_pair": {"quality": "clip_cosine", "diversity": "dreamsim_mean_pair_distance"},
            "optional_detail_curves": [],
        },
    }


def _make_run(root: Path, config, blocks, conditions, *, shared_paths=None):
    run_dir = root / "run"
    write_json(run_dir / "run_manifest.json", {"config_hash": "frozen", "resolved_config": config})
    write_jsonl(run_dir / "blocks.jsonl", blocks)
    image_root = run_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    if shared_paths is None:
        shared_paths = []
        for index in range(4):
            path = image_root / f"image_{index:02d}.png"
            Image.new("RGB", (4, 4), color=(index * 40, 0, 0)).save(path)
            shared_paths.append(path)
    for block in blocks:
        for family, alpha, gamma in conditions:
            candidate = condition_id(family, alpha, gamma)
            records = []
            for index, path in enumerate(shared_paths):
                records.append({
                    "run_id": "run", "model_id": "fake", "block_id": block["block_id"],
                    "prompt_id": block["prompt_id"], "prompt": block["prompt"],
                    "seed_batch_id": block.get("seed_batch_id", "s"), "base_index": index,
                    "condition_id": candidate, "family": family,
                    "method": "white" if family == "baseline" and alpha == 0 else family,
                    "alpha": alpha, "gamma": gamma, "normalization_profile": "none",
                    "image_path": str(path.resolve()),
                })
            write_jsonl(run_dir / "generations" / "fake" / block["block_id"] / candidate / "samples.jsonl", records)
    return run_dir


def _score(run_dir: Path, config):
    with patch.object(MetricRunner, "metric_versions", return_value={"fake": "1"}), \
            patch.object(MetricRunner, "clip_cosine", autospec=True,
                         side_effect=lambda self, path, prompt: float(len(prompt))), \
            patch.object(MetricRunner, "dreamsim_distance", autospec=True,
                         side_effect=lambda self, left, right: float(left.name != right.name)):
        evaluate_run(run_dir, config)


class ResultIntegrityTests(unittest.TestCase):
    def test_expected_condition_helper_matches_generation_grid(self):
        config = _config(baseline_alphas=(0.0, 0.5), same_phase=True)
        self.assertEqual(list(expected_condition_ids(config)),
                         [condition.identifier for condition in all_conditions(config)])

    def test_complete_small_run_scores_and_analyzes_without_disabled_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(clip=True, dreamsim=True)
            blocks = [{"block_id": "b", "prompt_id": "p", "prompt": "hello"}]
            run_dir = _make_run(Path(directory), config, blocks, [("baseline", 0.0, None)])
            _score(run_dir, config)
            with patch.object(analysis, "_plots") as plots:
                analyze_run(run_dir, config)
            plots.assert_called_once()
            self.assertEqual(len(_read_csv(run_dir / "metrics" / "per_image.csv")), 4)
            self.assertEqual(len(_read_csv(run_dir / "metrics" / "per_pair.csv")), 6)
            self.assertEqual({row["metric"] for row in _read_csv(run_dir / "metrics" / "per_group.csv")},
                             {"clip_cosine", "dreamsim_mean_pair_distance"})

    def test_metrics_rejects_bad_present_gallery_before_scoring(self):
        for case in ("three", "duplicate", "missing_png"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                config = _config()
                blocks = [{"block_id": "b", "prompt_id": "p", "prompt": "hello"}]
                run_dir = _make_run(Path(directory), config, blocks, [("baseline", 0.0, None)])
                samples = next(run_dir.glob("generations/**/samples.jsonl"))
                rows = read_jsonl(samples)
                if case == "three":
                    rows.pop()
                    write_jsonl(samples, rows)
                elif case == "duplicate":
                    rows[-1]["base_index"] = 2
                    write_jsonl(samples, rows)
                else:
                    Path(rows[-1]["image_path"]).unlink()
                with patch.object(MetricRunner, "clip_cosine") as scorer:
                    with self.assertRaisesRegex(RuntimeError, "Generation integrity check failed"):
                        evaluate_run(run_dir, config)
                scorer.assert_not_called()

    def test_partial_metrics_allowed_but_formal_analysis_requires_all_groups(self):
        for missing in ("condition", "block"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                if missing == "condition":
                    config = _config(baseline_alphas=(0.0, 0.5))
                    blocks = [{"block_id": "b", "prompt_id": "p", "prompt": "hello"}]
                    generated = [("baseline", 0.0, None)]
                else:
                    config = _config()
                    blocks = [
                        {"block_id": "b1", "prompt_id": "p1", "prompt": "hello"},
                        {"block_id": "b2", "prompt_id": "p2", "prompt": "world"},
                    ]
                    generated = [("baseline", 0.0, None)]
                run_dir = _make_run(Path(directory), config, blocks, generated)
                if missing == "block":
                    next((run_dir / "generations" / "fake" / "b2").glob("**/samples.jsonl")).unlink()
                _score(run_dir, config)
                with patch.object(analysis, "_plots") as plots:
                    with self.assertRaisesRegex(RuntimeError, "missing expected galleries"):
                        analyze_run(run_dir, config)
                plots.assert_not_called()

    def test_prompt_aware_alias_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config(same_phase=True, hps=True)
            blocks = [
                {"block_id": "b1", "prompt_id": "p1", "prompt": "short"},
                {"block_id": "b2", "prompt_id": "p2", "prompt": "a different prompt"},
            ]
            conditions = [("baseline", 0.0, None), ("same_phase", 0.0, 0.1)]
            run_dir = _make_run(Path(directory), config, blocks, conditions)
            with patch.object(MetricRunner, "metric_versions", return_value={"fake": "1"}), \
                    patch.object(MetricRunner, "clip_cosine", autospec=True,
                                 side_effect=lambda self, path, prompt: float(len(prompt))) as clip_scorer, \
                    patch.object(MetricRunner, "hpsv3", autospec=True,
                                 side_effect=lambda self, path, prompt: float(100 + len(prompt))) as hps_scorer:
                evaluate_run(run_dir, config)
                self.assertEqual(clip_scorer.call_count, 8)
                self.assertEqual(hps_scorer.call_count, 8)
                evaluate_run(run_dir, config)
                self.assertEqual(clip_scorer.call_count, 8)
                self.assertEqual(hps_scorer.call_count, 8)
            rows = _read_csv(run_dir / "metrics" / "per_image.csv")
            self.assertEqual(len(rows), 32)
            by_block = {block: {float(row["score"]) for row in rows if row["block_id"] == block}
                        for block in ("b1", "b2")}
            self.assertEqual(by_block, {"b1": {5.0, 105.0}, "b2": {18.0, 118.0}})
            self.assertTrue(all(row.get("prompt_hash") for row in rows))

    def test_old_per_image_rows_do_not_skip_prompt_aware_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            config = _config()
            blocks = [{"block_id": "b", "prompt_id": "p", "prompt": "hello"}]
            run_dir = _make_run(Path(directory), config, blocks, [("baseline", 0.0, None)])
            _score(run_dir, config)
            path = run_dir / "metrics" / "per_image.csv"
            old = _read_csv(path)
            for row in old:
                row.pop("prompt_hash", None)
                row["score"] = "999"
            _write_csv(path, old)
            with patch.object(MetricRunner, "metric_versions", return_value={"fake": "1"}), \
                    patch.object(MetricRunner, "clip_cosine", autospec=True, return_value=7.0) as scorer:
                evaluate_run(run_dir, config)
            self.assertEqual(scorer.call_count, 4)
            self.assertEqual({float(row["score"]) for row in _read_csv(path)}, {7.0})

    def test_analysis_rejects_metric_corruption_without_plotting(self):
        cases = ("missing_metric", "missing_pair", "duplicate_pair", "duplicate_group", "nan_score", "inf_score")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                config = _config(clip=True, dreamsim=True)
                blocks = [{"block_id": "b", "prompt_id": "p", "prompt": "hello"}]
                run_dir = _make_run(Path(directory), config, blocks, [("baseline", 0.0, None)])
                _score(run_dir, config)
                if case == "missing_metric":
                    path = run_dir / "metrics" / "per_image.csv"
                    rows = _read_csv(path)
                    rows.pop()
                    _write_csv(path, rows)
                elif case in {"missing_pair", "duplicate_pair"}:
                    path = run_dir / "metrics" / "per_pair.csv"
                    rows = _read_csv(path)
                    rows.pop()
                    if case == "duplicate_pair":
                        rows.append(dict(rows[0]))
                    _write_csv(path, rows)
                elif case == "duplicate_group":
                    path = run_dir / "metrics" / "per_group.csv"
                    rows = _read_csv(path)
                    rows.append(dict(rows[0]))
                    _write_csv(path, rows)
                elif case == "nan_score":
                    path = run_dir / "metrics" / "per_image.csv"
                    rows = _read_csv(path)
                    rows[0]["score"] = "nan"
                    _write_csv(path, rows)
                else:
                    path = run_dir / "metrics" / "per_group.csv"
                    rows = _read_csv(path)
                    rows[0]["score"] = "inf"
                    _write_csv(path, rows)
                sentinel = run_dir / "analysis" / "tables" / "all_curve_points.csv"
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text("existing analysis\n", encoding="utf-8")
                with patch.object(analysis, "_plots") as plots:
                    with self.assertRaisesRegex(RuntimeError, "Metric integrity check failed"):
                        analyze_run(run_dir, config)
                plots.assert_not_called()
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing analysis\n")


if __name__ == "__main__":
    unittest.main()
