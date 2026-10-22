"""CPU fixtures: saved tiny tensors/images and mocked metrics, never real models."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analysis
from analysis import analyze_run, prompt_values, _bootstrap_matched_improvements
from analyze_prompt_subset import classify, analyze_subsets
from io_utils import write_json, write_jsonl, read_json, read_jsonl, ensure_immutable_run, canonical_json, sha256_text
from run_experiment import load_config, generate, _analysis_config_for_existing_run
from run_support import conditions, display_selection, presentation_config, resolve_run
from result_integrity import validate_analysis_inputs
from test_pipeline_small import FakeAdapter
from metric_runner import MetricRunner, evaluate_run
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def fixture(root, *, variant=None, baseline=True, same=True, independent=True, single=False):
    cfg = load_config(ROOT / "configs" / (variant or "sdxl_turbo_full_finer.yaml"))
    cfg["paths"]["outputs_root"] = str(root / "outputs")
    cfg["model"]["device"] = "cpu"
    cfg["baseline"]["enabled"] = baseline
    cfg["same_phase_floor"]["enabled"] = same
    cfg["independent_white"]["enabled"] = independent
    if not variant:
        cfg["baseline"]["alpha_values"] = [0., .8] if not single else [0.]
        cfg["same_phase_floor"].update(alpha_values=[.8] if single else [.8, .9], gamma_values=[.05])
        cfg["independent_white"].update(alpha_values=[.7], gamma_values=[.75])
    cfg["quality_metrics"] = {"clip": {"enabled": True, "checkpoint": "fake"}, "hpsv3": {"enabled": False}}
    cfg["diversity_metrics"] = {"dreamsim": {"enabled": True}, "lpips": {"enabled": False}, "vendi_clip": {"enabled": False}}
    cfg["analysis"].update(bootstrap_replicates=12, primary_metric_pair={"quality": "clip_cosine", "diversity": "dreamsim_mean_pair_distance"}, optional_detail_curves=[])
    blocks = [{"block_id": b, "prompt_id": p, "prompt": text, "batch_seed": seed, "seed_batch_id": b,
               "geneval_metadata": {"tag": tag}}
              for b, p, text, seed, tag in (("b0", "p0", "a cat", 10, "普通/类别"), ("b1", "p0", "a cat", 20, "普通/类别"), ("b2", "p1", "two blue cars", 30, "two objects"))]
    class SyntheticAdapter(FakeAdapter):
        def generate(self, prompt, latents, generation_config):
            self.calls += 1
            return [Image.fromarray(((x[0].numpy() + 3) / 6 * 255).clip(0, 255).astype(np.uint8)).convert("RGB") for x in latents]
    run = generate(cfg, ROOT / "configs" / "fake.yaml", blocks, run_id="fixture", selected_conditions=None, force=False, adapter=SyntheticAdapter())
    def pixels(path):
        with Image.open(path) as image:
            return np.array(image).astype(float) / 255
    with patch.object(MetricRunner, "metric_versions", return_value={"synthetic": "1"}), \
         patch.object(MetricRunner, "clip_cosine", autospec=True,
                      side_effect=lambda self, path, prompt: float(pixels(path)[:3].mean()) + .001 * len(prompt)), \
         patch.object(MetricRunner, "dreamsim_distance", autospec=True,
                      side_effect=lambda self, left, right: float(np.abs(pixels(left)-pixels(right)).mean())):
        evaluate_run(run, cfg)
    return run, cfg


class FormalAnalysisTests(unittest.TestCase):
    def test_all_yaml_grids_and_full_default_selection(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            with self.subTest(path=path.name):
                cfg = load_config(path)
                grid = conditions(cfg); display = display_selection(cfg)
                for c in grid:
                    self.assertIn(c.alpha, display[c.family]["alpha_values"])
                    if c.gamma is not None:
                        self.assertIn(c.gamma, display[c.family]["gamma_values"])

    def test_family_priority_disjoint_and_invalid_ranges(self):
        cfg = load_config(ROOT / "configs/sdxl_turbo_geneval400_1group.yaml")
        cfg["same_phase_floor"]["alpha_values"] = [.3]
        display = display_selection(cfg, {"plot_gamma_range": "0:1", "plot_same_gamma_range": "0.05"})
        self.assertEqual(display["same_phase"]["gamma_values"], [.05])
        self.assertEqual(display["independent_white"]["gamma_values"], [.75, .825, .9])
        self.assertEqual(display["same_phase"]["alpha_values"], [.3])
        for spec in ("nan:1", "0:inf", "1:0", "0.6:0.7", "nan", ""):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                display_selection(cfg, {"plot_gamma_range": spec})
        cfg["same_phase_floor"]["enabled"] = False
        self.assertNotIn("same_phase", display_selection(cfg))

    def test_prompt_equal_and_paired_bootstrap(self):
        blocks = {"b0": {"prompt_id": "p"}, "b1": {"prompt_id": "p"}, "b2": {"prompt_id": "q"}}
        rows = [{"block_id": b, "condition_id": c, "metric": "q", "score": v + delta}
                for b, v in (("b0", 0), ("b1", 2), ("b2", 9)) for c, delta in (("base", 0), ("ours", 3))]
        values, _ = prompt_values(rows, blocks)
        self.assertEqual(np.mean([values[p, "base"]["q"] for p in ("p", "q")]), 5)
        self.assertNotEqual(5, np.mean([0, 2, 9]))
        curves = {}
        for pid, offset in (("p", 0), ("q", 10)):
            for c, d, q in (("b0", .2, .8), ("b1", .6, .4), ("o0", .2, 1.1), ("o1", .6, .7)):
                curves[pid, c] = {"q": q + offset, "d": d}
        args = (curves, ["b0", "b1"], ["o0", "o1"], "q", "d", [.3, .5], {"bootstrap_seed": 1, "bootstrap_replicates": 25, "confidence_level": .95})
        a = _bootstrap_matched_improvements(*args)
        self.assertEqual(a, _bootstrap_matched_improvements(*args))
        self.assertAlmostEqual(a[1]["bootstrap_ci_low"], .3)
        self.assertAlmostEqual(a[1]["bootstrap_ci_high"], .3)

    def test_classification_external_identity_conflict_and_missing(self):
        blocks = {"b": {"block_id": "b", "prompt_id": "p", "prompt": "text"}}
        self.assertEqual(classify(blocks), ({}, ["b"]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "categories.jsonl"
            row = {**blocks["b"], "category": "中文 / text"}
            write_jsonl(path, [row]); self.assertEqual(classify(blocks, manifest=path)[0], {"中文 / text": ["b"]})
            blocks["b"]["category"] = "conflict"
            with self.assertRaisesRegex(ValueError, "conflict"):
                classify(blocks, manifest=path)
            row["prompt"] = "other"; write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "identity/text"):
                classify(blocks, manifest=path)
        blocks["c"] = {"block_id": "c", "prompt_id": "q", "prompt": "other"}
        with self.assertRaisesRegex(ValueError, "Missing categories"):
            classify(blocks)
        self.assertEqual(classify(blocks, exclude_missing=True)[1], ["c"])

    def test_manifest_presentation_identity_is_narrow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"experiment": {"name": "x"}, "analysis": {"bootstrap_seed": 1}}
            old = copy.deepcopy(config); old["analysis"]["two_panel_display"] = {"alpha_values": [1]}
            write_json(root / "run_manifest.json", {"resolved_config": old, "config_hash": sha256_text(canonical_json(old))})
            original = (root / "run_manifest.json").read_bytes()
            ensure_immutable_run(root, config, force=False)
            config["analysis"]["two_panel_display"] = {"alpha_values": [2]}
            ensure_immutable_run(root, config, force=False)
            self.assertEqual(original, (root / "run_manifest.json").read_bytes())
            config["analysis"]["bootstrap_seed"] = 2
            with self.assertRaises(RuntimeError):
                ensure_immutable_run(root, config, force=False)

    def test_display_does_not_change_statistics_and_frozen_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = fixture(Path(tmp))
            with patch.object(analysis, "_plots"):
                a = analyze_run(run)
                original_tables = {p.name: p.read_bytes() for p in (a / "tables").glob("*.csv")}
                shown = presentation_config(cfg, {"plot_same_alpha_range": ".9"})
                b = analyze_run(run, shown)
            self.assertEqual(a, run / "analysis")
            self.assertEqual(a, b)
            for name, data in original_tables.items():
                self.assertEqual(data, (b / "tables" / name).read_bytes())
            metadata = read_json(a / "analysis_manifest.json")
            self.assertEqual(metadata["bootstrap_unit"], "prompt")
            self.assertTrue(all("prompt_id" in row and "block_id" not in row for row in analysis._read_csv(a / "tables/paired_effects.csv")))
            edited = copy.deepcopy(cfg); edited["same_phase_floor"]["alpha_values"] = [99]
            frozen = _analysis_config_for_existing_run(run, edited)
            self.assertEqual(frozen["same_phase_floor"]["alpha_values"], [.8, .9])
            self.assertEqual(resolve_run(run_dir=run), run)
            self.assertEqual(read_json(a / "analysis_manifest.json")["prompt_count"], 2)
            self.assertEqual(read_json(a / "analysis_manifest.json")["block_count"], 3)
            contract = validate_analysis_inputs(run)
            expected, _ = prompt_values(analysis._read_csv(run / "metrics/per_group.csv"), contract.blocks)
            for row in analysis._read_csv(a / "tables/all_curve_points.csv"):
                for axis in ("quality", "diversity"):
                    scores = [values[row[axis + "_metric"]] for (prompt, condition), values in expected.items() if condition == row["condition_id"]]
                    self.assertAlmostEqual(float(row[axis]), np.mean(scores))

    def test_corrupt_conditions_hashes_prompt_and_metrics_rejected(self):
        for case in ("condition", "hash", "prompt", "duplicate", "missing", "collision"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                run, cfg = fixture(Path(tmp), same=False, independent=False, single=True)
                if case == "collision":
                    cfg["baseline"]["alpha_values"] = [0., 0.]
                    write_json(run / "run_manifest.json", {"resolved_config": cfg})
                else:
                    path = run / "metrics/per_group.csv"
                    rows = analysis._read_csv(path)
                    if case == "condition": rows[0]["alpha"] = "9"
                    elif case == "hash": rows[0]["metric_config_hash"] = "bad"
                    elif case == "prompt": rows[0]["prompt"] = "bad"
                    elif case == "duplicate": rows.append(rows[0])
                    elif case == "missing": rows.pop()
                    analysis._write_csv(path, rows)
                with self.assertRaises(RuntimeError): validate_analysis_inputs(run)

    def test_frozen_cli_without_external_manifest_and_zoom(self):
        import subprocess, sys, os
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = fixture(Path(tmp), same=False, independent=False, single=True)
            config_path = Path(tmp) / "moved.yaml"
            import yaml
            cfg["blocks"]["manifest"] = "does-not-exist.jsonl"
            cfg["baseline"]["alpha_values"] = [99.]
            config_path.write_text(yaml.safe_dump(cfg))
            commands = [
                ["run_experiment.py", "--stage", "analyze", "--run-dir", str(run), "--config", str(config_path)],
                ["plot_zoomed.py", "--run-dir", str(run)],
                ["analyze_prompt_default-subset.py", "--run-dir", str(run), "--category-field", "missing"],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, str(ROOT / command[0]), *command[1:]], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(list(run.glob("analysis/primary/methods_qd_two_panel_zoomed.png")))

    def test_all_stage_does_not_mutate_scientific_config(self):
        import sys
        import run_experiment as runner
        cfg = load_config(ROOT / "configs/sdxl_turbo_full_finer.yaml")
        original = copy.deepcopy(cfg)
        with patch.object(runner, "load_config", return_value=cfg), patch.object(runner, "validate_config", return_value=[]), \
             patch.object(runner, "generate", return_value=Path("/tmp/fake-run")) as generate_mock, \
             patch.object(runner, "evaluate_run") as evaluate_mock, patch.object(runner, "analyze_run", return_value=Path("/tmp/fake-analysis")) as analyze_mock, \
             patch.object(sys, "argv", ["run_experiment.py", "--config", "fake.yaml", "--stage", "all", "--plot-same-gamma-range", "0.05"]):
            runner.main()
        self.assertEqual(cfg, original)
        self.assertEqual(generate_mock.call_args.args[0], original)
        self.assertEqual(evaluate_mock.call_args.args[1], original)
        self.assertEqual(analyze_mock.call_args.args[1]["analysis"]["two_panel_display"]["same_phase"]["gamma_values"], [.05])

    def test_all_family_combinations(self):
        from itertools import product
        for baseline, same, independent in product((False, True), repeat=3):
            if not any((baseline, same, independent)): continue
            with self.subTest(baseline=baseline, same=same, independent=independent), tempfile.TemporaryDirectory() as tmp:
                run, _ = fixture(Path(tmp), baseline=baseline, same=same, independent=independent, single=True)
                with patch.object(analysis, "_plots"):
                    out = analyze_run(run)
                self.assertTrue((out / "tables/all_curve_points.csv").exists())

    def test_render_compositions_and_subsets(self):
        for label, kwargs in (("finer", {"variant": "sdxl_turbo_full_finer.yaml"}),
                              ("geneval", {"variant": "sdxl_turbo_geneval400_1group.yaml"}),
                              ("single", {"independent": False, "single": True}),
                              ("no_baseline", {"baseline": False}),
                              ("baseline_only", {"same": False, "independent": False, "single": True})):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                run, cfg = fixture(Path(tmp), **kwargs)
                output = analyze_run(run)
                self.assertTrue(list(output.rglob("*.png")))
                if label == "no_baseline":
                    self.assertTrue(all(r["status"] == "unavailable" for r in analysis._read_csv(output / "tables/bootstrap_results.csv")))
                if label == "single":
                    subsets = analyze_subsets(run)
                    self.assertEqual(len(read_json(subsets / "analysis_info.json")["categories"]), 2)


if __name__ == "__main__": unittest.main()
