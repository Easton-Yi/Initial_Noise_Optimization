import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import noise_validation as nv
from noise_methods import sample_noise_batch, construct_raw_noise, noise_statistics, normalize, pink, pink_filter
from io_utils import read_json, write_json, read_jsonl, write_jsonl
from run_support import Condition
from test_formal_analysis import fixture


class SpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_statistics_alignment_errors_and_empty(self):
        rows = noise_statistics(torch.ones(2, 2, 4, 4))
        self.assertEqual(nv.stats_relative_error(rows[::-1], rows), 0.)
        for saved in ([], rows[:-1], rows + [rows[0]], [dict(r, channel=r["channel"] + 2) for r in rows], [dict(r, mean=float("nan")) for r in rows]):
            with self.assertRaises(ValueError): nv.stats_relative_error(saved, rows)
        self.assertEqual(nv.stats_comparison([], [])["status"], "unavailable")

    def test_fourier_complex_relation_detects_phase_flip(self):
        batch = sample_noise_batch(5, "b", (4, 2, 9, 11))
        c = Condition("same_phase", .9, .05)
        raw = construct_raw_noise(batch, c.family, c.alpha, c.gamma, "none")
        base = torch.fft.rfft2(batch.base_white)
        eta = torch.fft.rfft2(batch.independent_eta)
        h = pink_filter(9, 11, c.alpha)
        check, _, _ = nv.construction_checks(torch.fft.rfft2(raw), base, eta, c, h)
        self.assertEqual(check["status"], "passed")
        self.assertLess(check["max_phase_error_radians"], 1e-4)
        bad, _, _ = nv.construction_checks(-torch.fft.rfft2(raw), base, eta, c, h)
        self.assertEqual(bad["status"], "failed")

    def test_b_cross_term_identity_and_nonzero_finite_sample(self):
        for profile in ("none", "divgen_compat", "per_sample_per_channel_zero_mean_unit_std"):
            for gamma in (0., .4, 1.):
                batch = sample_noise_batch(4, "b", (4, 2, 8, 8))
                c = Condition("independent_white", .8, gamma)
                component = pink(batch.base_white, c.alpha)
                if gamma != 0: component = normalize(component, profile)
                actual = torch.fft.rfft2(construct_raw_noise(batch, c.family, c.alpha, c.gamma, profile))
                check, pred, cross = nv.construction_checks(actual, torch.fft.rfft2(batch.base_white), torch.fft.rfft2(batch.independent_eta), c, torch.fft.rfft2(component))
                self.assertEqual(check["status"], "passed", check)
                if gamma == .4:
                    self.assertGreater(float(cross.abs().mean()), .01)
                    self.assertGreater(float((actual.abs().square() - pred).abs().mean()), .01)
                bad, _, _ = nv.construction_checks(actual * 2, torch.fft.rfft2(batch.base_white), torch.fft.rfft2(batch.independent_eta), c, torch.fft.rfft2(component))
                self.assertEqual(bad["status"], "failed")

    def test_float32_cancellation_and_larger_fft_roundoff(self):
        for size in (64, 128):
            batch = sample_noise_batch(7, "b", (4, 4, size, size))
            for family in ("same_phase", "independent_white"):
                for gamma in (.0125, .5, .9):
                    c = Condition(family, .9, gamma)
                    profile = "per_sample_per_channel_zero_mean_unit_std"
                    raw = construct_raw_noise(batch, family, .9, gamma, profile)
                    component = torch.fft.rfft2(normalize(pink(batch.base_white, .9), profile)) if family == "independent_white" else pink_filter(size, size, .9)
                    result, _, _ = nv.construction_checks(torch.fft.rfft2(raw), torch.fft.rfft2(batch.base_white),
                                                        torch.fft.rfft2(batch.independent_eta), c, component)
                    self.assertEqual(result["status"], "passed", (size, c, result))

    def test_full_spectrum_weights_shells_parseval(self):
        for width in (8, 9):
            x = torch.randn(2, 3, 7, width)
            layout = nv.SpectrumLayout(7, width, 9)
            half = torch.fft.rfft2(x).abs().square()
            full = torch.fft.fft2(x).abs().square()
            energy = layout.shell(half, energy=True).sum(-1) + half[..., 0, 0]
            self.assertTrue(torch.allclose(energy.float(), full.sum((-1, -2)), rtol=1e-5))
            self.assertTrue(torch.allclose((energy/(7*width)).float(), x.square().sum((-1, -2)), rtol=1e-5))
            self.assertEqual(float(layout.counts.sum()), 7*width - 1)

    def test_streaming_accumulation_equals_one_shot(self):
        batch = sample_noise_batch(2, "b", (8, 2, 8, 8))
        raw = construct_raw_noise(batch, "independent_white", .8, .6, "none")
        final = normalize(raw, "divgen_compat")
        ref = normalize(batch.base_white, "divgen_compat")
        args = [torch.fft.rfft2(raw), final, torch.fft.rfft2(final), ref, torch.fft.rfft2(ref), torch.fft.rfft2(raw).abs().square(), torch.zeros_like(torch.fft.rfft2(raw).real)]
        layout = nv.SpectrumLayout(8, 8)
        whole, stream = nv.Accumulator(layout), nv.Accumulator(layout)
        whole.add(*args)
        stream.add(*(arg[:4] for arg in args)); stream.add(*(arg[4:] for arg in args))
        for key in whole.curves:
            self.assertTrue(torch.allclose(whole.curves[key], stream.curves[key], atol=1e-6))
        for key, value in whole.summary().items():
            self.assertAlmostEqual(value, stream.summary()[key], places=6)
        self.assertEqual(nv.aggregate_conditions({}), ([], []))

    def test_complete_validation_hashes_selection_and_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, cfg = fixture(Path(tmp))
            with patch.object(nv, "plot_diagnostics"):
                a = nv.run_validation(run_dir=run)
                saved = {name: (Path(a["out_dir"]) / name).read_bytes() for name in ("noise_frequency_summary.csv", "radial_psd.csv")}
                b = nv.run_validation(run_dir=run, options={"plot_same_alpha_range": ".9"})
            self.assertEqual(a["status"], "passed", a.get("checks"))
            self.assertEqual(a["checks"]["final_noise_bit_exact"]["status"], "passed")
            self.assertEqual(a["block_condition_evidence"], b["block_condition_evidence"])
            for filename in ("noise_frequency_summary.csv", "radial_psd.csv"):
                self.assertEqual(saved[filename], (Path(b["out_dir"]) / filename).read_bytes())
            self.assertEqual(a["counts"]["unique_conditions"], 5)
            self.assertEqual(a["counts"]["block_conditions"], 15)
            manifest = read_json(run / "run_manifest.json")
            manifest["resolved_config"]["experiment"]["noise_construction_version"] = "unknown_old_b"
            write_json(run / "run_manifest.json", manifest)
            with self.assertRaisesRegex(RuntimeError, "Unsupported"):
                nv.run_validation(run_dir=run)

    def test_cache_coverage_corruption_and_partial(self):
        for case in ("missing", "extra", "incomplete", "hash", "metadata", "shape", "finite"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                run, _ = fixture(Path(tmp), same=False, independent=False, single=True)
                cache = run / "noise_cache/b0"
                if case == "missing": cache.rename(run / "removed_cache")
                elif case == "extra": (run / "noise_cache/extra").mkdir()
                elif case == "incomplete": (cache / "independent_eta.pt").unlink()
                elif case == "metadata":
                    meta = read_json(cache / "noise_metadata.json"); meta["sample_seeds"] = [99]*4
                    write_json(cache / "noise_metadata.json", meta)
                else:
                    tensor = torch.load(cache / "base_white.pt", weights_only=True)
                    if case == "hash": tensor[0, 0, 0, 0] += 1
                    elif case == "shape": tensor = tensor[:3]
                    else: tensor[0, 0, 0, 0] = float("nan")
                    torch.save(tensor, cache / "base_white.pt")
                with self.assertRaises(RuntimeError): nv.run_validation(run_dir=run)
                with patch.object(nv, "plot_diagnostics"):
                    if case in ("missing", "extra", "incomplete"):
                        result = nv.run_validation(run_dir=run, allow_partial=True)
                        self.assertEqual(result["status"], "partial")
                        self.assertTrue(result["coverage"]["omissions"] or result["coverage"]["extra_blocks"])
                    else:
                        with self.assertRaises(RuntimeError): nv.run_validation(run_dir=run, allow_partial=True)

    def test_generation_missing_duplicate_hash_and_summary_evidence(self):
        for case in ("absent", "missing_index", "duplicate", "hash", "missing_hash", "empty_stats", "extra_index"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                run, _ = fixture(Path(tmp), same=False, independent=False, single=True)
                paths = list(run.glob("generations/**/samples.jsonl"))
                for path in paths:
                    rows = read_jsonl(path)
                    if case == "absent": path.unlink(); continue
                    if case == "missing_index": rows.pop()
                    elif case == "duplicate": rows[1]["base_index"] = 0
                    elif case == "extra_index": rows[-1]["base_index"] = 99
                    elif case == "hash": rows[0]["final_noise_hash"] = "different"
                    elif case == "missing_hash":
                        for r in rows: r.pop("final_noise_hash")
                    elif case == "empty_stats": write_json(path.parent / "noise_statistics.json", {"post_normalization": []})
                    write_jsonl(path, rows)
                with patch.object(nv, "plot_diagnostics"):
                    result = nv.run_validation(run_dir=run)
                checks = result["checks"]
                if case in ("missing_index", "duplicate", "extra_index"):
                    self.assertEqual(checks["sample_index_coverage"]["status"], "failed")
                    self.assertEqual(result["status"], "failed")
                elif case == "hash":
                    self.assertEqual(checks["final_noise_bit_exact"]["status"], "failed")
                    self.assertEqual(checks["saved_summary_statistics"]["status"], "passed")
                    self.assertIn("无法", result["block_condition_evidence"][0]["interpretation"])
                elif case == "empty_stats":
                    self.assertEqual(checks["saved_summary_statistics"]["status"], "unavailable")
                else:
                    self.assertEqual(checks["final_noise_bit_exact"]["status"], "unavailable")

    def test_cli_nonzero_for_corrupt_cache_and_missing_all_blocks(self):
        import subprocess, sys
        from test_formal_analysis import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = fixture(Path(tmp), same=False, independent=False, single=True)
            meta = run / "noise_cache/b0/noise_metadata.json"
            rows = read_json(meta); rows["base_hashes"][0] = "broken"; write_json(meta, rows)
            result = subprocess.run([sys.executable, str(ROOT / "noise_validation.py"), "--run-dir", str(run)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("raw cache hash mismatch", result.stderr)
            for path in (run / "noise_cache").glob("*/base_white.pt"): path.unlink()
            with self.assertRaisesRegex(RuntimeError, "No complete expected"):
                nv.run_validation(run_dir=run, allow_partial=True)

    def test_actual_psd_and_heatmap_rendering_all_compositions(self):
        for variant in ({"variant": "sdxl_turbo_full_finer.yaml"}, {"variant": "sdxl_turbo_geneval400_1group.yaml"},
                        {"baseline": False, "same": False, "single": True}, {"same": False, "independent": False, "single": True}):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                run, _ = fixture(Path(tmp), **variant)
                result = nv.run_validation(run_dir=run)
                self.assertEqual(result["status"], "passed", result["checks"])
                self.assertEqual(len(list(Path(result["out_dir"]).glob("*.png"))), 4)
                if not variant.get("baseline", True):
                    self.assertEqual(result["checks"]["nearest_baseline_alpha"]["status"], "unavailable")


if __name__ == "__main__": unittest.main()
