import copy
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from model_adapters import GenerationConfig, LatentSpec
from io_utils import read_jsonl
from run_experiment import all_conditions, generate


class FakeAdapter:
    model_id = "fake"
    def __init__(self): self.calls = 0
    def latent_spec(self, height, width, batch_size): return LatentSpec((batch_size, 2, 8, 8), "none")
    def generate(self, prompt, latents, generation_config):
        self.calls += 1
        return [Image.new("RGB", (8, 8), color=(int(index * 50), 0, 0)) for index in range(len(latents))]
    def close(self): pass


class PipelineSmallTests(unittest.TestCase):
    def test_two_conditions_produce_eight_images_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = {
                "experiment": {"name": "small", "master_seed": 7, "gallery_size": 4, "normalization_profile": "per_sample_per_channel_zero_mean_unit_std"},
                "paths": {"outputs_root": "out"}, "model": {"adapter": "sdxl_turbo", "device": "cpu", "dtype": "float32"},
                "generation": {"height": 8, "width": 8, "num_inference_steps": 1, "guidance_scale": 0., "generation_batch_size": 1, "output_format": "png", "release_model_after_generation": True},
                "baseline": {"enabled": True, "alpha_values": [0., .5]}, "same_phase_floor": {"enabled": False, "alpha_values": [.5], "gamma_values": [0.]}, "independent_white": {"enabled": False, "alpha_values": [.5], "gamma_values": [0.]},
                "quality_metrics": {"clip": {"enabled": False}, "hpsv3": {"enabled": False}}, "diversity_metrics": {"dreamsim": {"enabled": False}, "lpips": {"enabled": False}, "vendi_clip": {"enabled": False}}, "analysis": {"bootstrap_replicates": 2, "bootstrap_seed": 1, "confidence_level": .95}}
            config_path = root / "configs" / "small.yaml"; config_path.parent.mkdir(); config_path.write_text("# test\n")
            blocks = [{"block_id": "b", "prompt_id": "p", "prompt": "test", "seed_batch_id": "s", "batch_seed": 1}]
            run_dir = generate(config, config_path, blocks, run_id="r", selected_conditions="white,pink:0.5", force=False, adapter=FakeAdapter())
            images = sorted(run_dir.glob("generations/**/*.png"))
            self.assertEqual(len([path for path in images if path.name.startswith("image_")]), 8)
            before = {path: path.stat().st_mtime_ns for path in images}
            generate(config, config_path, blocks, run_id="r", selected_conditions="white,pink:0.5", force=False, adapter=FakeAdapter())
            self.assertEqual(before, {path: path.stat().st_mtime_ns for path in images})

    def test_same_phase_white_is_a_provenance_alias_not_a_duplicate_gallery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = {
                "experiment": {"name": "small", "master_seed": 7, "gallery_size": 4, "normalization_profile": "per_sample_per_channel_zero_mean_unit_std"},
                "paths": {"outputs_root": "out"}, "model": {"adapter": "sdxl_turbo", "device": "cpu", "dtype": "float32"},
                "generation": {"height": 8, "width": 8, "num_inference_steps": 1, "guidance_scale": 0., "generation_batch_size": 1, "output_format": "png", "release_model_after_generation": True},
                "baseline": {"enabled": True, "alpha_values": [0., .5]}, "same_phase_floor": {"enabled": True, "alpha_values": [0., .5], "gamma_values": [.1]}, "independent_white": {"enabled": False, "alpha_values": [0., .5], "gamma_values": [.1]},
                "quality_metrics": {"clip": {"enabled": False}, "hpsv3": {"enabled": False}}, "diversity_metrics": {"dreamsim": {"enabled": False}, "lpips": {"enabled": False}, "vendi_clip": {"enabled": False}}, "analysis": {"bootstrap_replicates": 2, "bootstrap_seed": 1, "confidence_level": .95}}
            config_path = root / "configs" / "small.yaml"; config_path.parent.mkdir(); config_path.write_text("# test\n")
            adapter = FakeAdapter()
            run_dir = generate(config, config_path, [{"block_id": "b", "prompt_id": "p", "prompt": "test", "seed_batch_id": "s", "batch_seed": 1}], run_id="r", selected_conditions=None, force=False, adapter=adapter)
            # 2 baseline galleries + same-phase alpha=.5; alpha=.0 is an alias.
            self.assertEqual(adapter.calls, 3)
            alias = read_jsonl(run_dir / "generations/fake/b/same_phase/alpha_0p0_gamma_0p1/samples.jsonl")
            self.assertEqual(len(alias), 4)
            self.assertTrue(all(row["is_exact_alias"] for row in alias))
            self.assertTrue(all(row["alias_of_condition_id"] == "baseline/alpha_0p0" for row in alias))

    def test_full_grid_has_152_logical_conditions_and_143_unique_galleries(self):
        config = {"baseline": {"enabled": True, "alpha_values": [round(i / 10, 1) for i in range(8)]},
                  "same_phase_floor": {"enabled": True, "alpha_values": [round(i / 10, 1) for i in range(8)], "gamma_values": [round(i / 10, 1) for i in range(1, 10)]},
                  "independent_white": {"enabled": True, "alpha_values": [round(i / 10, 1) for i in range(8)], "gamma_values": [round(i / 10, 1) for i in range(1, 10)]}}
        conditions = all_conditions(config)
        self.assertEqual(len(conditions), 152)
        aliases = sum(condition.family == "same_phase" and condition.alpha == 0.0 for condition in conditions)
        self.assertEqual(len(conditions) - aliases, 143)


if __name__ == "__main__": unittest.main()
