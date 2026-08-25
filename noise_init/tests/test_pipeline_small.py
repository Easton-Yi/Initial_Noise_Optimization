import copy
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from model_adapters import GenerationConfig, LatentSpec
from run_experiment import generate


class FakeAdapter:
    model_id = "fake"
    def latent_spec(self, height, width, batch_size): return LatentSpec((batch_size, 2, 8, 8), "none")
    def generate(self, prompt, latents, generation_config):
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


if __name__ == "__main__": unittest.main()
