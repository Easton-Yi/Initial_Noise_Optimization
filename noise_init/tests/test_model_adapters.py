from types import SimpleNamespace
import unittest

import torch
from PIL import Image

from model_adapters import GenerationConfig, _DiffusersAdapter


class _FakePipeline:
    def prepare_latents(self, batch_size, latents=None):
        self.received = latents
        return latents

    def __call__(self, *, latents, **kwargs):
        self.prepare_latents(1, latents=latents)
        return SimpleNamespace(images=[Image.new("RGB", (2, 2))])


class _IgnoringPipeline(_FakePipeline):
    def __call__(self, *, latents, **kwargs):
        self.prepare_latents(1, latents=None)
        return SimpleNamespace(images=[Image.new("RGB", (2, 2))])


class _TestAdapter(_DiffusersAdapter):
    def _load(self):
        self.pipe = _FakePipeline()


class ModelAdapterTests(unittest.TestCase):
    config = GenerationConfig(height=16, width=16, num_inference_steps=1, guidance_scale=0., generation_batch_size=1)

    def test_adapter_verifies_pipeline_received_the_supplied_latent(self):
        adapter = _TestAdapter({"device": "cpu", "dtype": "float32"}, model_id="test")
        adapter.pipe = _FakePipeline()
        latent = torch.randn(1, 2, 4, 4)
        adapter._call_one("test", latent, self.config)
        self.assertEqual(adapter.last_supplied_latent_hash, adapter.last_prepared_latent_hash)
        images = adapter.generate("test", torch.randn(4, 2, 4, 4), self.config)
        self.assertEqual(len(images), len(adapter.last_generated_latent_hashes))

    def test_adapter_rejects_pipeline_that_discards_the_supplied_latent(self):
        adapter = _TestAdapter({"device": "cpu", "dtype": "float32"}, model_id="test")
        adapter.pipe = _IgnoringPipeline()
        with self.assertRaisesRegex(RuntimeError, "did not pass supplied latents"):
            adapter._call_one("test", torch.randn(1, 2, 4, 4), self.config)


if __name__ == "__main__":
    unittest.main()
