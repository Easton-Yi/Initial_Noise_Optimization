from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from PIL import Image

from pc_specific_psd import adapters
from pc_specific_psd.compat_generation import GenerationConfig, derived_seed


class _FakePipeline:
    def __init__(self):
        self.received_generators: list[torch.Generator | None] = []
        self.last_call_kwargs = None

    def prepare_latents(self, batch_size, latents=None):
        self.received = latents
        return latents

    def __call__(self, *, latents, generator=None, **kwargs):
        self.prepare_latents(1, latents=latents)
        self.received_generators.append(generator)
        self.last_call_kwargs = kwargs
        return SimpleNamespace(images=[Image.new("RGB", (2, 2))])


class _ScalingPipeline(_FakePipeline):
    def prepare_latents(self, batch_size, latents=None):
        self.received = latents
        return latents * 2.0


class _IgnoringPipeline(_FakePipeline):
    def __call__(self, *, latents, generator=None, **kwargs):
        self.prepare_latents(1, latents=None)
        self.received_generators.append(generator)
        return SimpleNamespace(images=[Image.new("RGB", (2, 2))])


class _TestAdapterPCA(adapters.SDXLTurboAdapterPCA):
    """Bypasses real diffusers loading (mirrors noise_init/tests/test_model_adapters.py's
    own _TestAdapter pattern) while keeping _verify_and_record_revision reachable directly.
    """

    def _load(self) -> None:
        self.pipe = _FakePipeline()
        self._verify_and_record_revision(self.model_config.get("revision"))


class _FakeVAE:
    """No diffusers, no real weights: a deterministic stand-in whose
    "posterior mean" is just a downsampled, per-channel-averaged function of
    the preprocessed input -- enough to check encode_images_for_basis's shape
    and scaling-factor arithmetic without a real VAE.
    """

    def __init__(self, scaling_factor: float = 0.13025, latent_channels: int = 4, downscale: int = 8):
        self.config = SimpleNamespace(scaling_factor=scaling_factor)
        self._latent_channels = latent_channels
        self._downscale = downscale

    def encode(self, batch: torch.Tensor):
        b, _c, h, w = batch.shape
        pooled = torch.nn.functional.avg_pool2d(batch.mean(dim=1, keepdim=True), self._downscale)
        mean = pooled.expand(b, self._latent_channels, h // self._downscale, w // self._downscale).clone()
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))


class _TestAdapterPCAWithVAE(adapters.SDXLTurboAdapterPCA):
    """Like _TestAdapterPCA, but the fake pipe also carries a fake .vae so
    encode_images_for_basis has something to call.
    """

    def _load(self) -> None:
        self.pipe = SimpleNamespace(vae=_FakeVAE())
        self._verify_and_record_revision(self.model_config.get("revision"))


def _config() -> dict:
    return {"device": "cpu", "dtype": "float32", "checkpoint": "org/sdxl-turbo-fake"}


GENERATION_CONFIG = GenerationConfig(height=16, width=16, num_inference_steps=1, guidance_scale=0.0, generation_batch_size=1)


class RevisionResolutionTests(unittest.TestCase):
    """resolve_cached_commit_hash reads the real huggingface_hub cache-on-disk
    layout, so it is exercised here against a hand-built fake cache directory
    -- no network, no GPU, no real model weights.
    """

    def _make_fake_cache(self, tmp_path: Path, repo_id: str, resolved_sha: str) -> Path:
        cache_dir = tmp_path / "hub"
        snapshot_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "snapshots" / resolved_sha
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "model_index.json").write_text("{}")
        refs_dir = cache_dir / f"models--{repo_id.replace('/', '--')}" / "refs"
        refs_dir.mkdir(parents=True)
        return cache_dir

    def test_resolves_branch_revision_to_the_cached_snapshot_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_id = "org/sdxl-turbo-fake"
            resolved_sha = "a" * 40
            cache_dir = self._make_fake_cache(tmp_path, repo_id, resolved_sha)
            (cache_dir / f"models--{repo_id.replace('/', '--')}" / "refs" / "main").write_text(resolved_sha)

            resolved = adapters.resolve_cached_commit_hash(repo_id, cache_dir=str(cache_dir), revision="main")
            self.assertEqual(resolved, resolved_sha)

    def test_raises_when_cache_has_no_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(adapters.RevisionResolutionError):
                adapters.resolve_cached_commit_hash("org/does-not-exist", cache_dir=tmp, revision="main")

    def test_resolves_a_full_sha_revision_to_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_id = "org/sdxl-turbo-fake"
            resolved_sha = "e" * 40
            cache_dir = self._make_fake_cache(tmp_path, repo_id, resolved_sha)

            resolved = adapters.resolve_cached_commit_hash(repo_id, cache_dir=str(cache_dir), revision=resolved_sha)
            self.assertEqual(resolved, resolved_sha)


class VerifyAndRecordRevisionTests(unittest.TestCase):
    def test_no_requested_revision_records_none_without_touching_the_cache(self):
        adapter = _TestAdapterPCA(_config())
        adapter._verify_and_record_revision(None)
        self.assertEqual(adapter.revision_provenance, adapters.RevisionProvenance(None, None))

    def test_requested_revision_resolved_from_cache_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_id = "org/sdxl-turbo-fake"
            resolved_sha = "d" * 40
            model_dir = tmp_path / f"models--{repo_id.replace('/', '--')}"
            snapshot_dir = model_dir / "snapshots" / resolved_sha
            snapshot_dir.mkdir(parents=True)
            (snapshot_dir / "model_index.json").write_text("{}")
            refs_dir = model_dir / "refs"
            refs_dir.mkdir(parents=True)
            (refs_dir / "main").write_text(resolved_sha)

            config = _config()
            config["cache_dir"] = str(tmp_path)
            adapter = _TestAdapterPCA(config)
            adapter._verify_and_record_revision("main")
            self.assertEqual(adapter.revision_provenance.requested_revision, "main")
            self.assertEqual(adapter.revision_provenance.resolved_commit_hash, resolved_sha)

    def test_requested_revision_unresolvable_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = _config()
            config["cache_dir"] = tmp
            adapter = _TestAdapterPCA(config)
            with self.assertRaises(adapters.RevisionResolutionError):
                adapter._verify_and_record_revision("main")

    def test_load_calls_verify_and_record_revision(self):
        """_load() (even the fake-pipeline test override) must route revision
        confirmation through _verify_and_record_revision -- the real override
        does this immediately after from_pretrained finishes.
        """
        adapter = _TestAdapterPCA(_config())
        self.assertIsNone(adapter.revision_provenance)
        adapter._load()
        self.assertIsNotNone(adapter.revision_provenance)


class PairedGeneratorTests(unittest.TestCase):
    def test_512_image_generation_keeps_64_square_latent_and_512_pipeline_size(self):
        adapter = _TestAdapterPCA(_config())
        latents = torch.randn(1, 4, 64, 64)
        generation = GenerationConfig(
            height=512, width=512, num_inference_steps=1, guidance_scale=0.0,
            generation_batch_size=1,
        )
        adapter.generate(
            "prompt", latents, [("p000", "p000_s000", 0)], seed=999,
            generation_config=generation,
        )
        self.assertEqual(tuple(adapter.pipe.received.shape), (1, 4, 64, 64))
        self.assertEqual(adapter.pipe.last_call_kwargs["height"], 512)
        self.assertEqual(adapter.pipe.last_call_kwargs["width"], 512)

    def test_same_pair_key_across_two_calls_gives_identical_generator_seed(self):
        adapter = _TestAdapterPCA(_config())
        pair_key = ("p000", "p000_s000", 0)
        latents = torch.randn(1, 4, 4, 4)

        adapter.generate("prompt A", latents, [pair_key], seed=999, generation_config=GENERATION_CONFIG)
        first_seed = adapter.last_generator_seeds[0]

        adapter2 = _TestAdapterPCA(_config())
        adapter2.generate("prompt B (different condition, same draw)", latents, [pair_key], seed=999, generation_config=GENERATION_CONFIG)
        second_seed = adapter2.last_generator_seeds[0]

        self.assertEqual(first_seed, second_seed)

    def test_different_pair_keys_never_collide(self):
        adapter = _TestAdapterPCA(_config())
        latents = torch.randn(2, 4, 4, 4)
        pair_keys = [("p000", "p000_s000", 0), ("p000", "p000_s000", 1)]
        adapter.generate("prompt", latents, pair_keys, seed=999, generation_config=GENERATION_CONFIG)
        self.assertNotEqual(adapter.last_generator_seeds[0], adapter.last_generator_seeds[1])

    def test_generator_seed_matches_the_documented_derivation_formula(self):
        adapter = _TestAdapterPCA(_config())
        pair_key = ("p001", "p001_s001", 2)
        latents = torch.randn(1, 4, 4, 4)
        adapter.generate("prompt", latents, [pair_key], seed=777, generation_config=GENERATION_CONFIG)
        expected = derived_seed(777, adapters.PAIRED_GENERATOR_NAMESPACE, pair_key)
        self.assertEqual(adapter.last_generator_seeds[0], expected)

    def test_generator_object_actually_reaches_the_pipeline_call(self):
        adapter = _TestAdapterPCA(_config())
        pair_key = ("p000", "p000_s000", 0)
        latents = torch.randn(1, 4, 4, 4)
        adapter.generate("prompt", latents, [pair_key], seed=999, generation_config=GENERATION_CONFIG)
        received_generator = adapter.pipe.received_generators[0]
        self.assertIsInstance(received_generator, torch.Generator)
        self.assertEqual(received_generator.initial_seed(), adapter.last_generator_seeds[0])

    def test_fresh_generator_is_constructed_every_call_not_cached(self):
        adapter = _TestAdapterPCA(_config())
        latents = torch.randn(1, 4, 4, 4)
        adapter.generate("prompt", latents, [("p000", "b0", 0)], seed=1, generation_config=GENERATION_CONFIG)
        first_generator = adapter.pipe.received_generators[0]
        adapter.generate("prompt", latents, [("p000", "b0", 1)], seed=1, generation_config=GENERATION_CONFIG)
        second_generator = adapter.pipe.received_generators[1]
        self.assertIsNot(first_generator, second_generator)

    def test_mismatched_pair_keys_length_raises(self):
        adapter = _TestAdapterPCA(_config())
        latents = torch.randn(2, 4, 4, 4)
        with self.assertRaises(ValueError):
            adapter.generate("prompt", latents, [("p000", "b0", 0)], seed=1, generation_config=GENERATION_CONFIG)

    def test_non_4d_latents_rejected(self):
        adapter = _TestAdapterPCA(_config())
        latents = torch.randn(4, 4, 4)
        with self.assertRaises(ValueError):
            adapter.generate("prompt", latents, [("p000", "b0", 0)], seed=1, generation_config=GENERATION_CONFIG)

    def test_generate_verifies_injected_latent_reaches_prepare_latents(self):
        adapter = _TestAdapterPCA(_config())
        adapter._load()
        adapter.pipe = _IgnoringPipeline()
        latents = torch.randn(1, 4, 4, 4)
        with self.assertRaisesRegex(RuntimeError, "did not pass supplied latents"):
            adapter.generate("prompt", latents, [("p000", "b0", 0)], seed=1, generation_config=GENERATION_CONFIG)

    def test_result_count_matches_latent_hash_record_count(self):
        adapter = _TestAdapterPCA(_config())
        latents = torch.randn(3, 4, 4, 4)
        pair_keys = [("p000", "b0", 0), ("p000", "b0", 1), ("p000", "b0", 2)]
        images = adapter.generate("prompt", latents, pair_keys, seed=1, generation_config=GENERATION_CONFIG)
        self.assertEqual(len(images), 3)
        self.assertEqual(len(adapter.last_generated_latent_hashes), 3)
        self.assertEqual(adapter.last_pair_keys, pair_keys)

    def test_records_actual_prepare_latents_return_when_pipeline_scales_it(self):
        adapter = _TestAdapterPCA(_config())
        adapter._load()
        adapter.pipe = _ScalingPipeline()
        latent = torch.randn(1, 4, 4, 4)
        adapter.generate(
            "prompt", latent, [("p000", "b0", 0)], seed=1,
            generation_config=GENERATION_CONFIG,
        )
        self.assertNotEqual(
            adapter.last_prepare_latents_input_hash,
            adapter.last_prepare_latents_return_hash,
        )
        self.assertEqual(
            adapter.last_injection_record["prepare_latents_return_hash"],
            adapter.last_prepare_latents_return_hash,
        )
        self.assertEqual(adapter.last_injection_record["prepare_latents_return_shape"], [1, 4, 4, 4])


class LoadAndPreprocessImageTests(unittest.TestCase):
    def test_output_shape_is_512_and_range_is_minus1_to_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (800, 600), color=(0, 128, 255)).save(path)
            tensor = adapters._load_and_preprocess_image(str(path))
        self.assertEqual(tuple(tensor.shape), (3, 512, 512))
        self.assertGreaterEqual(float(tensor.min()), -1.0 - 1e-6)
        self.assertLessEqual(float(tensor.max()), 1.0 + 1e-6)

    def test_non_square_image_is_center_cropped_after_equal_ratio_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (1024, 512), color=(10, 20, 30)).save(path)
            tensor = adapters._load_and_preprocess_image(str(path))
        self.assertEqual(tuple(tensor.shape), (3, 512, 512))


class EncodeImagesForBasisTests(unittest.TestCase):
    def _write_image(self, tmp: Path, name: str, color: tuple[int, int, int], size=(640, 480)) -> str:
        path = tmp / name
        Image.new("RGB", size, color=color).save(path)
        return str(path)

    def test_returns_one_row_per_image_as_float32_cpu_tensor(self):
        adapter = _TestAdapterPCAWithVAE(_config())
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = [
                self._write_image(tmp_path, "a.png", (10, 20, 30)),
                self._write_image(tmp_path, "b.png", (200, 210, 220)),
                self._write_image(tmp_path, "c.png", (0, 0, 0)),
            ]
            latents = adapter.encode_images_for_basis(paths)
        self.assertEqual(tuple(latents.shape), (3, 4, 64, 64))
        self.assertEqual(latents.dtype, torch.float32)
        self.assertEqual(latents.device.type, "cpu")

    def test_lazily_loads_pipe_when_not_already_loaded(self):
        adapter = _TestAdapterPCAWithVAE(_config())
        self.assertIsNone(adapter.pipe)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_image(Path(tmp), "a.png", (5, 5, 5))
            adapter.encode_images_for_basis([path])
        self.assertIsNotNone(adapter.pipe)

    def test_posterior_mean_is_scaled_by_vae_scaling_factor(self):
        adapter = _TestAdapterPCAWithVAE(_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_image(Path(tmp), "a.png", (50, 100, 150))
            latents = adapter.encode_images_for_basis([path])
            preprocessed = adapters._load_and_preprocess_image(path).unsqueeze(0)
            raw_mean = adapter.pipe.vae.encode(preprocessed).latent_dist.mean
            expected = (raw_mean * adapter.pipe.vae.config.scaling_factor).to(torch.float32)
        self.assertTrue(torch.allclose(latents, expected, atol=1e-5))

    def test_different_images_produce_different_latents(self):
        adapter = _TestAdapterPCAWithVAE(_config())
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = [
                self._write_image(tmp_path, "a.png", (0, 0, 0)),
                self._write_image(tmp_path, "b.png", (255, 255, 255)),
            ]
            latents = adapter.encode_images_for_basis(paths)
        self.assertFalse(torch.allclose(latents[0], latents[1]))


if __name__ == "__main__":
    unittest.main()
