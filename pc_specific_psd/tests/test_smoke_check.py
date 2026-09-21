"""Change 5: ``adapters.smoke_check`` must prove the tau=0 latent path is
wired correctly end to end (finite/correctly-shaped latent, the same tensor
actually injected into the pipeline, agreement with the independently scaled
``same_phase_floor`` ground truth) *without* requiring any calibration
result -- it runs before ``calibrate`` ever produces one. Any failure is
reported as ``passed=False``, never raised. ``smoke_check_cache_key`` also
includes the frozen reference-scale profile.
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from pc_specific_psd import adapters, patch_codec, psd_editor, runner
from pc_specific_psd.compat_generation import same_phase_floor
from pc_specific_psd.tests._runner_test_support import CHANNELS, FakeAdapterPCA, load_test_config


def _build(tmp_path: Path):
    loaded = load_test_config(tmp_path)
    codec = runner.load_codec(loaded, allow_synthetic=True)
    adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
    return loaded, codec, adapter


class SmokeCheckPassingTests(unittest.TestCase):
    def test_passes_with_no_calibration_result_present_at_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))
            self.assertFalse(loaded.resolve_root(loaded.psd.calibration_result_path).exists())
            result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertTrue(result.passed)
            self.assertIsNone(result.reason)

    def test_actually_calls_the_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))
            adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertEqual(adapter.call_count, 1)

    def test_512_image_config_builds_64_square_latent_but_keeps_512_pipeline_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, _adapter = _build(Path(tmp))
            generation_config = replace(loaded.generation.config, height=512, width=512)
            loaded = replace(loaded, generation=replace(loaded.generation, config=generation_config))
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertTrue(result.passed, msg=result.reason)
            self.assertEqual(tuple(adapter.pipe.received.shape), (1, CHANNELS, 64, 64))
            self.assertEqual(adapter.pipe.last_call_kwargs["height"], 512)
            self.assertEqual(adapter.pipe.last_call_kwargs["width"], 512)


class SmokeCheckCodecMismatchTests(unittest.TestCase):
    def test_codec_channel_patch_size_mismatch_fails_without_raising(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, _codec, adapter = _build(Path(tmp))
            wrong_codec = patch_codec.OverlapCodec(torch.eye(4, dtype=torch.float32), patch_size=2, channels=1)
            result = adapters.smoke_check(loaded, adapter, codec=wrong_codec)
            self.assertFalse(result.passed)
            self.assertIn("codec", result.reason)


class SmokeCheckLatentAssertionTests(unittest.TestCase):
    def test_non_finite_latent_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))
            channels, height, width = CHANNELS, loaded.generation.config.height, loaded.generation.config.width
            nan_latent = torch.full((1, channels, height, width), float("nan"))
            with patch.object(adapters.psd_editor, "apply_psd_edit_tau_zero", return_value=nan_latent):
                result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertFalse(result.passed)
            self.assertIn("non-finite", result.reason)

    def test_wrong_shape_latent_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))
            wrong_shape = torch.zeros((1, CHANNELS, 4, 4))
            with patch.object(adapters.psd_editor, "apply_psd_edit_tau_zero", return_value=wrong_shape):
                result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertFalse(result.passed)
            self.assertIn("shape", result.reason)


class SmokeCheckInjectionIdentityTests(unittest.TestCase):
    def test_tensor_actually_injected_differs_from_computed_latent_fails(self):
        class _TamperingAdapter(FakeAdapterPCA):
            def generate(self, *args, **kwargs):
                images = super().generate(*args, **kwargs)
                self.last_generated_latent_hashes = ["not_the_real_hash"]
                return images

        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            codec = runner.load_codec(loaded, allow_synthetic=True)
            adapter = _TamperingAdapter(loaded.model.as_model_config_dict())
            result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertFalse(result.passed)
            self.assertIn("different latent", result.reason)


class SmokeCheckGroundTruthAgreementTests(unittest.TestCase):
    def test_current_scaled_reference_passes_actual_latent_rms_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))
            result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertTrue(result.passed, msg=result.reason)

    def test_disagreement_with_independent_same_phase_floor_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))
            with patch("pc_specific_psd.adapters.same_phase_floor", side_effect=lambda x, *_: torch.zeros_like(x)):
                result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertFalse(result.passed)
            self.assertIn("ground truth", result.reason)

    def test_old_unscaled_reference_is_rejected_by_actual_latent_rms_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))

            def old_reference(base_white):
                return same_phase_floor(
                    base_white, psd_editor.SAME_PHASE_ALPHA, psd_editor.SAME_PHASE_GAMMA
                )

            with patch.object(adapters.psd_editor, "apply_psd_edit_tau_zero", side_effect=old_reference):
                result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertFalse(result.passed)
            self.assertIn("RMS", result.reason)


class SmokeCheckExceptionWrappingTests(unittest.TestCase):
    def test_unexpected_exception_is_caught_and_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, codec, adapter = _build(Path(tmp))
            with patch.object(adapters.psd_editor, "base_white_for_draw", side_effect=RuntimeError("boom")):
                result = adapters.smoke_check(loaded, adapter, codec=codec)
            self.assertFalse(result.passed)
            self.assertEqual(result.reason, "boom")


class SmokeCheckCacheKeyTests(unittest.TestCase):
    def test_returns_config_hash_basis_hash_and_scale_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, _codec, _adapter = _build(Path(tmp))
            key = adapters.smoke_check_cache_key(loaded)
            self.assertEqual(len(key), 3)
            config_hash, basis_hash, profile = key
            self.assertIsInstance(config_hash, str)
            self.assertIsInstance(basis_hash, str)
            self.assertEqual(profile, psd_editor.REFERENCE_SCALE_PROFILE)

    def test_stable_across_repeated_calls_with_unchanged_config_and_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, _codec, _adapter = _build(Path(tmp))
            self.assertEqual(adapters.smoke_check_cache_key(loaded), adapters.smoke_check_cache_key(loaded))

    def test_unaffected_by_calibration_presence(self):
        """The whole point of excluding calibration_hash: the key must be
        identical before and after a calibration registry is written.
        """
        from pc_specific_psd.tests._runner_test_support import freeze_selected_calibration

        with tempfile.TemporaryDirectory() as tmp:
            loaded, _codec, _adapter = _build(Path(tmp))
            before = adapters.smoke_check_cache_key(loaded)
            freeze_selected_calibration(loaded, group_ids=("B5",))
            after = adapters.smoke_check_cache_key(loaded)
            self.assertEqual(before, after)

    def test_raises_when_basis_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, _codec, _adapter = _build(Path(tmp))
            loaded.resolve_root(loaded.basis.basis_output_path).unlink()
            with self.assertRaises(RuntimeError):
                adapters.smoke_check_cache_key(loaded)


if __name__ == "__main__":
    unittest.main()
