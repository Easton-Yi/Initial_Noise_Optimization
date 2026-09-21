"""Frozen same-phase reference (plan sections 5.1/14): SAME_PHASE_ALPHA=0.9 and
SAME_PHASE_GAMMA=0.05 are psd_editor.py module constants, never a config field.
This file asserts that directly against the source constants, independent of
config.py's own (more thorough) structural tests in test_config.py.
"""
import tempfile
import unittest
from pathlib import Path

import yaml

from pc_specific_psd import config, psd_editor

MINIMAL_RAW = {
    "run": {"name": "test_run", "master_seed": 1, "outputs_root": "outputs"},
    "model": {"adapter": "sdxl_turbo", "checkpoint": "org/sdxl-turbo-fake", "dtype": "float32", "device": "cpu", "cpu_offload": False},
    "generation": {"height": 16, "width": 16, "num_inference_steps": 1, "guidance_scale": 0.0, "generation_batch_size": 1},
    "basis": {"patch_size": 5, "channels": 4, "patches_per_image": 8, "sampling_seed": 1, "split_seed": 2, "basis_output_path": "basis.pt"},
}


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


class FrozenConstantValuesTests(unittest.TestCase):
    def test_same_phase_alpha_is_exactly_zero_point_nine(self):
        self.assertEqual(psd_editor.SAME_PHASE_ALPHA, 0.9)

    def test_same_phase_gamma_is_exactly_zero_point_zero_five(self):
        self.assertEqual(psd_editor.SAME_PHASE_GAMMA, 0.05)

    def test_reference_scale_profile_is_frozen(self):
        self.assertEqual(psd_editor.REFERENCE_SCALE_PROFILE, "expected_unit_rms_rfft_v1")

    def test_config_frozen_constants_default_to_the_same_source_values(self):
        self.assertEqual(config.FrozenConstants().same_phase_alpha, psd_editor.SAME_PHASE_ALPHA)
        self.assertEqual(config.FrozenConstants().same_phase_gamma, psd_editor.SAME_PHASE_GAMMA)
        self.assertEqual(
            config.FrozenConstants().reference_scale_profile, psd_editor.REFERENCE_SCALE_PROFILE
        )


class FrozenConstantOverrideRejectionTests(unittest.TestCase):
    def test_alpha_override_attempt_is_rejected(self):
        raw = {**MINIMAL_RAW, "psd": {"same_phase_alpha": 0.5}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "frozen constant"):
                config.load_config(path)

    def test_gamma_override_attempt_is_rejected(self):
        raw = {**MINIMAL_RAW, "psd": {"same_phase_gamma": 0.1}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "frozen constant"):
                config.load_config(path)

    def test_reference_scale_profile_override_attempt_is_rejected(self):
        raw = {**MINIMAL_RAW, "psd": {"reference_scale_profile": "legacy"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "frozen constant"):
                config.load_config(path)


class FrozenConstantProvenanceTests(unittest.TestCase):
    def test_resolved_config_echoes_both_frozen_values_regardless_of_yaml_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), MINIMAL_RAW)
            loaded = config.load_config(path)
            self.assertEqual(loaded.frozen.same_phase_alpha, 0.9)
            self.assertEqual(loaded.frozen.same_phase_gamma, 0.05)
            self.assertEqual(loaded.frozen.reference_scale_profile, "expected_unit_rms_rfft_v1")

    def test_provenance_dict_echoes_both_frozen_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), MINIMAL_RAW)
            loaded = config.load_config(path)
            provenance = loaded.provenance()
            self.assertEqual(provenance["same_phase_alpha"], 0.9)
            self.assertEqual(provenance["same_phase_gamma"], 0.05)
            self.assertEqual(provenance["reference_scale_profile"], "expected_unit_rms_rfft_v1")


if __name__ == "__main__":
    unittest.main()
