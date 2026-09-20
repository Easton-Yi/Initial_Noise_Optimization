import tempfile
import unittest
from pathlib import Path

import yaml

from pc_specific_psd import config

MINIMAL_RAW = {
    "run": {"name": "test_run", "master_seed": 1, "outputs_root": "outputs"},
    "model": {"adapter": "sdxl_turbo", "checkpoint": "org/sdxl-turbo-fake", "dtype": "float32", "device": "cpu", "cpu_offload": False},
    "generation": {"height": 16, "width": 16, "num_inference_steps": 1, "guidance_scale": 0.0, "generation_batch_size": 1},
    "basis": {"patch_size": 5, "channels": 4, "patches_per_image": 8, "sampling_seed": 1, "split_seed": 2, "basis_output_path": "basis.pt"},
}


def _write_config(tmp_path: Path, raw: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(raw))
    return path


class LoadConfigStructureTests(unittest.TestCase):
    def test_minimal_config_loads_with_needs_calibration_fields_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), MINIMAL_RAW)
            loaded = config.load_config(path)
            self.assertEqual(loaded.run.name, "test_run")
            self.assertIsNone(loaded.probing.rho)
            self.assertIsNone(loaded.psd.protocol)
            self.assertEqual(loaded.psd.groups, ())

    def test_frozen_constants_are_echoed_regardless_of_yaml_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), MINIMAL_RAW)
            loaded = config.load_config(path)
            self.assertEqual(loaded.frozen.same_phase_alpha, 0.9)
            self.assertEqual(loaded.frozen.same_phase_gamma, 0.05)
            provenance = loaded.provenance()
            self.assertEqual(provenance["same_phase_alpha"], 0.9)
            self.assertEqual(provenance["same_phase_gamma"], 0.05)

    def test_generation_missing_required_field_raises_config_error(self):
        raw = {**MINIMAL_RAW, "generation": {"height": 16, "width": 16}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "generation missing required"):
                config.load_config(path)

    def test_generation_rejects_unknown_field(self):
        raw = {**MINIMAL_RAW, "generation": {**MINIMAL_RAW["generation"], "bogus_field": 1}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "unknown field"):
                config.load_config(path)

    def test_generation_release_model_after_generation_is_not_part_of_generation_config(self):
        raw = {**MINIMAL_RAW, "generation": {**MINIMAL_RAW["generation"], "release_model_after_generation": False}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            loaded = config.load_config(path)
            self.assertFalse(loaded.generation.release_model_after_generation)
            self.assertFalse(hasattr(loaded.generation.config, "release_model_after_generation"))

    def test_model_missing_required_field_raises(self):
        raw = {**MINIMAL_RAW, "model": {"adapter": "sdxl_turbo"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "model.checkpoint"):
                config.load_config(path)

    def test_model_adapter_flux2klein_is_refused_as_not_verified(self):
        raw = {**MINIMAL_RAW, "model": {**MINIMAL_RAW["model"], "adapter": "flux2klein"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "not_verified"):
                config.load_config(path)

    def test_model_adapter_unknown_value_is_refused_as_not_verified(self):
        raw = {**MINIMAL_RAW, "model": {**MINIMAL_RAW["model"], "adapter": "some_other_model"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "not_verified"):
                config.load_config(path)

    def test_model_optional_fields_default_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), MINIMAL_RAW)
            loaded = config.load_config(path)
            self.assertIsNone(loaded.model.revision)
            self.assertIsNone(loaded.model.cache_dir)
            self.assertIsNone(loaded.model.vae_checkpoint)

    def test_basis_patch_size_inconsistent_with_pc_groups_raises(self):
        raw = {**MINIMAL_RAW, "basis": {**MINIMAL_RAW["basis"], "patch_size": 2}}  # 4*2*2=16 < 100
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "PC_GROUPS"):
                config.load_config(path)

    def test_psd_rejects_frozen_constant_override(self):
        raw = {**MINIMAL_RAW, "psd": {"same_phase_alpha": 0.5}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "frozen constant"):
                config.load_config(path)

    def test_psd_rejects_unknown_group_id(self):
        raw = {**MINIMAL_RAW, "psd": {"groups": [
            {"group_id": "Z9", "tau_plus_candidates": [1.0], "tau_minus_candidates": [-1.0], "target_rms": 1.0}
        ]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaisesRegex(config.ConfigError, "unknown group_id"):
                config.load_config(path)

    def test_top_level_must_be_a_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("- just\n- a\n- list\n")
            with self.assertRaisesRegex(config.ConfigError, "must be a mapping"):
                config.load_config(path)


class ResolveRootTests(unittest.TestCase):
    def test_relative_paths_resolve_against_config_file_directory_not_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subdir = tmp_path / "sub"
            subdir.mkdir()
            path = _write_config(subdir, MINIMAL_RAW)
            loaded = config.load_config(path)

            elsewhere = tmp_path / "elsewhere"
            elsewhere.mkdir()
            original_cwd = Path.cwd()
            import os
            os.chdir(elsewhere)
            try:
                resolved = loaded.resolve_root("basis.pt")
            finally:
                os.chdir(original_cwd)
            self.assertEqual(resolved, (subdir / "basis.pt").resolve())

    def test_absolute_path_passes_through_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), MINIMAL_RAW)
            loaded = config.load_config(path)
            absolute = "/tmp/some/absolute/path.pt"
            self.assertEqual(loaded.resolve_root(absolute), Path(absolute))

    def test_none_resolves_to_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), MINIMAL_RAW)
            loaded = config.load_config(path)
            self.assertIsNone(loaded.resolve_root(None))


if __name__ == "__main__":
    unittest.main()
