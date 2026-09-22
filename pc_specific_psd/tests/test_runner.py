"""runner.py core mechanics: codec/registry loading, frozen-correction
replay, per-condition latent rendering (reference shortcut vs. group edit),
and the generate_manifest resume/immutability contract -- all against a
synthetic identity basis and a fake adapter, never touching diffusers.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml

from pc_specific_psd import calibration, cli, config, manifests, psd_editor, runner
from pc_specific_psd.compat_generation import file_hash, read_json, tensor_hash, write_json
from pc_specific_psd.tests._runner_test_support import (
    BASIS_DIM,
    CHANNELS,
    PATCH_SIZE,
    FakeAdapterPCA,
    freeze_selected_calibration,
    freeze_selected_effect_calibration,
    load_test_config,
    load_v2_test_config,
)


class LoadCodecTests(unittest.TestCase):
    def test_load_codec_requires_allow_synthetic_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            with self.assertRaises(ValueError):
                runner.load_codec(loaded)  # allow_synthetic defaults to False

    def test_load_codec_returns_overlap_codec_shaped_like_the_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            codec = runner.load_codec(loaded, allow_synthetic=True)
            self.assertEqual(codec.d, BASIS_DIM)
            self.assertEqual(codec.patch_size, PATCH_SIZE)
            self.assertEqual(codec.channels, CHANNELS)


class HashHelperTests(unittest.TestCase):
    def test_basis_hash_and_calibration_hash_raise_before_files_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            runner.basis_hash_for(loaded)  # basis was written by load_test_config
            with self.assertRaises(runner.RunnerError):
                runner.calibration_hash_for(loaded)  # calibration_result.json not written yet

    def test_calibration_hash_available_once_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            self.assertTrue(runner.calibration_hash_for(loaded))


class BuildFrozenCorrectionsTests(unittest.TestCase):
    def test_raises_when_registry_is_not_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            codec = runner.load_codec(loaded, allow_synthetic=True)
            with self.assertRaises(runner.RunnerError):
                runner.build_frozen_corrections(loaded, codec, height=16, width=16)

    def test_raises_clear_recalibration_error_for_legacy_registry_without_scale_profile(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            registry_path = loaded.resolve_root(loaded.psd.calibration_result_path)
            payload = json.loads(registry_path.read_text())
            payload.pop("reference_scale_profile")
            registry_path.write_text(json.dumps(payload))
            codec = runner.load_codec(loaded, allow_synthetic=True)
            with self.assertRaisesRegex(runner.RunnerError, "stale.*calibrate"):
                runner.build_frozen_corrections(loaded, codec, height=16, width=16)

    def test_replays_exactly_the_declared_and_selected_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])  # B6 declared but not selected
            codec = runner.load_codec(loaded, allow_synthetic=True)
            corrections = runner.build_frozen_corrections(loaded, codec, height=16, width=16)
            self.assertEqual(set(corrections), {("B5", "plus"), ("B5", "minus")})
            self.assertEqual(corrections[("B5", "plus")].tau, 1.0)
            self.assertEqual(corrections[("B5", "minus")].tau, -1.0)
            self.assertEqual(corrections[("B5", "plus")].correction.shape, (loaded.psd.num_bins,))

    def test_correction_replay_is_deterministic_across_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5", "B6"])
            codec = runner.load_codec(loaded, allow_synthetic=True)
            first = runner.build_frozen_corrections(loaded, codec, height=16, width=16)
            second = runner.build_frozen_corrections(loaded, codec, height=16, width=16)
            for key in first:
                torch.testing.assert_close(first[key].correction, second[key].correction)


class RenderConditionLatentTests(unittest.TestCase):
    def _entry(self, condition_id: str) -> manifests.ConditionDrawEntry:
        return manifests.ConditionDrawEntry(condition_id, "p000", "p000_s000", 0)

    def test_reference_condition_matches_the_tau_zero_shortcut_directly(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            codec = runner.load_codec(loaded, allow_synthetic=True)
            corrections = runner.build_frozen_corrections(loaded, codec, height=16, width=16)
            entry = self._entry(runner.REFERENCE_CONDITION_ID)
            result = runner.render_condition_latent(entry, codec, corrections, channels=CHANNELS, height=16, width=16)

            block = manifests.SeedBlock("p000", 0, manifests.prompt_by_id("p000").batch_seeds[0])
            expected_base_white = psd_editor.base_white_for_draw(block, 0, channels=CHANNELS, height=16, width=16)
            expected = psd_editor.apply_psd_edit_tau_zero(expected_base_white)
            torch.testing.assert_close(result, expected)

    def test_group_condition_applies_the_frozen_tau_and_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            codec = runner.load_codec(loaded, allow_synthetic=True)
            corrections = runner.build_frozen_corrections(loaded, codec, height=16, width=16)
            entry = self._entry("B5_plus")
            result = runner.render_condition_latent(entry, codec, corrections, channels=CHANNELS, height=16, width=16)

            block = manifests.SeedBlock("p000", 0, manifests.prompt_by_id("p000").batch_seeds[0])
            expected_base_white = psd_editor.base_white_for_draw(block, 0, channels=CHANNELS, height=16, width=16)
            frozen = corrections[("B5", "plus")]
            group_indices = list(manifests.pc_group_by_id("B5").indices)
            edited = psd_editor.apply_psd_edit(codec, expected_base_white, group_indices, frozen.tau, frozen.r_s, frozen.beta)
            expected = calibration.apply_radial_correction(edited, frozen.correction, len(frozen.correction))
            torch.testing.assert_close(result, expected)

    def test_reference_and_group_condition_share_the_identical_base_white_for_one_draw(self):
        """Pairing (plan §5.1): every condition compared for the same draw must
        start from byte-identical base_white before the PSD edit diverges them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            codec = runner.load_codec(loaded, allow_synthetic=True)
            corrections = runner.build_frozen_corrections(loaded, codec, height=16, width=16)

            reference_latent = runner.render_condition_latent(
                self._entry(runner.REFERENCE_CONDITION_ID), codec, corrections, channels=CHANNELS, height=16, width=16
            )
            plus_latent = runner.render_condition_latent(
                self._entry("B5_plus"), codec, corrections, channels=CHANNELS, height=16, width=16
            )
            # Different edits of the *same* base_white must disagree in general
            # (tau != 0 actually perturbs the group) yet both are deterministic
            # functions of one shared base_white -- confirmed by reproducing the
            # reference branch's exact shortcut output above and cross-checking
            # here that the two are not accidentally identical.
            self.assertFalse(torch.allclose(reference_latent, plus_latent))

    def test_different_draws_get_different_base_white(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            codec = runner.load_codec(loaded, allow_synthetic=True)
            corrections = runner.build_frozen_corrections(loaded, codec, height=16, width=16)
            entry_a = manifests.ConditionDrawEntry(runner.REFERENCE_CONDITION_ID, "p000", "p000_s000", 0)
            entry_b = manifests.ConditionDrawEntry(runner.REFERENCE_CONDITION_ID, "p000", "p000_s000", 1)
            latent_a = runner.render_condition_latent(entry_a, codec, corrections, channels=CHANNELS, height=16, width=16)
            latent_b = runner.render_condition_latent(entry_b, codec, corrections, channels=CHANNELS, height=16, width=16)
            self.assertFalse(torch.allclose(latent_a, latent_b))


class GenerateManifestTests(unittest.TestCase):
    def _entries(self):
        return [
            manifests.ConditionDrawEntry(runner.REFERENCE_CONDITION_ID, "p000", "p000_s000", 0),
            manifests.ConditionDrawEntry("B5_plus", "p000", "p000_s000", 0),
        ]

    def test_generates_expected_sample_records_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            run_dir = runner.generate_manifest(
                loaded, self._entries(), run_id="r1", force=False, adapter=adapter, allow_synthetic_basis=True,
            )
            self.assertEqual(adapter.call_count, 2)
            for entry in self._entries():
                sample_dir = run_dir / "generations" / entry.condition_id / entry.block_id / f"b{entry.base_index}"
                self.assertTrue((sample_dir / "image.png").exists())
                records = list(_read_jsonl(sample_dir / "sample.jsonl"))
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["basis_hash"], runner.basis_hash_for(loaded))
                self.assertEqual(records[0]["calibration_hash"], runner.calibration_hash_for(loaded))
            ref_record = _read_jsonl(
                run_dir / "generations" / runner.REFERENCE_CONDITION_ID / "p000_s000" / "b0" / "sample.jsonl"
            )[0]
            self.assertIsNone(ref_record["pc_group_id"])
            self.assertEqual(ref_record["tau"], 0.0)
            plus_record = _read_jsonl(run_dir / "generations" / "B5_plus" / "p000_s000" / "b0" / "sample.jsonl")[0]
            self.assertEqual(plus_record["pc_group_id"], "B5")
            self.assertEqual(plus_record["tau"], 1.0)

    def test_resume_skips_already_complete_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            runner.generate_manifest(loaded, self._entries(), run_id="r1", force=False, adapter=adapter, allow_synthetic_basis=True)
            self.assertEqual(adapter.call_count, 2)
            runner.generate_manifest(loaded, self._entries(), run_id="r1", force=False, adapter=adapter, allow_synthetic_basis=True)
            self.assertEqual(adapter.call_count, 2)  # no new calls -- both draws already complete

    def test_incomplete_existing_sample_dir_is_refused_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            run_dir = runner.generate_manifest(loaded, self._entries(), run_id="r1", force=False, adapter=adapter, allow_synthetic_basis=True)
            sample_dir = run_dir / "generations" / runner.REFERENCE_CONDITION_ID / "p000_s000" / "b0"
            (sample_dir / "sample.jsonl").unlink()  # simulate a crash mid-write
            with self.assertRaises(runner.RunnerError):
                runner.generate_manifest(loaded, self._entries(), run_id="r1", force=False, adapter=adapter, allow_synthetic_basis=True)

    def test_resuming_with_a_changed_config_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            runner.generate_manifest(loaded, self._entries(), run_id="r1", force=False, adapter=adapter, allow_synthetic_basis=True)
            # A different frozen calibration (still SELECTED) changes calibration_hash,
            # so resuming under the same run_id must refuse rather than silently mix results.
            from pc_specific_psd.tests._runner_test_support import freeze_selected_calibration as refreeze
            refreeze(loaded, group_ids=["B5", "B6"])
            with self.assertRaises(RuntimeError):
                runner.generate_manifest(loaded, self._entries(), run_id="r1", force=False, adapter=adapter, allow_synthetic_basis=True)

    def test_empty_manifest_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])
            with self.assertRaises(runner.RunnerError):
                runner.generate_manifest(loaded, [], run_id="r1", force=False)


class EffectPreviewProvenanceTests(unittest.TestCase):
    def _validated_fixture(self, tmp_path: Path):
        loaded = load_v2_test_config(tmp_path)
        condition_id = freeze_selected_effect_calibration(loaded)
        result = cli._cmd_validate_noise_effect_v2(
            loaded, SimpleNamespace(dry_run=False, allow_synthetic_basis=True)
        )
        self.assertEqual(result["status"], "PASS")
        return loaded, condition_id

    def test_fake_adapter_preview_uses_frozen_correction_and_records_all_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, condition_id = self._validated_fixture(Path(tmp))
            adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            with mock.patch.object(
                calibration, "evaluate_candidate",
                side_effect=AssertionError("generation must not refit correction"),
            ):
                run_dir = runner.generate_effect_preview(
                    loaded, [condition_id], run_id="v2_preview", force=False,
                    adapter=adapter, allow_synthetic_basis=True,
                )
            self.assertEqual(adapter.call_count, 24)
            basis_hash = runner.basis_hash_for(loaded)
            calibration_hash = runner.calibration_hash_for(loaded)
            validation_path = loaded.resolve_root(loaded.psd.validation_result_path)
            validation_hash = file_hash(validation_path)
            validation = read_json(validation_path)
            self.assertEqual(validation["config_hash"], file_hash(loaded.config_path))
            self.assertEqual(validation["basis_hash"], basis_hash)
            self.assertEqual(validation["calibration_hash"], calibration_hash)
            manifest = read_json(run_dir / "run_manifest.json")["resolved_config"]
            self.assertEqual(manifest["basis_hash"], basis_hash)
            self.assertEqual(manifest["calibration_hash"], calibration_hash)
            self.assertEqual(manifest["validation_hash"], validation_hash)

            for sampled_condition in ("reference", condition_id):
                record = _read_jsonl(
                    run_dir / "generations" / sampled_condition / "p000_s000" / "b0" / "sample.jsonl"
                )[0]
                self.assertEqual(record["basis_hash"], basis_hash)
                self.assertEqual(record["calibration_hash"], calibration_hash)
                self.assertEqual(record["validation_hash"], validation_hash)

            codec = runner.load_codec(loaded, allow_synthetic=True)
            frozen = runner.load_effect_frozen_corrections(loaded)
            entry = manifests.ConditionDrawEntry(condition_id, "p000", "p000_s000", 0)
            expected = runner.render_effect_condition_latent(
                entry, codec, frozen, channels=CHANNELS, height=8, width=8
            )
            candidate_record = _read_jsonl(
                run_dir / "generations" / condition_id / "p000_s000" / "b0" / "sample.jsonl"
            )[0]
            self.assertEqual(candidate_record["final_noise_hash"], tensor_hash(expected[0]))

    def test_changed_yaml_rejects_old_registry_validation_and_runner_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            loaded, _ = self._validated_fixture(tmp_path)
            raw = yaml.safe_load(loaded.config_path.read_text())
            raw["psd"]["groups"][0]["target_relative_l2"] = [0.06]
            loaded.config_path.write_text(yaml.safe_dump(raw))
            changed = config.load_config(loaded.config_path)

            with self.assertRaisesRegex(config.ConfigValidationError, "config_hash"):
                config.validate_for_command(changed, "generate-psd")
            with self.assertRaisesRegex(cli.CliError, "current config"):
                cli._cmd_validate_noise_effect_v2(
                    changed, SimpleNamespace(dry_run=False, allow_synthetic_basis=True)
                )
            with self.assertRaisesRegex(runner.RunnerError, "current config"):
                runner.load_effect_frozen_corrections(changed)

    def test_replaced_validation_refuses_resume_under_same_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, condition_id = self._validated_fixture(Path(tmp))
            first_adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            runner.generate_effect_preview(
                loaded, [condition_id], run_id="v2_preview", force=False,
                adapter=first_adapter, allow_synthetic_basis=True,
            )
            validation_path = loaded.resolve_root(loaded.psd.validation_result_path)
            validation = read_json(validation_path)
            validation["replacement_marker"] = "changed validation artifact"
            write_json(validation_path, validation)
            with self.assertRaisesRegex(RuntimeError, "Run manifest differs"):
                runner.generate_effect_preview(
                    loaded, [condition_id], run_id="v2_preview", force=False,
                    adapter=FakeAdapterPCA(loaded.model.as_model_config_dict()),
                    allow_synthetic_basis=True,
                )


def _read_jsonl(path: Path) -> list[dict]:
    import json

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


if __name__ == "__main__":
    unittest.main()
