import argparse
import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from pc_specific_psd import cli, config, expected_rms, manifests, runner
from pc_specific_psd.compat_generation import file_hash, read_jsonl, write_json
from pc_specific_psd.tests import _runner_test_support as support


class ExpectedRMSRunnerTests(unittest.TestCase):
    def _config(self, root: Path):
        raw = copy.deepcopy(support.V2_RAW_CONFIG)
        raw["run"]["name"] = "expected_runner_test"
        raw["psd"]["calibration_profile"] = "expected_rms_v1"
        raw["psd"]["energy_constraint"] = "expected_rms"
        raw["psd"]["processing_batch_size"] = 3
        raw["psd"]["gate_candidates"] = [{"r_s": 4.0, "beta": 2.0}]
        raw["psd"]["groups"] = [{
            "group_id": "B1",
            "tau_plus_candidates": [0.1],
            "tau_minus_candidates": [-0.1],
            "target_relative_l2": [0.05],
            "effect_target_tolerance": 10.0,
        }]
        raw["psd"]["calibration_result_path"] = "expected_calibration.json"
        raw["psd"]["validation_result_path"] = "expected_validation.json"
        path = root / "expected_config.yaml"
        path.write_text(yaml.safe_dump(raw))
        loaded = config.load_config(path)
        support.write_synthetic_basis(loaded.resolve_root(loaded.basis.basis_output_path))
        return loaded

    def _freeze(self, loaded):
        codec = runner.load_codec(loaded, allow_synthetic=True)
        height = loaded.generation.config.height // 8
        width = loaded.generation.config.width // 8
        group = manifests.pc_group_by_id("B1")
        candidate = expected_rms.construct_candidate(
            codec, "B1", tuple(group.indices), 0.1, 4.0, 2.0,
            height=height, width=width, num_bins=loaded.psd.num_bins,
        )
        candidate = dataclasses.replace(
            candidate,
            operator=dataclasses.replace(
                candidate.operator,
                target_relative_l2=(0.05,), calibration_relative_l2=0.05,
            ),
        )
        reference_response = expected_rms.spectral.linear_operator_frequency_response(
            expected_rms.psd_editor.apply_psd_edit_tau_zero,
            loaded.basis.channels, height, width,
        )
        control = expected_rms.construct_fourier_control(candidate, reference_response)
        control = dataclasses.replace(
            control,
            operator=dataclasses.replace(control.operator, calibration_relative_l2=0.04),
        )
        operators = (candidate.operator, control.operator)
        registry_path = config.write_expected_rms_calibration_registry(loaded, {
            "status": "SELECTED",
            "protocol": "operator_clean",
            "basis_hash": runner.basis_hash_for(loaded),
            "config_hash": file_hash(loaded.config_path),
            "code_source": {
                "files": {
                    "expected_rms.py": file_hash(Path(expected_rms.__file__)),
                    "psd_editor.py": file_hash(Path(expected_rms.psd_editor.__file__)),
                    "spectral.py": file_hash(Path(expected_rms.spectral.__file__)),
                    "runner.py": file_hash(Path(runner.__file__)),
                },
            },
            "frozen_operators": {
                operator.condition_id: expected_rms.operator_to_payload(operator)
                for operator in operators
            },
            "valid_operator_pairs": [[operators[0].condition_id, operators[1].condition_id]],
        })
        validation_path = loaded.resolve_root(loaded.psd.validation_result_path)
        write_json(validation_path, {
            "status": "PASS",
            "calibration_profile": expected_rms.CALIBRATION_PROFILE,
            "energy_constraint": expected_rms.ENERGY_CONSTRAINT,
            "config_hash": file_hash(loaded.config_path),
            "basis_hash": runner.basis_hash_for(loaded),
            "calibration_hash": file_hash(registry_path),
            "bank": {
                "role": "validation", "seed": loaded.psd.validation_bank_seed,
                "size": loaded.psd.validation_bank_size,
            },
            "valid_operator_pairs": [[operators[0].condition_id, operators[1].condition_id]],
            "preview_condition_ids": [operator.condition_id for operator in operators],
            "condition_exclusions": [],
            "conditions": {
                operator.condition_id: {
                    "numerically_valid": True,
                    "actual_relative_l2": 0.05 if operator.operator_type == "pca_candidate" else 0.04,
                }
                for operator in operators
            },
        })
        return codec, operators, registry_path, validation_path

    def test_registry_render_and_preview_use_frozen_operator_without_refit(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = self._config(Path(tmp))
            codec, operators, _, _ = self._freeze(loaded)
            loaded_operators = runner.load_expected_rms_operators(loaded)
            self.assertEqual(set(loaded_operators), {operator.condition_id for operator in operators})
            entry = manifests.ConditionDrawEntry(
                operators[0].condition_id,
                manifests.PROMPTS[0].prompt_id,
                manifests.seed_blocks_for_prompt(manifests.PROMPTS[0])[0].block_id,
                manifests.PROBING_BASE_INDEX,
            )
            with mock.patch("pc_specific_psd.expected_rms.construct_candidate", side_effect=AssertionError("refit")):
                actual = runner.render_expected_rms_condition_latent(
                    entry, codec, loaded_operators,
                    channels=loaded.basis.channels,
                    height=loaded.generation.config.height // 8,
                    width=loaded.generation.config.width // 8,
                )
            block = runner._seed_block_for_entry(entry)
            base = expected_rms.psd_editor.base_white_for_draw(
                block, entry.base_index, channels=loaded.basis.channels,
                height=loaded.generation.config.height // 8,
                width=loaded.generation.config.width // 8,
            )
            direct = expected_rms.apply_frozen_operator(codec, base, operators[0])
            self.assertTrue(actual.equal(direct))

    def test_registry_loader_rejects_missing_duplicate_and_wrong_control_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = self._config(Path(tmp))
            _, operators, registry_path, _ = self._freeze(loaded)
            original = json.loads(registry_path.read_text())
            candidate, control = operators

            missing = copy.deepcopy(original)
            del missing["frozen_operators"][control.condition_id]
            config.write_expected_rms_calibration_registry(loaded, missing)
            with self.assertRaisesRegex(
                runner.RunnerError, "damaged.*exactly one Fourier control"
            ):
                runner.load_expected_rms_operators(loaded)

            duplicate = copy.deepcopy(original)
            duplicate["frozen_operators"]["duplicate_storage_key"] = (
                expected_rms.operator_to_payload(candidate)
            )
            config.write_expected_rms_calibration_registry(loaded, duplicate)
            with self.assertRaisesRegex(
                runner.RunnerError, "condition key does not match frozen operator ID"
            ):
                runner.load_expected_rms_operators(loaded)

            wrong_link = dataclasses.replace(control, candidate_id="missing_candidate")
            malformed = copy.deepcopy(original)
            malformed["frozen_operators"][control.condition_id] = (
                expected_rms.operator_to_payload(wrong_link)
            )
            config.write_expected_rms_calibration_registry(loaded, malformed)
            with self.assertRaisesRegex(
                runner.RunnerError, "damaged.*refers to missing candidate"
            ):
                runner.load_expected_rms_operators(loaded)

            inconsistent = copy.deepcopy(original)
            inconsistent["valid_operator_pairs"] = []
            config.write_expected_rms_calibration_registry(loaded, inconsistent)
            with self.assertRaisesRegex(
                runner.RunnerError, "damaged.*valid_operator_pairs"
            ):
                runner.load_expected_rms_operators(loaded)

    def test_preview_records_hashes_pairing_and_prepare_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = self._config(Path(tmp))
            _, operators, registry_path, validation_path = self._freeze(loaded)
            adapter = support.FakeAdapterPCA(loaded.model.as_model_config_dict())
            condition_ids = [operator.condition_id for operator in operators]
            run_dir = runner.generate_expected_rms_preview(
                loaded, condition_ids, run_id="expected_preview", adapter=adapter,
                allow_synthetic_basis=True,
            )
            records = [read_jsonl(path)[0] for path in sorted(run_dir.rglob("sample.jsonl"))]
            self.assertEqual(len(records), 36)
            for row in records:
                self.assertEqual(row["basis_hash"], runner.basis_hash_for(loaded))
                self.assertEqual(row["calibration_hash"], file_hash(registry_path))
                self.assertEqual(row["validation_hash"], file_hash(validation_path))
                self.assertIn("adapter_prepare_latents_return_hash", row)
            candidate = next(row for row in records if row["condition_id"] == operators[0].condition_id)
            control = next(
                row for row in records
                if row["condition_id"] == operators[1].condition_id
                and row["prompt_id"] == candidate["prompt_id"]
                and row["block_id"] == candidate["block_id"]
            )
            self.assertEqual(candidate["generator_seed"], control["generator_seed"])
            self.assertNotEqual(candidate["final_noise_hash"], control["final_noise_hash"])
            self.assertEqual(control["matched_candidate_id"], operators[0].condition_id)

    def test_changed_config_rejects_registry_and_changed_validation_rejects_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loaded = self._config(root)
            _, operators, _, validation_path = self._freeze(loaded)
            adapter = support.FakeAdapterPCA(loaded.model.as_model_config_dict())
            condition_ids = [operator.condition_id for operator in operators]
            runner.generate_expected_rms_preview(
                loaded, condition_ids, run_id="immutable_preview", adapter=adapter,
                allow_synthetic_basis=True,
            )
            validation = json.loads(validation_path.read_text())
            validation["replacement_marker"] = True
            validation_path.write_text(json.dumps(validation, sort_keys=True))
            with self.assertRaisesRegex(Exception, "provenance|immutable|mismatch|exists"):
                runner.generate_expected_rms_preview(
                    loaded, condition_ids, run_id="immutable_preview",
                    adapter=support.FakeAdapterPCA(loaded.model.as_model_config_dict()),
                    allow_synthetic_basis=True,
                )

            raw = yaml.safe_load(loaded.config_path.read_text())
            raw["psd"]["gate_candidates"][0]["r_s"] = 3.0
            loaded.config_path.write_text(yaml.safe_dump(raw))
            with self.assertRaisesRegex(runner.RunnerError, "stale relative to the current config"):
                runner.load_expected_rms_operators(loaded)

    def test_one_validation_control_failure_keeps_other_pair_and_preview_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = self._config(Path(tmp))
            codec, positive_operators, registry_path, _ = self._freeze(loaded)
            height = loaded.generation.config.height // 8
            width = loaded.generation.config.width // 8
            group = manifests.pc_group_by_id("B1")
            negative = expected_rms.construct_candidate(
                codec, "B1", tuple(group.indices), -0.1, 4.0, 2.0,
                height=height, width=width, num_bins=loaded.psd.num_bins,
            )
            negative = dataclasses.replace(
                negative,
                operator=dataclasses.replace(
                    negative.operator,
                    target_relative_l2=(0.05,),
                    calibration_relative_l2=0.05,
                ),
            )
            reference_response = expected_rms.spectral.linear_operator_frequency_response(
                expected_rms.psd_editor.apply_psd_edit_tau_zero,
                loaded.basis.channels, height, width,
            )
            negative_control = expected_rms.construct_fourier_control(
                negative, reference_response
            )
            negative_control = dataclasses.replace(
                negative_control,
                operator=dataclasses.replace(
                    negative_control.operator,
                    calibration_relative_l2=0.04,
                ),
            )
            registry = json.loads(registry_path.read_text())
            for operator in (negative.operator, negative_control.operator):
                registry["frozen_operators"][operator.condition_id] = (
                    expected_rms.operator_to_payload(operator)
                )
            registry["valid_operator_pairs"].append([
                negative.operator.condition_id,
                negative_control.operator.condition_id,
            ])
            registry["valid_operator_pairs"].sort()
            config.write_expected_rms_calibration_registry(loaded, registry)

            original = expected_rms.diagnose_operator

            def reject_positive_control(*args, **kwargs):
                diagnostic = original(*args, **kwargs)
                construction = args[2]
                operator = construction.operator
                if operator.operator_type == "fourier_control" and operator.tau > 0:
                    return dataclasses.replace(
                        diagnostic,
                        status="rejected",
                        failure_reason="synthetic_validation_control_failure",
                    )
                return diagnostic

            args = argparse.Namespace(
                dry_run=False, allow_synthetic_basis=True, candidates_file=None,
            )
            with mock.patch(
                "pc_specific_psd.expected_rms.diagnose_operator",
                side_effect=reject_positive_control,
            ):
                validation = cli._cmd_validate_noise(loaded, args)
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(
                validation["valid_operator_pairs"],
                [[negative.operator.condition_id, negative_control.operator.condition_id]],
            )
            excluded_ids = {
                item["condition_id"] for item in validation["condition_exclusions"]
            }
            self.assertIn(positive_operators[0].condition_id, excluded_ids)
            self.assertIn(positive_operators[1].condition_id, excluded_ids)

            plan = cli._cmd_generate_psd(
                loaded,
                argparse.Namespace(
                    stage="preview", dry_run=True, run_id=None, force=False,
                    allow_synthetic_basis=True, candidates_file=None,
                    approved_conditions_file=None,
                ),
            )
            self.assertEqual(plan["condition_ids"], validation["preview_condition_ids"])
            self.assertEqual(plan["image_count"], 36)

    def test_v2_and_expected_registries_stamp_shared_diagnostic_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            effect_root = root / "effect"
            effect_root.mkdir()
            effect_config = support.load_v2_test_config(effect_root)
            effect_path = config.write_effect_calibration_registry(
                effect_config, {"status": "FAIL"}
            )
            expected_root = root / "expected"
            expected_root.mkdir()
            expected_config = self._config(expected_root)
            _, _, expected_path, _ = self._freeze(expected_config)
            expected_versions = expected_rms.spectral.diagnostic_definition_versions()
            self.assertEqual(
                json.loads(effect_path.read_text())["diagnostic_versions"],
                expected_versions,
            )
            self.assertEqual(
                json.loads(expected_path.read_text())["diagnostic_versions"],
                expected_versions,
            )

    def test_cli_calibration_validation_chain_freezes_scale_and_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = self._config(Path(tmp))
            args = argparse.Namespace(
                dry_run=False, allow_synthetic_basis=True, candidates_file=None,
            )
            calibration_result = cli._cmd_calibrate(loaded, args)
            self.assertEqual(calibration_result["status"], "SELECTED")
            registry = json.loads(loaded.resolve_root(loaded.psd.calibration_result_path).read_text())
            operators = registry["frozen_operators"]
            self.assertEqual(
                {raw["operator_type"] for raw in operators.values()},
                {"pca_candidate", "fourier_control"},
            )
            self.assertTrue(all("correction" not in raw for raw in operators.values()))
            validation = cli._cmd_validate_noise(loaded, args)
            self.assertEqual(validation["status"], "PASS")
            self.assertTrue(validation["preview_condition_ids"])
            self.assertEqual(validation["config_hash"], registry["config_hash"])
            self.assertEqual(validation["basis_hash"], registry["basis_hash"])

if __name__ == "__main__":    unittest.main()
