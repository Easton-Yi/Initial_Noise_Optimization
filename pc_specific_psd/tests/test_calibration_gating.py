"""Per-command config validation (plan's config.py section): probe/export-review/
select-candidates must succeed while every needs_calibration PSD field is
still unset, calibrate/validate-noise require the finite candidate sets to be
*declared* but not yet *selected*, and generate-psd/export-gallery-review/
ingest-gallery-review/metrics/analyze require the full needs_calibration
registry to hold a selected candidate for every declared group.
"""
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from pc_specific_psd import config

BASE_RAW = {
    "run": {"name": "test_run", "master_seed": 1, "outputs_root": "outputs"},
    "model": {"adapter": "sdxl_turbo", "checkpoint": "org/sdxl-turbo-fake", "dtype": "float32", "device": "cpu", "cpu_offload": False},
    "generation": {"height": 16, "width": 16, "num_inference_steps": 1, "guidance_scale": 0.0, "generation_batch_size": 1},
    "basis": {"patch_size": 5, "channels": 4, "patches_per_image": 8, "sampling_seed": 1, "split_seed": 2, "basis_output_path": "basis.pt"},
}

DECLARED_PSD = {
    "protocol": "operator_clean",
    "num_bins": 8,
    "psd_tolerance": 0.05,
    "correction_gain_bound": 10.0,
    "calibration_bank_size": 16,
    "validation_bank_size": 16,
    "calibration_bank_seed": 3,
    "validation_bank_seed": 4,
    "gate_candidates": [{"r_s": 0.2, "beta": 6.0}],
    "groups": [{"group_id": "B5", "tau_plus_candidates": [1.0], "tau_minus_candidates": [-1.0], "target_rms": 1.0}],
    "calibration_result_path": "calibration_result.json",
}


def _write_config(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _write_calibration_result(
    tmp_path: Path,
    relative: str,
    *,
    status: str,
    group_ids: list[str],
    include_reference_scale_profile: bool = True,
) -> None:
    payload = {
        "status": status,
        "gate": {"r_s": 0.2, "beta": 6.0},
        "protocol": "operator_clean",
        "group_selections": {group_id: {"tau_plus": 1.0, "tau_minus": -1.0} for group_id in group_ids},
    }
    if include_reference_scale_profile:
        payload["reference_scale_profile"] = "expected_unit_rms_rfft_v1"
    (tmp_path / relative).write_text(json.dumps(payload))


class BasisTierCommandsTests(unittest.TestCase):
    def test_validate_config_build_basis_inspect_basis_succeed_with_no_probing_or_psd_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), BASE_RAW)
            for command in ("validate-config", "build-basis", "inspect-basis"):
                config.resolve_config(path, command)  # must not raise


class ProbingTierCommandsTests(unittest.TestCase):
    def test_probe_export_review_select_candidates_refuse_without_rho(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), BASE_RAW)
            for command in ("probe", "export-review", "select-candidates"):
                with self.assertRaises(config.ConfigValidationError) as ctx:
                    config.resolve_config(path, command)
                self.assertIn("probing.rho", ctx.exception.missing_fields)

    def test_probe_succeeds_once_rho_is_set_despite_needs_calibration_fields_being_unset(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            for command in ("probe", "export-review", "select-candidates"):
                loaded = config.resolve_config(path, command)  # must not raise
                self.assertIsNone(loaded.psd.protocol)


class CalibrationTierCommandsTests(unittest.TestCase):
    def test_calibrate_and_validate_noise_require_declared_candidate_sets_but_not_a_selection(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": DECLARED_PSD}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            for command in ("calibrate", "validate-noise"):
                config.resolve_config(path, command)  # must not raise even though nothing is selected yet

    def test_calibrate_refuses_when_candidate_sets_are_not_yet_declared(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaises(config.ConfigValidationError) as ctx:
                config.resolve_config(path, "calibrate")
            self.assertIn("psd.protocol", ctx.exception.missing_fields)
            self.assertIn("psd.gate_candidates", ctx.exception.missing_fields)
            self.assertIn("psd.groups", ctx.exception.missing_fields)

    def test_calibrate_refuses_when_a_declared_group_has_no_tau_candidates(self):
        psd = {**DECLARED_PSD, "groups": [{"group_id": "B5", "tau_plus_candidates": [], "tau_minus_candidates": [-1.0], "target_rms": 1.0}]}
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": psd}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            with self.assertRaises(config.ConfigValidationError) as ctx:
                config.resolve_config(path, "calibrate")
            self.assertIn("psd.groups[B5].tau_plus_candidates", ctx.exception.missing_fields)


class FullTierCommandsTests(unittest.TestCase):
    FULL_COMMANDS = ("generate-psd", "export-preview-review", "ingest-preview-review",
                      "export-gallery-review", "ingest-gallery-review", "metrics", "analyze")

    def test_full_tier_refuses_when_calibration_result_file_does_not_exist(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": DECLARED_PSD}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_config(Path(tmp), raw)
            for command in self.FULL_COMMANDS:
                with self.assertRaises(config.ConfigValidationError) as ctx:
                    config.resolve_config(path, command)
                self.assertTrue(any("calibration_result" in field for field in ctx.exception.missing_fields))

    def test_full_tier_refuses_when_calibration_rejected_all_gates(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": DECLARED_PSD}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = _write_config(tmp_path, raw)
            _write_calibration_result(tmp_path, DECLARED_PSD["calibration_result_path"], status="REJECTED_ALL_GATES", group_ids=[])
            with self.assertRaises(config.ConfigValidationError) as ctx:
                config.resolve_config(path, "generate-psd")
            self.assertTrue(any("REJECTED_ALL_GATES" in field for field in ctx.exception.missing_fields))

    def test_full_tier_refuses_when_a_declared_group_is_missing_from_the_selection(self):
        psd = {**DECLARED_PSD, "groups": [
            *DECLARED_PSD["groups"],
            {"group_id": "B6", "tau_plus_candidates": [1.0], "tau_minus_candidates": [-1.0], "target_rms": 1.0},
        ]}
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": psd}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = _write_config(tmp_path, raw)
            _write_calibration_result(tmp_path, psd["calibration_result_path"], status="SELECTED", group_ids=["B5"])
            with self.assertRaises(config.ConfigValidationError) as ctx:
                config.resolve_config(path, "generate-psd")
            self.assertIn("calibration_result.group_selections['B6']", ctx.exception.missing_fields)

    def test_full_tier_succeeds_once_every_declared_group_is_selected(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": DECLARED_PSD}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = _write_config(tmp_path, raw)
            _write_calibration_result(tmp_path, DECLARED_PSD["calibration_result_path"], status="SELECTED", group_ids=["B5"])
            for command in self.FULL_COMMANDS:
                config.resolve_config(path, command)  # must not raise

    def test_full_tier_recognizes_legacy_registry_as_stale_and_requires_recalibration(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": DECLARED_PSD}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = _write_config(tmp_path, raw)
            _write_calibration_result(
                tmp_path,
                DECLARED_PSD["calibration_result_path"],
                status="SELECTED",
                group_ids=["B5"],
                include_reference_scale_profile=False,
            )
            legacy = config.load_calibration_registry(config.load_config(path))
            self.assertTrue(legacy.loaded)
            self.assertIsNone(legacy.reference_scale_profile)
            with self.assertRaises(config.ConfigValidationError) as ctx:
                config.resolve_config(path, "generate-psd")
            message = " ".join(ctx.exception.missing_fields)
            self.assertIn("reference_scale_profile", message)
            self.assertIn("calibrate", message)


class CalibrationRegistryRoundTripTests(unittest.TestCase):
    def test_write_calibration_registry_round_trips_through_load(self):
        raw = {**BASE_RAW, "probing": {"rho": 0.10}, "psd": DECLARED_PSD}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path = _write_config(tmp_path, raw)
            loaded = config.load_config(path)
            written_path = config.write_calibration_registry(
                loaded, status="SELECTED",
                gate=config.GateCandidateConfig(r_s=0.2, beta=6.0), protocol="operator_clean",
                group_taus={"B5": (1.0, -1.0)},
            )
            self.assertTrue(written_path.exists())
            registry = config.load_calibration_registry(loaded)
            self.assertTrue(registry.loaded)
            self.assertEqual(registry.status, "SELECTED")
            self.assertEqual(registry.gate, config.GateCandidateConfig(r_s=0.2, beta=6.0))
            self.assertEqual(registry.reference_scale_profile, "expected_unit_rms_rfft_v1")
            self.assertEqual(registry.group_taus, {"B5": (1.0, -1.0)})


if __name__ == "__main__":
    unittest.main()
