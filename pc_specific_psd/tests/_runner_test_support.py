"""Shared fixtures for test_runner.py / test_runner_config_branch.py /
test_budget_accounting.py: a 100-dim identity basis (valid orthonormal
columns, patch_size=5 x channels=4 matches PC_GROUPS' fixed dimension), a
frozen "SELECTED" calibration registry, and a fake adapter that never touches
diffusers (mirrors test_adapters.py's own _FakePipeline/_TestAdapterPCA
pattern, duplicated rather than imported so a change to that file's fixture
cannot silently change runner test behavior).
"""
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from PIL import Image

from pc_specific_psd import adapters, basis, config
from pc_specific_psd.compat_generation import file_hash

PATCH_SIZE = 5
CHANNELS = 4
BASIS_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS  # 100, matches PC_GROUPS

RAW_CONFIG = {
    "run": {"name": "runner_test", "master_seed": 42, "outputs_root": "outputs"},
    "model": {"adapter": "sdxl_turbo", "checkpoint": "org/sdxl-turbo-fake", "dtype": "float32", "device": "cpu", "cpu_offload": False},
    # CPU fixture: image-space dimensions remain distinct from latent-space
    # dimensions (64 / VAE scale 8 = 8), while staying small enough for the
    # full suite. Production's 512 -> 64 contract has a dedicated regression.
    "generation": {"height": 64, "width": 64, "num_inference_steps": 1, "guidance_scale": 0.0, "generation_batch_size": 1},
    "basis": {
        "patch_size": PATCH_SIZE, "channels": CHANNELS, "patches_per_image": 8,
        "sampling_seed": 1, "split_seed": 2, "basis_output_path": "basis.pt",
    },
    "probing": {"rho": 0.10},
    "psd": {
        "protocol": "operator_clean",
        "num_bins": 8,
        "psd_tolerance": 1.0,
        "correction_gain_bound": 1.0e6,
        "calibration_bank_size": 8,
        "calibration_bank_seed": 123,
        "validation_bank_size": 8,
        "validation_bank_seed": 456,
        "gate_candidates": [{"r_s": 0.2, "beta": 6.0}],
        "groups": [
            {"group_id": "B5", "tau_plus_candidates": [1.0], "tau_minus_candidates": [-1.0], "target_rms": 1.0},
            {"group_id": "B6", "tau_plus_candidates": [0.5], "tau_minus_candidates": [-0.5], "target_rms": 1.0},
        ],
        "calibration_result_path": "calibration_result.json",
    },
}

V2_RAW_CONFIG = {
    **RAW_CONFIG,
    "run": {**RAW_CONFIG["run"], "name": "runner_v2_test"},
    "psd": {
        **RAW_CONFIG["psd"],
        "calibration_profile": "effect_size_v2",
        "minimum_covariance_distance": 0.0,
        "psd_tolerance": 1.0e6,
        "gate_candidates": [{"r_s": 0.2, "beta": 6.0}],
        "groups": [{
            "group_id": "B1", "tau_plus_candidates": [0.1],
            "tau_minus_candidates": [-0.1], "target_relative_l2": [0.05],
            "effect_target_tolerance": 10.0,
        }],
        "calibration_result_path": "effect_calibration_result.json",
        "validation_result_path": "effect_validation_result.json",
    },
}


def write_config(tmp_path: Path, *, groups=None) -> Path:
    raw = {**RAW_CONFIG, "psd": {**RAW_CONFIG["psd"]}}
    if groups is not None:
        raw["psd"]["groups"] = groups
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def write_synthetic_basis(path: Path) -> None:
    basis.save_basis(
        basis.PCABasis(
            components=torch.eye(BASIS_DIM, dtype=torch.float32),
            eigenvalues=torch.arange(BASIS_DIM, 0, -1, dtype=torch.float64),
            mean=torch.zeros(BASIS_DIM, dtype=torch.float64),
            patch_size=PATCH_SIZE, channels=CHANNELS, num_samples=1,
            metadata={"format_version": basis.BASIS_FORMAT_VERSION, "synthetic": True},
        ),
        path,
    )


def load_test_config(tmp_path: Path, *, groups=None) -> config.PCASpecificPSDConfig:
    path = write_config(tmp_path, groups=groups)
    loaded = config.load_config(path)
    write_synthetic_basis(loaded.resolve_root(loaded.basis.basis_output_path))
    return loaded


def load_v2_test_config(tmp_path: Path) -> config.PCASpecificPSDConfig:
    path = tmp_path / "config_v2.yaml"
    path.write_text(yaml.safe_dump(V2_RAW_CONFIG))
    loaded = config.load_config(path)
    write_synthetic_basis(loaded.resolve_root(loaded.basis.basis_output_path))
    return loaded


def freeze_selected_effect_calibration(loaded: config.PCASpecificPSDConfig) -> str:
    """Write one persisted v2 condition without fitting anything in the runner."""
    condition_id = "B1_plus_tau_0p1"
    correction = [1.0] * loaded.psd.num_bins
    config.write_effect_calibration_registry(loaded, {
        "status": "SELECTED",
        "protocol": "operator_clean",
        "gate": {"r_s": 0.2, "beta": 6.0},
        "bank": {
            "role": "calibration", "seed": loaded.psd.calibration_bank_seed,
            "size": loaded.psd.calibration_bank_size,
        },
        "basis_hash": file_hash(loaded.resolve_root(loaded.basis.basis_output_path)),
        "config_hash": file_hash(loaded.config_path),
        "all_targets_reached": True,
        "unreachable_target_count": 0,
        "selections": [], "candidate_diagnostics": [], "tau_zero_diagnostics": [],
        "condition_selections": {
            condition_id: {
                "condition_id": condition_id, "group_id": "B1", "sign": "plus",
                "tau": 0.1, "actual_relative_l2": 0.05,
                "target_relative_l2": [0.05], "correction": correction,
                "diagnostics": {},
            },
        },
    })
    return condition_id


def freeze_selected_calibration(loaded: config.PCASpecificPSDConfig, *, group_ids) -> None:
    config.write_calibration_registry(
        loaded, status="SELECTED",
        gate=config.GateCandidateConfig(r_s=0.2, beta=6.0), protocol="operator_clean",
        group_taus={
            group.group_id: (group.tau_plus_candidates[0], group.tau_minus_candidates[0])
            for group in loaded.psd.groups if group.group_id in group_ids
        },
    )


class _FakePipeline:
    def __init__(self):
        self.received_generators: list[torch.Generator | None] = []
        self.vae_scale_factor = 8
        self.last_call_kwargs = None

    def prepare_latents(self, batch_size, latents=None):
        self.received = latents
        return latents

    def __call__(self, *, latents, generator=None, **kwargs):
        self.prepare_latents(1, latents=latents)
        self.received_generators.append(generator)
        self.last_call_kwargs = kwargs
        return SimpleNamespace(images=[Image.new("RGB", (2, 2))])


class FakeAdapterPCA(adapters.SDXLTurboAdapterPCA):
    """Bypasses real diffusers loading entirely -- no network, no GPU. Counts
    calls to ``generate`` so tests can assert exactly which draws were
    (re)generated, e.g. to prove a resume or a preview->full reuse actually
    skipped work rather than merely producing the right file layout.
    """

    def __init__(self, model_config: dict):
        super().__init__(model_config)
        self.call_count = 0

    def _load(self) -> None:
        self.pipe = _FakePipeline()
        self._verify_and_record_revision(self.model_config.get("revision"))

    def generate(self, *args, **kwargs):
        self.call_count += 1
        return super().generate(*args, **kwargs)
