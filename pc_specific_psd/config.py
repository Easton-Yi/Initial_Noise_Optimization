"""Per-command-validated configuration for the PCA-specific PSD study.

Deliberately a dataclass tree, not the flat dict ``noise_init.run_experiment``
loads via ``yaml.safe_load`` -- this package's commands each need a different
subset of the schema *resolved* before they may run (plan section on
``config.py``): ``probe`` must succeed while every PSD/gate field is still
undeclared, while ``generate-psd`` must refuse and name exactly which
``needs_calibration`` fields are missing. A flat dict cannot express "this
field is required for command X but not command Y" without ad hoc code at
every call site, so the schema is typed and ``validate_for_command`` centralizes
the per-command gating instead.

Six PC groups (``manifests.PC_GROUPS``) are fixed pipeline constants, not a
config field -- this module only validates the basis dimension is consistent
with them, never accepts an override. ``psd_editor.SAME_PHASE_ALPHA``/
``SAME_PHASE_GAMMA`` are likewise frozen constants: any YAML attempt to set
them under ``psd:`` is rejected, and ``FrozenConstants`` echoes their real
values into every resolved config for provenance instead.

The ``needs_calibration`` PSD/gate registry is two layers, matching the plan's
strict calibration-bank-tunes / validation-bank-accepts-once split:
``PSDConfig`` holds the finite, pre-declared candidate sets (authored in the
YAML, before any generation), while the *selected* gate/tau values -- written
only once ``calibrate`` runs ``calibration.select_gate_and_taus`` and
``validate_on_bank`` -- live in a separate small JSON file at
``psd.calibration_result_path``, loaded on demand by ``load_calibration_registry``.
Keeping the selection out of the YAML means re-running ``calibrate`` (a new
calibration version) never requires hand-editing the declared candidate sets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml

from pc_specific_psd import manifests, probing, psd_editor
from pc_specific_psd.compat_generation import GenerationConfig

Protocol = Literal["legacy_matched", "operator_clean"]

_FROZEN_PSD_KEYS = ("same_phase_alpha", "same_phase_gamma")

_GENERATION_REQUIRED_FIELDS = ("height", "width", "num_inference_steps", "guidance_scale", "generation_batch_size")
_GENERATION_ALLOWED_FIELDS = frozenset(_GENERATION_REQUIRED_FIELDS) | {"output_format"}

_MODEL_REQUIRED_FIELDS = ("adapter", "checkpoint", "dtype", "device", "cpu_offload")
_MODEL_OPTIONAL_FIELDS = ("vae_checkpoint", "revision", "cache_dir")

# Plan §2.3/§10/§11: "FLUX如未完成codec确认，应在CLI明确标 not_verified，拒绝正式PCA实验;
# 不能复制SDXL basis/PC编号或猜C值" -- the basis, six PC groups, and both patch codecs in
# this package are fit and dimensioned (patch_size=5, channels=4 => d=100) against SDXL-Turbo's
# VAE only. There is no FLUX basis, no FLUX PC group partition, and no validated patch geometry
# for FLUX's latent space, so any adapter other than the one verified here is refused at config
# load time -- for every command, not just generation ones -- rather than silently running with
# a mismatched basis under a FLUX-labeled config.
_SUPPORTED_ADAPTERS = ("sdxl_turbo",)

_BASIS_REQUIRED_FIELDS = ("patch_size", "channels", "patches_per_image", "sampling_seed", "split_seed", "basis_output_path")
_BASIS_OPTIONAL_FIELDS = ("num_leading_components", "dataset_manifest_path")

_CALIBRATION_TIER_SCALAR_FIELDS = (
    "protocol", "num_bins", "psd_tolerance", "correction_gain_bound",
    "calibration_bank_size", "validation_bank_size", "calibration_bank_seed", "validation_bank_seed",
)


class ConfigError(ValueError):
    """Structural YAML/schema error -- a malformed or self-contradictory config."""


class ConfigValidationError(ValueError):
    """Raised by ``validate_for_command`` when fields the command needs are unresolved."""

    def __init__(self, command: str, missing_fields: Sequence[str]):
        self.command = command
        self.missing_fields = tuple(missing_fields)
        super().__init__(f"config unresolved for command {command!r}: {', '.join(self.missing_fields)}")


def _required(raw: dict, section: str, key: str) -> Any:
    try:
        return raw[key]
    except KeyError as exc:
        raise ConfigError(f"config.{section}.{key} is required") from exc


@dataclass(frozen=True)
class RunConfig:
    name: str
    master_seed: int
    outputs_root: str


@dataclass(frozen=True)
class ModelConfig:
    adapter: str
    checkpoint: str
    dtype: str
    device: str
    cpu_offload: bool
    vae_checkpoint: str | None = None
    revision: str | None = None
    cache_dir: str | None = None

    def as_model_config_dict(self) -> dict[str, Any]:
        """The plain-dict shape ``adapters.SDXLTurboAdapterPCA.__init__`` expects."""
        return {
            "adapter": self.adapter,
            "checkpoint": self.checkpoint,
            "dtype": self.dtype,
            "device": self.device,
            "cpu_offload": self.cpu_offload,
            "vae_checkpoint": self.vae_checkpoint,
            "revision": self.revision,
            "cache_dir": self.cache_dir,
        }


@dataclass(frozen=True)
class GenerationSection:
    config: GenerationConfig
    release_model_after_generation: bool = True


@dataclass(frozen=True)
class BasisConfig:
    patch_size: int
    channels: int
    patches_per_image: int
    sampling_seed: int
    split_seed: int
    basis_output_path: str
    num_leading_components: int | None = None
    dataset_manifest_path: str | None = None  # None until a real dataset path is supplied; never auto-downloaded


@dataclass(frozen=True)
class ProbingConfig:
    rho: float | None = None


@dataclass(frozen=True)
class GateCandidateConfig:
    r_s: float
    beta: float


@dataclass(frozen=True)
class GroupCandidateConfig:
    group_id: str
    tau_plus_candidates: tuple[float, ...]
    tau_minus_candidates: tuple[float, ...]
    target_rms: float


@dataclass(frozen=True)
class PSDConfig:
    protocol: Protocol | None = None
    num_bins: int | None = None
    psd_tolerance: float | None = None
    correction_gain_bound: float | None = None
    condition_number_threshold: float | None = None
    calibration_bank_size: int | None = None
    validation_bank_size: int | None = None
    calibration_bank_seed: int | None = None
    validation_bank_seed: int | None = None
    gate_candidates: tuple[GateCandidateConfig, ...] = ()
    groups: tuple[GroupCandidateConfig, ...] = ()
    calibration_result_path: str | None = None


@dataclass(frozen=True)
class FrozenConstants:
    same_phase_alpha: float = psd_editor.SAME_PHASE_ALPHA
    same_phase_gamma: float = psd_editor.SAME_PHASE_GAMMA


@dataclass(frozen=True)
class PCASpecificPSDConfig:
    config_path: Path
    run: RunConfig
    model: ModelConfig
    generation: GenerationSection
    basis: BasisConfig
    probing: ProbingConfig
    psd: PSDConfig
    frozen: FrozenConstants = field(default_factory=FrozenConstants)

    def resolve_root(self, relative: str | None) -> Path | None:
        """Resolves ``relative`` against this config file's own directory --
        never ``Path.cwd()`` -- so the CLI is invocable from anywhere.
        """
        if relative is None:
            return None
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.config_path.parent / candidate).resolve()

    def provenance(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "same_phase_alpha": self.frozen.same_phase_alpha,
            "same_phase_gamma": self.frozen.same_phase_gamma,
        }


def _build_model_config(raw_model: dict) -> ModelConfig:
    kwargs: dict[str, Any] = {key: _required(raw_model, "model", key) for key in _MODEL_REQUIRED_FIELDS}
    kwargs.update({key: raw_model.get(key) for key in _MODEL_OPTIONAL_FIELDS})
    unknown = set(raw_model) - set(_MODEL_REQUIRED_FIELDS) - set(_MODEL_OPTIONAL_FIELDS)
    if unknown:
        raise ConfigError(f"config.model has unknown field(s): {sorted(unknown)}")
    if kwargs["adapter"] not in _SUPPORTED_ADAPTERS:
        raise ConfigError(
            f"config.model.adapter={kwargs['adapter']!r} is not_verified for PCA-specific PSD "
            f"editing: this package's basis, PC group partition, and patch codecs are fit and "
            f"validated against SDXL-Turbo's VAE only (patch_size=5, channels=4) and are not "
            f"transferable to any other adapter, including FLUX, without a new basis, new PC "
            f"groups, and full revalidation -- refusing to run a formal PCA experiment; only "
            f"{_SUPPORTED_ADAPTERS!r} is supported"
        )
    return ModelConfig(**kwargs)


def _build_generation_section(raw_generation: dict) -> GenerationSection:
    working = dict(raw_generation)
    release_model_after_generation = working.pop("release_model_after_generation", True)
    missing = [key for key in _GENERATION_REQUIRED_FIELDS if key not in working]
    if missing:
        raise ConfigError(f"config.generation missing required field(s): {', '.join(missing)}")
    unknown = set(working) - _GENERATION_ALLOWED_FIELDS
    if unknown:
        raise ConfigError(f"config.generation has unknown field(s): {sorted(unknown)}")
    return GenerationSection(
        config=GenerationConfig(**working), release_model_after_generation=bool(release_model_after_generation)
    )


def _build_basis_config(raw_basis: dict) -> BasisConfig:
    kwargs: dict[str, Any] = {key: _required(raw_basis, "basis", key) for key in _BASIS_REQUIRED_FIELDS}
    kwargs.update({key: raw_basis.get(key) for key in _BASIS_OPTIONAL_FIELDS})
    unknown = set(raw_basis) - set(_BASIS_REQUIRED_FIELDS) - set(_BASIS_OPTIONAL_FIELDS)
    if unknown:
        raise ConfigError(f"config.basis has unknown field(s): {sorted(unknown)}")
    return BasisConfig(**kwargs)


def _build_probing_config(raw_probing: dict) -> ProbingConfig:
    unknown = set(raw_probing) - {"rho"}
    if unknown:
        raise ConfigError(f"config.probing has unknown field(s): {sorted(unknown)}")
    return ProbingConfig(rho=raw_probing.get("rho"))


def _build_psd_config(raw_psd: dict) -> PSDConfig:
    for key in _FROZEN_PSD_KEYS:
        if key in raw_psd:
            raise ConfigError(
                f"config.psd.{key} is a frozen constant (psd_editor.SAME_PHASE_ALPHA/"
                f"SAME_PHASE_GAMMA) and cannot be set from config"
            )
    gate_candidates = tuple(GateCandidateConfig(r_s=g["r_s"], beta=g["beta"]) for g in raw_psd.get("gate_candidates", []))
    groups = tuple(
        GroupCandidateConfig(
            group_id=g["group_id"],
            tau_plus_candidates=tuple(g["tau_plus_candidates"]),
            tau_minus_candidates=tuple(g["tau_minus_candidates"]),
            target_rms=g["target_rms"],
        )
        for g in raw_psd.get("groups", [])
    )
    known_group_ids = {g.group_id for g in manifests.PC_GROUPS}
    unknown_group_ids = {g.group_id for g in groups} - known_group_ids
    if unknown_group_ids:
        raise ConfigError(f"config.psd.groups declares unknown group_id(s): {sorted(unknown_group_ids)}")

    allowed = set(_CALIBRATION_TIER_SCALAR_FIELDS) | {
        "condition_number_threshold", "gate_candidates", "groups", "calibration_result_path",
    }
    unknown = set(raw_psd) - allowed
    if unknown:
        raise ConfigError(f"config.psd has unknown field(s): {sorted(unknown)}")

    return PSDConfig(
        protocol=raw_psd.get("protocol"),
        num_bins=raw_psd.get("num_bins"),
        psd_tolerance=raw_psd.get("psd_tolerance"),
        correction_gain_bound=raw_psd.get("correction_gain_bound"),
        condition_number_threshold=raw_psd.get("condition_number_threshold"),
        calibration_bank_size=raw_psd.get("calibration_bank_size"),
        validation_bank_size=raw_psd.get("validation_bank_size"),
        calibration_bank_seed=raw_psd.get("calibration_bank_seed"),
        validation_bank_seed=raw_psd.get("validation_bank_seed"),
        gate_candidates=gate_candidates,
        groups=groups,
        calibration_result_path=raw_psd.get("calibration_result_path"),
    )


def load_config(path: str | Path) -> PCASpecificPSDConfig:
    """Parses ``path`` into a ``PCASpecificPSDConfig``. Structural errors (a
    missing required key, an unknown key, a frozen-constant override attempt,
    an unknown PC group id) always raise ``ConfigError`` here, regardless of
    which command will run -- only the *needs_calibration* fields are
    legitimately allowed to be absent at parse time; see ``validate_for_command``.
    """
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top-level YAML must be a mapping")

    run = RunConfig(**{key: _required(raw.get("run", {}), "run", key) for key in ("name", "master_seed", "outputs_root")})
    model = _build_model_config(raw.get("model", {}))
    generation = _build_generation_section(raw.get("generation", {}))
    basis = _build_basis_config(raw.get("basis", {}))
    probing_config = _build_probing_config(raw.get("probing", {}) or {})
    psd = _build_psd_config(raw.get("psd", {}) or {})

    basis_dimension = basis.channels * basis.patch_size * basis.patch_size
    try:
        manifests.validate_pc_groups(basis_dimension=basis_dimension)
    except ValueError as exc:
        raise ConfigError(f"config.basis patch_size/channels inconsistent with manifests.PC_GROUPS: {exc}") from exc

    return PCASpecificPSDConfig(
        config_path=config_path, run=run, model=model, generation=generation,
        basis=basis, probing=probing_config, psd=psd,
    )


@dataclass(frozen=True)
class CalibrationRegistry:
    """Whether ``calibrate`` has produced a frozen, ``SELECTED`` calibration
    result covering every group declared in ``config.psd.groups``, loaded from
    ``psd.calibration_result_path`` if that file exists.
    """
    loaded: bool
    status: str | None
    gate: GateCandidateConfig | None
    protocol: Protocol | None
    group_taus: dict[str, tuple[float, float]]  # group_id -> (tau_plus, tau_minus)


_EMPTY_REGISTRY = CalibrationRegistry(loaded=False, status=None, gate=None, protocol=None, group_taus={})


def load_calibration_registry(config: PCASpecificPSDConfig) -> CalibrationRegistry:
    path = config.resolve_root(config.psd.calibration_result_path)
    if path is None or not path.exists():
        return _EMPTY_REGISTRY
    payload = json.loads(path.read_text())
    gate_payload = payload.get("gate")
    gate = GateCandidateConfig(r_s=gate_payload["r_s"], beta=gate_payload["beta"]) if gate_payload else None
    group_taus = {
        group_id: (float(selection["tau_plus"]), float(selection["tau_minus"]))
        for group_id, selection in payload.get("group_selections", {}).items()
    }
    return CalibrationRegistry(
        loaded=True, status=payload.get("status"), gate=gate, protocol=payload.get("protocol"), group_taus=group_taus
    )


def write_calibration_registry(
    config: PCASpecificPSDConfig,
    *,
    status: str,
    gate: GateCandidateConfig | None,
    protocol: Protocol | None,
    group_taus: dict[str, tuple[float, float]],
) -> Path:
    """Persists the frozen selection ``calibrate`` produced. Never called by
    ``load_config``/``validate_for_command`` themselves -- only the ``calibrate``
    command writes this file.
    """
    path = config.resolve_root(config.psd.calibration_result_path)
    if path is None:
        raise ConfigError("config.psd.calibration_result_path must be set before calibrate can write a result")
    payload = {
        "status": status,
        "gate": {"r_s": gate.r_s, "beta": gate.beta} if gate is not None else None,
        "protocol": protocol,
        "group_selections": {
            group_id: {"tau_plus": tau_plus, "tau_minus": tau_minus}
            for group_id, (tau_plus, tau_minus) in group_taus.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


# -- Per-command validation ---------------------------------------------------

COMMAND_TIERS: dict[str, str] = {
    "validate-config": "basis",
    "build-basis": "basis",
    "inspect-basis": "basis",
    "probe": "probing",
    "export-review": "probing",
    "select-candidates": "probing",
    "calibrate": "calibration",
    "validate-noise": "calibration",
    "generate-psd": "full",
    "export-preview-review": "full",
    "ingest-preview-review": "full",
    "export-gallery-review": "full",
    "ingest-gallery-review": "full",
    "metrics": "full",
    "analyze": "full",
    "workflow": "basis",
}

ALL_COMMANDS = tuple(COMMAND_TIERS)


def _probing_tier_missing(config: PCASpecificPSDConfig) -> list[str]:
    return ["probing.rho"] if config.probing.rho is None else []


def _calibration_tier_missing(config: PCASpecificPSDConfig) -> list[str]:
    psd = config.psd
    missing = [f"psd.{name}" for name in _CALIBRATION_TIER_SCALAR_FIELDS if getattr(psd, name) is None]
    if not psd.gate_candidates:
        missing.append("psd.gate_candidates")
    if not psd.groups:
        missing.append("psd.groups")
    for group in psd.groups:
        if not group.tau_plus_candidates:
            missing.append(f"psd.groups[{group.group_id}].tau_plus_candidates")
        if not group.tau_minus_candidates:
            missing.append(f"psd.groups[{group.group_id}].tau_minus_candidates")
    if psd.calibration_result_path is None:
        missing.append("psd.calibration_result_path")
    return missing


def _full_tier_missing(config: PCASpecificPSDConfig, candidate_group_ids: Sequence[str] | None = None) -> list[str]:
    missing = _calibration_tier_missing(config)
    if missing:
        return missing
    registry = load_calibration_registry(config)
    if not registry.loaded:
        resolved = config.resolve_root(config.psd.calibration_result_path)
        return [f"calibration_result at {resolved} does not exist -- run `calibrate` first"]
    if registry.status != "SELECTED":
        return [f"calibration_result status is {registry.status!r} (expected 'SELECTED') -- run `calibrate` again"]
    groups = (
        config.psd.groups if candidate_group_ids is None
        else [group for group in config.psd.groups if group.group_id in candidate_group_ids]
    )
    return [
        f"calibration_result.group_selections[{group.group_id!r}]"
        for group in groups
        if group.group_id not in registry.group_taus
    ]


_TIER_CHECKS: dict[str, tuple] = {
    "basis": (),
    "probing": (_probing_tier_missing,),
    "calibration": (_probing_tier_missing, _calibration_tier_missing),
    "full": (_probing_tier_missing, _full_tier_missing),
}


def validate_for_command(config: PCASpecificPSDConfig, command: str, candidate_group_ids: Sequence[str] | None = None) -> None:
    """Raises ``ConfigValidationError`` naming exactly the unresolved fields
    ``command`` needs. Basis/model/generation fields are structural
    requirements already enforced by ``load_config`` for every command, so
    they never appear in this function's output.

    ``candidate_group_ids``, when given, narrows the "full" tier's
    calibration-completeness check to only the named groups -- so a run that
    only ever calibrates the PC groups probing selected doesn't get blocked
    on every other declared-but-uncalibrated group. ``None`` (the default)
    preserves the original behaviour of requiring every declared group.
    """
    try:
        tier = COMMAND_TIERS[command]
    except KeyError as exc:
        raise ConfigError(f"unknown command {command!r}; expected one of {ALL_COMMANDS}") from exc
    missing: list[str] = []
    for check in _TIER_CHECKS[tier]:
        if check is _full_tier_missing:
            missing.extend(check(config, candidate_group_ids))
        else:
            missing.extend(check(config))
    if missing:
        raise ConfigValidationError(command, missing)


def resolve_config(path: str | Path, command: str, candidate_group_ids: Sequence[str] | None = None) -> PCASpecificPSDConfig:
    config = load_config(path)
    validate_for_command(config, command, candidate_group_ids)
    return config
