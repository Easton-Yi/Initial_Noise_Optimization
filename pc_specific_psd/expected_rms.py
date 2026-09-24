"""Expected-RMS PCA operators and matched radial Fourier controls.

This profile deliberately does not restore the reference radial PSD.  A PCA
candidate is the raw fixed linear edit multiplied by one deterministic scalar
computed from its complete impulse-response transfer matrix.  Its Fourier
control applies one frozen radial multiplier to the reference operator.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Sequence

import torch

from pc_specific_psd import calibration_v2, psd_editor, spectral
from pc_specific_psd.patch_codec import OverlapCodec

CALIBRATION_PROFILE = "expected_rms_v1"
ENERGY_CONSTRAINT = "expected_rms"
OPERATOR_SCHEMA_VERSION = "expected_rms_operator_v1"
DIAGNOSTIC_VERSION = "expected_rms_diagnostics_v1"
BINNING_VERSION = "integer_radius_rfft_conjugate_weighted_v2"
DEFAULT_ANNULAR_MATCH_TOLERANCE = 1e-5
DEFAULT_TAU_ZERO_TOLERANCE = 1e-5

OperatorType = Literal["reference", "pca_candidate", "fourier_control"]


@dataclass(frozen=True)
class FrozenOperator:
    condition_id: str
    operator_type: OperatorType
    candidate_id: str | None
    group_id: str | None
    group_indices: tuple[int, ...]
    tau: float
    gate_r_s: float
    gate_beta: float
    scale: float
    radial_multiplier: torch.Tensor | None
    num_bins: int
    latent_shape: tuple[int, int, int]
    expected_mean_square: float
    raw_expected_mean_square: float
    target_relative_l2: tuple[float, ...] = ()
    calibration_relative_l2: float | None = None
    schema_version: str = OPERATOR_SCHEMA_VERSION
    energy_constraint: str = ENERGY_CONSTRAINT


@dataclass(frozen=True)
class OperatorConstruction:
    operator: FrozenOperator
    response: torch.Tensor
    radial_psd: spectral.ExpectedRadialPSD


@dataclass(frozen=True)
class OperatorDiagnostics:
    diagnostic_version: str
    condition_id: str
    operator_type: str
    status: str
    failure_reason: str
    expected_rms_raw: float
    expected_rms_final: float
    fixed_scale: float
    measured_rms: float
    measured_mean_square: float
    measured_mean_square_standard_error: float
    paired_relative_l2_pre_scale: float
    paired_relative_l2_final_fp32: float
    paired_relative_l2_injection_dtype: float
    paired_relative_l2_per_sample: torch.Tensor
    covariance_distance_from_reference: float
    theoretical_radial_power: torch.Tensor
    measured_radial_power: torch.Tensor
    annular_counts: torch.Tensor
    low_mid_high_energy_fractions: tuple[float, float, float]
    low_mid_high_bin_ranges: tuple[tuple[int, int], ...]
    minimum_singular_value: float
    maximum_singular_value: float
    worst_per_frequency_condition_number: float
    worst_condition_frequency: tuple[int, int]
    coefficient_projection: calibration_v2.CoefficientEnergyDiagnostics | None
    processing_batch_size: int
    elapsed_seconds: float


@dataclass(frozen=True)
class TargetSelection:
    group_id: str
    sign: Literal["plus", "minus"]
    target_relative_l2: float
    tolerance: float
    status: str
    reason: str
    selected_tau: float | None
    actual_relative_l2: float | None
    condition_id: str | None
    nearest_feasible_tau: float | None
    nearest_feasible_relative_l2: float | None


@dataclass(frozen=True)
class ConditionExclusion:
    condition_id: str
    stage: str
    reason: str
    paired_condition_id: str | None = None


@dataclass(frozen=True)
class FrozenOperatorPair:
    candidate: FrozenOperator
    control: FrozenOperator


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    selections: tuple[TargetSelection, ...]
    operators: tuple[FrozenOperator, ...]
    diagnostics: tuple[OperatorDiagnostics, ...]
    attempted_taus: dict[str, tuple[float, ...]]
    all_targets_reached: bool
    unreachable_target_count: int
    exclusions: tuple[ConditionExclusion, ...]


def _condition_id(group_id: str, sign: str, tau: float) -> str:
    encoded = format(abs(float(tau)), ".8g").replace(".", "p")
    return f"{group_id}_{sign}_tau_{encoded}_expected_rms"


def control_condition_id(candidate_id: str) -> str:
    return f"{candidate_id}_fourier_control"


def _operator_dict(operator: FrozenOperator) -> dict[str, Any]:
    payload = asdict(operator)
    multiplier = operator.radial_multiplier
    payload["radial_multiplier"] = None if multiplier is None else multiplier.detach().to(torch.float64).tolist()
    payload["group_indices"] = list(operator.group_indices)
    payload["latent_shape"] = list(operator.latent_shape)
    payload["target_relative_l2"] = list(operator.target_relative_l2)
    return payload


def operator_hash(operator: FrozenOperator) -> str:
    encoded = json.dumps(_operator_dict(operator), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def operator_to_payload(operator: FrozenOperator) -> dict[str, Any]:
    return {**_operator_dict(operator), "operator_hash": operator_hash(operator)}


def operator_from_payload(payload: dict[str, Any]) -> FrozenOperator:
    if payload.get("schema_version") != OPERATOR_SCHEMA_VERSION:
        raise ValueError(f"unsupported frozen operator schema {payload.get('schema_version')!r}")
    if payload.get("energy_constraint") != ENERGY_CONSTRAINT:
        raise ValueError("frozen operator does not use expected_rms energy constraint")
    if payload.get("correction") is not None:
        raise ValueError("expected_rms frozen operators must not contain a radial correction")
    multiplier = payload.get("radial_multiplier")
    operator = FrozenOperator(
        condition_id=payload["condition_id"],
        operator_type=payload["operator_type"],
        candidate_id=payload.get("candidate_id"),
        group_id=payload.get("group_id"),
        group_indices=tuple(int(value) for value in payload.get("group_indices", ())),
        tau=float(payload.get("tau", 0.0)),
        gate_r_s=float(payload.get("gate_r_s", 0.0)),
        gate_beta=float(payload.get("gate_beta", 0.0)),
        scale=float(payload.get("scale", 1.0)),
        radial_multiplier=None if multiplier is None else torch.tensor(multiplier, dtype=torch.float64),
        num_bins=int(payload["num_bins"]),
        latent_shape=tuple(int(value) for value in payload["latent_shape"]),
        expected_mean_square=float(payload["expected_mean_square"]),
        raw_expected_mean_square=float(payload["raw_expected_mean_square"]),
        target_relative_l2=tuple(float(value) for value in payload.get("target_relative_l2", ())),
        calibration_relative_l2=(
            None if payload.get("calibration_relative_l2") is None
            else float(payload["calibration_relative_l2"])
        ),
    )
    if operator.operator_type not in ("pca_candidate", "fourier_control"):
        raise ValueError(f"invalid persisted operator type {operator.operator_type!r}")
    if len(operator.latent_shape) != 3 or any(value <= 0 for value in operator.latent_shape):
        raise ValueError("frozen operator has an invalid latent shape")
    if operator.num_bins <= 0:
        raise ValueError("frozen operator num_bins must be positive")
    if len(operator.group_indices) != len(set(operator.group_indices)):
        raise ValueError("frozen operator group indices contain duplicates")
    if not math.isfinite(operator.scale) or operator.scale <= 0:
        raise ValueError("frozen operator scale must be finite and positive")
    if not math.isfinite(operator.expected_mean_square) or operator.expected_mean_square <= 0:
        raise ValueError("frozen operator expected energy must be finite and positive")
    if not math.isfinite(operator.raw_expected_mean_square) or operator.raw_expected_mean_square <= 0:
        raise ValueError("frozen operator raw expected energy must be finite and positive")
    if operator.operator_type == "pca_candidate" and (operator.radial_multiplier is not None or operator.candidate_id is not None):
        raise ValueError("PCA candidate has incompatible control fields")
    if operator.operator_type == "fourier_control":
        if operator.candidate_id is None or operator.radial_multiplier is None:
            raise ValueError("Fourier control is missing candidate link or multiplier")
        if operator.radial_multiplier.numel() != operator.num_bins:
            raise ValueError("Fourier-control multiplier length does not match num_bins")
        if bool((~torch.isfinite(operator.radial_multiplier)).any() | (operator.radial_multiplier < 0).any()):
            raise ValueError("Fourier-control multiplier is non-finite or negative")
    if payload.get("operator_hash") != operator_hash(operator):
        raise ValueError(f"frozen operator hash mismatch for {operator.condition_id!r}")
    return operator


def validate_operator_pairs(operators: Sequence[FrozenOperator]) -> tuple[FrozenOperatorPair, ...]:
    """Validate registry integrity and return canonical candidate/control pairs.

    Numerical rejection is deliberately not handled here. Duplicate IDs,
    malformed links, non-canonical control names, and mismatched frozen
    definitions indicate a damaged registry and must fail closed.
    """
    by_id: dict[str, FrozenOperator] = {}
    for operator in operators:
        if operator.condition_id in by_id:
            raise ValueError(f"duplicate frozen operator condition_id {operator.condition_id!r}")
        by_id[operator.condition_id] = operator
    candidates = {
        condition_id: operator
        for condition_id, operator in by_id.items()
        if operator.operator_type == "pca_candidate"
    }
    controls = [operator for operator in by_id.values() if operator.operator_type == "fourier_control"]
    if not candidates:
        raise ValueError("frozen operator registry has no PCA candidates")
    if len(candidates) > 4:
        raise ValueError("frozen operator registry contains more than four PCA candidates")
    controls_by_candidate: dict[str, list[FrozenOperator]] = {condition_id: [] for condition_id in candidates}
    for control in controls:
        if control.candidate_id not in candidates:
            raise ValueError(
                f"Fourier control {control.condition_id!r} refers to missing candidate {control.candidate_id!r}"
            )
        expected_id = control_condition_id(control.candidate_id)
        if control.condition_id != expected_id:
            raise ValueError(
                f"Fourier control {control.condition_id!r} has non-canonical ID; expected {expected_id!r}"
            )
        controls_by_candidate[control.candidate_id].append(control)

    pairs: list[FrozenOperatorPair] = []
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        linked = controls_by_candidate[candidate_id]
        if len(linked) != 1:
            raise ValueError(
                f"PCA candidate {candidate_id!r} must have exactly one Fourier control, found {len(linked)}"
            )
        control = linked[0]
        candidate_definition = (
            candidate.group_id, candidate.group_indices, candidate.tau,
            candidate.gate_r_s, candidate.gate_beta, candidate.num_bins,
            candidate.latent_shape, candidate.target_relative_l2,
        )
        control_definition = (
            control.group_id, control.group_indices, control.tau,
            control.gate_r_s, control.gate_beta, control.num_bins,
            control.latent_shape, control.target_relative_l2,
        )
        if control_definition != candidate_definition:
            raise ValueError(
                f"Fourier control {control.condition_id!r} does not match candidate {candidate_id!r} metadata"
            )
        pairs.append(FrozenOperatorPair(candidate=candidate, control=control))
    return tuple(pairs)


def reference_operator(channels: int, height: int, width: int, num_bins: int) -> FrozenOperator:
    return FrozenOperator(
        condition_id="reference", operator_type="reference", candidate_id=None,
        group_id=None, group_indices=(), tau=0.0, gate_r_s=0.0, gate_beta=0.0,
        scale=1.0, radial_multiplier=None, num_bins=num_bins,
        latent_shape=(channels, height, width), expected_mean_square=1.0,
        raw_expected_mean_square=1.0,
    )


def _apply_radial_multiplier(latent: torch.Tensor, multiplier: torch.Tensor, num_bins: int) -> torch.Tensor:
    height, width = latent.shape[-2:]
    _, bins = spectral.radial_bin_index(height, width, num_bins, rfft=True)
    if multiplier.numel() != num_bins:
        raise ValueError("radial multiplier length does not match num_bins")
    grid = multiplier.to(device=latent.device, dtype=latent.dtype)[bins.to(latent.device)]
    transformed = torch.fft.rfft2(latent, dim=(-2, -1)) * grid
    return torch.fft.irfft2(transformed, s=(height, width), dim=(-2, -1)).to(latent.dtype)


def apply_frozen_operator(codec: OverlapCodec, base_white: torch.Tensor, operator: FrozenOperator) -> torch.Tensor:
    """The single application path used by calibration, validation and runner."""
    expected = (codec.channels, base_white.shape[-2], base_white.shape[-1])
    if tuple(operator.latent_shape) != expected:
        raise ValueError(f"operator latent shape {operator.latent_shape} does not match input {expected}")
    if operator.operator_type == "reference":
        return psd_editor.apply_psd_edit_tau_zero(base_white)
    if operator.operator_type == "pca_candidate":
        if operator.radial_multiplier is not None:
            raise ValueError("PCA expected-RMS candidate unexpectedly contains a radial multiplier")
        if operator.tau == 0.0:
            return psd_editor.apply_psd_edit_tau_zero(base_white)
        raw = psd_editor.apply_psd_edit(
            codec, base_white, operator.group_indices, operator.tau,
            operator.gate_r_s, operator.gate_beta,
        )
        return raw * operator.scale
    if operator.operator_type == "fourier_control":
        if operator.radial_multiplier is None or operator.candidate_id is None:
            raise ValueError("Fourier control is missing its frozen multiplier or candidate link")
        reference = psd_editor.apply_psd_edit_tau_zero(base_white)
        return _apply_radial_multiplier(reference, operator.radial_multiplier, operator.num_bins)
    raise ValueError(f"unknown operator type {operator.operator_type!r}")


def _response_for(codec: OverlapCodec, operator: FrozenOperator) -> torch.Tensor:
    channels, height, width = operator.latent_shape
    return spectral.linear_operator_frequency_response(
        lambda tensor: apply_frozen_operator(codec, tensor, operator), channels, height, width
    )


def construction_for_frozen_operator(codec: OverlapCodec, operator: FrozenOperator) -> OperatorConstruction:
    """Recreate deterministic diagnostics from frozen fields, never fitting."""
    response = _response_for(codec, operator)
    radial = spectral.expected_radial_psd_from_frequency_response(
        response, width=operator.latent_shape[2], num_bins=operator.num_bins
    )
    return OperatorConstruction(operator=operator, response=response, radial_psd=radial)


def construct_candidate(
    codec: OverlapCodec,
    group_id: str,
    group_indices: Sequence[int],
    tau: float,
    gate_r_s: float,
    gate_beta: float,
    *,
    height: int,
    width: int,
    num_bins: int,
    reference_response: torch.Tensor | None = None,
    tau_zero_tolerance: float = DEFAULT_TAU_ZERO_TOLERANCE,
) -> OperatorConstruction:
    channels = codec.channels
    reference_response = reference_response if reference_response is not None else spectral.linear_operator_frequency_response(
        psd_editor.apply_psd_edit_tau_zero, channels, height, width
    )
    q_reference = spectral.expected_mean_square_from_frequency_response(reference_response, width=width)
    if float(tau) == 0.0:
        raw_response = spectral.linear_operator_frequency_response(
            lambda tensor: psd_editor.apply_psd_edit(
                codec, tensor, group_indices, 0.0, gate_r_s, gate_beta, reference=True
            ), channels, height, width,
        )
        q_raw = spectral.expected_mean_square_from_frequency_response(raw_response, width=width)
        relative = float((raw_response - reference_response).abs().norm() / reference_response.abs().norm().clamp(min=1e-30))
        if relative > tau_zero_tolerance or not math.isclose(q_raw, q_reference, rel_tol=tau_zero_tolerance, abs_tol=tau_zero_tolerance):
            raise ValueError("tau=0 full codec path does not match the frozen reference shortcut")
        scale = 1.0
        final_response = reference_response
    else:
        raw_response = spectral.linear_operator_frequency_response(
            lambda tensor: psd_editor.apply_psd_edit(
                codec, tensor, group_indices, tau, gate_r_s, gate_beta
            ), channels, height, width,
        )
        q_raw = spectral.expected_mean_square_from_frequency_response(raw_response, width=width)
        if not math.isfinite(q_raw) or q_raw <= 0 or not math.isfinite(q_reference) or q_reference <= 0:
            raise ValueError("candidate/reference expected energy must be finite and positive")
        scale = math.sqrt(q_reference / q_raw)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("expected-RMS scale must be finite and positive")
        final_response = raw_response * scale
    condition_id = _condition_id(group_id, "plus" if tau > 0 else "minus" if tau < 0 else "zero", tau)
    q_final = spectral.expected_mean_square_from_frequency_response(final_response, width=width)
    operator = FrozenOperator(
        condition_id=condition_id, operator_type="pca_candidate", candidate_id=None,
        group_id=group_id, group_indices=tuple(int(value) for value in group_indices),
        tau=float(tau), gate_r_s=float(gate_r_s), gate_beta=float(gate_beta),
        scale=scale, radial_multiplier=None, num_bins=num_bins,
        latent_shape=(channels, height, width), expected_mean_square=q_final,
        raw_expected_mean_square=q_raw,
    )
    return OperatorConstruction(
        operator=operator,
        response=final_response,
        radial_psd=spectral.expected_radial_psd_from_frequency_response(final_response, width=width, num_bins=num_bins),
    )


def construct_fourier_control(
    candidate: OperatorConstruction,
    reference_response: torch.Tensor,
) -> OperatorConstruction:
    operator = candidate.operator
    _, height, width = operator.latent_shape
    reference_psd = spectral.expected_radial_psd_from_frequency_response(
        reference_response, width=width, num_bins=operator.num_bins
    )
    candidate_psd = candidate.radial_psd
    nonempty = reference_psd.counts > 0
    invalid = nonempty & ((~torch.isfinite(reference_psd.power)) | (reference_psd.power <= 0))
    if bool(invalid.any()):
        raise ValueError("nonempty reference annulus has zero or non-finite expected power")
    multiplier = torch.ones(operator.num_bins, dtype=torch.float64)
    multiplier[nonempty] = torch.sqrt(candidate_psd.power[nonempty] / reference_psd.power[nonempty])
    if bool((~torch.isfinite(multiplier[nonempty])).any() | (multiplier[nonempty] < 0).any()):
        raise ValueError("Fourier-control multiplier is non-finite or negative")
    grid = multiplier[candidate_psd.bin_index].to(torch.complex128)
    response = reference_response.to(torch.complex128) * grid[..., None, None]
    control_psd = spectral.expected_radial_psd_from_frequency_response(
        response, width=width, num_bins=operator.num_bins
    )
    relative = torch.zeros_like(control_psd.power)
    valid = nonempty & (candidate_psd.power > 0)
    relative[valid] = (control_psd.power[valid] - candidate_psd.power[valid]).abs() / candidate_psd.power[valid]
    if valid.any() and float(relative[valid].max()) > DEFAULT_ANNULAR_MATCH_TOLERANCE:
        raise ValueError("Fourier control failed deterministic annular PSD matching")
    frozen = FrozenOperator(
        condition_id=control_condition_id(operator.condition_id),
        operator_type="fourier_control", candidate_id=operator.condition_id,
        group_id=operator.group_id, group_indices=operator.group_indices,
        tau=operator.tau, gate_r_s=operator.gate_r_s, gate_beta=operator.gate_beta,
        scale=1.0, radial_multiplier=multiplier, num_bins=operator.num_bins,
        latent_shape=operator.latent_shape,
        expected_mean_square=control_psd.total_expected_mean_square,
        raw_expected_mean_square=reference_psd.total_expected_mean_square,
        target_relative_l2=operator.target_relative_l2,
        calibration_relative_l2=None,
    )
    return OperatorConstruction(frozen, response, control_psd)


def _dtype_from_name(name: str) -> torch.dtype:
    values = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported injection dtype {name!r}") from exc


def diagnose_operator(
    codec: OverlapCodec,
    base_white_bank: torch.Tensor,
    construction: OperatorConstruction,
    reference_response: torch.Tensor,
    *,
    batch_size: int,
    injection_dtype: str,
    condition_number_threshold: float | None,
    minimum_covariance_distance: float,
) -> OperatorDiagnostics:
    """Measure a frozen operator without refitting any field."""
    if batch_size <= 0:
        raise ValueError("processing batch_size must be positive")
    started = time.perf_counter()
    operator = construction.operator
    numerator_pre = numerator_final = numerator_injection = denominator = denominator_injection = 0.0
    final_squares = 0.0
    sample_mean_squares: list[torch.Tensor] = []
    per_sample: list[torch.Tensor] = []
    measured_psd_sum = torch.zeros(operator.num_bins, dtype=torch.float64)
    sample_count = 0
    finite = True
    coefficient: calibration_v2.CoefficientEnergyDiagnostics | None = None
    coefficient_sums = {
        "reference_band": 0.0,
        "candidate_band": 0.0,
        "reference_complement": 0.0,
        "candidate_complement": 0.0,
    }
    coefficient_position_count = 0
    group = tuple(int(index) for index in operator.group_indices)
    group_set = set(group)
    complement = tuple(index for index in range(codec.d) if index not in group_set)
    inject_dtype = _dtype_from_name(injection_dtype)
    for batch in base_white_bank.split(batch_size):
        reference = psd_editor.apply_psd_edit_tau_zero(batch)
        final = apply_frozen_operator(codec, batch, operator)
        if operator.operator_type == "pca_candidate" and operator.tau != 0:
            raw = psd_editor.apply_psd_edit(
                codec, batch, operator.group_indices, operator.tau,
                operator.gate_r_s, operator.gate_beta,
            )
        else:
            raw = final
        finite = finite and bool(torch.isfinite(final).all())
        ref64, final64, raw64 = reference.to(torch.float64), final.to(torch.float64), raw.to(torch.float64)
        numerator_pre += float((raw64 - ref64).square().sum())
        numerator_final += float((final64 - ref64).square().sum())
        denominator += float(ref64.square().sum())
        reference_injection64 = reference.to(inject_dtype).to(torch.float64)
        final_injection64 = final.to(inject_dtype).to(torch.float64)
        numerator_injection += float((final_injection64 - reference_injection64).square().sum())
        denominator_injection += float(reference_injection64.square().sum())
        final_squares += float(final64.square().sum())
        sample_ms = final64.reshape(final.shape[0], -1).square().mean(dim=1)
        sample_mean_squares.append(sample_ms)
        per_sample.append(calibration_v2.per_sample_relative_l2(final, reference))
        batch_psd = spectral.radial_psd(final, operator.num_bins)
        measured_psd_sum += batch_psd.power / float(operator.latent_shape[1] * operator.latent_shape[2]) * final.shape[0]
        sample_count += final.shape[0]
        if group:
            reference_coefficients = codec.encode(reference).to(torch.float64)
            final_coefficients = codec.encode(final).to(torch.float64)
            coefficient_sums["reference_band"] += float(reference_coefficients[:, list(group)].square().sum())
            coefficient_sums["candidate_band"] += float(final_coefficients[:, list(group)].square().sum())
            if complement:
                coefficient_sums["reference_complement"] += float(
                    reference_coefficients[:, list(complement)].square().sum()
                )
                coefficient_sums["candidate_complement"] += float(
                    final_coefficients[:, list(complement)].square().sum()
                )
            coefficient_position_count += reference.shape[0] * reference.shape[-2] * reference.shape[-1]
            del reference_coefficients, final_coefficients
    if sample_count == 0:
        raise ValueError("diagnostic bank must be non-empty")
    if group:
        if coefficient_position_count <= 0:
            raise ValueError("coefficient projection accumulator has no samples")
        ref_band = coefficient_sums["reference_band"] / coefficient_position_count
        cand_band = coefficient_sums["candidate_band"] / coefficient_position_count
        ref_complement = coefficient_sums["reference_complement"] / coefficient_position_count
        cand_complement = coefficient_sums["candidate_complement"] / coefficient_position_count
        ref_total = ref_band + ref_complement
        cand_total = cand_band + cand_complement
        coefficient = calibration_v2.CoefficientEnergyDiagnostics(
            measurement_domain=(
                "OverlapCodec analysis coefficient maps; band and complement means are not a "
                "latent-energy orthogonal decomposition"
            ),
            reference_band_energy=ref_band,
            candidate_band_energy=cand_band,
            reference_complement_energy=ref_complement,
            candidate_complement_energy=cand_complement,
            reference_band_fraction=ref_band / ref_total if ref_total > 0 else 0.0,
            candidate_band_fraction=cand_band / cand_total if cand_total > 0 else 0.0,
            band_energy_ratio=cand_band / ref_band if ref_band > 0 else float("inf"),
            complement_energy_ratio=(
                cand_complement / ref_complement if ref_complement > 0 else float("inf")
            ),
        )
    transfer = spectral.transfer_matrix_diagnostics_from_frequency_response(construction.response)
    covariance = spectral.covariance_distance_from_frequency_responses(
        reference_response, construction.response, width=operator.latent_shape[2]
    )
    final_delta = math.sqrt(numerator_final / denominator) if denominator > 0 else float("inf")
    pre_delta = math.sqrt(numerator_pre / denominator) if denominator > 0 else float("inf")
    injection_delta = (
        math.sqrt(numerator_injection / denominator_injection)
        if denominator_injection > 0 else float("inf")
    )
    mean_squares = torch.cat(sample_mean_squares)
    measured_mean_square = final_squares / float(base_white_bank.numel())
    standard_error = float(mean_squares.std(unbiased=True) / math.sqrt(sample_count)) if sample_count > 1 else 0.0
    power = construction.radial_psd.power
    weighted = power * construction.radial_psd.counts
    ranges = spectral.low_mid_high_bin_ranges(operator.num_bins)
    pieces = tuple(weighted[start:end] for start, end in ranges)
    total = float(weighted.sum())
    fractions = tuple(float(piece.sum()) / total if total > 0 else 0.0 for piece in pieces)
    reason = ""
    if not finite:
        reason = "nonfinite_output"
    elif not math.isfinite(operator.scale) or operator.scale <= 0:
        reason = "invalid_expected_rms_scale"
    elif condition_number_threshold is not None and transfer.worst_condition_number > condition_number_threshold:
        reason = "condition_number_exceeded"
    elif operator.operator_type == "pca_candidate" and operator.tau != 0 and covariance <= minimum_covariance_distance:
        reason = "distribution_change_not_resolved"
    return OperatorDiagnostics(
        diagnostic_version=DIAGNOSTIC_VERSION,
        condition_id=operator.condition_id,
        operator_type=operator.operator_type,
        status="accepted" if not reason else "rejected",
        failure_reason=reason,
        expected_rms_raw=math.sqrt(operator.raw_expected_mean_square),
        expected_rms_final=math.sqrt(operator.expected_mean_square),
        fixed_scale=operator.scale,
        measured_rms=math.sqrt(measured_mean_square),
        measured_mean_square=measured_mean_square,
        measured_mean_square_standard_error=standard_error,
        paired_relative_l2_pre_scale=pre_delta,
        paired_relative_l2_final_fp32=final_delta,
        paired_relative_l2_injection_dtype=injection_delta,
        paired_relative_l2_per_sample=torch.cat(per_sample),
        covariance_distance_from_reference=covariance,
        theoretical_radial_power=power,
        measured_radial_power=measured_psd_sum / sample_count,
        annular_counts=construction.radial_psd.counts,
        low_mid_high_energy_fractions=fractions,
        low_mid_high_bin_ranges=ranges,
        minimum_singular_value=float(transfer.minimum_singular_value),
        maximum_singular_value=float(transfer.maximum_singular_value),
        worst_per_frequency_condition_number=transfer.worst_condition_number,
        worst_condition_frequency=transfer.worst_frequency,
        coefficient_projection=coefficient,
        processing_batch_size=batch_size,
        elapsed_seconds=time.perf_counter() - started,
    )


def calibrate(
    codec: OverlapCodec,
    calibration_bank: torch.Tensor,
    *,
    group_id: str,
    group_indices: Sequence[int],
    tau_plus_candidates: Sequence[float],
    tau_minus_candidates: Sequence[float],
    targets: Sequence[float],
    gate_r_s: float,
    gate_beta: float,
    num_bins: int,
    batch_size: int,
    injection_dtype: str,
    condition_number_threshold: float | None,
    minimum_covariance_distance: float,
    configured_tolerance: float | None = None,
    max_refinements_per_sign: int = 6,
) -> CalibrationResult:
    """Select reachable doses and construct their matched Fourier controls."""
    channels, height, width = calibration_bank.shape[1:]
    reference_response = spectral.linear_operator_frequency_response(
        psd_editor.apply_psd_edit_tau_zero, channels, height, width
    )
    constructions: dict[float, OperatorConstruction] = {}
    diagnostics: dict[float, OperatorDiagnostics] = {}

    def evaluate(tau: float) -> None:
        tau = float(tau)
        if tau in constructions:
            return
        built = construct_candidate(
            codec, group_id, group_indices, tau, gate_r_s, gate_beta,
            height=height, width=width, num_bins=num_bins,
            reference_response=reference_response,
        )
        constructions[tau] = built
        diagnostics[tau] = diagnose_operator(
            codec, calibration_bank, built, reference_response,
            batch_size=batch_size, injection_dtype=injection_dtype,
            condition_number_threshold=condition_number_threshold,
            minimum_covariance_distance=minimum_covariance_distance,
        )

    evaluate(0.0)
    for tau in dict.fromkeys((*tau_plus_candidates, *tau_minus_candidates)):
        evaluate(float(tau))

    selections: list[TargetSelection] = []
    selected_targets: dict[float, list[float]] = {}
    control_diagnostics: list[OperatorDiagnostics] = []
    exclusions: list[ConditionExclusion] = []
    attempted_by_sign: dict[str, tuple[float, ...]] = {}
    for sign, declared in (("plus", tau_plus_candidates), ("minus", tau_minus_candidates)):
        refinements = 0
        for target in targets:
            tolerance = calibration_v2.target_tolerance(float(target), configured_tolerance)
            while refinements < max_refinements_per_sign:
                feasible = sorted(
                    (tau for tau, item in diagnostics.items() if (tau > 0 if sign == "plus" else tau < 0) and item.status == "accepted"),
                    key=abs,
                )
                points = [(0.0, 0.0)] + [(tau, diagnostics[tau].paired_relative_l2_final_fp32) for tau in feasible]
                bracket = next(
                    ((left, right) for left, right in zip(points, points[1:])
                     if min(left[1], right[1]) <= target <= max(left[1], right[1])),
                    None,
                )
                if bracket is None:
                    break
                best_now = min(points[1:], key=lambda pair: (abs(pair[1] - target), abs(pair[0])))
                if abs(best_now[1] - target) <= tolerance:
                    break
                midpoint = (bracket[0][0] + bracket[1][0]) / 2.0
                if midpoint in diagnostics:
                    break
                evaluate(midpoint)
                refinements += 1
            feasible_items = [
                (tau, item) for tau, item in diagnostics.items()
                if (tau > 0 if sign == "plus" else tau < 0) and item.status == "accepted"
            ]
            if not feasible_items:
                selections.append(TargetSelection(
                    group_id, sign, float(target), tolerance, "NO_FEASIBLE_CANDIDATE",
                    "no numerically valid tau on this side", None, None, None, None, None,
                ))
                continue
            tau, best = min(
                feasible_items,
                key=lambda pair: (abs(pair[1].paired_relative_l2_final_fp32 - target), abs(pair[0])),
            )
            actual = best.paired_relative_l2_final_fp32
            reached = abs(actual - target) <= tolerance
            selections.append(TargetSelection(
                group_id=group_id, sign=sign, target_relative_l2=float(target), tolerance=tolerance,
                status="SELECTED" if reached else "TARGET_UNREACHABLE",
                reason="" if reached else "nearest feasible tau is outside target tolerance",
                selected_tau=tau if reached else None,
                actual_relative_l2=actual if reached else None,
                condition_id=constructions[tau].operator.condition_id if reached else None,
                nearest_feasible_tau=tau,
                nearest_feasible_relative_l2=actual,
            ))
            if reached:
                selected_targets.setdefault(tau, []).append(float(target))
        attempted_by_sign[sign] = tuple(sorted(
            (tau for tau in diagnostics if (tau > 0 if sign == "plus" else tau < 0)),
            key=abs,
        ))

    frozen: list[FrozenOperator] = []
    for tau, target_values in selected_targets.items():
        candidate = constructions[tau]
        candidate_operator = replace(
            candidate.operator,
            target_relative_l2=tuple(target_values),
            calibration_relative_l2=diagnostics[tau].paired_relative_l2_final_fp32,
        )
        candidate = replace(candidate, operator=candidate_operator)
        frozen.append(candidate_operator)
        control_id = control_condition_id(candidate_operator.condition_id)
        try:
            control = construct_fourier_control(candidate, reference_response)
        except ValueError as exc:
            reason = f"control_construction_failed: {exc}"
            exclusions.extend((
                ConditionExclusion(candidate_operator.condition_id, "calibration", reason, control_id),
                ConditionExclusion(control_id, "calibration", reason, candidate_operator.condition_id),
            ))
            frozen.pop()
            continue
        control_diag = diagnose_operator(
            codec, calibration_bank, control, reference_response,
            batch_size=batch_size, injection_dtype=injection_dtype,
            condition_number_threshold=condition_number_threshold,
            minimum_covariance_distance=minimum_covariance_distance,
        )
        control_diagnostics.append(control_diag)
        if control_diag.status != "accepted":
            reason = control_diag.failure_reason or "control_numerical_validation_failed"
            exclusions.extend((
                ConditionExclusion(
                    candidate_operator.condition_id, "calibration",
                    f"paired control {control.operator.condition_id!r} failed: {reason}",
                    control.operator.condition_id,
                ),
                ConditionExclusion(
                    control.operator.condition_id, "calibration", reason, candidate_operator.condition_id,
                ),
            ))
            frozen.pop()
            continue
        frozen.append(replace(
            control.operator,
            calibration_relative_l2=control_diag.paired_relative_l2_final_fp32,
        ))
    selected_count = sum(item.status == "SELECTED" for item in selections)
    unreachable = len(selections) - selected_count
    zero_ok = diagnostics[0.0].status == "accepted" and diagnostics[0.0].paired_relative_l2_final_fp32 <= DEFAULT_TAU_ZERO_TOLERANCE
    accepted_pair_count = len(frozen) // 2
    return CalibrationResult(
        status="SELECTED" if zero_ok and accepted_pair_count else "FAIL",
        selections=tuple(selections),
        operators=tuple(frozen),
        diagnostics=(
            tuple(diagnostics[tau] for tau in sorted(diagnostics, key=lambda value: (value != 0, value)))
            + tuple(control_diagnostics)
        ),
        attempted_taus=attempted_by_sign,
        all_targets_reached=bool(selections) and unreachable == 0 and not exclusions,
        unreachable_target_count=unreachable,
        exclusions=tuple(exclusions),
    )
