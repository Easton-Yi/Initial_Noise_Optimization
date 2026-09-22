"""Effect-size calibration and final-operator diagnostics (v2).

The v1 calibration path ranks candidates by final RMS after radial matching.
That quantity is intentionally almost constant, so it cannot identify a
meaningful intervention dose.  This module keeps the formal PSD editor,
fixed gate, correction and safety checks, but ranks independently feasible
tau values by their *final paired relative L2* to the reference.  It also
measures analytic output-covariance and overlap-analysis band-energy changes,
because paired sample movement alone does not prove a distribution change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from pc_specific_psd import calibration, psd_editor, spectral
from pc_specific_psd.patch_codec import OverlapCodec

CALIBRATION_PROFILE = "effect_size_v2"
DEFAULT_TARGET_ABSOLUTE_FLOOR = 0.005
DEFAULT_TARGET_RELATIVE_FRACTION = 0.10
DEFAULT_MINIMUM_COVARIANCE_DISTANCE = 1e-6


def target_tolerance(target: float, configured: float | None = None) -> float:
    if target <= 0:
        raise ValueError("effect target must be positive")
    if configured is not None:
        if configured <= 0:
            raise ValueError("configured effect-target tolerance must be positive")
        return float(configured)
    return max(DEFAULT_TARGET_ABSOLUTE_FLOOR, DEFAULT_TARGET_RELATIVE_FRACTION * float(target))


def paired_relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must have the same shape")
    numerator = (candidate.to(torch.float64) - reference.to(torch.float64)).square().sum()
    denominator = reference.to(torch.float64).square().sum()
    if denominator <= 0:
        return 0.0 if numerator <= 0 else float("inf")
    return float((numerator / denominator).sqrt())


def paired_cosine(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    candidate64, reference64 = candidate.to(torch.float64), reference.to(torch.float64)
    denominator = candidate64.square().sum().sqrt() * reference64.square().sum().sqrt()
    if denominator <= 0:
        return 1.0 if torch.equal(candidate64, reference64) else 0.0
    return float((candidate64 * reference64).sum() / denominator)


def per_sample_relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    candidate64, reference64 = candidate.to(torch.float64), reference.to(torch.float64)
    delta = (candidate64 - reference64).reshape(candidate.shape[0], -1)
    ref = reference64.reshape(reference.shape[0], -1)
    return delta.square().sum(dim=1).sqrt() / ref.square().sum(dim=1).sqrt().clamp(min=1e-30)


@dataclass(frozen=True)
class CoefficientEnergyDiagnostics:
    measurement_domain: str
    reference_band_energy: float
    candidate_band_energy: float
    reference_complement_energy: float
    candidate_complement_energy: float
    reference_band_fraction: float
    candidate_band_fraction: float
    band_energy_ratio: float
    complement_energy_ratio: float


def coefficient_energy_diagnostics(
    codec: OverlapCodec,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    group_indices: Sequence[int],
) -> CoefficientEnergyDiagnostics:
    group = tuple(int(index) for index in group_indices)
    complement = tuple(index for index in range(codec.d) if index not in set(group))
    ref_coeff = codec.encode(reference).to(torch.float64)
    candidate_coeff = codec.encode(candidate).to(torch.float64)

    def energy(tensor: torch.Tensor, indices: tuple[int, ...]) -> float:
        # Sum over coefficient dimensions, then average over bank/spatial
        # positions so band + complement is meaningful in this analysis
        # domain even though it is not a latent-energy decomposition.
        return float(tensor[:, list(indices)].square().sum(dim=1).mean()) if indices else 0.0

    ref_band = energy(ref_coeff, group)
    cand_band = energy(candidate_coeff, group)
    ref_complement = energy(ref_coeff, complement)
    cand_complement = energy(candidate_coeff, complement)
    ref_total = ref_band + ref_complement
    cand_total = cand_band + cand_complement
    return CoefficientEnergyDiagnostics(
        measurement_domain="OverlapCodec analysis coefficient maps; band and complement means are not a latent-energy orthogonal decomposition",
        reference_band_energy=ref_band,
        candidate_band_energy=cand_band,
        reference_complement_energy=ref_complement,
        candidate_complement_energy=cand_complement,
        reference_band_fraction=ref_band / ref_total if ref_total > 0 else 0.0,
        candidate_band_fraction=cand_band / cand_total if cand_total > 0 else 0.0,
        band_energy_ratio=cand_band / ref_band if ref_band > 0 else float("inf"),
        complement_energy_ratio=cand_complement / ref_complement if ref_complement > 0 else float("inf"),
    )


@dataclass(frozen=True)
class FinalOperatorDiagnostics:
    group_id: str
    group_indices: tuple[int, ...]
    sign: Literal["plus", "minus", "zero"]
    tau: float
    gate_r_s: float
    gate_beta: float
    status: str
    failure_reason: str
    correction: torch.Tensor
    valid_bin_mask: torch.Tensor
    final_power_valid_mask: torch.Tensor
    paired_relative_l2_pre: float
    paired_relative_l2_final: float
    paired_relative_l2_per_sample: torch.Tensor
    paired_cosine_final: float
    reference_rms: float
    candidate_rms: float
    min_correction: float
    max_correction: float
    symmetric_correction_factor: float
    final_max_radial_psd_error: float
    worst_condition_number: float | None
    covariance_distance: float
    covariance_method: str
    coefficient_energy: CoefficientEnergyDiagnostics


def _correction_summary(correction: torch.Tensor, valid: torch.Tensor) -> tuple[float, float, float, bool]:
    values = correction.to(torch.float64)[valid]
    invalid = bool((~torch.isfinite(values)).any() | (values <= 0).any()) if values.numel() else False
    if values.numel() == 0:
        return 1.0, 1.0, 1.0, False
    if invalid:
        return float("inf"), float("inf"), float("inf"), True
    minimum, maximum = float(values.min()), float(values.max())
    return minimum, maximum, max(maximum, 1.0 / minimum), False


def diagnose_fixed_operator(
    codec: OverlapCodec,
    base_white_bank: torch.Tensor,
    group_id: str,
    group_indices: Sequence[int],
    gate: calibration.GateCandidate,
    tau: float,
    correction: torch.Tensor,
    *,
    num_bins: int,
    psd_tolerance: float,
    correction_gain_bound: float,
    condition_number_threshold: float | None,
    minimum_covariance_distance: float = DEFAULT_MINIMUM_COVARIANCE_DISTANCE,
    min_power: float = calibration.DEFAULT_MIN_POWER,
) -> FinalOperatorDiagnostics:
    """Measure the exact corrected operator that would be injected."""
    reference = psd_editor.apply_psd_edit_tau_zero(base_white_bank)
    edited = psd_editor.apply_psd_edit(
        codec, base_white_bank, group_indices, tau, gate.r_s, gate.beta
    )
    final = calibration.apply_radial_correction(edited, correction, num_bins)
    reference_power = calibration.radial_power(reference, num_bins)
    final_power = calibration.radial_power(final, num_bins)
    valid_reference = reference_power >= min_power
    valid_final = final_power >= min_power
    valid_comparison = valid_reference & valid_final
    relative_error = calibration.psd_relative_error(reference_power, final_power, min_power=min_power)
    max_psd_error = float(relative_error[valid_reference].max()) if valid_reference.any() else 0.0
    minimum, maximum, symmetric_factor, correction_invalid = _correction_summary(correction, valid_reference)
    height, width = base_white_bank.shape[-2:]

    def reference_operator(x: torch.Tensor) -> torch.Tensor:
        return psd_editor.apply_psd_edit_tau_zero(x)

    def candidate_operator(x: torch.Tensor) -> torch.Tensor:
        edited_x = psd_editor.apply_psd_edit(codec, x, group_indices, tau, gate.r_s, gate.beta)
        return calibration.apply_radial_correction(edited_x, correction, num_bins)

    covariance_distance = spectral.linear_operator_covariance_distance(
        reference_operator, candidate_operator, codec.channels, height, width
    )
    transfer = spectral.impulse_response_transfer_matrix(
        candidate_operator, codec.channels, height, width
    )
    failure_reason = ""
    if not valid_reference.any() or bool((~valid_comparison & valid_reference).any()):
        failure_reason = "invalid_power_bin"
    elif max_psd_error > psd_tolerance:
        failure_reason = "psd_tolerance_exceeded"
    elif correction_invalid:
        failure_reason = "correction_nonfinite_or_nonpositive"
    elif symmetric_factor > correction_gain_bound:
        failure_reason = "correction_gain_exceeded_symmetric"
    elif condition_number_threshold is not None and transfer.worst_condition_number > condition_number_threshold:
        failure_reason = "condition_number_exceeded"
    elif float(tau) != 0.0 and covariance_distance <= minimum_covariance_distance:
        failure_reason = "distribution_change_not_resolved"
    status = "accepted" if not failure_reason else "rejected"
    sign: Literal["plus", "minus", "zero"] = "plus" if tau > 0 else "minus" if tau < 0 else "zero"
    return FinalOperatorDiagnostics(
        group_id=group_id,
        group_indices=tuple(group_indices),
        sign=sign,
        tau=float(tau),
        gate_r_s=gate.r_s,
        gate_beta=gate.beta,
        status=status,
        failure_reason=failure_reason,
        correction=correction.detach().to(torch.float64),
        valid_bin_mask=valid_reference,
        final_power_valid_mask=valid_final,
        paired_relative_l2_pre=paired_relative_l2(edited, reference),
        paired_relative_l2_final=paired_relative_l2(final, reference),
        paired_relative_l2_per_sample=per_sample_relative_l2(final, reference),
        paired_cosine_final=paired_cosine(final, reference),
        reference_rms=calibration._measured_rms(reference),
        candidate_rms=calibration._measured_rms(final),
        min_correction=minimum,
        max_correction=maximum,
        symmetric_correction_factor=symmetric_factor,
        final_max_radial_psd_error=max_psd_error,
        worst_condition_number=transfer.worst_condition_number,
        covariance_distance=covariance_distance,
        covariance_method=(
            "analytic per-frequency M(omega)M(omega)^* over the complete rFFT grid; "
            "Hermitian column multiplicity 1 for DC/even Nyquist and 2 otherwise"
        ),
        coefficient_energy=coefficient_energy_diagnostics(codec, reference, final, group_indices),
    )


def fit_and_diagnose_candidate(
    codec: OverlapCodec,
    calibration_base_white: torch.Tensor,
    group_id: str,
    group_indices: Sequence[int],
    gate: calibration.GateCandidate,
    tau: float,
    reference_power: torch.Tensor,
    *,
    num_bins: int,
    protocol: calibration.Protocol,
    psd_tolerance: float,
    correction_gain_bound: float,
    condition_number_threshold: float | None,
    minimum_covariance_distance: float = DEFAULT_MINIMUM_COVARIANCE_DISTANCE,
) -> FinalOperatorDiagnostics:
    if protocol != "operator_clean":
        raise ValueError("effect_size_v2 requires the linear operator_clean protocol")
    if float(tau) == 0.0:
        # Identity is definition-driven, never fitted to sampling noise.
        return diagnose_fixed_operator(
            codec, calibration_base_white, group_id, group_indices, gate, 0.0,
            calibration.zero_tau_correction(num_bins), num_bins=num_bins,
            psd_tolerance=psd_tolerance, correction_gain_bound=correction_gain_bound,
            condition_number_threshold=condition_number_threshold,
            minimum_covariance_distance=minimum_covariance_distance,
        )
    fitted = calibration.evaluate_candidate(
        codec,
        calibration_base_white,
        group_indices,
        gate,
        tau,
        reference_power,
        num_bins=num_bins,
        protocol=protocol,
        psd_tolerance=psd_tolerance,
        correction_gain_bound=correction_gain_bound,
        condition_number_threshold=condition_number_threshold,
    )
    measured = diagnose_fixed_operator(
        codec,
        calibration_base_white,
        group_id,
        group_indices,
        gate,
        tau,
        fitted.correction,
        num_bins=num_bins,
        psd_tolerance=psd_tolerance,
        correction_gain_bound=correction_gain_bound,
        condition_number_threshold=condition_number_threshold,
        minimum_covariance_distance=minimum_covariance_distance,
    )
    if not fitted.accepted and measured.status == "accepted":
        return FinalOperatorDiagnostics(
            **{**measured.__dict__, "status": "rejected", "failure_reason": fitted.reason}
        )
    return measured


@dataclass(frozen=True)
class EffectGroupSpec:
    group_id: str
    group_indices: tuple[int, ...]
    tau_plus_candidates: tuple[float, ...]
    tau_minus_candidates: tuple[float, ...]
    target_relative_l2: tuple[float, ...]
    effect_target_tolerance: float | None = None


@dataclass(frozen=True)
class EffectTargetSelection:
    group_id: str
    sign: Literal["plus", "minus"]
    target_relative_l2: float
    tolerance: float
    status: str
    reason: str
    selected_tau: float | None
    actual_relative_l2: float | None
    condition_id: str | None
    feasible_delta_min: float | None
    feasible_delta_max: float | None
    evaluation: FinalOperatorDiagnostics | None


@dataclass(frozen=True)
class EffectCalibrationResult:
    status: str
    profile: str
    gate: calibration.GateCandidate
    protocol: calibration.Protocol
    selections: tuple[EffectTargetSelection, ...]
    candidate_diagnostics: tuple[FinalOperatorDiagnostics, ...]
    tau_zero_diagnostics: tuple[FinalOperatorDiagnostics, ...]
    all_targets_reached: bool
    unreachable_target_count: int


def _condition_id(group_id: str, sign: str, tau: float) -> str:
    encoded = format(abs(float(tau)), ".6g").replace(".", "p")
    return f"{group_id}_{sign}_tau_{encoded}"


def _select_target(
    spec: EffectGroupSpec,
    sign: Literal["plus", "minus"],
    target: float,
    diagnostics: Sequence[FinalOperatorDiagnostics],
) -> EffectTargetSelection:
    tolerance = target_tolerance(target, spec.effect_target_tolerance)
    feasible = [item for item in diagnostics if item.sign == sign and item.status == "accepted"]
    if not feasible:
        return EffectTargetSelection(
            spec.group_id, sign, target, tolerance, "NO_FEASIBLE_CANDIDATE",
            "no independently feasible tau on this side", None, None, None, None, None, None,
        )
    distances = [abs(item.paired_relative_l2_final - target) for item in feasible]
    best_distance = min(distances)
    # Values equal at ordinary float round-off are a tie; prefer the smaller
    # absolute dose, then preserve declaration order.
    tied = [
        (index, item) for index, (item, distance) in enumerate(zip(feasible, distances))
        if math.isclose(distance, best_distance, rel_tol=1e-12, abs_tol=1e-12)
    ]
    selected = min(tied, key=lambda pair: (abs(pair[1].tau), pair[0]))[1]
    actual = selected.paired_relative_l2_final
    delta_values = [item.paired_relative_l2_final for item in feasible]
    reached = abs(actual - target) <= tolerance
    return EffectTargetSelection(
        group_id=spec.group_id,
        sign=sign,
        target_relative_l2=target,
        tolerance=tolerance,
        status="SELECTED" if reached else "TARGET_UNREACHABLE",
        reason="" if reached else "nearest feasible tau is outside target tolerance",
        selected_tau=selected.tau,
        actual_relative_l2=actual,
        condition_id=_condition_id(spec.group_id, sign, selected.tau) if reached else None,
        feasible_delta_min=min(delta_values),
        feasible_delta_max=max(delta_values),
        evaluation=selected,
    )


def select_effect_size_targets(
    codec: OverlapCodec,
    calibration_base_white: torch.Tensor,
    reference_power: torch.Tensor,
    gate: calibration.GateCandidate,
    group_specs: Sequence[EffectGroupSpec],
    *,
    protocol: calibration.Protocol,
    num_bins: int,
    psd_tolerance: float,
    correction_gain_bound: float,
    condition_number_threshold: float | None,
    minimum_covariance_distance: float = DEFAULT_MINIMUM_COVARIANCE_DISTANCE,
) -> EffectCalibrationResult:
    """Evaluate every tau independently and select by final effect size."""
    if not group_specs:
        raise ValueError("group_specs must be non-empty")
    all_diagnostics: list[FinalOperatorDiagnostics] = []
    zero_diagnostics: list[FinalOperatorDiagnostics] = []
    selections: list[EffectTargetSelection] = []
    for spec in group_specs:
        if not spec.target_relative_l2:
            raise ValueError(f"group {spec.group_id!r} must declare effect targets")
        zero_diagnostics.append(
            fit_and_diagnose_candidate(
                codec, calibration_base_white, spec.group_id, spec.group_indices, gate, 0.0,
                reference_power, num_bins=num_bins, protocol=protocol,
                psd_tolerance=psd_tolerance, correction_gain_bound=correction_gain_bound,
                condition_number_threshold=condition_number_threshold,
                minimum_covariance_distance=minimum_covariance_distance,
            )
        )
        for tau in dict.fromkeys((*spec.tau_plus_candidates, *spec.tau_minus_candidates)):
            if tau == 0:
                continue
            all_diagnostics.append(
                fit_and_diagnose_candidate(
                    codec, calibration_base_white, spec.group_id, spec.group_indices, gate, tau,
                    reference_power, num_bins=num_bins, protocol=protocol,
                    psd_tolerance=psd_tolerance, correction_gain_bound=correction_gain_bound,
                    condition_number_threshold=condition_number_threshold,
                    minimum_covariance_distance=minimum_covariance_distance,
                )
            )
        group_diagnostics = [item for item in all_diagnostics if item.group_id == spec.group_id]
        for target in spec.target_relative_l2:
            selections.append(_select_target(spec, "plus", target, group_diagnostics))
            selections.append(_select_target(spec, "minus", target, group_diagnostics))
    zero_ok = all(
        item.status == "accepted"
        and item.paired_relative_l2_final <= 1e-5
        and item.final_max_radial_psd_error <= psd_tolerance
        and item.covariance_distance <= 1e-5
        for item in zero_diagnostics
    )
    selected = tuple(item for item in selections if item.status == "SELECTED")
    all_targets_reached = bool(selections) and len(selected) == len(selections)
    unreachable_target_count = sum(item.status != "SELECTED" for item in selections)
    status = "SELECTED" if zero_ok and bool(selected) else "FAIL"
    return EffectCalibrationResult(
        status=status,
        profile=CALIBRATION_PROFILE,
        gate=gate,
        protocol=protocol,
        selections=tuple(selections),
        candidate_diagnostics=tuple(all_diagnostics),
        tau_zero_diagnostics=tuple(zero_diagnostics),
        all_targets_reached=all_targets_reached,
        unreachable_target_count=unreachable_target_count,
    )
