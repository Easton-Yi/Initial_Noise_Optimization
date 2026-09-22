"""Output-PSD calibration for the stride-1 PSD editor (plan section 5.4/6.1).

The editor's frequency-domain edit (``psd_editor.apply_psd_edit``) preserves
the reference radial PSD only through the ``h_ref(r)`` factor applied to
every coefficient map; the group-only ``t_B(r, tau)`` factor and the
non-overlapping-patch cross terms introduced by ``Center()`` synthesis can
still shift the *reconstructed* output's radial PSD away from the reference.
This module measures that shift on an independent calibration bank and
freezes a fixed per-radial-bin correction ``c_{B,tau}(b) =
sqrt(P_ref(b) / P~_{B,tau}(b))`` to cancel it, then selects, from finite
pre-declared candidate lists, the one gate ``(r_s, beta)`` and per-group
tau+/tau- that survive tolerance/gain/condition-number checks -- entirely
against the calibration bank, never against generated images, reward/quality
metrics, or the validation bank. The independent validation bank is only
ever consulted once, afterward, as a pure accept/reject gate over the frozen
selection (``validate_on_bank``): it has no fallback path to a different
candidate, by construction -- a retry means building a brand new
``CalibrationResult`` (a new calibration version) with fresh validation-bank
seeds, not re-consulting this one.

``legacy_matched`` also applies ``noise_init``'s existing per-sample
normalize() inside the refinement loop (nonlinear, so a single correction
pass is not guaranteed to still match after normalize -- hence the bounded
fixed-point refinement, up to ``MAX_CALIBRATION_ITERATIONS`` rounds);
``operator_clean`` never applies it, so refinement should already converge
on its first round (a linear radial filter's effect on radial PSD is exact,
not approximate).

tau=0 is a special-cased *definition*, not a measured output of the
selection procedure above: ``zero_tau_correction`` is exactly ``1.0`` for
every bin, matching ``psd_editor.apply_psd_edit_tau_zero``'s exact-identity
shortcut and avoiding any chance that calibration-bank sampling noise
manufactures a spurious nonzero correction at the identity point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

from pc_specific_psd import psd_editor, spectral
from pc_specific_psd.compat_generation import normalize
from pc_specific_psd.patch_codec import OverlapCodec

MAX_CALIBRATION_ITERATIONS = 3
DEFAULT_MIN_POWER = 1e-12
LEGACY_NORMALIZATION_PROFILE = "per_sample_per_channel_zero_mean_unit_std"

Protocol = Literal["legacy_matched", "operator_clean"]


def _measured_rms(tensor: torch.Tensor) -> float:
    return float(tensor.to(torch.float64).square().mean().sqrt())


def radial_power(tensor: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Cross-channel-averaged radial PSD power per bin; thin wrapper over spectral.radial_psd."""
    return spectral.radial_psd(tensor, num_bins).power.to(torch.float64)


def compute_reference_power(
    reference_base_white_bank: torch.Tensor,
    num_bins: int,
    *,
    protocol: Protocol = "operator_clean",
    normalization_profile: str = LEGACY_NORMALIZATION_PROFILE,
) -> torch.Tensor:
    """P_ref(b): the radial PSD of the tau=0-for-every-group reference output,
    measured at the same pipeline position as the candidate it will be
    compared against (plan section 5.4: "参考目标是在同一 pipeline 位置定义的
    P_ref(b)"). ``legacy_matched`` candidates are measured after
    ``normalize()``, so the reference must pass through the same normalize()
    step here too, or the correction would be chasing an absolute output
    scale that normalize() resets every iteration regardless of correction
    (verified empirically: without this, legacy_matched's correction shrinks
    every round without ever reducing the relative error). ``operator_clean``
    never normalizes anywhere, so the reference stays unnormalized to match.

    Reference is exactly ``psd_editor.apply_psd_edit_tau_zero``
    (``same_phase_floor`` times its fixed analytic expected-unit-RMS scale),
    never the raw editor path evaluated at tau=0 --
    this keeps the reference target definition-driven rather than an
    ordinary measurement subject to the same sampling noise as candidates.
    """
    reference_output = psd_editor.apply_psd_edit_tau_zero(reference_base_white_bank)
    if protocol == "legacy_matched":
        reference_output = normalize(reference_output, normalization_profile)
    return radial_power(reference_output, num_bins)


def zero_tau_correction(num_bins: int) -> torch.Tensor:
    """c_{B,0}(b) is exactly 1.0 for every bin, by definition -- not measured."""
    return torch.ones(num_bins, dtype=torch.float64)


@dataclass(frozen=True)
class RadialCorrection:
    correction: torch.Tensor    # (num_bins,) float64, 1.0 at invalid bins
    invalid_bins: torch.Tensor  # (num_bins,) bool
    min_gain: float             # min positive amplitude correction over valid bins
    max_gain: float             # max(correction) over valid bins only (1.0 if none valid)
    symmetric_factor: float     # max(max(c), 1/min(c)) over valid bins
    has_invalid_values: bool    # non-finite or non-positive value in a valid bin
    exceeds_gain_bound: bool


def compute_radial_correction(
    reference_power: torch.Tensor,
    measured_power: torch.Tensor,
    *,
    max_gain_bound: float,
    min_power: float = DEFAULT_MIN_POWER,
) -> RadialCorrection:
    """c(b) = sqrt(P_ref(b) / P_measured(b)), restricted to valid (non-negligible-energy) bins.

    A bin is invalid whenever either side is below ``min_power`` -- an
    ill-defined ratio is forced to a no-op correction (1.0) rather than an
    arbitrarily large one, matching plan section 5.4's "explicit threshold
    for failure, never an arbitrarily large gain standing in for a match."
    """
    if reference_power.shape != measured_power.shape:
        raise ValueError("reference_power and measured_power must have the same shape")
    invalid = (reference_power < min_power) | (measured_power < min_power)
    safe_measured = measured_power.clamp(min=min_power)
    raw_correction = (reference_power.clamp(min=0.0) / safe_measured).sqrt()
    correction = torch.where(invalid, torch.ones_like(raw_correction), raw_correction)
    valid_values = correction[~invalid]
    has_invalid_values = bool(
        (~torch.isfinite(valid_values)).any() | (valid_values <= 0).any()
    ) if valid_values.numel() > 0 else False
    if valid_values.numel() == 0:
        min_gain = max_gain = symmetric_factor = 1.0
    elif has_invalid_values:
        min_gain = max_gain = symmetric_factor = float("inf")
    else:
        min_gain = float(valid_values.min())
        max_gain = float(valid_values.max())
        symmetric_factor = max(max_gain, 1.0 / min_gain)
    return RadialCorrection(
        correction, invalid, min_gain, max_gain, symmetric_factor,
        has_invalid_values, has_invalid_values or symmetric_factor > max_gain_bound,
    )


def psd_relative_error(reference_power: torch.Tensor, measured_power: torch.Tensor, *, min_power: float = DEFAULT_MIN_POWER) -> torch.Tensor:
    return (measured_power - reference_power).abs() / reference_power.clamp(min=min_power)


def apply_radial_correction(tensor: torch.Tensor, correction: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Applies a per-radial-bin scalar multiplier c(b) to tensor's spectrum and inverse-transforms back.

    Uses the same integer-FFT-bin radial grid as spectral.radial_psd, so a
    correction measured via radial_power on this (height, width) is applied
    on the identical bin boundaries it was fit on.
    """
    height, width = tensor.shape[-2], tensor.shape[-1]
    _, bin_index = spectral.radial_bin_index(height, width, num_bins, rfft=True)
    multiplier = correction.to(torch.float64)[bin_index]
    spectrum = torch.fft.rfft2(tensor.to(torch.float64), dim=(-2, -1))
    corrected = torch.fft.irfft2(spectrum * multiplier, s=(height, width), dim=(-2, -1))
    return corrected.to(tensor.dtype)


@dataclass(frozen=True)
class CandidateEvaluation:
    accepted: bool
    reason: str  # "" if accepted, else "psd_tolerance_exceeded" | "correction_gain_exceeded" | "condition_number_exceeded"
    correction: torch.Tensor  # (num_bins,) float64, the final cumulative correction
    measured_rms: float
    min_gain: float
    max_gain: float
    symmetric_correction_factor: float
    correction_invalid: bool
    iterations_used: int
    worst_condition_number: float | None


def evaluate_candidate(
    codec: OverlapCodec,
    calibration_base_white: torch.Tensor,
    group_indices: Sequence[int],
    gate: "GateCandidate",
    tau: float,
    reference_power: torch.Tensor,
    *,
    num_bins: int,
    protocol: Protocol,
    psd_tolerance: float,
    correction_gain_bound: float,
    condition_number_threshold: float | None = None,
    min_power: float = DEFAULT_MIN_POWER,
    max_iterations: int = MAX_CALIBRATION_ITERATIONS,
    normalization_profile: str = LEGACY_NORMALIZATION_PROFILE,
) -> CandidateEvaluation:
    """Bounded fixed-point refinement of one (gate, group, tau) candidate's correction.

    Each round: apply the current cumulative correction to a fresh edit of
    the calibration bank (plus normalize() under ``legacy_matched``), measure
    the resulting radial PSD, and stop if it already matches the reference
    within ``psd_tolerance``; otherwise fold in ``sqrt(P_ref/P_measured)``
    and try again, up to ``max_iterations`` rounds. The candidate is rejected
    if it still fails tolerance after the cap, if the final cumulative
    correction's gain exceeds ``correction_gain_bound``, or (when checked) if
    the operator's transfer-matrix condition number exceeds
    ``condition_number_threshold``.
    """
    if protocol not in ("legacy_matched", "operator_clean"):
        raise ValueError(f"unknown protocol {protocol!r}")
    valid_reference = reference_power >= min_power
    correction = torch.ones(num_bins, dtype=torch.float64)
    measured_power = reference_power
    corrected = calibration_base_white
    converged = False
    iterations_used = 0

    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration
        edited = psd_editor.apply_psd_edit(codec, calibration_base_white, group_indices, tau, gate.r_s, gate.beta)
        corrected = apply_radial_correction(edited, correction, num_bins)
        if protocol == "legacy_matched":
            corrected = normalize(corrected, normalization_profile)
        measured_power = radial_power(corrected, num_bins)
        rel_error = psd_relative_error(reference_power, measured_power, min_power=min_power)
        # Deliberately checked over every bin where the *reference* carries
        # real energy, even ones where measured_power has collapsed near
        # zero -- a candidate that destroys a bin's energy must fail
        # tolerance (a large but finite relative error) rather than have
        # that bin silently excluded from the check.
        converged = bool((rel_error[valid_reference] <= psd_tolerance).all()) if valid_reference.any() else True
        if converged:
            break
        step = compute_radial_correction(reference_power, measured_power, max_gain_bound=correction_gain_bound, min_power=min_power)
        correction = correction * step.correction

    valid_correction = correction[valid_reference]
    correction_invalid = bool(
        (~torch.isfinite(valid_correction)).any() | (valid_correction <= 0).any()
    ) if valid_correction.numel() > 0 else False
    if valid_correction.numel() == 0:
        min_gain = max_gain = symmetric_factor = 1.0
    elif correction_invalid:
        min_gain = max_gain = symmetric_factor = float("inf")
    else:
        min_gain = float(valid_correction.min())
        max_gain = float(valid_correction.max())
        symmetric_factor = max(max_gain, 1.0 / min_gain)
    exceeds_gain = correction_invalid or symmetric_factor > correction_gain_bound
    measured_rms = _measured_rms(corrected)

    worst_condition_number = None
    condition_ok = True
    if condition_number_threshold is not None:
        height, width = calibration_base_white.shape[-2], calibration_base_white.shape[-1]

        def _operator(x: torch.Tensor) -> torch.Tensor:
            e = psd_editor.apply_psd_edit(codec, x, group_indices, tau, gate.r_s, gate.beta)
            return apply_radial_correction(e, correction, num_bins)

        diagnostics = spectral.impulse_response_transfer_matrix(_operator, codec.channels, height, width)
        worst_condition_number = diagnostics.worst_condition_number
        condition_ok = worst_condition_number <= condition_number_threshold

    if not converged:
        reason = "psd_tolerance_exceeded"
    elif correction_invalid:
        reason = "correction_nonfinite_or_nonpositive"
    elif exceeds_gain:
        # Retain the legacy machine-readable prefix while making the new
        # two-sided meaning explicit for existing artifact consumers.
        reason = "correction_gain_exceeded_symmetric"
    elif not condition_ok:
        reason = "condition_number_exceeded"
    else:
        reason = ""
    accepted = converged and not exceeds_gain and condition_ok
    return CandidateEvaluation(
        accepted, reason, correction, measured_rms, min_gain, max_gain,
        symmetric_factor, correction_invalid, iterations_used, worst_condition_number,
    )


@dataclass(frozen=True)
class GateCandidate:
    r_s: float
    beta: float


@dataclass(frozen=True)
class GroupCandidateSpec:
    group_id: str
    group_indices: tuple[int, ...]
    tau_plus_candidates: tuple[float, ...]
    tau_minus_candidates: tuple[float, ...]
    target_rms: float


@dataclass(frozen=True)
class GroupSelection:
    group_id: str
    group_indices: tuple[int, ...]
    tau_plus: float
    tau_minus: float
    evaluation_plus: CandidateEvaluation
    evaluation_minus: CandidateEvaluation


@dataclass(frozen=True)
class CalibrationResult:
    status: str  # "SELECTED" | "REJECTED_ALL_GATES"
    gate: GateCandidate | None
    protocol: Protocol
    group_selections: dict[str, GroupSelection]
    rejected_gates: tuple[tuple[GateCandidate, str], ...]


def _select_closest_rms(candidates: Sequence[float], evaluations: dict[float, CandidateEvaluation], target_rms: float) -> float:
    """First candidate (in declared order) achieving the smallest |measured_rms - target_rms|."""
    best_tau, best_distance = None, None
    for tau in candidates:
        distance = abs(evaluations[tau].measured_rms - target_rms)
        # A target constructed as the midpoint of two binary floating-point
        # RMS values can leave the two mathematically equal distances one ulp
        # apart. Treat numerical ties as ties so declared order remains the
        # deterministic tiebreaker promised by this function.
        if best_distance is None or (
            distance < best_distance
            and not math.isclose(distance, best_distance, rel_tol=1e-12, abs_tol=1e-12)
        ):
            best_tau, best_distance = tau, distance
    assert best_tau is not None  # candidates is non-empty by construction (validated by caller)
    return best_tau


def select_gate_and_taus(
    codec: OverlapCodec,
    calibration_base_white: torch.Tensor,
    reference_power: torch.Tensor,
    gate_candidates: Sequence[GateCandidate],
    group_specs: Sequence[GroupCandidateSpec],
    *,
    protocol: Protocol,
    num_bins: int,
    psd_tolerance: float,
    correction_gain_bound: float,
    condition_number_threshold: float | None = None,
    min_power: float = DEFAULT_MIN_POWER,
    max_iterations: int = MAX_CALIBRATION_ITERATIONS,
    normalization_profile: str = LEGACY_NORMALIZATION_PROFILE,
) -> CalibrationResult:
    """Deterministic calibration-bank-only selection (plan section 5.4).

    Walks ``gate_candidates`` in declared order. A gate survives only if
    every declared tau candidate (both signs, every group in ``group_specs``)
    is individually accepted by ``evaluate_candidate``; the first surviving
    gate wins and scanning stops there. For that gate, each group
    independently picks its tau+/tau- as the declared candidate whose
    calibration-bank measured RMS is closest to its ``target_rms`` (ties
    broken by declared order via ``_select_closest_rms``). If no gate
    survives, returns a ``REJECTED_ALL_GATES`` result carrying every gate's
    rejection reason rather than raising -- the caller (CLI) decides how to
    report that.
    """
    if not gate_candidates:
        raise ValueError("gate_candidates must be non-empty")
    if not group_specs:
        raise ValueError("group_specs must be non-empty")
    for spec in group_specs:
        if not spec.tau_plus_candidates or not spec.tau_minus_candidates:
            raise ValueError(f"group {spec.group_id!r} must declare at least one tau+ and one tau- candidate")

    rejected_gates: list[tuple[GateCandidate, str]] = []
    for gate in gate_candidates:
        per_group_evaluations: dict[str, dict[float, CandidateEvaluation]] = {}
        gate_failure_reason = ""
        for spec in group_specs:
            per_group_evaluations[spec.group_id] = {}
            for tau in (*spec.tau_plus_candidates, *spec.tau_minus_candidates):
                if tau in per_group_evaluations[spec.group_id]:
                    continue
                evaluation = evaluate_candidate(
                    codec, calibration_base_white, spec.group_indices, gate, tau, reference_power,
                    num_bins=num_bins, protocol=protocol, psd_tolerance=psd_tolerance,
                    correction_gain_bound=correction_gain_bound, condition_number_threshold=condition_number_threshold,
                    min_power=min_power, max_iterations=max_iterations, normalization_profile=normalization_profile,
                )
                per_group_evaluations[spec.group_id][tau] = evaluation
                if not evaluation.accepted and not gate_failure_reason:
                    gate_failure_reason = f"group {spec.group_id} tau={tau}: {evaluation.reason}"
        if gate_failure_reason:
            rejected_gates.append((gate, gate_failure_reason))
            continue

        group_selections: dict[str, GroupSelection] = {}
        for spec in group_specs:
            evaluations = per_group_evaluations[spec.group_id]
            tau_plus = _select_closest_rms(spec.tau_plus_candidates, evaluations, spec.target_rms)
            tau_minus = _select_closest_rms(spec.tau_minus_candidates, evaluations, spec.target_rms)
            group_selections[spec.group_id] = GroupSelection(
                group_id=spec.group_id,
                group_indices=spec.group_indices,
                tau_plus=tau_plus,
                tau_minus=tau_minus,
                evaluation_plus=evaluations[tau_plus],
                evaluation_minus=evaluations[tau_minus],
            )
        return CalibrationResult(
            status="SELECTED", gate=gate, protocol=protocol,
            group_selections=group_selections, rejected_gates=tuple(rejected_gates),
        )

    return CalibrationResult(status="REJECTED_ALL_GATES", gate=None, protocol=protocol, group_selections={}, rejected_gates=tuple(rejected_gates))


@dataclass(frozen=True)
class ValidationOutcome:
    passed: bool
    reason: str
    measured_rel_error: torch.Tensor


def validate_on_bank(
    codec: OverlapCodec,
    validation_base_white: torch.Tensor,
    calibration_result: CalibrationResult,
    group_id: str,
    sign: Literal["plus", "minus"],
    reference_power: torch.Tensor,
    *,
    num_bins: int,
    psd_tolerance: float,
    min_power: float = DEFAULT_MIN_POWER,
    normalization_profile: str = LEGACY_NORMALIZATION_PROFILE,
) -> ValidationOutcome:
    """Consults the independent validation bank exactly once, as a pure accept/reject gate.

    Re-applies the already-frozen gate/tau/correction from ``calibration_result``
    (never re-selecting or re-tuning anything) and checks the resulting radial
    PSD against ``reference_power`` within ``psd_tolerance``. There is no
    fallback here to a different candidate: a failing outcome marks this
    calibration version FAIL outright, by construction -- a retry means the
    caller builds an entirely new ``CalibrationResult`` (a new calibration
    version) and calls this again with fresh validation-bank seeds, never
    re-consulting this bank/result pair for a second opinion.
    """
    if calibration_result.status != "SELECTED":
        raise ValueError("validate_on_bank requires a SELECTED calibration_result")
    if sign not in ("plus", "minus"):
        raise ValueError("sign must be 'plus' or 'minus'")
    selection = calibration_result.group_selections[group_id]
    tau, correction = (selection.tau_plus, selection.evaluation_plus.correction) if sign == "plus" else (selection.tau_minus, selection.evaluation_minus.correction)
    gate = calibration_result.gate
    assert gate is not None  # implied by status == "SELECTED"

    edited = psd_editor.apply_psd_edit(codec, validation_base_white, selection.group_indices, tau, gate.r_s, gate.beta)
    corrected = apply_radial_correction(edited, correction, num_bins)
    if calibration_result.protocol == "legacy_matched":
        corrected = normalize(corrected, normalization_profile)
    measured_power = radial_power(corrected, num_bins)
    rel_error = psd_relative_error(reference_power, measured_power, min_power=min_power)
    valid = reference_power >= min_power
    passed = bool((rel_error[valid] <= psd_tolerance).all()) if valid.any() else True
    reason = "" if passed else "validation_bank_tolerance_exceeded"
    return ValidationOutcome(passed, reason, rel_error)
