#!/usr/bin/env python3
"""Streaming frozen-cache reconstruction and spectral diagnostics (no resampling).

Hash equality, summary-statistic agreement, algebraic checks and finite-sample
PSD diagnostics are independent evidence. Only saved final hashes can establish
bit-exact agreement with the recorded pre-adapter input tensors.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from io_utils import read_json, read_jsonl, write_json, tensor_hash, file_hash, derived_seed
from noise_methods import NoiseBatch, construct_raw_noise, noise_statistics, normalize, pink, pink_filter, radial_frequency_grid
from result_integrity import load_run_contract, validate_condition_metadata
from run_support import (Condition as ConditionSpec, conditions, resolve_run, display_selection, parse_selection,
                         selected_condition, add_display_arguments, new_output,
                         NOISE_CONSTRUCTION_VERSION, NOISE_FREQUENCY_GRID_VERSION)

VALIDATION_VERSION = "spectral_v3"
NUM_RADIAL_BINS = 24
EXACT_RTOL, EXACT_ATOL = 2e-5, 2e-5
NUMERIC_HASH_TOLERANCE = 1e-4
STATS_KEYS_FOR_NUMERIC_CHECK = ("std", "l2_norm", "psd_mean", "psd_high_frequency_mean")
EPS = 1e-12
FAMILIES = ("same_phase", "independent_white")


# Legacy helpers delegate to the same grid/selection implementation.
def full_condition_grid(baseline_alphas, proposed_alphas, proposed_gammas):
    config = {"baseline": {"enabled": bool(baseline_alphas), "alpha_values": baseline_alphas}}
    for family, key in (("same_phase", "same_phase_floor"), ("independent_white", "independent_white")):
        config[key] = {"enabled": bool(proposed_alphas[family]), "alpha_values": proposed_alphas[family],
                       "gamma_values": proposed_gammas[family]}
    return conditions(config)


def _selected_gamma_grids(gamma_spec, same_phase_gamma_spec, independent_gamma_spec, grids):
    specs = {"same_phase": same_phase_gamma_spec, "independent_white": independent_gamma_spec}
    return {family: tuple(parse_selection(specs[family] if specs[family] is not None else gamma_spec,
                                         grid, family + " gamma")) if grid else () for family, grid in grids.items()}


def stats_comparison(saved_rows, mine_rows):
    """Align exact sample/channel coverage before comparing selected finite fields."""
    def index(rows):
        result = {}
        for row in rows:
            key = (row["base_index"], row["channel"])
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in key) or key in result:
                raise ValueError(f"Duplicate/invalid sample-channel index: {key}")
            if any(not math.isfinite(float(value)) for name, value in row.items() if name not in ("base_index", "channel")):
                raise ValueError(f"Non-finite statistics at {key}")
            if any(k not in row for k in STATS_KEYS_FOR_NUMERIC_CHECK):
                raise ValueError(f"Missing comparison fields at {key}")
            result[key] = row
        return result
    result = {"fields": list(STATS_KEYS_FOR_NUMERIC_CHECK), "tolerance": NUMERIC_HASH_TOLERANCE,
              "error_definition": "max abs(reconstructed-saved)/max(abs(saved),1e-6)",
              "saved_rows": len(saved_rows), "reconstructed_rows": len(mine_rows), "compared_rows": 0}
    if not saved_rows or not mine_rows:
        return {**result, "status": "unavailable", "reason": "empty statistics", "max_relative_error": None}
    try:
        saved, mine = index(saved_rows), index(mine_rows)
        if saved.keys() != mine.keys():
            return {**result, "status": "failed", "reason": "sample/channel coverage mismatch",
                    "missing": sorted(mine.keys() - saved.keys()), "extra": sorted(saved.keys() - mine.keys()),
                    "max_relative_error": None}
        error = max(abs(float(mine[k][f]) - float(saved[k][f])) / max(abs(float(saved[k][f])), 1e-6)
                    for k in mine for f in STATS_KEYS_FOR_NUMERIC_CHECK)
        return {**result, "status": "passed" if error <= NUMERIC_HASH_TOLERANCE else "failed",
                "compared_rows": len(mine), "max_relative_error": error}
    except (KeyError, TypeError, ValueError) as exc:
        return {**result, "status": "failed", "reason": str(exc), "max_relative_error": None}


def stats_relative_error(saved_rows, mine_rows):
    result = stats_comparison(saved_rows, mine_rows)
    if result["max_relative_error"] is None:
        raise ValueError(result.get("reason", "Invalid statistics"))
    return result["max_relative_error"]


def frequency_column_weights(width, width_freq=None):
    weights = torch.full((width // 2 + 1,), 2., dtype=torch.float64)
    weights[0] = 1
    if width % 2 == 0:
        weights[-1] = 1
    return weights


class SpectrumLayout:
    def __init__(self, height, width, bins=NUM_RADIAL_BINS):
        if bins < 1:
            raise ValueError("radial bins must be positive")
        self.height, self.width, self.bins = height, width, bins
        self.r = radial_frequency_grid(height, width).double()
        self.edges = torch.linspace(0, float(self.r.max()) + 1e-6, bins + 1, dtype=torch.float64)
        self.index = torch.bucketize(self.r.flatten(), self.edges[1:-1], right=False)
        self.weights = frequency_column_weights(width).expand_as(self.r).clone()
        # Report full-spectrum non-DC shells; DC is explicitly reported separately.
        self.weights[0, 0] = 0
        self.counts = torch.zeros(bins, dtype=torch.float64).index_add_(0, self.index, self.weights.flatten())
        self.valid = self.counts > 0

    def shell(self, power, *, energy=False):
        flat = (power.double() * self.weights).flatten(-2)
        out = torch.zeros((*flat.shape[:-1], self.bins), dtype=torch.float64)
        out.index_add_(-1, self.index, flat)
        return out if energy else out / self.counts.clamp_min(1)

    def settings(self):
        return {"fft": "torch.fft.rfft2", "fft_norm": "backward (unnormalized forward)",
                "power": "abs(F)^2; sum(full_power)/(H*W) equals spatial sum of squares",
                "height": self.height, "width": self.width, "radial_bins": self.bins,
                "radial_edges_fft_bin_units": self.edges.tolist(),
                "bin_boundary": "first [0,e1], then (ei,e(i+1)]", "dc": "excluded from shells/bands/coherence, reported separately",
                "empty_bins": "CSV null; excluded from normalization, distances and plots",
                "rfft_weights": "1 for fx=0/even-width Nyquist, 2 otherwise (DC weight set to zero)",
                "shell_mean": "weighted power sum / full-spectrum coefficient count",
                "shell_energy": "weighted power sum (divide by H*W for spatial energy)",
                "normalized_log_psd": "ln(max(shell_mean/sum(nonempty shell_means),1e-12)); shell-equal normalization",
                "distance": "mean squared difference of normalized log-PSD over nonempty shells",
                "coherence": "sum_f w_f |sum_sample,channel Fz conj(Fbase)|^2 / sum_f w_f (sum |Fz|^2)(sum |Fbase|^2)",
                "coherence_weight": "full-spectrum multiplicity times product of pooled powers; ensemble magnitude-squared coherence, not phase proof",
                "aggregation_unit": "sample/channel, equally weighted; no prompt bootstrap",
                "bands": "non-DC radius <=rmax/3, (rmax/3,2*rmax/3], >2*rmax/3; mean per-sample/channel energy fraction"}


def normalize_log_curve(curve, valid=None):
    valid = (torch.isfinite(curve) & (curve > 0)) if valid is None else valid
    if not bool(valid.any()) or not bool(torch.isfinite(curve[valid]).all()) or float(curve[valid].sum()) <= 0:
        raise ValueError("No finite positive spectral energy")
    result = torch.full_like(curve, float("nan"))
    result[valid] = torch.log((curve[valid] / curve[valid].sum()).clamp_min(EPS))
    return result


def coefficient_check(actual, expected, *, component_scale=None, absolute_error_bound=0.):
    """Mixed float32 tolerance; avoid relative-error singularities at cancellation.

    For complex coefficients, use an absolute roundoff floor of eight float32
    epsilons times per-sample/channel RMS Fourier amplitude. For the power
    identity, relative tolerance uses the pre-cancellation component magnitude.
    These are numerical algebra checks, not bit-exact tensor checks.
    """
    actual = actual.to(torch.complex128) if actual.is_complex() else actual.double()
    expected = expected.to(torch.complex128) if expected.is_complex() else expected.double()
    delta = (actual - expected).abs()
    scale = expected.abs() if component_scale is None else component_scale.double()
    floor = (8 * torch.finfo(torch.float32).eps * scale.square().mean((-1, -2), keepdim=True).sqrt()
             if component_scale is None else 0.)
    allowance = EXACT_ATOL + floor + EXACT_RTOL * scale + absolute_error_bound
    return {"status": "passed" if bool((delta <= allowance).all()) else "failed",
            "max_scaled_error": float((delta / allowance).max()), "max_absolute_error": float(delta.max()),
            "tolerance_scale": "pre-cancellation component power plus propagated complex roundoff bound" if component_scale is not None else "expected amplitude plus 8*float32_eps*per-sample/channel RMS amplitude floor"}


def construction_checks(raw_fft, base_fft, eta_fft, condition, pink_fft=None):
    gamma = condition.gamma
    if condition.family in ("baseline", "same_phase"):
        if pink_fft is None:
            raise ValueError("Supply the actual H×W filter; width cannot be inferred from rFFT shape")
        h = pink_fft
        multiplier = h if condition.family == "baseline" else ((1 - gamma) * h.square() + gamma).sqrt()
        expected = base_fft * multiplier
        check = coefficient_check(raw_fft, expected)
        mask = expected.abs() > max(EXACT_ATOL * 10, float(expected.abs().max()) * 1e-6)
        check["phase_bins_checked"] = int(mask.sum())
        check["near_zero_bins_excluded_from_phase"] = int(mask.numel() - mask.sum())
        check["max_phase_error_radians"] = float(torch.angle(raw_fft[mask] * expected[mask].conj()).abs().max()) if bool(mask.any()) else None
        return check, expected.abs().square(), torch.zeros_like(raw_fft.real)
    pink_fft, eta_fft = pink_fft.to(torch.complex128), eta_fft.to(torch.complex128)
    expected = (1 - gamma) ** .5 * pink_fft + gamma ** .5 * eta_fft
    prediction = (1 - gamma) * pink_fft.abs().square() + gamma * eta_fft.abs().square()
    cross = 2 * (gamma * (1 - gamma)) ** .5 * (pink_fft * eta_fft.conj()).real
    check = coefficient_check(raw_fft, expected)
    amplitude_tolerance = (EXACT_ATOL + EXACT_RTOL * expected.abs() +
                           8 * torch.finfo(torch.float32).eps * expected.abs().square().mean((-1, -2), keepdim=True).sqrt())
    # Propagate the complex-coefficient error bound through |F|^2:
    # ||F+dF|^2-|F|^2| <= 2|F| |dF| + |dF|^2.
    power_bound = 2 * expected.abs() * amplitude_tolerance + amplitude_tolerance.square()
    power_check = coefficient_check(raw_fft.to(torch.complex128).abs().square(), prediction + cross,
                                    component_scale=prediction + cross.abs(), absolute_error_bound=power_bound)
    check["power_identity"] = power_check
    if power_check["status"] == "failed":
        check["status"] = "failed"
    return check, prediction, cross


def cache_inventory(run, blocks):
    root = run / "noise_cache"
    actual = {p.name for p in root.iterdir()} if root.is_dir() else set()
    expected = set(blocks)
    incomplete = {b: [name for name in ("base_white.pt", "independent_eta.pt", "noise_metadata.json")
                      if not (root / b / name).is_file()] for b in expected & actual}
    incomplete = {b: names for b, names in incomplete.items() if names}
    return {"expected_blocks": sorted(expected), "missing_blocks": sorted(expected - actual),
            "extra_blocks": sorted(actual - expected), "incomplete_blocks": incomplete,
            "candidate_blocks": sorted(expected & actual - set(incomplete))}


def load_block_batch(run, block_id, contract):
    root = run / "noise_cache" / block_id
    metadata = read_json(root / "noise_metadata.json")
    base = torch.load(root / "base_white.pt", map_location="cpu", weights_only=True)
    eta = torch.load(root / "independent_eta.pt", map_location="cpu", weights_only=True)
    if not isinstance(base, torch.Tensor) or not isinstance(eta, torch.Tensor):
        raise RuntimeError(f"Cache {block_id}: expected tensors")
    if base.ndim != 4 or base.shape != eta.shape or base.shape[0] != len(contract.base_indices) or min(base.shape) < 1:
        raise RuntimeError(f"Cache {block_id}: invalid base/eta shape")
    if base.dtype != torch.float32 or eta.dtype != torch.float32 or not bool(torch.isfinite(base).all() and torch.isfinite(eta).all()):
        raise RuntimeError(f"Cache {block_id}: non-finite values or unexpected dtype")
    block = contract.blocks[block_id]
    if metadata.get("block_id") != block_id or metadata.get("shape") != list(base.shape) or metadata.get("batch_seed") != block.get("batch_seed"):
        raise RuntimeError(f"Cache {block_id}: metadata shape/identity mismatch")
    for name in ("sample_seeds", "eta_sample_seeds", "base_hashes", "eta_hashes"):
        if not isinstance(metadata.get(name), list) or len(metadata[name]) != base.shape[0]:
            raise RuntimeError(f"Cache {block_id}: incomplete {name}")
    if any(type(seed) is not int for name in ("sample_seeds", "eta_sample_seeds") for seed in metadata[name]):
        raise RuntimeError(f"Cache {block_id}: invalid seed metadata")
    if "batch_seed" in block and metadata["sample_seeds"] != [int(block["batch_seed"]) + i for i in contract.base_indices]:
        raise RuntimeError(f"Cache {block_id}: sample seed/index mismatch")
    master = contract.config["experiment"].get("master_seed")
    if master is not None and metadata["eta_sample_seeds"] != [derived_seed(int(master), block_id, "independent_eta", i) for i in contract.base_indices]:
        raise RuntimeError(f"Cache {block_id}: eta seed/index mismatch")
    if [tensor_hash(x) for x in base] != metadata["base_hashes"] or [tensor_hash(x) for x in eta] != metadata["eta_hashes"]:
        raise RuntimeError(f"Cache {block_id}: raw cache hash mismatch; formal validation stopped")
    return NoiseBatch(base, eta, metadata["sample_seeds"], metadata["eta_sample_seeds"], metadata["base_hashes"], metadata["eta_hashes"])


def generation_evidence(run, block_id, condition, final, batch, contract):
    paths = list((run / "generations").glob(f"*/{block_id}/{condition.identifier}/samples.jsonl"))
    evidence = {"hash_status": "unavailable", "sample_index_status": "unavailable", "sample_count": final.shape[0],
                "hash_checked_samples": 0, "hash_matched_samples": 0, "stats_status": "unavailable", "confirmed_recorded_tensor": False, "source_hashes": {}}
    if len(paths) > 1:
        return {**evidence, "sample_index_status": "failed", "reason": "ambiguous model records"}
    if not paths:
        evidence["reason"] = "samples.jsonl missing"
        return evidence
    path = paths[0]
    evidence["source_hashes"][str(path.relative_to(run))] = file_hash(path)
    rows = read_jsonl(path)
    try:
        indices = [r["base_index"] for r in rows]
        if any(type(i) is not int for i in indices) or sorted(indices) != list(contract.base_indices):
            raise ValueError(f"missing/duplicate/extra sample indices: {indices}")
        block = contract.blocks[block_id]
        for row in rows:
            validate_condition_metadata(row, context=str(path))
            if row["condition_id"] != condition.identifier or row.get("block_id") != block_id or any(row.get(k) != block[k] for k in ("prompt_id", "prompt")):
                raise ValueError("sample identity mismatch")
            i = row["base_index"]
            for key, expected in (("base_noise_hash", batch.base_hashes[i]), ("eta_noise_hash", batch.eta_hashes[i]),
                                  ("normalization_profile", contract.config["experiment"]["normalization_profile"]),
                                  ("sample_seed", batch.sample_seeds[i])):
                if key in row and row[key] != expected:
                    raise ValueError(f"generation record {key} mismatch")
        evidence["sample_index_status"] = "passed"
        for row in rows:
            if row.get("final_noise_hash"):
                evidence["hash_checked_samples"] += 1
                evidence["hash_matched_samples"] += int(row["final_noise_hash"] == tensor_hash(final[row["base_index"]]))
        checked, matched = evidence["hash_checked_samples"], evidence["hash_matched_samples"]
        evidence["hash_status"] = "failed" if matched < checked else "passed" if checked == len(indices) else "partial" if checked else "unavailable"
    except (ValueError, KeyError, RuntimeError, TypeError) as exc:
        evidence.update(sample_index_status="failed", reason=str(exc))
    stats_path = path.parent / "noise_statistics.json"
    if stats_path.is_file():
        evidence["source_hashes"][str(stats_path.relative_to(run))] = file_hash(stats_path)
        comparison = stats_comparison(read_json(stats_path).get("post_normalization", []), noise_statistics(final))
        evidence["stats_comparison"] = comparison
        evidence["stats_status"] = comparison["status"]
    if evidence["hash_status"] == "failed" and evidence["stats_status"] == "passed":
        evidence["interpretation"] = "汇总统计一致，但无法据此确认张量逐元素一致；跨环境数值差异是可能原因之一。"
    evidence["confirmed_recorded_tensor"] = evidence["hash_status"] == "passed" and evidence["sample_index_status"] == "passed"
    return evidence


class Accumulator:
    """Only condition-level sums persist; no cached blocks or sample tensors retained."""
    def __init__(self, layout):
        self.layout = layout
        self.n = 0
        self.curves = {k: torch.zeros(layout.bins, dtype=torch.float64) for k in ("raw", "final", "prediction_without_cross", "cross", "identity_prediction", "empirical_minus_prediction", "identity_residual")}
        self.cross = torch.zeros_like(layout.r, dtype=torch.complex128)
        self.xx = torch.zeros_like(layout.r); self.yy = torch.zeros_like(layout.r)
        self.scalar = {k: 0. for k in ("mean", "std", "l2_norm", "cosine_to_base", "low_frac", "mid_frac", "high_frac", "raw_dc_power", "final_dc_power")}

    def add(self, raw_fft, final, final_fft, reference, reference_fft, prediction, cross):
        layout = self.layout
        n = final.shape[0] * final.shape[1]
        power = final_fft.abs().square()
        for key, value in (("raw", raw_fft.abs().square()), ("final", power), ("prediction_without_cross", prediction),
                           ("cross", cross), ("identity_prediction", prediction + cross),
                           ("empirical_minus_prediction", raw_fft.abs().square() - prediction),
                           ("identity_residual", raw_fft.abs().square() - prediction - cross)):
            self.curves[key] += layout.shell(value).sum((0, 1))
        self.cross += (final_fft.to(torch.complex128) * reference_fft.to(torch.complex128).conj()).sum((0, 1))
        self.xx += power.double().sum((0, 1)); self.yy += reference_fft.abs().square().double().sum((0, 1))
        flat, ref = final.double().flatten(-2), reference.double().flatten(-2)
        for key, value in (("mean", flat.mean(-1)), ("std", flat.std(-1, unbiased=False)),
                           ("l2_norm", flat.norm(dim=-1)), ("cosine_to_base", (flat * ref).sum(-1) / (flat.norm(dim=-1) * ref.norm(dim=-1)).clamp_min(EPS)),
                           ("raw_dc_power", raw_fft[..., 0, 0].abs().square()), ("final_dc_power", power[..., 0, 0])):
            self.scalar[key] += float(value.double().sum())
        weighted = power.double() * layout.weights
        total = weighted.sum((-1, -2)).clamp_min(EPS)
        r, top = layout.r, float(layout.r.max())
        for key, mask in (("low_frac", r <= top/3), ("mid_frac", (r > top/3) & (r <= 2*top/3)), ("high_frac", r > 2*top/3)):
            self.scalar[key] += float((weighted[..., mask].sum(-1) / total).sum())
        self.n += n

    def summary(self):
        if not self.n:
            raise ValueError("Empty accumulation")
        denom = float((self.layout.weights * self.xx * self.yy).sum())
        return {**{k: v / self.n for k, v in self.scalar.items()}, "sample_channel_count": self.n,
                "coherence_to_base": float((self.layout.weights * self.cross.abs().square()).sum()) / denom if denom > 0 else None}


def combined_status(statuses):
    if "failed" in statuses:
        return "failed"
    if not statuses or all(s == "unavailable" for s in statuses):
        return "unavailable"
    return "passed" if all(s == "passed" for s in statuses) else "partial"


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def run_validation(run_id=None, alpha_spec=None, gamma_spec=None, same_phase_gamma_spec=None,
                   independent_gamma_spec=None, *, run_dir=None, options=None, allow_partial=False, num_bins=NUM_RADIAL_BINS):
    run = resolve_run(run_id, run_dir)
    output = new_output(run, "frequency_validation")
    summary = {"validation_code_version": VALIDATION_VERSION, "output_format_version": 3,
               "plot_style": "original noise_validation.py",
               "plot_label_aliases": {"equivalent baseline alpha": "nearest_baseline_alpha on the discrete run grid"},
               "legacy_region_caption": "retained original title text; does not establish preregistration or a configured region", "run_dir": str(run),
               "code_hashes": {name: file_hash(Path(__file__).parent / name) for name in
                               ("noise_validation.py", "noise_methods.py", "run_support.py", "result_integrity.py")},
               "out_dir": str(output), "status": "unavailable", "allow_partial": allow_partial,
               "source_hashes": {name: file_hash(run / name) for name in ("run_manifest.json", "blocks.jsonl") if (run / name).exists()},
               "checks": {}, "thresholds": {"complex_and_power_identity_rtol": EXACT_RTOL, "atol": EXACT_ATOL,
                                             "complex_roundoff_floor": "8*float32_eps*per-sample/channel RMS Fourier amplitude",
                                             "power_identity_rtol_scale": "component prediction + abs(actual cross term), computed in float64",
                                             "power_absolute_bound": "atol + 2*abs(expected_F)*complex_tolerance + complex_tolerance^2 (propagated Fourier roundoff)",
                                             "saved_statistics_relative_tolerance": NUMERIC_HASH_TOLERANCE,
                                             "psd_shape_pass_threshold": None}}
    try:
        contract = load_run_contract(run)
        config = contract.config
        summary["resolved_config"] = config
        summary["source_run_manifest"] = read_json(run / "run_manifest.json")
        experiment = config["experiment"]
        summary["noise_construction_version"] = experiment.get("noise_construction_version")
        if experiment.get("noise_construction_version") != NOISE_CONSTRUCTION_VERSION or experiment.get("noise_frequency_grid") != NOISE_FREQUENCY_GRID_VERSION:
            summary["checks"]["construction_version"] = {"status": "unavailable", "reason": "unsupported noise construction/frequency version"}
            raise RuntimeError("Unsupported frozen noise construction/frequency version; refusing to guess old Design B")
        summary["checks"]["construction_version"] = {"status": "passed"}
        profile = experiment["normalization_profile"]
        grid = conditions(config)
        options = dict(options or {})
        for key, value in (("plot_alpha_range", alpha_spec), ("plot_gamma_range", gamma_spec),
                           ("plot_same_gamma_range", same_phase_gamma_spec), ("plot_independent_gamma_range", independent_gamma_spec)):
            if value is not None:
                options.setdefault(key, value)
        selection = display_selection(config, options)
        summary["display_selection"] = selection
        inventory = cache_inventory(run, contract.blocks)
        summary["coverage"] = {**inventory, "actual_blocks": [], "coverage_fraction": 0., "omissions": {}}
        coverage = summary["coverage"]
        coverage["omissions"] = {**{b: "missing cache directory" for b in inventory["missing_blocks"]},
                                  **{b: "missing " + ", ".join(v) for b, v in inventory["incomplete_blocks"].items()}}
        incomplete = bool(inventory["missing_blocks"] or inventory["extra_blocks"] or inventory["incomplete_blocks"])
        summary["checks"]["cache_coverage"] = {"status": "partial" if incomplete and allow_partial else "failed" if incomplete else "passed"}
        if incomplete and not allow_partial:
            raise RuntimeError(f"Cache coverage failed: missing={inventory['missing_blocks']}, extra={inventory['extra_blocks']}, incomplete={inventory['incomplete_blocks']}")
        if not inventory["candidate_blocks"]:
            raise RuntimeError("No complete expected cache blocks to validate")
        expected_paths = {(bid, c.identifier) for bid in contract.blocks for c in grid}
        extras = []
        for path in (run / "generations").glob("*/**/samples.jsonl"):
            parts = path.relative_to(run / "generations").parts
            if len(parts) != 5 or (parts[1], "/".join(parts[2:4])) not in expected_paths:
                extras.append(str(path.relative_to(run)))
        summary["checks"]["unexpected_generation_records"] = {"status": "failed" if extras else "passed", "paths": extras}
        accumulators, records, shape = {}, [], None
        summary["block_condition_evidence"] = records
        for bid in inventory["candidate_blocks"]:
            # Corruption is fatal even in partial mode. Partial permits only absent files/blocks.
            batch = load_block_batch(run, bid, contract)
            for name in ("base_white.pt", "independent_eta.pt", "noise_metadata.json"):
                path = run / "noise_cache" / bid / name
                summary["source_hashes"][str(path.relative_to(run))] = file_hash(path)
            if shape is None:
                shape = tuple(batch.base_white.shape)
                layout = SpectrumLayout(*shape[-2:], num_bins)
                summary["spectral_settings"] = layout.settings()
                accumulators = {c: Accumulator(layout) for c in grid}
            elif tuple(batch.base_white.shape) != shape:
                raise RuntimeError(f"Cache {bid}: shape differs across blocks")
            base_fft = torch.fft.rfft2(batch.base_white, norm="backward")
            eta_fft = torch.fft.rfft2(batch.independent_eta, norm="backward")
            reference = normalize(batch.base_white, profile)
            reference_fft = torch.fft.rfft2(reference, norm="backward")
            # Reuse normalized pink FFT for every gamma of an alpha in this block.
            pink_components = {}
            for c in grid:
                raw = construct_raw_noise(batch, c.family, c.alpha, c.gamma, profile)
                final = normalize(raw, profile)
                raw_fft, final_fft = torch.fft.rfft2(raw, norm="backward"), torch.fft.rfft2(final, norm="backward")
                if c.family == "independent_white":
                    key = (c.alpha, c.gamma == 0)
                    if key not in pink_components:
                        component = pink(batch.base_white, c.alpha)
                        if c.gamma != 0:
                            component = normalize(component, profile)
                        pink_components[key] = torch.fft.rfft2(component, norm="backward")
                    component_fft = pink_components[key]
                else:
                    component_fft = pink_filter(shape[-2], shape[-1], c.alpha)
                exact, prediction, cross = construction_checks(raw_fft, base_fft, eta_fft, c, component_fft)
                accumulators[c].add(raw_fft, final, final_fft, reference, reference_fft, prediction, cross)
                evidence = generation_evidence(run, bid, c, final, batch, contract)
                summary["source_hashes"].update(evidence.pop("source_hashes"))
                records.append({"block_id": bid, "condition_id": c.identifier, "exact_construction": exact, **evidence})
                del raw, final, raw_fft, final_fft, prediction, cross, component_fft
            coverage["actual_blocks"].append(bid)
            coverage["coverage_fraction"] = len(coverage["actual_blocks"]) / len(contract.blocks)
            del batch, base_fft, eta_fft, reference, reference_fft, pink_components
        summary["checks"]["raw_cache_integrity"] = {"status": "passed"}
        summary["checks"]["cache_reconstruction"] = {"status": "partial" if incomplete else "passed"}
        for name, key in (("final_noise_bit_exact", "hash_status"), ("saved_summary_statistics", "stats_status"), ("sample_index_coverage", "sample_index_status")):
            summary["checks"][name] = {"status": combined_status([r[key] for r in records])}
        summary["checks"]["recorded_generation_tensor"] = {"status": combined_status([
            "passed" if r["confirmed_recorded_tensor"] else "failed" if r["sample_index_status"] == "failed" or r["hash_status"] == "failed" else "unavailable"
            for r in records])}
        summary["checks"]["exact_construction"] = {"status": combined_status([r["exact_construction"]["status"] for r in records])}
        summary["counts"] = {"unique_conditions": len(grid), "validated_blocks": len(coverage["actual_blocks"]),
                             "block_conditions": len(records), "samples": len(records) * shape[0],
                             "sample_channels": len(records) * shape[0] * shape[1],
                             "hash_checked_samples": sum(r["hash_checked_samples"] for r in records),
                             "hash_matched_samples": sum(r["hash_matched_samples"] for r in records),
                             "block_conditions_with_hashes": sum(r["hash_checked_samples"] > 0 for r in records),
                             "unique_conditions_with_hashes": len({r["condition_id"] for r in records if r["hash_checked_samples"]})}
        summary["block_condition_evidence"] = records
        aggregate, spectral_rows = aggregate_conditions(accumulators)
        summary["checks"]["nearest_baseline_alpha"] = {"status": "passed" if any(c.family == "baseline" for c in grid) and any(c.family != "baseline" for c in grid) else "unavailable", "meaning": "Nearest discrete baseline grid point by normalized log-PSD distance; not distribution/generation equivalence or continuous optimum"}
        summary["checks"]["finite_sample_psd"] = {"status": "passed", "meaning": "diagnostics computed, no empirical-shape correctness threshold"}
        _write_csv(output / "noise_frequency_summary.csv", aggregate)
        _write_csv(output / "radial_psd.csv", spectral_rows)
        failed = any(v["status"] == "failed" for v in summary["checks"].values())
        evidence_incomplete = any(summary["checks"][name]["status"] in ("unavailable", "partial")
                                  for name in ("final_noise_bit_exact", "saved_summary_statistics", "sample_index_coverage"))
        summary["status"] = "failed" if failed else "partial" if incomplete or evidence_incomplete else "passed"
        summary["status_scope"] = "execution/integrity and evidence completeness; see separate checks for tensor identity and diagnostic applicability"
        plot_diagnostics(output, accumulators, aggregate, selection, partial=incomplete)
        return summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        if "raw_cache_integrity" not in summary["checks"]:
            summary["checks"]["raw_cache_integrity"] = {"status": "failed" if "Cache " in str(exc) else "unavailable"}
        raise RuntimeError(f"{exc}; failed validation summary: {output / 'validation_summary.json'}") from exc
    finally:
        write_json(output / "validation_summary.json", summary)


def aggregate_conditions(accumulators):
    if not accumulators:
        return [], []
    layout = next(iter(accumulators.values())).layout
    baseline = {c.alpha: normalize_log_curve(a.curves["final"] / a.n, layout.valid) for c, a in accumulators.items() if c.family == "baseline"}
    baseline_grid = sorted(baseline)
    rows, spectra = [], []
    for c, acc in accumulators.items():
        curves = {k: v / acc.n for k, v in acc.curves.items()}
        final_log = normalize_log_curve(curves["final"], layout.valid)
        raw_log = normalize_log_curve(curves["raw"], layout.valid)
        prediction_log = normalize_log_curve(curves["prediction_without_cross"], layout.valid)
        distances = {a: float((final_log[layout.valid] - log[layout.valid]).square().mean()) for a, log in baseline.items()}
        nearest = min(distances, key=lambda a: (distances[a], a)) if distances and c.family != "baseline" else None
        meta = {"condition_id": c.identifier, "family": c.family, "alpha": c.alpha, "gamma": c.gamma}
        rows.append({**meta, **acc.summary(), "nearest_baseline_alpha": nearest,
                     "nearest_baseline_status": "passed" if nearest is not None else "unavailable",
                     "nearest_baseline_distance": distances[nearest] if nearest is not None else None,
                     "nearest_baseline_at_boundary": nearest in (baseline_grid[0], baseline_grid[-1]) if nearest is not None else None,
                     "baseline_search_grid": str(baseline_grid), "distance_definition": "nonempty-shell mean squared normalized log-PSD difference",
                     "raw_prediction_log_psd_mse_diagnostic": float((raw_log[layout.valid] - prediction_log[layout.valid]).square().mean()),
                     "cross_shell_energy_l1_fraction_diagnostic": float((curves["cross"].abs() * layout.counts).sum() / (curves["raw"] * layout.counts).sum().clamp_min(EPS))})
        for i in range(layout.bins):
            valid = bool(layout.valid[i])
            spectra.append({**meta, "bin": i, "left": float(layout.edges[i]), "right": float(layout.edges[i+1]),
                            "full_spectrum_count": float(layout.counts[i]),
                            **{k + "_shell_mean_psd": float(v[i]) if valid else None for k, v in curves.items()},
                            **{k + "_shell_energy": float(v[i] * layout.counts[i]) if valid else None for k, v in curves.items()}})
    return rows, spectra


def plot_diagnostics(output, accumulators, rows, selection, *, partial=False):
    from types import SimpleNamespace
    if not accumulators:
        return
    selected = [c for c in accumulators if selected_condition(c, selection)]
    if not selected:
        return
    layout = next(iter(accumulators.values())).layout
    alphas = {family: tuple(grid["alpha_values"]) for family, grid in selection.items()}
    gammas = {family: tuple(grid.get("gamma_values", ())) for family, grid in selection.items()}
    curves = {c: normalize_log_curve(accumulators[c].curves["final"] / accumulators[c].n, layout.valid) for c in selected}
    views = {c: SimpleNamespace(raw_shell_power_sum=accumulators[c].curves["raw"],
                               sample_count=accumulators[c].n,
                               prediction_shell=accumulators[c].curves["prediction_without_cross"] / accumulators[c].n)
             for c in selected}
    baseline = sorted(c.alpha for c in accumulators if c.family == "baseline")
    ids = {c.identifier for c in selected}
    nearest = {(row["family"], row["alpha"], row["gamma"]): row["nearest_baseline_alpha"] for row in rows
               if row["condition_id"] in ids and row["nearest_baseline_alpha"] is not None}
    coherence = {(row["family"], row["alpha"], row["gamma"]): row["coherence_to_base"] for row in rows
                 if row["condition_id"] in ids and row["coherence_to_base"] is not None}
    _plot_radial_psd_selected(output / "radial_psd_selected.png", curves, float(layout.r.max()), alphas, gammas, partial=partial)
    _plot_theoretical_vs_empirical(output / "theoretical_vs_empirical_psd.png", selected, views, layout.r,
                                  layout.index, layout.counts, float(layout.r.max()), alphas, gammas, partial=partial)
    _plot_equivalent_alpha_heatmap(output / "equivalent_alpha_heatmap.png", nearest, alphas, gammas, baseline, partial=partial)
    _plot_coherence_heatmap(output / "coherence_heatmap.png", coherence, alphas, gammas, partial=partial)


LINESTYLE_CYCLE = ("solid", (0, (4, 1.5)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1, 3, 1)))


def _linestyle_for(index):
    return LINESTYLE_CYCLE[index % len(LINESTYLE_CYCLE)]


def _format_grid_value(value):
    return f"{value:.4f}".rstrip("0").rstrip(".")

def _shell_centers(num_bins: int, r_max: float) -> np.ndarray:
    edges = np.linspace(0.0, r_max + 1e-6, num_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def _save_figure(figure, target: Path, *, partial=False) -> None:
    if partial and figure._suptitle is not None:
        figure._suptitle.set_text("PARTIAL — " + figure._suptitle.get_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=220, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(figure)


def _plot_radial_psd_selected(target: Path, curves: dict[ConditionSpec, torch.Tensor], r_max: float,
                               alphas: tuple[float, ...],
                               gammas: dict[str, tuple[float, ...]], *, partial=False) -> None:
    import matplotlib.pyplot as plt
    conditions = list(curves)
    num_bins = len(next(iter(curves.values())))
    families = [f for f in FAMILIES if any(c.family == f for c in conditions)] or ["baseline"]
    figure, axes = plt.subplots(1, len(families), figsize=(12 if len(families) == 2 else 6, 4.8), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    cmap = plt.colormaps["viridis"]
    x = _shell_centers(num_bins, r_max)
    for axis, family in zip(axes, families):
        family_gammas = gammas.get(family, ())
        family_alphas = alphas.get(family, ()) if isinstance(alphas, dict) else alphas
        baseline_conditions = sorted([c for c in conditions if c.family == "baseline"], key=lambda c: c.alpha)
        for index, bc in enumerate(baseline_conditions):
            axis.plot(x, curves[bc].numpy(), color="0.75", linewidth=0.8, zorder=1,
                      label="baseline (all α)" if index == 0 else None)
        # Only the two extreme baseline curves (alpha=0 and alpha=max) are labelled directly on the
        # plot -- the intermediate ones are visually ordered between them and don't need their own tag.
        for bc in dict.fromkeys((baseline_conditions[0], baseline_conditions[-1])) if baseline_conditions else ():
            axis.annotate(f"α={_format_grid_value(bc.alpha)}", xy=(x[-1], curves[bc].numpy()[-1]),
                          xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color="0.4",
                          annotation_clip=False)
        for alpha in family_alphas:
            for gamma_index, gamma in enumerate(family_gammas):
                match = next((c for c in conditions if c.family == family and c.alpha == alpha and c.gamma == gamma), None)
                if match is None:
                    continue
                color = cmap(0.1 + 0.8 * family_alphas.index(alpha) / max(len(family_alphas) - 1, 1))
                axis.plot(x, curves[match].numpy(), color=color, linestyle=_linestyle_for(gamma_index), linewidth=1.6,
                          label=f"α={_format_grid_value(alpha)}, γ={_format_grid_value(gamma)}", zorder=2)
        axis.set(xlabel="Radial frequency r (FFT-bin units)", ylabel="log normalized PSD" if axis is axes[0] else "",
                 title={"same_phase": "Same-phase", "independent_white": "Independent-white", "baseline": "Baseline"}[family])
        axis.grid(alpha=0.2)
        axis.margins(x=0.14)
        axis.legend(fontsize=6, ncol=2, frameon=False)
    figure.suptitle("Representative radial PSD curves; grey lines are the baseline alpha sweep")
    _save_figure(figure, target, partial=partial)


def _plot_theoretical_vs_empirical(target: Path, conditions: list[ConditionSpec], accumulators: dict[ConditionSpec, Accumulator],
                                    r: torch.Tensor, bin_index_flat: torch.Tensor, counts: torch.Tensor,
                                    r_max: float, alphas: tuple[float, ...],
                                    gammas: dict[str, tuple[float, ...]], *, partial=False) -> None:
    # Overlays every requested alpha x gamma pair in one figure per family: color encodes alpha,
    # linestyle (LINESTYLE_CYCLE) encodes gamma, and empirical vs theoretical is distinguished by
    # weight/opacity only (empirical = full weight, theoretical = thinner and faded) so linestyle
    # stays free to encode gamma. Only the empirical curve carries a legend label -- the faded
    # theoretical curves reuse each empirical curve's own color, so a single generic swatch could
    # never represent them; that distinction is spelled out in the suptitle instead.
    import matplotlib.pyplot as plt
    families = [f for f in FAMILIES if any(c.family == f for c in conditions)] or ["baseline"]
    figure, axes = plt.subplots(1, len(families), figsize=(12 if len(families) == 2 else 6, 4.8), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    cmap = plt.colormaps["plasma"]
    x = _shell_centers(len(next(iter(accumulators.values())).raw_shell_power_sum), r_max)
    for axis, family in zip(axes, families):
        family_gammas = (None,) if family == "baseline" else gammas.get(family, ())
        family_alphas = alphas.get(family, ()) if isinstance(alphas, dict) else alphas
        for alpha_index, alpha in enumerate(family_alphas):
            color = cmap(0.1 + 0.8 * alpha_index / max(len(family_alphas) - 1, 1))
            for gamma_index, gamma in enumerate(family_gammas):
                match = next((c for c in conditions if c.family == family and c.alpha == alpha and c.gamma == gamma), None)
                if match is None:
                    continue
                acc = accumulators[match]
                empirical = normalize_log_curve(acc.raw_shell_power_sum / max(acc.sample_count, 1)).numpy()
                theoretical_shell = acc.prediction_shell
                theoretical = normalize_log_curve(theoretical_shell).numpy()
                linestyle = _linestyle_for(gamma_index)
                axis.plot(x, empirical, color=color, linestyle=linestyle, linewidth=1.6, alpha=1.0,
                          label=f"α={_format_grid_value(alpha)}" + (f", γ={_format_grid_value(gamma)}" if gamma is not None else ""))
                axis.plot(x, theoretical, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.45)
        axis.set(xlabel="Radial frequency r (FFT-bin units)", ylabel="log normalized PSD" if axis is axes[0] else "",
                 title={"same_phase": "Same-phase", "independent_white": "Independent-white", "baseline": "Baseline"}[family])
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6, ncol=2, frameon=False)
    figure.suptitle("Theoretical vs empirical PSD shape, pre-final-normalization; "
                     "color=α, linestyle=γ, solid full-weight=empirical, faded thin=theory")
    _save_figure(figure, target, partial=partial)


def _grid_matrix(mapping: dict[tuple[str, float, float], float], family: str,
                  alphas: list[float], gammas: list[float]) -> np.ndarray:
    return np.array([[mapping.get((family, alpha, gamma), np.nan) for gamma in gammas] for alpha in alphas])


def _draw_primary_region_box(axis, alphas, gammas):
    # A legacy display region is drawn only if it is explicitly frozen in config.
    region = getattr(axis, "_saved_primary_region", None)
    if not region:
        return
    from matplotlib.patches import Rectangle
    if not all(value in alphas for value in region["alpha_range"]) or region["gamma_max"] not in gammas:
        return
    lo, hi = (alphas.index(value) for value in region["alpha_range"])
    axis.add_patch(Rectangle((-.5, lo-.5), gammas.index(region["gamma_max"])+1, hi-lo+1,
                             fill=False, edgecolor="white", linewidth=1.8, zorder=5))


def _heatmap_panel(axis, matrix: np.ndarray, title: str, cmap: str, vmin: float | None, vmax: float | None,
                    alphas: list[float], gammas: list[float]):
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axis.set_xticks(range(len(gammas))); axis.set_xticklabels([_format_grid_value(g) for g in gammas], fontsize=7)
    axis.set_yticks(range(len(alphas))); axis.set_yticklabels([_format_grid_value(a) for a in alphas], fontsize=7)
    axis.set(xlabel="γ", ylabel="α", title=title)
    _draw_primary_region_box(axis, alphas, gammas)
    return image


def _plot_equivalent_alpha_heatmap(target: Path, mapping: dict[tuple[str, float, float], float],
                                    alphas: dict[str, list[float]], gammas: dict[str, list[float]],
                                    baseline_alphas: list[float], *, partial=False) -> None:
    import matplotlib.pyplot as plt
    families = [f for f in FAMILIES if alphas.get(f) and gammas.get(f)] or ["baseline"]
    figure, axes = plt.subplots(1, len(families), figsize=(12 if len(families) == 2 else 6, 5), squeeze=False)
    axes = axes[0]
    for axis, family in zip(axes, families):
        family_alphas, family_gammas = alphas.get(family, []), gammas.get(family, [])
        if not family_alphas or not family_gammas or not mapping:
            axis.text(.5, .5, "Unavailable", transform=axis.transAxes, ha="center")
            axis.set_axis_off()
            continue
        matrix = _grid_matrix(mapping, family, family_alphas, family_gammas)
        image = _heatmap_panel(axis, matrix, "Same-phase" if family == "same_phase" else "Independent-white",
                                "viridis", min(baseline_alphas), max(baseline_alphas),
                                family_alphas, family_gammas)
        figure.colorbar(image, ax=axis, label="equivalent baseline α", shrink=0.85)
    figure.suptitle("Equivalent baseline α by closest normalized log-PSD; white box = primary region")
    _save_figure(figure, target, partial=partial)


def _plot_coherence_heatmap(target: Path, mapping: dict[tuple[str, float, float], float],
                             alphas: dict[str, list[float]], gammas: dict[str, list[float]], *, partial=False) -> None:
    import matplotlib.pyplot as plt
    families = [f for f in FAMILIES if alphas.get(f) and gammas.get(f)] or ["baseline"]
    figure, axes = plt.subplots(1, len(families), figsize=(12 if len(families) == 2 else 6, 5), squeeze=False)
    axes = axes[0]
    for axis, family in zip(axes, families):
        family_alphas, family_gammas = alphas.get(family, []), gammas.get(family, [])
        if not family_alphas or not family_gammas or not mapping:
            axis.text(.5, .5, "Unavailable", transform=axis.transAxes, ha="center")
            axis.set_axis_off()
            continue
        matrix = _grid_matrix(mapping, family, family_alphas, family_gammas)
        image = _heatmap_panel(axis, matrix, "Same-phase" if family == "same_phase" else "Independent-white",
                                "viridis", 0.0, 1.0, family_alphas, family_gammas)
        figure.colorbar(image, ax=axis, label="Fourier coherence with base white", shrink=0.85)
    figure.suptitle("Coherence with base white noise; white box = primary region")
    _save_figure(figure, target, partial=partial)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id"); parser.add_argument("--run-dir")
    parser.add_argument("--allow-partial", action="store_true", help="Permit missing/extra/incomplete cache blocks; never permit corrupt caches")
    parser.add_argument("--radial-bins", type=int, default=NUM_RADIAL_BINS)
    add_display_arguments(parser, include_metrics=False)
    # Compatibility aliases use the same priority and do not constrain validation.
    for flag, dest in (("--alphas", "plot_alpha_range"), ("--gammas", "plot_gamma_range"),
                       ("--same-phase-gammas", "plot_same_gamma_range"), ("--independent-gammas", "plot_independent_gamma_range")):
        parser.add_argument(flag, dest=dest, default=argparse.SUPPRESS,
                            help="Compatibility display alias; family-specific overrides common; default full enabled grids")
    args = parser.parse_args()
    try:
        result = run_validation(args.run_id, run_dir=args.run_dir, options=vars(args),
                                allow_partial=args.allow_partial, num_bins=args.radial_bins)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"{result['status']}: {result['out_dir']}")
    for name, check in result["checks"].items():
        print(f"  {name}: {check['status']}")
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__": main()
