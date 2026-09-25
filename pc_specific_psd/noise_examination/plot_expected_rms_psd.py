#!/usr/bin/env python3
"""Read-only, CPU-only PSD figures for frozen expected_rms_v1 operators.

Run from any directory:
  .venv-generation/bin/python   pc_specific_psd/noise_examination/plot_expected_rms_psd.py   --all-strengths

Defaults to the largest |tau| VALIDATED pair on each side. --all-strengths
puts all validated doses into panels, still producing four figure files.
No calibration, scale fitting, model loading, or production file mutation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from pc_specific_psd import expected_rms, psd_editor, runner, spectral
from pc_specific_psd.compat_generation import file_hash, read_json
from pc_specific_psd.config import load_config

VERSION = "expected_rms_four_psd_figures_v1"
COLORS = {"reference": "#555555", "plus": "#C44E52", "minus": "#2878B5", "fourier": "#D89000"}


def annular_per_pc(response, width, num_bins):
    """E[|FFT(map)|^2]/(H*W), per PC, including rFFT multiplicities."""
    height, wf, d, _ = response.shape
    _, bins = spectral.radial_bin_index(height, width, num_bins, rfft=True)
    weights = spectral.rfft_conjugate_weights(width).expand(height, wf)
    counts = torch.zeros(num_bins, dtype=torch.float64).scatter_add_(
        0, bins.flatten(), weights.flatten())
    power = response.to(torch.complex128).abs().square().sum(-1).permute(2, 0, 1)
    sums = torch.zeros((d, num_bins), dtype=torch.float64).scatter_add_(
        1, bins.flatten().expand(d, -1), (power * weights).reshape(d, -1))
    return sums / counts.clamp_min(1), counts


def edited_maps(codec, white, operator=None):
    """Actual pre-Center edit stage, with the frozen final scalar included."""
    height, width = white.shape[-2:]
    coeff = codec.encode(white)
    spectrum = torch.fft.rfft2(coeff)
    spectrum = spectrum * psd_editor.reference_amplitude_response(height, width).to(coeff.dtype)
    if operator is not None:
        transfer = psd_editor.group_transfer_multiplier(
            height, width, operator.gate_r_s, operator.gate_beta, operator.tau).to(coeff.dtype)
        spectrum[:, list(operator.group_indices)] *= transfer
    result = torch.fft.irfft2(spectrum, s=(height, width)).to(coeff.dtype)
    return result if operator is None else result * operator.scale


def five_rows(pc_power):
    # An average of 96 PCs, NOT a selected representative or their sum.
    return torch.cat((pc_power[:4], pc_power[4:].mean(0, keepdim=True)), dim=0).numpy()


def authenticate(cfg):
    if cfg.psd.calibration_profile != "expected_rms_v1":
        raise ValueError("Use sdxl_turbo_pca_expected_rms.yaml, not the old v1/v2 config.")
    operators = runner.load_expected_rms_operators(cfg)
    cal_path = cfg.resolve_root(cfg.psd.calibration_result_path)
    val_path = cfg.resolve_root(cfg.psd.validation_result_path)
    val = read_json(val_path)
    required = {
        "status": "PASS", "calibration_profile": "expected_rms_v1",
        "energy_constraint": "expected_rms", "config_hash": file_hash(cfg.config_path),
        "basis_hash": runner.basis_hash_for(cfg), "calibration_hash": file_hash(cal_path),
    }
    for key, value in required.items():
        if val.get(key) != value:
            raise ValueError(f"Validation is missing/stale: {key}. Restore matching artifacts; do not bypass hashes.")
    canonical = {(p.candidate.condition_id, p.control.condition_id): p
                 for p in expected_rms.validate_operator_pairs(tuple(operators.values()))}
    ids = val.get("valid_operator_pairs", [])
    if not ids or any(not isinstance(p, list) or len(p) != 2 for p in ids):
        raise ValueError("Validation contains no well-formed candidate/control pairs.")
    flat = [cid for p in ids for cid in p]
    if len(flat) != len(set(flat)) or flat != val.get("preview_condition_ids"):
        raise ValueError("Validation pair and preview manifests disagree.")
    pairs = []
    for pair in ids:
        if tuple(pair) not in canonical:
            raise ValueError(f"Unknown validated pair: {pair}")
        for cid in pair:
            if not val.get("conditions", {}).get(cid, {}).get("numerically_valid", False):
                raise ValueError(f"Condition did not pass validation: {cid}")
        pairs.append(canonical[tuple(pair)])
    return pairs, val


def collect(codec, pair, reference, pc_reference):
    op = pair.candidate
    c, h, w = op.latent_shape
    if tuple(op.group_indices) != (0, 1, 2, 3) or codec.d <= 4:
        raise ValueError("These PC1-4/Other-PC figures require the frozen B1 group [0,1,2,3].")
    candidate = expected_rms.construction_for_frozen_operator(codec, op)
    control = expected_rms.construction_for_frozen_operator(codec, pair.control)
    pc_response = spectral.linear_operator_frequency_response(
        lambda x: edited_maps(codec, x, op), c, h, w)
    pc_power, counts = annular_per_pc(pc_response, w, op.num_bins)

    # Check the displayed edit stage reconstructs the actual shared operator.
    generator = torch.Generator().manual_seed(74129)
    white = torch.randn((1, c, h, w), generator=generator)
    actual = expected_rms.apply_frozen_operator(codec, white, op)
    reconstructed = codec.decode_center(edited_maps(codec, white, op))
    torch.testing.assert_close(reconstructed, actual, rtol=3e-5, atol=3e-6)
    p_ref = reference.radial_psd.power.numpy()
    p_pca = candidate.radial_psd.power.numpy()
    p_fourier = control.radial_psd.power.numpy()
    weights = counts.numpy()
    energy = [float(np.dot(p, weights) / (h * w)) for p in (p_ref, p_pca, p_fourier)]
    match = float(np.max(np.abs(p_fourier - p_pca) / np.maximum(p_pca, 1e-30)))
    # Existing deterministic match tolerance; no new efficacy threshold.
    if match > expected_rms.DEFAULT_ANNULAR_MATCH_TOLERANCE:
        raise ValueError(f"Frozen Fourier match no longer reproduces: relative error {match:g}")
    if not math.isclose(energy[1], op.expected_mean_square, rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError("Recomputed candidate energy disagrees with the frozen operator.")
    edges = np.array(spectral.radial_binning_metadata(h, w, op.num_bins)["bin_edges"])
    return {
        "condition_id": op.condition_id, "control_id": pair.control.condition_id,
        "tau": op.tau, "scale": op.scale, "target_relative_l2": list(op.target_relative_l2),
        "radius": (edges[:-1] + edges[1:]) / 2, "counts": weights,
        "reference": p_ref, "pca": p_pca, "fourier": p_fourier,
        "pc_reference": pc_reference.numpy(), "pc_pca": pc_power.numpy(),
        "five_reference": five_rows(pc_reference), "five_pca": five_rows(pc_power),
        "mean_square": energy, "energy_ratio": [v / energy[0] for v in energy],
        "max_annular_match_relative_error": match,
        "cumulative_reference": np.cumsum(weights * p_ref) / (h * w * energy[0]),
        "cumulative_pca": np.cumsum(weights * p_pca) / (h * w * energy[0]),
        "cumulative_fourier": np.cumsum(weights * p_fourier) / (h * w * energy[0]),
    }


def save_figure(fig, path, dpi):
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalize_log_curve(power):
    """Historical white-floor display: ln(P / sum of annular mean powers).

    Display only. This unweighted sum is not total latent energy.
    Raw powers remain unchanged for energy diagnostics and ratio panels.
    """
    power = np.asarray(power, dtype=np.float64)
    normalized = power / max(float(power.sum()), 1e-12)
    return np.log(np.maximum(normalized, 1e-12))


def overall_figure(rows, sign, path, dpi, limits):
    fig, axes = plt.subplots(2, len(rows), figsize=(8 * len(rows), 8),
                             squeeze=False, sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    for col, row in enumerate(rows):
        ax, ratio_ax = axes[:, col]
        r = row["radius"]
        ax.plot(r, normalize_log_curve(row["reference"]), color=COLORS["reference"], lw=2.4,
                label="Same-phase white-floor")
        ax.plot(r, normalize_log_curve(row["pca"]), color=COLORS[sign], lw=2.7, label="PCA candidate")
        ax.plot(r, normalize_log_curve(row["fourier"]), color=COLORS["fourier"], lw=1.9, ls="--",
                label="Matched Fourier control")
        ax.set_ylim(*limits)
        ax.set_ylabel("log normalized PSD")
        ax.set_title(f"{sign.capitalize()}: tau = {row['tau']:+g}")
        ax.legend(loc="upper right", fontsize=10)
        ratios = row["energy_ratio"]
        ax.text(.98, .48, "Expected total energy / white-floor\n"
                f"White-floor: {ratios[0]:.8f}\nPCA: {ratios[1]:.8f}\nFourier: {ratios[2]:.8f}\n"
                f"Max annular match error: {row['max_annular_match_relative_error']:.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=.92, edgecolor="#DDDDDD"))
        ratio_ax.axhline(1, color=COLORS["reference"], lw=1.3)
        ratio_ax.plot(r, row["pca"] / row["reference"], color=COLORS[sign], lw=2.2)
        ratio_ax.plot(r, row["fourier"] / row["reference"], color=COLORS["fourier"], ls="--", lw=1.6)
        ratio_ax.set_ylabel("Power / white-floor")
        ratio_ax.set_xlabel("Radial frequency r (FFT-bin units)")
        for a in (ax, ratio_ax):
            a.grid(alpha=.2)
            a.set_xlim(0, float(r[-1]) + float(r[0]))
    fig.suptitle("Final latent PSD: equal expected energy, different spectral allocation", fontsize=15)
    fig.text(.5, .015, "Same-phase alpha=0.9, gamma=0.05; fixed expected-RMS scaling. "
             "Full-grid theoretical PSD before injection-dtype cast; DC included.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .95))
    save_figure(fig, path, dpi)


def pc_figure(rows, sign, path, dpi, zmode, limits, elev, azim):
    fig = plt.figure(figsize=(11 * len(rows), 8.5))
    labels = ["PC1", "PC2", "PC3", "PC4", "Other PCs"]
    for col, row in enumerate(rows):
        ax = fig.add_subplot(1, len(rows), col + 1, projection="3d")
        for pc in range(5):
            for key, color, style, lw in (("five_reference", COLORS["reference"], "--", 1.8),
                                           ("five_pca", COLORS[sign], "-", 2.2)):
                power = row[key][pc]
                z = np.log10(np.maximum(power, 1e-30)) if zmode == "log10" else power
                ax.plot(row["radius"], np.full(len(z), pc), z, color=color, ls=style, lw=lw)
        ax.set_yticks(range(5), labels)
        ax.set_xlabel("Radial frequency r (FFT-bin units)", labelpad=12)
        ax.set_ylabel("PC / group", labelpad=16)
        ax.set_zlabel("log10 expected power" if zmode == "log10" else "Expected power", labelpad=12)
        ax.set_zlim(*limits)
        ax.set_xlim(0, float(row["radius"][-1]) + float(row["radius"][0]))
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        ax.set_box_aspect((1.7, 1.25, 1))
        ax.tick_params(axis="y", labelsize=9)
        ax.set_title(f"{sign.capitalize()}: tau = {row['tau']:+g}; fixed scale = {row['scale']:.6f}", pad=22)
        ax.legend(handles=[Line2D([0], [0], color=COLORS["reference"], ls="--", label="Same-phase white-floor"),
                           Line2D([0], [0], color=COLORS[sign], label="PCA edited maps")],
                  loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=10)
    fig.suptitle("PC-space PSD at the edit stage (before Center reconstruction)", fontsize=15, y=.97)
    fig.text(.5, .035, "Final expected-RMS scalar included. Other PCs = mean of PC5-100, not sum.\n"
             "Maps are correlated: these powers are NOT an additive decomposition of final latent energy.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.02, right=.91, top=.86, bottom=.14, wspace=.18)
    save_figure(fig, path, dpi)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "pc_specific_psd/configs/sdxl_turbo_pca_expected_rms.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--all-strengths", action="store_true", help="Plot both frozen doses per sign in panels.")
    parser.add_argument("--plus-tau", type=float, help="Select an existing positive frozen tau; does not fit it.")
    parser.add_argument("--minus-tau", type=float, help="Select an existing negative frozen tau; does not fit it.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--pc-z", choices=("log10", "linear"), default="log10")
    parser.add_argument("--elev", type=float, default=25)
    parser.add_argument("--azim", type=float, default=-58)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.dpi < 50:
        parser.error("threads must be positive; dpi must be at least 50")
    if args.all_strengths and (args.plus_tau is not None or args.minus_tau is not None):
        parser.error("Use --all-strengths OR explicit tau selectors.")
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    cfg = load_config(args.config.resolve())
    pairs, validation = authenticate(cfg)
    codec = runner.load_codec(cfg)  # formal basis required, no synthetic fallback
    if codec.d != 100 or codec.channels != 4 or codec.patch_size != 5:
        raise ValueError("This plot layout is for the 100-PC, 4-channel, 5x5 patch experiment.")
    selected = {}
    for sign, requested in (("plus", args.plus_tau), ("minus", args.minus_tau)):
        available = sorted([p for p in pairs if (p.candidate.tau > 0) == (sign == "plus")
                            and p.candidate.tau != 0], key=lambda p: abs(p.candidate.tau))
        if requested is not None:
            available = [p for p in available if math.isclose(p.candidate.tau, requested, abs_tol=1e-9)]
        if not available:
            raise ValueError(f"No validated frozen {sign} condition matches the request.")
        selected[sign] = available if args.all_strengths else available[-1:]
    dims = {(p.candidate.latent_shape, p.candidate.num_bins) for ps in selected.values() for p in ps}
    if len(dims) != 1:
        raise ValueError("Selected operators use incompatible latent shapes or annular grids.")
    ((c, h, w), bins) = next(iter(dims))
    if (c, h, w) != (cfg.basis.channels, cfg.generation.config.height // 8, cfg.generation.config.width // 8):
        raise ValueError("Frozen latent shape differs from config.")
    ref_op = expected_rms.reference_operator(c, h, w, bins)
    reference = expected_rms.construction_for_frozen_operator(codec, ref_op)
    pc_ref_response = spectral.linear_operator_frequency_response(lambda x: edited_maps(codec, x), c, h, w)
    pc_reference, _ = annular_per_pc(pc_ref_response, w, bins)
    rows = {}
    for sign, ps in selected.items():
        rows[sign] = []
        for pair in ps:
            print(f"Computing {pair.candidate.condition_id} ...", flush=True)
            rows[sign].append(collect(codec, pair, reference, pc_reference))
    all_rows = [r for rs in rows.values() for r in rs]
    powers = np.concatenate([normalize_log_curve(r[k]) for r in all_rows
                             for k in ("reference", "pca", "fourier")])
    overall_span = max(float(np.ptp(powers)), .1)
    overall_limits = (float(powers.min()) - .05 * overall_span,
                      float(powers.max()) + .08 * overall_span)
    pc_values = np.concatenate([r[k].flatten() for r in all_rows for k in ("five_reference", "five_pca")])
    if args.pc_z == "log10":
        pc_values = np.log10(np.maximum(pc_values, 1e-30))
    span = max(float(np.ptp(pc_values)), .1)
    pc_limits = (float(pc_values.min()) - .05 * span, float(pc_values.max()) + .08 * span)
    if args.pc_z == "linear":
        pc_limits = (0, pc_limits[1])
    output = (args.output_dir or Path(__file__).resolve().parent / "psd_figures").resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("01_plus_overall_psd", "02_minus_overall_psd", "03_plus_pc_psd_3d", "04_minus_pc_psd_3d")
    targets = [output / (n + ext) for n in names for ext in (".png", ".pdf")]
    targets += [output / "psd_plot_data.json", output / "overall_psd.csv"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Plots already exist. Choose a NEW --output-dir; existing files are never overwritten.")
    for sign, name in (("plus", names[0]), ("minus", names[1])):
        overall_figure(rows[sign], sign, output / name, args.dpi, overall_limits)
    for sign, name in (("plus", names[2]), ("minus", names[3])):
        pc_figure(rows[sign], sign, output / name, args.dpi, args.pc_z, pc_limits, args.elev, args.azim)
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = {"script_version": VERSION, "script_sha256": file_hash(Path(__file__)),
                  "git_commit": commit, "config_sha256": file_hash(cfg.config_path),
                  "basis_sha256": runner.basis_hash_for(cfg),
                  "calibration_sha256": file_hash(cfg.resolve_root(cfg.psd.calibration_result_path)),
                  "validation_sha256": file_hash(cfg.resolve_root(cfg.psd.validation_result_path)),
                  "torch_version": torch.__version__, "selection": "all" if args.all_strengths else "strongest or explicit",
                  "pc_z": args.pc_z, "binning": spectral.radial_binning_metadata(h, w, bins)}
    payload = {"provenance": provenance, "reference": "same-phase alpha=0.9 gamma=0.05, fixed expected RMS",
               "measurement": "theoretical response PSD, pre-injection-cast; no sample normalization",
               "pc_domain": "edited coefficient maps before Center, final frozen scalar included",
               "power_units": "expected squared unnormalized FFT amplitude divided by H*W",
               "notes": ["PC powers are not additive latent-energy shares.",
                         "Annular PSD matching does not ensure within-annulus angular or channel covariance matching.",
                         "This is not the historical per-sample/per-channel-normalized white-floor."],
               "conditions": all_rows}
    (output / "psd_plot_data.json").write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n")
    with (output / "overall_psd.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition_id", "tau", "radius", "count", "whitefloor", "pca", "fourier",
                         "pca_over_whitefloor", "fourier_over_whitefloor"])
        for row in all_rows:
            for r, n, a, b, f in zip(row["radius"], row["counts"], row["reference"], row["pca"], row["fourier"]):
                writer.writerow([row["condition_id"], row["tau"], r, n, a, b, f, b/a, f/a])
    print(f"Saved four PNG + four PDF figures and numerical data to: {output}")
    for row in all_rows:
        print(f"tau={row['tau']:+g}: energy ratios={row['energy_ratio']}, "
              f"max annular error={row['max_annular_match_relative_error']:.3e}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, runner.RunnerError) as exc:
        raise SystemExit(f"PSD plotting stopped: {exc}") from exc
