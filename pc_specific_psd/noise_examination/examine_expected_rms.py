#!/usr/bin/env python3
"""CPU-only examination of frozen PCA expected-RMS operators.

run from the repository root:
  .venv-generation/bin/python   pc_specific_psd/noise_examination/examine_expected_rms.py

Reads the existing config, formal basis, calibration and validation. Uses
the repository's authenticated loader and frozen application function.
Does NOT build a basis, recalibrate, change registry files or load a model.
Outputs descriptive diagnostics, not an additional pass/fail or Q-D test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_repo(explicit=None):
    if explicit:
        roots = [Path(explicit).expanduser().resolve()]
    else:
        here = Path(__file__).resolve().parent
        roots = [Path.cwd(), here, *here.parents]
    for root in roots:
        if (root / 'pc_specific_psd/expected_rms.py').is_file() and (root / 'noise_init').is_dir():
            return root.resolve()
    raise ValueError('Repository root not found. Place this script under pc_specific_psd/ or specify --repo-root.')


def relative_root(value, root):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def norm2(x, weights):
    """Full-plane Frobenius norm squared from an rFFT response."""
    return float((x.abs().square() * weights[..., None, None]).sum())


def ratio_sqrt(a, b):
    return math.sqrt(max(a, 0.0) / b) if b > 0 else None


def scalar_residual(response, weights):
    """Best scalar*I independently at EACH frequency; not a radial fit."""
    import torch
    c = response.shape[-1]
    scalar = response.diagonal(dim1=-2, dim2=-1).sum(-1) / c
    residual = response - scalar[..., None, None] * torch.eye(c, dtype=response.dtype)
    return ratio_sqrt(norm2(residual, weights), norm2(response, weights))


def pair_metrics(reference, candidate, control, weights):
    import torch
    ref2 = norm2(reference, weights)
    change2 = norm2(candidate - reference, weights)
    pair2 = norm2(candidate - control, weights)
    sigma_p = candidate @ candidate.conj().transpose(-2, -1)
    sigma_f = control @ control.conj().transpose(-2, -1)
    diff = sigma_p - sigma_f
    c = candidate.shape[-1]
    isotropic = (diff.diagonal(dim1=-2, dim2=-1).sum(-1) / c)[..., None, None]
    isotropic = isotropic * torch.eye(c, dtype=diff.dtype)
    anisotropic = diff - isotropic
    denominator = norm2(sigma_f, weights)
    return {
        'analytic_candidate_vs_reference_l2': ratio_sqrt(change2, ref2),
        'analytic_control_vs_reference_l2': ratio_sqrt(norm2(control - reference, weights), ref2),
        'analytic_pair_l2_over_reference': ratio_sqrt(pair2, ref2),
        'pair_l2_over_candidate_intervention': ratio_sqrt(pair2, change2),
        'covariance_pair_distance': ratio_sqrt(norm2(diff, weights), denominator),
        'covariance_pair_scalar_component': ratio_sqrt(norm2(isotropic, weights), denominator),
        'covariance_pair_channel_component': ratio_sqrt(norm2(anisotropic, weights), denominator),
        'covariance_orthogonal_split_relative_error': abs(
            norm2(diff, weights) - norm2(isotropic, weights) - norm2(anisotropic, weights)
        ) / max(norm2(diff, weights), 1e-30),
    }


def empirical_pair(codec, candidate, control, bank, batch_size, injection_dtype, apply, ref_op):
    """Small raw bank, batched coefficient maps; use exactly the held-out draw order."""
    accum = dict(ref=0., pc=0., fourier=0., pair=0., ref_cast=0., pc_cast=0., pair_cast=0.)
    per_sample = []
    for batch in bank.split(batch_size):
        ref = apply(codec, batch, ref_op)
        pc = apply(codec, batch, candidate)
        fourier = apply(codec, batch, control)
        r, p, f = ref.double(), pc.double(), fourier.double()
        accum['ref'] += float(r.square().sum())
        accum['pc'] += float((p-r).square().sum())
        accum['fourier'] += float((f-r).square().sum())
        accum['pair'] += float((p-f).square().sum())
        denom = r.flatten(1).square().sum(1)
        per_sample.extend(((p-f).flatten(1).square().sum(1) / denom).sqrt().tolist())
        r, p, f = [x.to(injection_dtype).double() for x in (ref, pc, fourier)]
        accum['ref_cast'] += float(r.square().sum())
        accum['pc_cast'] += float((p-r).square().sum())
        accum['pair_cast'] += float((p-f).square().sum())
    return {
        'empirical_candidate_vs_reference_l2': ratio_sqrt(accum['pc'], accum['ref']),
        'empirical_control_vs_reference_l2': ratio_sqrt(accum['fourier'], accum['ref']),
        'empirical_pair_l2_over_reference': ratio_sqrt(accum['pair'], accum['ref']),
        'empirical_pair_l2_over_reference_injection_dtype': ratio_sqrt(accum['pair_cast'], accum['ref_cast']),
        'empirical_candidate_l2_injection_dtype': ratio_sqrt(accum['pc_cast'], accum['ref_cast']),
        'per_sample_pair_l2_over_reference': per_sample,
    }


def pc_diagnostics(codec, basis, reference, shape, indices, gate, bins, spectral, psd_editor):
    import torch
    channels, height, width = shape
    weights = spectral.rfft_conjugate_weights(width).view(1, -1).expand(height, -1)
    h = psd_editor.reference_amplitude_response(height, width).double()
    w = psd_editor.low_frequency_gate(height, width, gate['r_s'], gate['beta']).double()
    gate_weights = weights * (h*w).square()
    kernels = []
    entries = []
    p = codec.patch_size
    center = p // 2

    def restricted(x, selected):
        coefficients = codec.encode(x)
        keep = torch.zeros_like(coefficients)
        keep[:, list(selected)] = coefficients[:, list(selected)]
        return codec.decode_center(keep)

    for index in indices:
        kernel = spectral.linear_operator_frequency_response(
            lambda x: restricted(x, [index]), channels, height, width
        )
        kernels.append(kernel)
    pooled = spectral.linear_operator_frequency_response(
        lambda x: restricted(x, indices), channels, height, width
    )
    ref2 = norm2(reference, weights)
    for label, index, kernel in [
        *[(f'PC{i+1}', i, k) for i, k in zip(indices, kernels)],
        ('B1 pooled', None, pooled),
    ]:
        # Derivative at tau=0 includes the derivative of expected-RMS scale.
        derivative_raw = .5 * (h*w)[..., None, None] * kernel
        projection = float((
            (reference.conj() * derivative_raw).real * weights[..., None, None]
        ).sum()) / ref2
        derivative = derivative_raw - projection * reference
        radial = spectral.expected_radial_psd_from_frequency_response(kernel, width=width, num_bins=bins)
        entry = {
            'pc': label,
            'pc_index_1based': None if index is None else index + 1,
            'center_vector_norm': None,
            'spatial_constant_subspace_fraction': None,
            'relative_eigenvalue_gap_to_next': None,
            'white_input_expected_output_mean_square': radial.total_expected_mean_square,
            'kernel_non_scalar_fraction_all_frequencies': scalar_residual(kernel, weights),
            'kernel_non_scalar_fraction_reference_gate_weighted': scalar_residual(kernel, gate_weights),
            'first_order_l2_per_unit_tau_after_expected_rms': ratio_sqrt(norm2(derivative, weights), ref2),
            'kernel_theoretical_radial_power': radial.power.tolist(),
        }
        if index is not None:
            vector = basis.components[:, index].double().reshape(channels, p, p)
            vnorm2 = float(vector.square().sum())
            entry['center_vector_norm'] = float(vector[:, center, center].norm())
            entry['spatial_constant_subspace_fraction'] = float((vector.sum((-2, -1))/p).square().sum()) / vnorm2
            eig = basis.eigenvalues.double()
            if index + 1 < len(eig) and float(eig[index]) > 0:
                entry['relative_eigenvalue_gap_to_next'] = float((eig[index]-eig[index+1])/eig[index])
        entries.append(entry)
        print(f"  {label}: gate-weighted non-scalar={fmt(entry['kernel_non_scalar_fraction_reference_gate_weighted'])}", flush=True)

    constants = torch.zeros((channels*p*p, channels), dtype=torch.float64)
    for channel in range(channels):
        constants[channel*p*p:(channel+1)*p*p, channel] = 1./p
    overlap = basis.components[:, list(indices)].double().T @ constants
    cosines = torch.linalg.svdvals(overlap).clamp(0, 1)
    summary = {
        'pc_indices_1based': [i+1 for i in indices],
        'sum_of_individual_kernels_vs_pooled_relative_error': ratio_sqrt(norm2(sum(kernels)-pooled, weights), norm2(pooled, weights)),
        'principal_angles_to_channelwise_spatial_constants_degrees': torch.rad2deg(torch.acos(cosines)).tolist(),
        'constant_subspace_average_overlap': float(cosines.square().sum()/min(len(indices), channels)),
        'note': 'Individual PC gains differ from the pooled B1 gain; geometric differences are not evidence of generation quality or semantic roles.',
    }
    return entries, summary


def fmt(x):
    return 'N/A' if x is None else f'{100*x:.4f}%'


def write_csv(path, rows):
    fields = [k for k, v in rows[0].items() if not isinstance(v, (list, dict))]
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output, data):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return 'matplotlib/numpy is unavailable. JSON, CSV and Markdown outputs are still produced; plots are skipped.'
    pairs = data['pairs']
    labels = [f"tau={x['tau']:+g}" for x in pairs]
    x = np.arange(len(pairs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for dx, key, label in [(-.2, 'analytic_candidate_vs_reference_l2', 'PCA vs reference'),
                           (.2, 'analytic_pair_l2_over_reference', 'PCA vs Fourier')]:
        axes[0].bar(x+dx, [100*r[key] for r in pairs], width=.38, label=label)
    axes[0].set_ylabel('Expected paired L2 / reference RMS (%)')
    axes[0].set_title('Same-seed noise movement')
    axes[0].legend(fontsize=8)
    for dx, key, label in [(-.25, 'covariance_pair_distance', 'Total'),
                           (0., 'covariance_pair_scalar_component', 'Scalar spectrum'),
                           (.25, 'covariance_pair_channel_component', 'Channel anisotropy')]:
        axes[1].bar(x+dx, [100*r[key] for r in pairs], width=.24, label=label)
    axes[1].set_ylabel('Covariance distance / Fourier covariance norm (%)')
    axes[1].set_title('Distribution difference: PCA vs Fourier')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=15)
        ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'01_pair_distances.png', dpi=170)
    plt.close(fig)

    edges = np.array(data['binning']['bin_edges'])
    radius = (edges[:-1]+edges[1:])/2
    ref = np.array(data['reference_radial_power'])

    def normalize_log_curve(power):
        # Match the historical white-floor plot: natural log of P / sum(P).
        # Display only: the unweighted sum of annular means is not total energy.
        power = np.asarray(power, dtype=np.float64)
        return np.log(np.maximum(power / max(float(power.sum()), 1e-12), 1e-12))

    displayed = np.concatenate([normalize_log_curve(ref), *[
        normalize_log_curve(pair[key]) for pair in pairs
        for key in ('candidate_radial_power', 'control_radial_power')]])
    span = max(float(np.ptp(displayed)), .1)
    limits = (float(displayed.min()) - .05*span, float(displayed.max()) + .08*span)
    rows = (len(pairs)+1)//2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.8*rows), squeeze=False, constrained_layout=True)
    for ax, pair in zip(axes.flat, pairs):
        ax.plot(radius, normalize_log_curve(ref), color='0.35',
                label='Same-phase white-floor', linewidth=2)
        for key, label, style in [('candidate_radial_power', 'PCA', '-'), ('control_radial_power', 'Fourier', '--')]:
            ax.plot(radius, normalize_log_curve(pair[key]), style, label=label, linewidth=2)
        ax.set_ylim(*limits)
        ax.set_title(f"tau={pair['tau']:+g}: matched annular PSD")
        ax.set_xlabel('Radial frequency r (FFT-bin units)')
        ax.set_ylabel('log normalized PSD')
        ax.legend()
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    fig.savefig(output/'02_radial_psd.png', dpi=170)
    plt.close(fig)

    pcs = data['pcs']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].bar([r['pc'] for r in pcs[:-1]], [r['center_vector_norm'] for r in pcs[:-1]])
    axes[0].set_title('Center synthesis weight')
    axes[0].set_ylabel('Norm of center channel vector')
    axes[1].bar([r['pc'] for r in pcs], [100*(r['kernel_non_scalar_fraction_reference_gate_weighted'] or 0) for r in pcs])
    axes[1].set_title('Departure from scalar channel filtering')
    axes[1].set_ylabel('Gate-weighted response residual (%)')
    axes[1].tick_params(axis='x', rotation=25)
    for r in pcs:
        axes[2].plot(radius, r['kernel_theoretical_radial_power'], label=r['pc'])
    axes[2].set_title('Each projection after Center synthesis')
    axes[2].set_xlabel('Radial frequency r (FFT-bin units)')
    axes[2].set_ylabel('Expected radial power (white input)')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.savefig(output/'03_pc_response.png', dpi=170)
    plt.close(fig)
    return None


def markdown_report(data):
    lines = [
        '# Expected-RMS Noise Diagnostics', '',
        'This report describes noise differences between frozen operators. It does not reselect tau, assess image quality/diversity, or introduce new acceptance thresholds.', '',
        '## Direct PCA-Fourier Comparison', '',
        '| tau | Analytic PCA-reference L2 | Analytic PCA-Fourier L2 | Empirical PCA-Fourier L2 | Pair covariance distance |',
        '|---:|---:|---:|---:|---:|',
    ]
    for r in data['pairs']:
        lines.append(f"| {r['tau']:+g} | {fmt(r['analytic_candidate_vs_reference_l2'])} | {fmt(r['analytic_pair_l2_over_reference'])} | {fmt(r['empirical_pair_l2_over_reference'])} | {fmt(r['covariance_pair_distance'])} |")
    lines += [
        '', 'All L2 columns use reference energy for normalization. Analytic values are sqrt(E||difference||^2 / E||reference||^2), not averages of per-sample L2 ratios.',
        f"Maximum absolute difference between candidate-reference L2 recomputed on the same validation bank and the recorded validation value: {max(r['reproduced_validation_candidate_l2_absolute_error'] for r in data['pairs']):.6g}. This checks numerical reproducibility, not method effectiveness.",
        'Covariance distances are normalized by the full covariance-spectrum Frobenius norm of the Fourier control, including all rFFT conjugate weights.',
        'Same-seed tensor differences and distribution differences are distinct: for example, an orthogonal rotation can change a tensor while preserving a standard Gaussian distribution.',
        'covariance_pair_scalar_component and covariance_pair_channel_component form an orthogonal decomposition: their squared distances sum to the squared total distance. The scalar term can include within-annulus frequency/angular variation; the channel term captures cross-channel correlations and channel power differences. Neither directly proves that a learned PCA basis is better than a random basis.',
        'pair_l2_over_candidate_intervention is a ratio of two distances and may exceed 1. It is not a percentage of PCA-specific contribution or a fraction explained by Fourier filtering.',
        '', '## Actual PC1-4 Responses', '',
        '| PC | Center vector norm | Reference/gate-weighted non-scalar response fraction | First-order intervention per unit tau at tau=0 |',
        '|---|---:|---:|---:|',
    ]
    for r in data['pcs']:
        center = '—' if r['center_vector_norm'] is None else f"{r['center_vector_norm']:.6f}"
        lines.append(f"| {r['pc']} | {center} | {fmt(r['kernel_non_scalar_fraction_reference_gate_weighted'])} | {fmt(r['first_order_l2_per_unit_tau_after_expected_rms'])} |")
    lines += [
        '', 'K_i is obtained by retaining only that PC in the analysis coefficient maps and applying Center reconstruction; it does not assume independent PC noise. B1 pooled reconstructs all four PCs together.',
        'The non-scalar fraction compares K(omega) with the best a(omega)I at each frequency, weighted by |h_ref(r)w(r)|^2, corresponding to the raw editing response near tau=0. A value near zero indicates approximately the same Fourier filter across channels, but does not imply radial symmetry. See the first table for differences from the actual radial control.',
        'The first-order intervention includes the derivative of the global expected-RMS scale. It describes behavior near tau=0 and cannot be extrapolated to large tau.',
        'A small center vector norm may weaken a PC output contribution. A large individual-PC non-scalar fraction does not establish a benefit for generation.',
        '', 'Principal angles between B1 and the subspace of channelwise spatially constant patches (degrees):',
        '`'+json.dumps(data['pc_group_geometry']['principal_angles_to_channelwise_spatial_constants_degrees'])+'`',
        '', 'Small angles indicate that B1 is close to this specific four-dimensional smooth subspace; they are not a general score of semantic or structural capability. Eigenvalue gaps do not replace split-half stability checks.',
        '', '## Interpreting Next Steps', '',
        '- If both PCA-Fourier tensor and covariance differences are small, the current methods are close at the noise level. Investigate whether pooling B1 suppresses differences between its individual directions.',
        '- If tensor differences are substantial but covariance differences are small, distinguish same-seed image changes from changes in the generated distribution.',
        '- If covariance differences are clear, differences beyond the overall annular PSD exist, but their value for generation still requires paired image experiments.',
        '- These are interpretation paths. This script imposes no arbitrary threshold for a sufficiently large difference and does not automatically recommend increasing tau.',
        '', '## Files and Provenance', '',
        'See `noise_examination.json` for complete provenance, hashes, bank details, formulas and arrays. Values in both CSV files are ratios, not percentages.',
        'In `02_radial_psd.png`, each overall PSD curve is displayed as ln(P / sum(P)), using the unweighted sum of annular mean powers to match the historical white-floor plots. This display normalization does not change the raw powers or energy diagnostics. `03_pc_response.png` retains the unnormalized power of each reconstructed PC response to white input.',
    ]
    if data.get('plot_note'):
        lines += ['', data['plot_note']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root')
    parser.add_argument('--config', default='pc_specific_psd/configs/sdxl_turbo_pca_expected_rms.yaml')
    parser.add_argument('--output', help='New output directory; defaults to an automatically timestamped directory.')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('--batch-size and --threads must be positive integers')
    root = find_repo(args.repo_root)
    sys.path.insert(0, str(root))
    import torch
    from pc_specific_psd import basis as basis_module, config, expected_rms, psd_editor, runner, spectral
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    cfg = config.resolve_config(relative_root(args.config, root), 'generate-psd')
    if cfg.psd.calibration_profile != 'expected_rms_v1':
        raise ValueError('This script supports expected_rms_v1 only.')
    operators = runner.load_expected_rms_operators(cfg)
    codec = runner.load_codec(cfg)  # No synthetic fallback; does not load VAE/model.
    basis_path = cfg.resolve_root(cfg.basis.basis_output_path)
    basis = basis_module.load_basis(basis_path)
    if codec.basis.device.type != 'cpu':
        raise ValueError('The basis is not on CPU. This script does not modify the basis; check the saved format of the formal basis.')
    cal_path = cfg.resolve_root(cfg.psd.calibration_result_path)
    val_path = cfg.resolve_root(cfg.psd.validation_result_path)
    cal, val = read_json(cal_path), read_json(val_path)
    pairs = {p.candidate.condition_id: p for p in expected_rms.validate_operator_pairs(tuple(operators.values()))}
    selected = []
    seen = set()
    for candidate_id, control_id in val['valid_operator_pairs']:
        if candidate_id in seen or candidate_id not in pairs or pairs[candidate_id].control.condition_id != control_id:
            raise ValueError('The validation pair manifest does not match the frozen registry.')
        seen.add(candidate_id)
        selected.append(pairs[candidate_id])
        for oid in (candidate_id, control_id):
            row = val['conditions'][oid]
            if not row['numerically_valid'] or not row['preview_eligible'] or row['operator_hash'] != expected_rms.operator_hash(operators[oid]):
                raise ValueError(f'Invalid validation condition or mismatched hash: {oid}')
    ids = [op.condition_id for pair in selected for op in (pair.candidate, pair.control)]
    if not selected or sorted(ids) != sorted(val['preview_condition_ids']):
        raise ValueError('Validation contains no valid pairs, or the preview manifest is inconsistent.')
    indices = tuple(selected[0].candidate.group_indices)
    if indices != (0, 1, 2, 3) or any(tuple(p.candidate.group_indices) != indices for p in selected):
        raise ValueError('This script targets B1=PC1-4. A different grouping was found; check the config.')
    shape = tuple(cal['latent_shape'])
    channels, height, width = shape
    if any(tuple(op.latent_shape) != shape for op in operators.values()):
        raise ValueError('Frozen conditions have inconsistent latent shapes.')
    bins = int(cal['binning']['num_bins'])
    bank_spec = val['bank']
    bank = torch.randn((int(bank_spec['size']), *shape), generator=torch.Generator('cpu').manual_seed(int(bank_spec['seed'])), dtype=torch.float32)
    ref_op = expected_rms.reference_operator(*shape, bins)
    reference = spectral.linear_operator_frequency_response(psd_editor.apply_psd_edit_tau_zero, *shape)
    weights = spectral.rfft_conjugate_weights(width).view(1, -1).expand(height, -1)
    ref_radial = spectral.expected_radial_psd_from_frequency_response(reference, width=width, num_bins=bins)
    injection_dtype = getattr(torch, cal['injection_dtype'])
    data = {
        'diagnostic_version': 'pca_fourier_direct_and_pc_response_v1',
        'status': 'DESCRIPTIVE_ONLY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'provenance': {
            'script_sha256': digest(__file__), 'torch_version': str(torch.__version__),
            'config_path': str(cfg.config_path), 'config_sha256': digest(cfg.config_path),
            'basis_path': str(basis_path), 'basis_sha256': digest(basis_path),
            'calibration_path': str(cal_path), 'calibration_sha256': digest(cal_path),
            'validation_path': str(val_path), 'validation_sha256': digest(val_path),
            'operator_source': cal['code_source'],
            'additional_source_sha256': {name: digest(root/'pc_specific_psd'/name) for name in ('patch_codec.py', 'basis.py', 'config.py')},
        },
        'bank': bank_spec, 'processing_batch_size': args.batch_size, 'threads': args.threads,
        'injection_dtype': cal['injection_dtype'], 'binning': cal['binning'],
        'calibration_exclusions': cal.get('condition_exclusions', []),
        'validation_exclusions': val.get('condition_exclusions', []),
        'reference_radial_power': ref_radial.power.tolist(), 'pairs': [],
    }
    print(f'CPU diagnostics: {len(selected)} pairs, validation bank={len(bank)}, batch={args.batch_size}', flush=True)
    for pair in selected:
        pc = expected_rms.construction_for_frozen_operator(codec, pair.candidate)
        fourier = expected_rms.construction_for_frozen_operator(codec, pair.control)
        metrics = pair_metrics(reference, pc.response, fourier.response, weights)
        metrics.update(empirical_pair(codec, pair.candidate, pair.control, bank, args.batch_size, injection_dtype, expected_rms.apply_frozen_operator, ref_op))
        metrics.update({
            'candidate_id': pair.candidate.condition_id, 'control_id': pair.control.condition_id,
            'tau': pair.candidate.tau, 'scale': pair.candidate.scale,
            'targets': list(pair.candidate.target_relative_l2),
            'validation_target_checks': val['conditions'][pair.candidate.condition_id]['target_checks'],
            'validation_candidate_l2': val['conditions'][pair.candidate.condition_id]['actual_relative_l2'],
            'reproduced_validation_candidate_l2_absolute_error': abs(metrics['empirical_candidate_vs_reference_l2']-val['conditions'][pair.candidate.condition_id]['actual_relative_l2']),
            'candidate_radial_power': pc.radial_psd.power.tolist(),
            'control_radial_power': fourier.radial_psd.power.tolist(),
            'annular_psd_pair_max_relative_error': float(((pc.radial_psd.power-fourier.radial_psd.power).abs()/pc.radial_psd.power.clamp(min=1e-30)).max()),
        })
        data['pairs'].append(metrics)
        print(f"  tau={pair.candidate.tau:+g}: PCA-ref={fmt(metrics['analytic_candidate_vs_reference_l2'])}; PCA-Fourier={fmt(metrics['analytic_pair_l2_over_reference'])}; covariance={fmt(metrics['covariance_pair_distance'])}", flush=True)
    data['pcs'], data['pc_group_geometry'] = pc_diagnostics(codec, basis, reference, shape, indices, cal['gate'], bins, spectral, psd_editor)
    output = relative_root(args.output, root) if args.output else root/'pc_specific_psd/noise_examination'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'pair_summary.csv', data['pairs'])
    write_csv(output/'pc_summary.csv', data['pcs'])
    data['plot_note'] = 'Plots skipped because --no-plots was specified.' if args.no_plots else make_plots(output, data)
    (output/'noise_examination.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'README.md').write_text(markdown_report(data), encoding='utf-8')
    print(f'Done. Output directory: {output}', flush=True)
    print('Read README.md first, then pair_summary.csv and the figures. No tau reselection or changes to existing registries were performed.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, KeyError) as exc:
        print(f'Examination did not complete: {exc}', file=sys.stderr)
        sys.exit(1)
