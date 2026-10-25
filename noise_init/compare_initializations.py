#!/usr/bin/env python3
"""Explicit, read-only source selection and paired categorical comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from coupled_run import _version
from io_utils import (canonical_json, file_hash, read_json, read_jsonl, sha256_text,
                      write_json)
from metric_runner import _read_csv, _write_csv
from result_integrity import (enabled_metric_names, load_generation_records,
                              validate_analysis_inputs)
from run_support import conditions, input_hashes

BOOTSTRAP = {'replicates': 2000, 'seed': 20261009, 'confidence_level': .95,
             'unit': 'prompt', 'interval': 'percentile', 'difference': 'method minus reference'}
METRICS = ('hpsv3', 'clip_cosine', 'dreamsim_mean_pair_distance',
           'lpips_alex_mean_pair_distance', 'vendi_clip')
ROOT = Path(__file__).resolve().parent


def generation_settings(config):
    return {'model': {k: config['model'].get(k) for k in
            ('adapter', 'checkpoint', 'vae_checkpoint', 'revision', 'dtype')},
            'generation': {k: config['generation'].get(k) for k in
            ('height', 'width', 'num_inference_steps', 'guidance_scale', 'generation_batch_size', 'output_format')},
            'gallery_size': config['experiment']['gallery_size']}


def block_identity(blocks):
    keys = ('block_id', 'prompt_id', 'prompt', 'category', 'seed_batch_id', 'batch_seed')
    return [{k: b.get(k) for k in keys} for b in blocks]


def paired_statistics(values, references):
    """values[method][metric] contains the same ordered prompt array everywhere."""
    first = next(iter(values.values()))
    count = len(next(iter(first.values())))
    indices = np.random.default_rng(BOOTSTRAP['seed']).integers(0, count, (BOOTSTRAP['replicates'], count))
    summary, differences = [], []
    for name, metrics in values.items():
        for metric, data in metrics.items():
            data = np.asarray(data, dtype=float)
            if data.shape != (count,) or not np.isfinite(data).all():
                raise ValueError('Bootstrap requires finite complete prompt coverage')
            low, high = np.quantile(data[indices].mean(axis=1), [.025, .975])
            summary.append({'method': name, 'metric': metric, 'mean': float(data.mean()),
                            'ci_low': float(low), 'ci_high': float(high), 'prompt_count': count})
            for reference in references:
                if reference == name:
                    continue
                delta = data - np.asarray(values[reference][metric])
                low, high = np.quantile(delta[indices].mean(axis=1), [.025, .975])
                differences.append({'method': name, 'reference': reference, 'metric': metric,
                                    'mean_difference': float(delta.mean()), 'ci_low': float(low),
                                    'ci_high': float(high), 'prompt_count': count})
    return summary, differences


def _verify_aggregates(root, selected):
    """Validate stored means against every image and unordered pair, not just n."""
    image = _read_csv(root / 'metrics/per_image.csv')
    pairs = _read_csv(root / 'metrics/per_pair.csv')
    raw = {}
    for row in image + pairs:
        if row['condition_id'] in selected:
            raw.setdefault((row['block_id'], row['condition_id'], row['metric']), []).append(float(row['score']))
    for row in _read_csv(root / 'metrics/per_group.csv'):
        key = (row['block_id'], row['condition_id'], row['metric'])
        if key in raw and not np.isclose(float(row['score']), np.mean(raw[key]), atol=1e-10, rtol=1e-10):
            raise RuntimeError(f'Group mean disagrees with individual scores: {root}/{key}')


def compare(spec, *, spec_dir=Path('.'), output_dir=None, make_plots=True):
    scope = spec.get('scope', 'formal')
    if scope not in ('formal', 'smoke', 'single_run_summary'):
        raise ValueError('scope must be formal, smoke, or single_run_summary')
    selections = spec.get('selections', [])
    if not selections:
        raise ValueError('Explicit run and condition selections are required')
    names = [s['name'] for s in selections]
    logical = [s['condition'] for s in selections]
    if len(set(names)) != len(names) or len(set(logical)) != len(logical):
        raise ValueError('Duplicate logical selections or method labels')
    references = spec.get('references', [])
    if not references or not set(references) <= set(names) or len(set(references)) != len(references):
        raise ValueError('references must explicitly name distinct selected methods')
    if scope == 'formal' and not any(c.startswith(('same_phase/', 'independent_white/')) for c in logical):
        raise ValueError('Formal comparison requires explicit prespecified white-floor condition selection')
    roots = [(spec_dir / s['run']).resolve() for s in selections]
    output = Path(output_dir).resolve() if output_dir else (spec_dir / spec['output']).resolve()
    if any(output.is_relative_to(r) or r.is_relative_to(output) for r in roots):
        raise ValueError('Comparison output must be separate from every source run')
    expected_path = (ROOT / 'manifests/blocks_geneval400_1group.jsonl' if scope == 'formal' else
                     (spec_dir / spec['expected_manifest']).resolve())
    expected = read_jsonl(expected_path)
    if not expected or (scope == 'formal' and len(expected) != 400):
        raise ValueError('Expected manifest is missing or has incorrect formal coverage')
    if len({b['prompt_id'] for b in expected}) != len(expected):
        raise ValueError('Comparison requires exactly one seed group per prompt')
    audits, contracts, generation_rows = {}, {}, {}
    settings, metric_identity = None, None
    known_runtimes, known_preprocessing = [], []
    for root in dict.fromkeys(roots):
        selected = [s['condition'] for s, r in zip(selections, roots) if r == root]
        contract = validate_analysis_inputs(root, condition_ids=selected)
        manifest = read_json(root / 'run_manifest.json')
        if manifest.get('config_hash') != sha256_text(canonical_json(manifest['resolved_config'])):
            raise RuntimeError(f'Corrupt run config hash: {root}')
        if block_identity(list(contract.blocks.values())) != block_identity(expected):
            raise RuntimeError(f'Prompt/category/seed coverage or manifest order differs: {root}')
        current = generation_settings(contract.config)
        if settings is not None and current != settings:
            raise RuntimeError(f'Incompatible generator settings: {root}')
        settings = current
        metrics = enabled_metric_names(contract.config)
        if scope == 'formal' and set(sum(metrics, ())) != set(METRICS):
            raise RuntimeError('Formal comparisons require all five core metrics')
        metric_manifest = read_json(root / 'metrics/metric_manifest.json')
        identity = {'quality_metrics': contract.config['quality_metrics'],
                    'diversity_metrics': contract.config['diversity_metrics'], **metric_manifest}
        if metric_identity is not None and identity != metric_identity:
            raise RuntimeError(f'Incompatible metric settings/versions/hash: {root}; re-evaluate its selected images in a separate evaluation output')
        metric_identity = identity
        _verify_aggregates(root, selected)
        rows = [r for r in load_generation_records(root) if r['condition_id'] in selected]
        from coupled_sampling import enabled
        from run_experiment import _generation_hash
        if len({r.get('generation_config_hash') for r in rows}) != 1 or not rows[0].get('generation_config_hash'):
            raise RuntimeError(f'Inconsistent sample generator hashes: {root}')
        for row in rows:
            block = contract.blocks[row['block_id']]
            if row['model_id'] != contract.config['model']['adapter']:
                raise RuntimeError(f'Sample model identity differs from frozen adapter: {root}')
            if any(row.get(k) != block.get(k) for k in ('prompt_id', 'prompt', 'category', 'seed_batch_id', 'batch_seed')):
                raise RuntimeError(f'Sample block/seed identity mismatch: {root}')
            if enabled(contract.config) and row.get('generation_config_hash') != _generation_hash(contract.config):
                raise RuntimeError(f'Sample generator hash mismatch: {root}')
            if file_hash(row['image_path']) != row.get('image_hash'):
                raise RuntimeError(f'Source image hash mismatch: {root}')
        runtime_path = root / 'generator_runtime.json'
        runtime = read_json(runtime_path) if runtime_path.exists() else 'unknown'
        if isinstance(runtime, dict) and runtime.get('scheduler_class'):
            known_runtimes.append(runtime)
        preprocessing_path = root / 'metrics/preprocessing_audit.json'
        preprocessing = read_json(preprocessing_path) if preprocessing_path.exists() else 'unknown'
        if isinstance(preprocessing, dict):
            known_preprocessing.append(preprocessing['versions'])
        audits[str(root)] = {'input_hashes': input_hashes(root), 'config_hash': manifest['config_hash'],
            'selected_conditions': selected, 'normalization': contract.config['experiment']['normalization_profile'],
            'generator_runtime': runtime, 'preprocessing_audit': preprocessing,
            'new_generation_provenance': {name: {'sha256': file_hash(root / name), 'record': read_json(root / name)}
                for name in ('coupled_source.json', 'generation_versions.json') if (root / name).exists()},
            'metric_manifest': metric_manifest,
            'sample_record_hashes': {str(p.relative_to(root)): file_hash(p)
                for p in sorted(root.glob('generations/**/samples.jsonl'))
                if any(str(p.parent).endswith('/' + c) for c in selected)}}
        contracts[root], generation_rows[root] = contract, rows
    if known_runtimes and any(r != known_runtimes[0] for r in known_runtimes[1:]):
        raise RuntimeError('Known scheduler/pipeline runtime settings differ between sources')
    if known_preprocessing and any(r != known_preprocessing[0] for r in known_preprocessing[1:]):
        raise RuntimeError('Known preprocessing versions differ; re-evaluate selected images in a separate output')
    values, per_prompt = {}, []
    for selection, root in zip(selections, roots):
        rows = [r for r in _read_csv(root / 'metrics/per_group.csv') if r['condition_id'] == selection['condition']]
        by_key = {(r['block_id'], r['metric']): float(r['score']) for r in rows}
        metrics = sum(enabled_metric_names(contracts[root].config), ())
        values[selection['name']] = {m: np.array([by_key[(b['block_id'], m)] for b in expected]) for m in metrics}
        for block in expected:
            for metric in metrics:
                per_prompt.append({'method': selection['name'], 'condition_id': selection['condition'],
                    'prompt_id': block['prompt_id'], 'block_id': block['block_id'], 'category': block.get('category'),
                    'metric': metric, 'score': by_key[(block['block_id'], metric)]})
    summary, differences = paired_statistics(values, references)
    # All validation precedes output creation; source directories are read-only.
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / 'summary_by_method.csv', summary)
    _write_csv(output / 'per_prompt.csv', per_prompt)
    _write_csv(output / 'paired_method_differences.csv', differences)
    audit = {'scope': scope, 'sources': audits, 'selections': selections, 'references': references,
             'expected_manifest': str(expected_path), 'expected_manifest_hash': file_hash(expected_path),
             'generation_settings': settings, 'bootstrap': BOOTSTRAP, 'prompt_count': len(expected),
             'current_audit_versions': {n: _version(n) for n in ('torch', 'torchvision', 'Pillow')},
             'limitations': ['Unknown historical scheduler/preprocessing versions are not verified identical.',
                 'Intervals reflect prompt variation for one fixed seed group per prompt.',
                 'Raw Gaussian coupling and normalized spectral conditions compare complete designs.',
                 'Perceptual distances are not proof of semantic diversity; a CI containing zero is not equivalence.']}
    write_json(output / 'comparison_manifest.json', audit)
    if make_plots:
        plot_points(summary, output, settings['model']['adapter'])
        save_gallery(selections, roots, generation_rows, expected, output)
    return output


def plot_points(summary, output, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = {(r['method'], r['metric']): r for r in summary}
    names = list(dict.fromkeys(r['method'] for r in summary))
    available = {r['metric'] for r in summary}
    for quality in ('hpsv3', 'clip_cosine'):
        for diversity in ('dreamsim_mean_pair_distance', 'lpips_alex_mean_pair_distance', 'vendi_clip'):
            if not {quality, diversity} <= available:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for name in names:
                q, d = rows[name, quality], rows[name, diversity]
                ax.errorbar(d['mean'], q['mean'],
                    xerr=[[max(0., d['mean']-d['ci_low'])], [max(0., d['ci_high']-d['mean'])]],
                    yerr=[[max(0., q['mean']-q['ci_low'])], [max(0., q['ci_high']-q['mean'])]],
                    fmt='o', capsize=3, label=name)
            ax.set(xlabel=diversity, ylabel=quality, title=f'{model}: prompt means and 95% bootstrap intervals')
            ax.legend(fontsize=8); fig.tight_layout()
            suffix = diversity.replace('_mean_pair_distance', '')
            stem = f'qd_{quality}_{suffix}'
            fig.savefig(output / (stem + '.png'), dpi=180)
            fig.savefig(output / (stem + '.pdf'))
            plt.close(fig)


def save_gallery(selections, roots, records, blocks, output):
    from PIL import Image, ImageDraw
    # First prompt in manifest order for each category; deterministic, never ranked by scores.
    chosen, seen = [], set()
    for b in blocks:
        category = b.get('category', 'uncategorized')
        if category not in seen:
            chosen.append(b); seen.add(category)
    size, label_width = 128, 260
    canvas = Image.new('RGB', (label_width + 4 * size, len(chosen) * len(selections) * (size + 24)), 'white')
    draw = ImageDraw.Draw(canvas)
    y = 0
    for block in chosen:
        for selection, root in zip(selections, roots):
            draw.text((8, y + 8), f"{block['prompt_id']}\n{selection['name']}", fill='black')
            gallery = sorted([r for r in records[root] if r['block_id'] == block['block_id'] and
                              r['condition_id'] == selection['condition']], key=lambda r: r['base_index'])
            for i, row in enumerate(gallery):
                with Image.open(row['image_path']) as im:
                    canvas.paste(im.convert('RGB').resize((size, size)), (label_width + i * size, y))
            y += size + 24
    canvas.save(output / 'gallery_comparison.png')
    write_json(output / 'gallery_selection.json', {'rule': 'first prompt per category in manifest order',
                                                  'blocks': chosen})


def analyze_coupled_run(run_dir):
    """A categorical summary of all conditions, clearly separate from formal comparison."""
    from result_integrity import load_run_contract
    root = Path(run_dir).resolve()
    contract = load_run_contract(root)
    selections = [{'name': c.identifier, 'run': str(root), 'condition': c.identifier} for c in conditions(contract.config)]
    return compare({'scope': 'single_run_summary', 'selections': selections,
                    'references': [selections[0]['name']], 'expected_manifest': str(root / 'blocks.jsonl'),
                    'output': str(root.parent / 'comparisons' / (root.name + '_categorical'))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, help='YAML with explicit run paths, condition IDs and references')
    parser.add_argument('--output-dir', help='Separate comparison destination; cannot be inside a source run')
    args = parser.parse_args()
    path = Path(args.spec).resolve()
    try:
        print(compare(yaml.safe_load(path.read_text()), spec_dir=path.parent, output_dir=args.output_dir))
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
