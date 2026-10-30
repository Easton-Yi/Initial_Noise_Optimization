"""Transactional generation and provenance for newly created coupled runs only."""
from __future__ import annotations

import importlib.metadata
import shutil
import time
from pathlib import Path

import torch

from coupled_sampling import basis_provenance, source_identity, validate_config
from io_utils import (canonical_json, ensure_immutable_run, environment_record, file_hash,
                      read_json, read_jsonl, tensor_hash, write_json, write_jsonl)
from model_adapters import GenerationConfig, _quick_tensor_hash, _torch_dtype, build_adapter
from noise_methods import construct_noise, load_or_create_noise_batch, noise_statistics


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_readings():
    result = {}
    if torch.cuda.is_available():
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    # These are host cgroup readings, not a peak attributable to this process.
    for name in ("memory.current", "memory.peak", "memory.max"):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result['host_cgroup_' + name.replace('.', '_')] = path.read_text().strip()
    result['host_peak_scope'] = 'cgroup lifetime; not a current-run peak'
    return result


def runtime_identity(adapter):
    pipe = getattr(adapter, 'pipe', None)
    scheduler = getattr(pipe, 'scheduler', None)
    if scheduler is None:
        return {'scheduler_class': None, 'scheduler_config': None, 'status': 'unavailable'}
    # Diffusers configs contain tuples and NumPy scalars in some versions.
    import json
    config = json.loads(json.dumps(dict(scheduler.config), default=str))
    return {'scheduler_class': type(scheduler).__name__, 'scheduler_config': config,
            'pipeline_class': type(pipe).__name__,
            'model_commit': getattr(getattr(pipe, 'config', None), '_commit_hash', None),
            'vae_class': type(getattr(pipe, 'vae', None)).__name__}


def check_scheduler(adapter, generation):
    """Check the configured deterministic cases without changing historical adapters."""
    scheduler = getattr(getattr(adapter, 'pipe', None), 'scheduler', None)
    if scheduler is None:
        return  # Test adapters have no diffusion scheduler; recorded as unavailable.
    name = type(scheduler).__name__
    if adapter.model_id == 'sdxl_turbo':
        if name != 'EulerAncestralDiscreteScheduler' or generation['num_inference_steps'] != 1:
            raise ValueError('Coupled SDXL requires the agreed one-step Euler ancestral schedule; other schedules need an explicit scheduler RNG policy')
        probe = type(scheduler).from_config(scheduler.config)
        probe.set_timesteps(1)
        sigma, next_sigma = probe.sigmas[:2]
        sigma_up = (next_sigma.square() * (sigma.square() - next_sigma.square()) / sigma.square()).sqrt()
        if float(sigma_up) != 0.0:
            raise RuntimeError('Configured final-step stochastic coefficient is not zero')
    elif adapter.model_id == 'flux2_klein':
        if name != 'FlowMatchEulerDiscreteScheduler' or scheduler.config.get('stochastic_sampling', False):
            raise ValueError('Coupled FLUX requires deterministic FlowMatchEulerDiscreteScheduler evaluation')


def freeze(path, payload):
    if path.exists():
        if read_json(path) != payload:
            raise RuntimeError(f'Frozen provenance differs at {path}; use a new run id')
    else:
        write_json(path, payload, replace=False)


def verify_group(directory, final_directory, block, condition, batch, latents, config, source):
    """Verify a completed gallery or a staged gallery before atomic publication."""
    from run_experiment import _generation_hash
    from result_integrity import validate_condition_metadata
    try:
        rows = read_jsonl(directory / 'samples.jsonl')
        completion = read_json(directory / 'completion.json')
        if len(rows) != 4 or [r['base_index'] for r in rows] != list(range(4)):
            raise ValueError('expected four ordered output slots')
        if (completion['source_identity'] != source or completion['complete'] is not True or
                completion['samples_hash'] != file_hash(directory / 'samples.jsonl') or
                completion['grid_hash'] != file_hash(directory / 'grid_1x4.png') or
                completion['statistics_hash'] != file_hash(directory / 'noise_statistics.json')):
            raise ValueError('completion digest mismatch')
        basis = basis_provenance(batch, condition.family) if condition.family != 'baseline' else None
        dtype = _torch_dtype(config['model']['dtype'])
        for index, row in enumerate(rows):
            validate_condition_metadata(row, context='coupled generation')
            if any(row.get(k) != v for k, v in block.items()):
                raise ValueError('block/seed identity mismatch')
            target = directory / f'image_{index:02d}.png'
            if (row['condition_id'] != condition.identifier or
                    row['generation_config_hash'] != _generation_hash(config) or
                    row['normalization_profile'] != 'none' or
                    row['final_noise_hash'] != tensor_hash(latents[index]) or
                    row['adapter_prepared_latent_hash'] != _quick_tensor_hash(latents[index:index+1].to(dtype)) or
                    row['image_hash'] != file_hash(target) or row.get('basis_provenance') != basis):
                raise ValueError('image, latent, or basis provenance mismatch')
            if basis is not None and (row.get('sample_seed') is not None or row.get('base_noise_hash') is not None or row.get('eta_noise_hash') is not None):
                raise ValueError('derived output incorrectly labelled as an independent draw')
            # A legitimate relocation may change only the prefix before generations/.
            from result_integrity import resolve_recorded_image
            run_root = next(p for p in final_directory.parents if p.name == 'generations').parent
            if resolve_recorded_image(run_root, row) != final_directory / target.name:
                raise ValueError('image path does not match output identity')
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        raise RuntimeError(f'Corrupt committed/staged group at {directory}: {exc}; repair explicitly or use a new run id') from exc


def generate_coupled(config, config_path, blocks, *, run_id, selected_conditions, force, adapter=None):
    from run_experiment import _run_dir, _sample_path, _save_grid, _generation_hash, all_conditions
    validate_config(config)
    conditions = all_conditions(config, selected_conditions)
    root = _run_dir(config, config_path, run_id)
    formal_blocks = read_jsonl(config_path.parent.parent / config.get('blocks', {}).get('manifest', '')) if config.get('blocks') else []
    if root.name == config['experiment']['name'] and formal_blocks and blocks != formal_blocks:
        raise ValueError('A prompt subset requires a separate --run-id; formal run IDs require the full frozen manifest')
    ensure_immutable_run(root, config, force=force)
    if (root / 'blocks.jsonl').exists():
        if read_jsonl(root / 'blocks.jsonl') != blocks:
            raise RuntimeError('Frozen blocks differ; choose a new run id')
    else:
        write_jsonl(root / 'blocks.jsonl', blocks)
    source = source_identity()
    freeze(root / 'coupled_source.json', source)
    freeze(root / 'generation_versions.json', {'torch': str(torch.__version__),
           **{name: _version(name) for name in ('diffusers', 'transformers', 'Pillow')}})
    synchronize()
    setup_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adapter = adapter or build_adapter(config['model'])
    try:
        spec = adapter.latent_spec(config['generation']['height'], config['generation']['width'], 4)
        check_scheduler(adapter, config['generation'])
        synchronize()
        runtime = runtime_identity(adapter)
        freeze(root / 'generator_runtime.json', runtime)
        setup = {'seconds': time.perf_counter() - setup_start, 'environment': environment_record(),
                 'versions': {name: _version(name) for name in ('diffusers', 'transformers', 'torchvision', 'Pillow')},
                 'memory': memory_readings()}
        write_json(root / 'timings' / f'setup_{time.time_ns()}.json', setup, replace=False)
        gen_config = GenerationConfig(**{k: config['generation'][k] for k in
            ('height', 'width', 'num_inference_steps', 'guidance_scale', 'generation_batch_size', 'output_format')})
        for block in blocks:
            start = time.perf_counter()
            batch = load_or_create_noise_batch(root / 'noise_cache' / block['block_id'],
                int(config['experiment']['master_seed']), block['block_id'], spec.shape, batch_seed=int(block['batch_seed']))
            if batch.sample_seeds != [int(block['batch_seed']) + i for i in range(4)]:
                raise RuntimeError('Cached basis seeds differ; use a new run id')
            basis_seconds = time.perf_counter() - start
            for condition in conditions:
                start = time.perf_counter()
                latents = construct_noise(batch, condition.family, condition.alpha, condition.gamma, 'none')
                noise_seconds = time.perf_counter() - start
                target = _sample_path(root, adapter.model_id, block['block_id'], condition)
                if target.exists():
                    verify_group(target, target, block, condition, batch, latents, config, source)
                    continue
                stage = root / '.coupled_staging' / adapter.model_id / block['block_id'] / condition.identifier
                if stage.exists():
                    shutil.rmtree(stage)  # Only this exact uncommitted group, outside metric discovery.
                stage.mkdir(parents=True)
                synchronize()
                start = time.perf_counter()
                images = adapter.generate(block['prompt'], latents, gen_config)
                synchronize()
                generation_seconds = time.perf_counter() - start
                prepared = getattr(adapter, 'last_generated_latent_hashes', None)
                if len(images) != 4 or prepared is None or len(prepared) != 4:
                    raise RuntimeError('Coupled generation requires four images and four verified injection hashes')
                start = time.perf_counter()
                basis = basis_provenance(batch, condition.family) if condition.family != 'baseline' else None
                records = []
                for i, image in enumerate(images):
                    filename = f'image_{i:02d}.png'
                    image.save(stage / filename, format='PNG')
                    records.append({'run_id': root.name, 'model_id': adapter.model_id, **block,
                        'base_index': i, 'sample_seed': None if basis else batch.sample_seeds[i],
                        'base_noise_hash': None if basis else batch.base_hashes[i], 'eta_noise_hash': None,
                        'basis_provenance': basis, 'method': condition.method, 'family': condition.family,
                        'condition_id': condition.identifier, 'alpha': condition.alpha, 'gamma': condition.gamma,
                        'normalization_profile': 'none', 'final_noise_hash': tensor_hash(latents[i]),
                        'adapter_prepared_latent_hash': prepared[i], 'image_path': str((target / filename).resolve()),
                        'image_hash': file_hash(stage / filename), 'generation_config_hash': _generation_hash(config)})
                _save_grid(images, stage / 'grid_1x4.png')
                write_jsonl(stage / 'samples.jsonl', records)
                stats = noise_statistics(latents)
                write_json(stage / 'noise_statistics.json', {'pre_normalization': stats, 'post_normalization': stats})
                scheduler = getattr(getattr(adapter, 'pipe', None), 'scheduler', None)
                schedule = {name: getattr(scheduler, name).detach().cpu().tolist()
                            for name in ('timesteps', 'sigmas') if isinstance(getattr(scheduler, name, None), torch.Tensor)}
                write_json(stage / 'completion.json', {'complete': True, 'source_identity': source,
                    'samples_hash': file_hash(stage / 'samples.jsonl'), 'grid_hash': file_hash(stage / 'grid_1x4.png'),
                    'statistics_hash': file_hash(stage / 'noise_statistics.json'), 'schedule': schedule,
                    'timings': {'basis_load_or_sample_seconds': basis_seconds, 'noise_construction_seconds': noise_seconds,
                                'gallery_generation_seconds': generation_seconds, 'file_write_seconds': time.perf_counter() - start},
                    'memory': memory_readings()})
                verify_group(stage, target, block, condition, batch, latents, config, source)
                target.parent.mkdir(parents=True, exist_ok=True)
                stage.rename(target)
        write_json(root / 'coupled_completion.json', {'selected_conditions': [c.identifier for c in conditions],
                   'block_count': len(blocks), 'completed_at': environment_record()['created_at_utc']})
    finally:
        adapter.close()
    return root


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return 'not-installed'


def validate_saved_groups(root, contract, groups):
    """Read-only verification; never recreate missing caches during evaluation."""
    root = root.resolve()
    from noise_methods import NoiseBatch, sample_noise_batch
    from run_support import conditions
    from io_utils import sha256_text, scientific_config
    saved = read_json(root / 'run_manifest.json')
    if saved.get('config_hash') != sha256_text(canonical_json(scientific_config(contract.config))):
        raise RuntimeError('Corrupt coupled run configuration hash')
    source = read_json(root / 'coupled_source.json')
    by_condition = {c.identifier: c for c in conditions(contract.config)}
    # One basis is retained at a time; comparison remains bounded in host memory.
    for block_id in sorted({key[0] for key in groups}):
        cache = root / 'noise_cache' / block_id
        base = torch.load(cache / 'base_white.pt', map_location='cpu', weights_only=True)
        eta = torch.load(cache / 'independent_eta.pt', map_location='cpu', weights_only=True)
        meta = read_json(cache / 'noise_metadata.json')
        block = contract.blocks[block_id]
        expected = sample_noise_batch(int(contract.config['experiment']['master_seed']), block_id,
                                     tuple(base.shape), batch_seed=int(block['batch_seed']))
        if (not torch.equal(base, expected.base_white) or not torch.equal(eta, expected.independent_eta) or
                meta['sample_seeds'] != expected.sample_seeds or meta['base_hashes'] != expected.base_hashes or
                meta['eta_hashes'] != expected.eta_hashes):
            raise RuntimeError(f'Corrupt coupled basis cache: {block_id}')
        batch = NoiseBatch(base, eta, expected.sample_seeds, expected.eta_sample_seeds,
                           expected.base_hashes, expected.eta_hashes)
        for (candidate, condition_id), rows in groups.items():
            if candidate != block_id:
                continue
            condition = by_condition[condition_id]
            latents = construct_noise(batch, condition.family, condition.alpha, condition.gamma, 'none')
            directory = root / 'generations' / rows[0]['model_id'] / block_id / condition_id
            verify_group(directory, directory, block, condition, batch, latents, contract.config, source)
