"""Small, separate PC1-PC4 diagnostic; leaves the six production groups intact.

Run from the repository root, in .venv-generation:
python -m pc_specific_psd.probe_b1_fine --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml

Default: first two configured prompts, their three existing probe draws,
four singleton edits = 24 new images. Reference and B1 images are reused.
Uses equal expected TOTAL perturbation energy (the existing rho convention).
Singleton theta is therefore larger than the original four-PC B1 theta.
This screens sensitivity, not each PC's contribution at the original angle.
Results are diagnostic only: they are NOT ingested into the B1-B6 review CSV.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw


def make_sheet(prompt, rows, destination):
    tile, left, top, row_header = 320, 125, 75, 30
    labels = ['Reference', 'B1: PC1-4', 'PC1', 'PC2', 'PC3', 'PC4']
    sheet = Image.new('RGB', (left + 6 * tile, top + len(rows) * (tile + row_header)), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f'{prompt.prompt_id}: {prompt.text}', fill='black')
    draw.text((12, 28), 'Same total perturbation budget; singleton PCs use a larger angle than B1.', fill='black')
    for col, label in enumerate(labels):
        draw.text((left + col * tile + 8, 55), label, fill='black')
    for row_index, row in enumerate(rows):
        y = top + row_index * (tile + row_header)
        draw.text((8, y + 10), str(row['block_id']), fill='black')
        draw.text((8, y + 28), f"seed {row['sample_seed']}", fill='black')
        for col, path in enumerate(row['images']):
            with Image.open(path) as original:
                thumb = original.convert('RGB')
                thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (left + col * tile, y + row_header))
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--prompt-ids', nargs=2, help='Exactly two existing prompt IDs; default: first two in PROMPTS')
    parser.add_argument('--probe-run-dir', type=Path, help='Existing completed probe run, if a custom run ID was used')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()

    from pc_specific_psd import config, manifests, probing, patch_codec
    from pc_specific_psd.adapters import SDXLTurboAdapterPCA
    from pc_specific_psd.basis import load_basis
    from pc_specific_psd.compat_generation import (
        ensure_immutable_run, file_hash, read_jsonl, tensor_hash, write_json, write_jsonl,
    )

    cfg = config.resolve_config(args.config, 'probe')
    height, width = cfg.generation.config.height // 8, cfg.generation.config.width // 8
    channels, patch_size, rho = cfg.basis.channels, cfg.basis.patch_size, cfg.probing.rho
    parent_group = manifests.pc_group_by_id('B1')
    if tuple(parent_group.indices) != (0, 1, 2, 3):
        raise ValueError('This diagnostic expects B1 = PC1-PC4; keep the production groups unchanged.')
    prompts = tuple(manifests.prompt_by_id(pid) for pid in args.prompt_ids) if args.prompt_ids else manifests.PROMPTS[:2]
    if len(prompts) != 2 or prompts[0].prompt_id == prompts[1].prompt_id:
        raise ValueError('Choose two distinct existing prompts.')
    prompt_ids = {p.prompt_id for p in prompts}
    root = cfg.resolve_root(cfg.run.outputs_root)
    source = args.probe_run_dir or root / f'{cfg.run.name}_probe'
    output = args.output_dir or root / f'{cfg.run.name}_b1_fine'
    basis_path = cfg.resolve_root(cfg.basis.basis_output_path)
    basis = load_basis(basis_path, expected_patch_size=patch_size, expected_channels=channels)
    grid = patch_codec.centered_grid(height, width, patch_size)
    codec = patch_codec.NonOverlapCodec(grid, basis.components, channels)
    theta = probing.fair_budget_angle(rho, channels * height * width, grid.n_patches, 1)
    entries = probing.build_probing_manifest(
        channels=channels, height=height, width=width, patch_size=patch_size,
        rho=rho, groups=(parent_group,),
    )
    entries = [e for e in entries if e.prompt_id in prompt_ids]
    plan, reused = [], []
    # Check all reused images before loading any GPU model.
    for entry in entries:
        condition = 'reference' if entry.condition_type == 'reference' else 'B1'
        sample_dir = source / 'probes' / condition / entry.block_id / f'b{entry.base_index}'
        rows = read_jsonl(sample_dir / 'sample.jsonl')
        realization = probing.realize_probe(entry, basis, channels=channels, height=height, width=width)
        if len(rows) != 1 or rows[0]['final_noise_hash'] != tensor_hash(realization.output_latent[0]):
            raise ValueError(f'Existing {condition} does not match the corrected 64x64 probe: {sample_dir}')
        image_path = Path(rows[0]['image_path'])
        if not image_path.is_file() or file_hash(image_path) != rows[0]['image_hash']:
            raise ValueError(f'Missing/changed source image: {image_path}')
        reused.append({'image_path': str(image_path.resolve()), 'image_hash': rows[0]['image_hash']})
        if condition == 'B1':
            plan.append(entry)
    provenance = {
        'diagnostic': 'b1_single_pc_equal_total_budget_v1', 'config_hash': file_hash(Path(args.config)),
        'basis_hash': file_hash(basis_path), 'prompt_ids': [p.prompt_id for p in prompts],
        'rho': rho, 'singleton_theta': theta, 'parent_theta': plan[0].theta,
        'latent_shape': [1, channels, height, width], 'reused': reused,
    }
    ensure_immutable_run(output, provenance, force=False)
    print(f'Prompts: {[(p.prompt_id, p.text) for p in prompts]}', flush=True)
    print(f'B1 theta={plan[0].theta:.6f}; singleton theta={theta:.6f}; up to {len(plan) * 4} new images', flush=True)
    adapter = SDXLTurboAdapterPCA(cfg.model.as_model_config_dict())
    sheets = {p.prompt_id: [] for p in prompts}
    completed = 0
    try:
        for entry in plan:
            base = probing.draw_base_latent(entry, channels=channels, height=height, width=width)
            donor = probing._draw_donor_latent(entry.donor_seed, channels=channels, height=height, width=width)
            coefficients, donor_coefficients = codec.encode(base), codec.encode(donor)
            old_images = []
            for label in ('reference', 'B1'):
                record = read_jsonl(source / 'probes' / label / entry.block_id / f'b{entry.base_index}' / 'sample.jsonl')[0]
                old_images.append(Path(record['image_path']))
            row = {'block_id': entry.block_id, 'sample_seed': entry.sample_seed, 'images': old_images}
            for index in range(4):
                rotated = codec.rotate_block(coefficients.clone(), donor_coefficients, [index], theta)
                latent = codec.decode(rotated, template=base)
                digest = tensor_hash(latent[0])
                folder = output / f'PC{index + 1}' / entry.block_id
                image_path, record_path = folder / 'image.png', folder / 'sample.jsonl'
                saved = read_jsonl(record_path)
                complete = (len(saved) == 1 and saved[0]['final_noise_hash'] == digest
                            and image_path.is_file() and file_hash(image_path) == saved[0]['image_hash'])
                if not complete:
                    folder.mkdir(parents=True, exist_ok=True)
                    pair_key = (entry.prompt_id, entry.block_id, entry.base_index)
                    image = adapter.generate(entry.prompt_text, latent, [pair_key], seed=cfg.run.master_seed,
                                             generation_config=cfg.generation.config)[0]
                    if image.size != (cfg.generation.config.width, cfg.generation.config.height):
                        raise ValueError(f'Unexpected output image size: {image.size}')
                    image.save(image_path)
                    write_jsonl(record_path, [{
                        'pc_1based': index + 1, 'theta': theta, 'rho': rho, 'entry': asdict(entry),
                        'final_noise_hash': digest, 'image_hash': file_hash(image_path),
                        'adapter_prepared_latent_hash': adapter.last_generated_latent_hashes[0],
                        'generator_seed': adapter.last_generator_seeds[0],
                    }])
                row['images'].append(image_path)
                completed += 1
                print(f'[{completed}/{len(plan) * 4}] {entry.block_id} PC{index + 1}' + (' cached' if complete else ''), flush=True)
            sheets[entry.prompt_id].append(row)
    finally:
        adapter.close()
    for prompt in prompts:
        destination = output / f'b1_fine_{prompt.prompt_id}.png'
        make_sheet(prompt, sheets[prompt.prompt_id], destination)
        print(f'Contact sheet: {destination.resolve()}', flush=True)
    write_json(output / 'diagnostic_summary.json', provenance)
    print('Done. Review these sheets before deciding on a subgroup; do not put PC1-PC4 IDs into the B1-B6 CSV.')


if __name__ == '__main__':
    main()
