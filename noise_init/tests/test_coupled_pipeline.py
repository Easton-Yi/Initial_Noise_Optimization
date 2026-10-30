import copy
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from compare_initializations import analyze_coupled_run, compare, paired_statistics
from io_utils import file_hash, read_json, read_jsonl, write_json, write_jsonl
from metric_runner import MetricRunner, _read_csv, _write_csv, evaluate_run
from model_adapters import LatentSpec, _quick_tensor_hash
from noise_methods import construct_noise, sample_noise_batch
from result_integrity import validate_analysis_inputs
from run_experiment import generate, load_config

ROOT = Path(__file__).resolve().parents[1]


class Adapter:
    model_id = 'sdxl_turbo'

    def __init__(self, fail_at=None):
        self.calls = 0; self.fail_at = fail_at; self.seen = []; self.closed = False

    def latent_spec(self, height, width, batch_size):
        return LatentSpec((batch_size, 2, 8, 8), 'none')

    def generate(self, prompt, latents, config):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError('simulated interruption')
        self.seen.append(latents.clone())
        self.last_generated_latent_hashes = [_quick_tensor_hash(x[None]) for x in latents]
        return [Image.new('RGB', (8, 8), (i * 50, 30, 20)) for i in range(4)]

    def close(self): self.closed = True


def setup(root):
    c = load_config(ROOT / 'configs/sdxl_turbo_geneval400_coupled.yaml')
    c['model']['device'] = 'cpu'; c['model']['dtype'] = 'float32'
    c['paths']['outputs_root'] = str(root / 'outputs')
    c['generation']['height'] = c['generation']['width'] = 8
    blocks = [{'block_id': f'b{i}', 'prompt_id': f'p{i}', 'prompt': f'test {i}',
               'seed_batch_id': 's000', 'batch_seed': 17 + 10*i, 'category': 'single_object'} for i in range(2)]
    manifest = root / 'blocks.jsonl'; write_jsonl(manifest, blocks)
    c['blocks']['manifest'] = str(manifest)
    return c, blocks, root / 'configs/test.yaml'


def score(root, config):
    def vendi(kernel):
        eigenvalues = np.linalg.eigvalsh(kernel) / len(kernel)
        v = eigenvalues[eigenvalues > 1e-12]
        return float(np.exp(-sum(v * np.log(v))))
    fake = types.ModuleType('vendi_score'); fake.vendi = types.SimpleNamespace(score_K=vendi)
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {'vendi_score': fake}))
        stack.enter_context(patch.object(MetricRunner, 'metric_versions', return_value={'fake': '1'}))
        for name in ['clip_cosine', 'hpsv3']:
            stack.enter_context(patch.object(MetricRunner, name, side_effect=lambda path, prompt: float(len(prompt) + int(path.stem[-2:]))))
        for name in ['dreamsim_distance', 'lpips_distance']:
            stack.enter_context(patch.object(MetricRunner, name,
                side_effect=lambda left, right: float(abs(int(left.stem[-2:])-int(right.stem[-2:])))))
        stack.enter_context(patch.object(MetricRunner, 'clip_embedding',
            side_effect=lambda path: np.array([np.cos(int(path.stem[-2:])), np.sin(int(path.stem[-2:]))])))
        evaluate_run(root, config)


def generate_fixture(root, *, legacy=False, adapter=None):
    c, blocks, path = setup(root)
    if legacy:
        c.pop('coupled_sampling')
        c['experiment']['normalization_profile'] = 'per_sample_per_channel_zero_mean_unit_std'
        c['baseline'] = {'enabled': True, 'alpha_values': [0.]}
        c['same_phase_floor'] = {'enabled': True, 'alpha_values': [.8], 'gamma_values': [.05]}
    run = generate(c, path, blocks, run_id='legacy' if legacy else 'coupled', selected_conditions=None,
                   force=False, adapter=adapter or Adapter())
    return run, c, blocks, path


def spec_for(root, new, old):
    return {'scope': 'smoke', 'expected_manifest': str(root / 'blocks.jsonl'), 'output': str(root / 'comparison'),
            'references': ['floor'], 'selections': [
                {'name': 'antithetic', 'run': str(new), 'condition': 'antithetic/pairs_01_23'},
                {'name': 'repulsive', 'run': str(new), 'condition': 'repulsive/strength_1'},
                {'name': 'floor', 'run': str(old), 'condition': 'same_phase/alpha_0p8_gamma_0p05'}]}


def snapshot(root):
    return {str(p.relative_to(root)): (file_hash(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


class CoupledPipelineTests(unittest.TestCase):
    def test_generate_metric_analysis_cross_run_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); adapter = Adapter()
            run, c, blocks, path = generate_fixture(root, adapter=adapter)
            self.assertEqual(adapter.calls, 4)
            for i, block in enumerate(blocks):
                batch = sample_noise_batch(c['experiment']['master_seed'], block['block_id'], (4, 2, 8, 8), batch_seed=block['batch_seed'])
                for j, method in enumerate(['antithetic', 'repulsive']):
                    self.assertTrue(torch.equal(adapter.seen[2*i+j], construct_noise(batch, method, None, None, 'none')))
            images = list(run.glob('generations/**/image_*.png'))
            self.assertEqual(len(images), 16)
            committed_before = snapshot(run / 'generations')
            retry = Adapter()
            generate(c, path, blocks, run_id='coupled', selected_conditions=None, force=True, adapter=retry)
            self.assertEqual(retry.calls, 0)
            self.assertEqual(committed_before, snapshot(run / 'generations'))
            score(run, c)
            self.assertEqual(len(_read_csv(run / 'metrics/per_pair.csv')), 4 * 6 * 2)
            self.assertNotIn('pyiqa', sys.modules)
            old, old_c, _, _ = generate_fixture(root, legacy=True); score(old, old_c)
            before = {run: snapshot(run), old: snapshot(old)}
            result = compare(spec_for(root, run, old))
            self.assertTrue((result / 'qd_hpsv3_dreamsim.png').exists())
            self.assertTrue((result / 'qd_hpsv3_dreamsim.pdf').exists())
            self.assertEqual(len(_read_csv(result / 'summary_by_method.csv')), 15)
            self.assertTrue((analyze_coupled_run(run) / 'summary_by_method.csv').exists())
            for source in [run, old]: self.assertEqual(before[source], snapshot(source))

    def test_interrupted_group_restarts_but_committed_group_is_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); c, blocks, path = setup(root); adapter = Adapter(fail_at=2)
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                generate(c, path, blocks, run_id='interrupted', selected_conditions=None, force=False, adapter=adapter)
            self.assertTrue(adapter.closed)
            run = root / 'outputs/interrupted'
            self.assertEqual(len(list(run.glob('generations/**/samples.jsonl'))), 1)
            before = snapshot(run / 'generations')
            retry = Adapter()
            generate(c, path, blocks, run_id='interrupted', selected_conditions=None, force=False, adapter=retry)
            self.assertEqual(retry.calls, 3)
            after = snapshot(run / 'generations')
            self.assertTrue(all(after[k] == v for k, v in before.items()))
            score(run, c); validate_analysis_inputs(run)

    def test_corrupt_completed_image_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, c, blocks, path = generate_fixture(Path(tmp))
            image = next(run.glob('generations/**/image_00.png')); image.write_bytes(b'corrupt')
            adapter = Adapter()
            with self.assertRaisesRegex(RuntimeError, 'Corrupt committed'):
                generate(c, path, blocks, run_id='coupled', selected_conditions=None, force=False, adapter=adapter)
            self.assertEqual(adapter.calls, 0); self.assertEqual(image.read_bytes(), b'corrupt')

    def test_resume_refuses_changed_config_seeds_or_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, c, blocks, path = generate_fixture(Path(tmp))
            changed = copy.deepcopy(c); changed['generation']['guidance_scale'] = 2.
            with self.assertRaises(RuntimeError):
                generate(changed, path, blocks, run_id='coupled', selected_conditions=None, force=False, adapter=Adapter())
            altered = copy.deepcopy(blocks); altered[0]['batch_seed'] += 1
            with self.assertRaises(RuntimeError):
                generate(c, path, altered, run_id='coupled', selected_conditions=None, force=False, adapter=Adapter())
            with patch('coupled_run.source_identity', return_value={'version': 'changed'}), self.assertRaises(RuntimeError):
                generate(c, path, blocks, run_id='coupled', selected_conditions=None, force=False, adapter=Adapter())

    def test_smoke_cannot_pollute_formal_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, blocks, path = setup(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'separate --run-id'):
                generate(c, path, blocks[:1], run_id=None, selected_conditions=None, force=False, adapter=Adapter())

    def test_relocated_sources_compare_without_rewrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, c, _, _ = generate_fixture(root); score(run, c)
            old, old_c, _, _ = generate_fixture(root, legacy=True); score(old, old_c)
            moved = root / 'relocated'; shutil.copytree(run, moved)
            before = snapshot(moved)
            compare(spec_for(root, moved, old), make_plots=False)
            self.assertEqual(before, snapshot(moved))

    def test_comparison_rejects_missing_metric_duplicate_selection_and_source_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, c, _, _ = generate_fixture(root); score(run, c)
            old, old_c, _, _ = generate_fixture(root, legacy=True); score(old, old_c)
            spec = spec_for(root, run, old)
            bad = copy.deepcopy(spec); bad['selections'].append({**bad['selections'][0], 'name': 'duplicate'})
            with self.assertRaisesRegex(ValueError, 'Duplicate'): compare(bad, make_plots=False)
            with self.assertRaisesRegex(ValueError, 'separate'): compare(spec, output_dir=run / 'comparison', make_plots=False)
            rows = _read_csv(run / 'metrics/per_pair.csv'); _write_csv(run / 'metrics/per_pair.csv', rows[1:])
            with self.assertRaisesRegex(RuntimeError, 'missing per_pair'): compare(spec, make_plots=False)

    def test_changed_prompt_and_metric_version_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, c, _, _ = generate_fixture(root); score(run, c)
            old, old_c, _, _ = generate_fixture(root, legacy=True); score(old, old_c)
            spec = spec_for(root, run, old)
            manifest_path = old / 'metrics/metric_manifest.json'; original = read_json(manifest_path)
            write_json(manifest_path, {**original, 'versions': {'fake': '2'}})
            with self.assertRaisesRegex(RuntimeError, 'Incompatible metric'): compare(spec, make_plots=False)
            write_json(manifest_path, original)
            path = old / 'blocks.jsonl'; rows = read_jsonl(path); rows[0]['prompt'] = 'changed'; write_jsonl(path, rows)
            with self.assertRaisesRegex(RuntimeError, 'prompt metadata'): compare(spec, make_plots=False)

    def test_constant_paired_difference_uses_prompts(self):
        summary, differences = paired_statistics({'a': {'q': np.arange(12)}, 'b': {'q': np.arange(12) + 3}}, ['a'])
        self.assertEqual(len(summary), 2)
        self.assertEqual(differences[0]['mean_difference'], 3.)
        self.assertEqual(differences[0]['ci_low'], 3.)
        self.assertEqual(differences[0]['ci_high'], 3.)
        self.assertEqual(differences[0]['prompt_count'], 12)

    def test_unselected_historical_conditions_need_not_be_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run, c, _, _ = generate_fixture(root); score(run, c)
            old, old_c, _, _ = generate_fixture(root, legacy=True); score(old, old_c)
            for table in ('per_image', 'per_pair', 'per_group'):
                path = old / 'metrics' / (table + '.csv')
                _write_csv(path, [r for r in _read_csv(path) if r['condition_id'] != 'baseline/alpha_0'])
            compare(spec_for(root, run, old), make_plots=False)

    def test_incompatible_generator_and_missing_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            from io_utils import canonical_json, sha256_text
            root = Path(tmp); run, c, _, _ = generate_fixture(root); score(run, c)
            old, old_c, _, _ = generate_fixture(root, legacy=True); score(old, old_c)
            manifest_path = old / 'run_manifest.json'; manifest = read_json(manifest_path)
            manifest['resolved_config']['generation']['num_inference_steps'] = 4
            manifest['config_hash'] = sha256_text(canonical_json(manifest['resolved_config']))
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(RuntimeError, 'Incompatible generator'):
                compare(spec_for(root, run, old), make_plots=False)

    def test_all_stage_dispatches_to_categorical_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            import yaml
            from run_experiment import main
            c, blocks, path = setup(Path(tmp)); path.parent.mkdir(parents=True)
            path.write_text(yaml.safe_dump(c, sort_keys=False))
            with patch.object(sys, 'argv', ['run_experiment.py', '--config', str(path), '--stage', 'all', '--run-id', 'cli_smoke']), \
                    patch('coupled_run.build_adapter', return_value=Adapter()), \
                    patch('run_experiment.evaluate_run', side_effect=lambda run, config, **kwargs: score(run, config)):
                main()
            self.assertTrue((Path(tmp) / 'outputs/comparisons/cli_smoke_categorical/summary_by_method.csv').exists())


if __name__ == '__main__': unittest.main()
