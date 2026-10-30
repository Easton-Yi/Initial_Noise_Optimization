import ast
import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from coupled_sampling import antithetic, couple_noise, formal_noise
from io_utils import canonical_json, scientific_config, sha256_text, tensor_hash
from noise_methods import construct_noise, sample_noise_batch
from run_experiment import all_conditions, load_config, validate_config
from result_integrity import validate_condition_metadata

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures'


class CoupledMathTests(unittest.TestCase):
    def test_antithetic_exact_pairing_no_mutation_no_rng(self):
        x = sample_noise_batch(7, 'b', (4, 2, 8, 8), batch_seed=31).base_white
        before, state = x.clone(), torch.random.get_rng_state().clone()
        y = antithetic(x)
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(y, torch.stack([x[0], -x[0], x[2], -x[2]])))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(y, antithetic(x)))

    def test_exact_vendored_upstream_reference_all_dtypes_and_endpoints(self):
        # Captured by executing the pinned author's function, not our local implementation.
        for case in json.loads((FIXTURES / 'coupled_upstream_reference.json').read_text()):
            with self.subTest(**case):
                x = torch.linspace(-2, 2, case['k'] * 12, dtype=getattr(torch, case['dtype'])).reshape(case['k'], 2, 2, 3)
                before, state = x.clone(), torch.random.get_rng_state().clone()
                y = couple_noise(x, case['strength'])
                self.assertEqual(tensor_hash(y), case['hash'])
                self.assertEqual(y.dtype, x.dtype)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.equal(y[0], x[0]))
                self.assertTrue(torch.equal(x, before))
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                self.assertTrue(torch.isfinite(y).all())
                if case['k'] == 1 or case['strength'] == 0:
                    self.assertIs(y, x)
                if case['k'] == 2 and case['strength'] == 1:
                    self.assertTrue(torch.equal(y[1], -x[0]))

    def test_distribution_covariance_and_group_sum(self):
        matrix = couple_noise(torch.eye(4, dtype=torch.float64).reshape(4, 4, 1, 1)).reshape(4, 4)
        expected = torch.full((4, 4), -1 / 3, dtype=torch.float64); expected.fill_diagonal_(1.)
        self.assertTrue(torch.allclose(matrix @ matrix.T, expected, atol=1e-14))
        x = torch.randn((4, 1, 256, 256), generator=torch.Generator().manual_seed(912))
        y = couple_noise(x).flatten(1)
        self.assertLess(y.sum(0).abs().max().item(), 2e-6)
        self.assertLess(y.mean(1).abs().max().item(), .015)
        self.assertTrue(torch.allclose(torch.cov(y).double(), expected, atol=.02))
        # Covariance is a distribution property, not an exact pairwise cosine constraint.

    def test_invalid_inputs_and_formal_guards(self):
        x = torch.ones(4, 2, 8, 8)
        for strength in [True, '1', -1, 1.1, float('nan'), float('inf')]:
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                couple_noise(x, strength)
        for value in [None, torch.ones(4), torch.ones(4, 1, 2, 2, dtype=torch.int64), torch.ones(0, 2, 3, 4)]:
            with self.assertRaises(ValueError):
                couple_noise(value)
        for value in [x[:1], x.double(), torch.full_like(x, float('nan'))]:
            with self.assertRaises(ValueError):
                formal_noise(value, 'repulsive', 'none')
        b = sample_noise_batch(7, 'b', (4, 2, 8, 8), batch_seed=31)
        for method in ['antithetic', 'repulsive']:
            with self.assertRaises(ValueError):
                construct_noise(b, method, None, None, 'divgen_compat')
            with self.assertRaises(ValueError):
                construct_noise(b, method, 0., None, 'none')
            expected = antithetic(b.base_white) if method == 'antithetic' else couple_noise(b.base_white)
            self.assertTrue(torch.equal(construct_noise(b, method, None, None, 'none'), expected))

    def test_formal_configs_and_rejected_changes(self):
        for model in ['sdxl_turbo', 'flux2_klein']:
            path = ROOT / 'configs' / f'{model}_geneval400_coupled.yaml'
            c = load_config(path)
            self.assertEqual(len(validate_config(c, path)), 400)
            self.assertEqual([x.identifier for x in all_conditions(c)], ['antithetic/pairs_01_23', 'repulsive/strength_1'])
            old = load_config(path.with_name(f'{model}_geneval400_1group.yaml'))
            for key in ['model', 'generation', 'blocks', 'quality_metrics', 'diversity_metrics', 'analysis']:
                self.assertEqual(c[key], old[key])
            for section, key, value in [('experiment', 'gallery_size', 2), ('experiment', 'normalization_profile', 'divgen_compat'),
                                         ('generation', 'generation_batch_size', 4)]:
                changed = copy.deepcopy(c); changed[section][key] = value
                with self.assertRaises(ValueError): all_conditions(changed)
            changed = copy.deepcopy(c); changed['coupled_sampling']['repulsive']['strength'] = .5
            with self.assertRaises(ValueError): all_conditions(changed)
            changed = copy.deepcopy(c); changed['coupled_sampling']['antithetic']['enabled'] = False
            with self.assertRaises(ValueError): all_conditions(changed, 'antithetic')

    def test_contradictory_metadata_rejected(self):
        row = {'family': 'repulsive', 'method': 'repulsive', 'condition_id': 'repulsive/strength_1',
               'alpha': None, 'gamma': None, 'normalization_profile': 'none'}
        validate_condition_metadata(row, context='test')
        for key, value in [('alpha', 0), ('gamma', 1), ('method', 'white'), ('condition_id', 'repulsive/strength_0p5')]:
            with self.assertRaises(RuntimeError): validate_condition_metadata({**row, key: value}, context='test')

    def test_pre_edit_legacy_outputs_seeds_conditions_and_hashes(self):
        frozen = json.loads((FIXTURES / 'coupled_legacy_reference.json').read_text())
        b = sample_noise_batch(7, 'reference', (4, 2, 8, 8), batch_seed=31)
        self.assertEqual(b.sample_seeds, frozen['seeds']); self.assertEqual(b.base_hashes, frozen['hashes'])
        for key, digest in frozen['tensor_hashes'].items():
            profile, method, alpha, gamma = ast.literal_eval(key)
            self.assertEqual(tensor_hash(construct_noise(b, method, alpha, gamma, profile)), digest)
        for model, reference in frozen['configs'].items():
            c = load_config(ROOT / 'configs' / f'{model}_geneval400_1group.yaml')
            self.assertNotIn('coupled_sampling', c)
            self.assertEqual([x.identifier for x in all_conditions(c)], reference['ids'])
            # Frozen hashes were captured at this path; replacing the cache location
            # here makes the regression independent of the test checkout's location.
            c['model']['cache_dir'] = '/home/zyw2004/comp3740_indiv_project/Initial_Noise_Optimization/noise_init/cache'
            self.assertEqual(sha256_text(canonical_json(scientific_config(c))), reference['hash'])

    def test_installed_schedulers_deterministic_configurations(self):
        try:
            from diffusers import EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler
        except ImportError:
            self.skipTest('Diffusers is unavailable; no package installation')
        scheduler = EulerAncestralDiscreteScheduler(timestep_spacing='trailing')
        scheduler.set_timesteps(1)
        a, b = scheduler.sigmas[:2]
        self.assertEqual(float((b ** 2 * (a ** 2 - b ** 2) / a ** 2).sqrt()), 0.)
        def trajectory(seed):
            flow = FlowMatchEulerDiscreteScheduler(stochastic_sampling=False)
            flow.set_timesteps(4)
            x = torch.ones((1, 2, 4, 4))
            for t in flow.timesteps:
                x = flow.step(x * .1, t, x, generator=torch.Generator().manual_seed(seed)).prev_sample
            return x
        self.assertTrue(torch.equal(trajectory(1), trajectory(900)))


if __name__ == '__main__': unittest.main()
