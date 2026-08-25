import tempfile
import unittest
from pathlib import Path

import torch

from noise_methods import (construct_noise, independent_white, load_or_create_noise_batch, pink,
                           radial_frequency_grid, sample_noise_batch, same_phase_floor)


class NoiseMethodTests(unittest.TestCase):
    shape = (4, 3, 32, 40)

    def setUp(self):
        self.batch = sample_noise_batch(20260825, "p000_s000", self.shape)

    def test_deterministic_distinct_and_paired(self):
        duplicate = sample_noise_batch(20260825, "p000_s000", self.shape)
        self.assertTrue(torch.equal(self.batch.base_white, duplicate.base_white))
        self.assertEqual(len(set(self.batch.base_hashes)), 4)
        self.assertNotEqual(self.batch.base_hashes, self.batch.eta_hashes)

    def test_manifest_batch_seed_controls_the_four_base_samples(self):
        batch = sample_noise_batch(20260825, "p000_s000", self.shape, batch_seed=10000)
        self.assertEqual(batch.sample_seeds, [10000, 10001, 10002, 10003])

    def test_endpoints_after_shared_normalization(self):
        profile = "per_sample_per_channel_zero_mean_unit_std"
        pink_norm = construct_noise(self.batch, "baseline", .5, None, profile)
        self.assertTrue(torch.allclose(construct_noise(self.batch, "same_phase", .5, 0., profile), pink_norm, atol=2e-6))
        self.assertTrue(torch.allclose(construct_noise(self.batch, "independent_white", .5, 0., profile), pink_norm, atol=2e-6))
        white_norm = construct_noise(self.batch, "baseline", 0., None, profile)
        self.assertTrue(torch.allclose(construct_noise(self.batch, "same_phase", .5, 1., profile), white_norm, atol=2e-6))
        eta_norm = construct_noise(type(self.batch)(self.batch.independent_eta, self.batch.independent_eta, self.batch.eta_sample_seeds, self.batch.eta_sample_seeds, self.batch.eta_hashes, self.batch.eta_hashes), "baseline", 0., None, profile)
        self.assertTrue(torch.allclose(construct_noise(self.batch, "independent_white", .5, 1., profile), eta_norm, atol=2e-6))

    def test_phase_and_frequency_domain_properties(self):
        raw = same_phase_floor(self.batch.base_white, .5, .35)
        original = torch.fft.rfft2(self.batch.base_white, dim=(-2, -1))
        transformed = torch.fft.rfft2(raw, dim=(-2, -1))
        mask = original.abs() > 1e-4
        # Positive-real multiplication retains Fourier phase.
        self.assertTrue(torch.allclose((transformed[mask] / original[mask]).imag, torch.zeros_like((transformed[mask] / original[mask]).imag), atol=2e-5))
        self.assertEqual(tuple(raw.shape), self.shape)
        self.assertTrue(torch.isfinite(raw).all())

    def test_integer_frequency_grid_matches_divgen_filter_strength(self):
        radial = radial_frequency_grid(128, 128)
        multiplier = (1 + radial).pow(-.5)
        self.assertGreater(float(radial.max()), 90.0)
        self.assertLess(float(multiplier.min()), .11)

    def test_independent_spatial_and_frequency_formula_agree(self):
        spatial = independent_white(self.batch.base_white, self.batch.independent_eta, .3, .4)
        h = (1 + radial_frequency_grid(32, 40)).pow(-.3)
        frequency = torch.fft.irfft2((1 - .4) ** .5 * torch.fft.rfft2(self.batch.base_white, dim=(-2, -1)) * h + .4 ** .5 * torch.fft.rfft2(self.batch.independent_eta, dim=(-2, -1)), s=(32, 40), dim=(-2, -1))
        self.assertTrue(torch.allclose(spatial, frequency, atol=2e-5))

    def test_cache_is_hash_checked_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            first = load_or_create_noise_batch(Path(directory), 1, "block", self.shape)
            second = load_or_create_noise_batch(Path(directory), 1, "block", self.shape)
            self.assertTrue(torch.equal(first.base_white, second.base_white))

    def test_empirical_psd_has_expected_direction(self):
        # Aggregate independent samples so random-periodogram variation does not dominate.
        white, pink_power = [], []
        for index in range(32):
            base = sample_noise_batch(index, "b", (1, 1, 48, 48)).base_white
            white.append(torch.fft.rfft2(base).abs().square())
            pink_power.append(torch.fft.rfft2(pink(base, .7)).abs().square())
        white, pink_mean = torch.cat(white).mean(0)[0], torch.cat(pink_power).mean(0)[0]
        self.assertGreater(float((pink_mean[0, 1] / pink_mean[-1, -1])), float((white[0, 1] / white[-1, -1])))


if __name__ == "__main__": unittest.main()
