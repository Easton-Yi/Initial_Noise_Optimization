import unittest

import torch

from pc_specific_psd import manifests, probing, psd_editor
from pc_specific_psd.compat_generation import sample_noise_batch

CHANNELS, HEIGHT, WIDTH = 4, 8, 8


def _block(prompt_id: str = "p000", block_index: int = 0) -> manifests.SeedBlock:
    prompt = manifests.prompt_by_id(prompt_id)
    return manifests.seed_blocks_for_prompt(prompt)[block_index]


class DeterminismTests(unittest.TestCase):
    def test_same_draw_called_twice_is_byte_identical(self):
        block = _block()
        first = psd_editor.base_white_for_draw(block, 2, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        second = psd_editor.base_white_for_draw(block, 2, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        self.assertTrue(torch.equal(first, second))


class SampleSeedFormulaTests(unittest.TestCase):
    def test_matches_batch_seed_plus_base_index_construction(self):
        block = _block("p001", 1)
        base_index = 3
        expected_seed = block.batch_seed + base_index
        self.assertEqual(block.sample_seed(base_index), expected_seed)

        actual = psd_editor.base_white_for_draw(block, base_index, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        expected = sample_noise_batch(
            expected_seed, block.block_id, (1, CHANNELS, HEIGHT, WIDTH), batch_seed=expected_seed
        ).base_white
        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_out_of_range_base_index(self):
        block = _block()
        with self.assertRaises(ValueError):
            psd_editor.base_white_for_draw(block, 8, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        with self.assertRaises(ValueError):
            psd_editor.base_white_for_draw(block, -1, channels=CHANNELS, height=HEIGHT, width=WIDTH)


class PairingAcrossConditionsTests(unittest.TestCase):
    """Every condition compared for one (prompt_id, block_id, base_index) draw
    (reference, candidate+, candidate-, ...) calls base_white_for_draw with
    the same block/base_index and must get back the identical tensor -- that
    pairing, not any single fixed triple, is the actual contract (plan
    section 5.2/14). The (p000, s000, base_index=0) triple used across this
    file is a test fixture only, never shared production noise: production
    code draws a fresh base_white per (prompt_id, block_id, base_index) via
    this same function, for every draw in the study independently.
    """

    def test_conditions_sharing_a_draw_get_identical_base_white(self):
        block = _block()
        reference_condition = psd_editor.base_white_for_draw(block, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        candidate_plus_condition = psd_editor.base_white_for_draw(block, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        candidate_minus_condition = psd_editor.base_white_for_draw(block, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        self.assertTrue(torch.equal(reference_condition, candidate_plus_condition))
        self.assertTrue(torch.equal(reference_condition, candidate_minus_condition))


class DifferentDrawsDifferTests(unittest.TestCase):
    def test_different_base_index_same_block_differs(self):
        block = _block()
        draws = [
            psd_editor.base_white_for_draw(block, i, channels=CHANNELS, height=HEIGHT, width=WIDTH) for i in range(4)
        ]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertFalse(torch.equal(draws[i], draws[j]))

    def test_different_block_same_base_index_differs(self):
        block_a = _block("p000", 0)
        block_b = _block("p000", 1)
        draw_a = psd_editor.base_white_for_draw(block_a, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        draw_b = psd_editor.base_white_for_draw(block_b, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        self.assertFalse(torch.equal(draw_a, draw_b))

    def test_different_prompt_same_base_index_and_block_index_differs(self):
        block_p0 = _block("p000", 0)
        block_p1 = _block("p001", 0)
        draw_p0 = psd_editor.base_white_for_draw(block_p0, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        draw_p1 = psd_editor.base_white_for_draw(block_p1, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        self.assertFalse(torch.equal(draw_p0, draw_p1))


class ConsistencyWithProbingDrawBaseLatentTests(unittest.TestCase):
    """probing.draw_base_latent is the base_index=0-only special case of this
    same rule (it bakes the block's batch_seed in directly instead of taking
    base_index explicitly); the two must agree exactly at base_index=0.
    """

    def test_agrees_with_probing_draw_base_latent_at_base_index_zero(self):
        block = _block("p002", 2)
        entry = probing.ProbeEntry(
            prompt_id=block.prompt_id,
            prompt_text="A photo of a dog",
            block_id=block.block_id,
            batch_seed=block.batch_seed,
            base_index=0,
            sample_seed=block.sample_seed(0),
            condition_type="reference",
        )
        via_probing = probing.draw_base_latent(entry, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        via_editor = psd_editor.base_white_for_draw(block, 0, channels=CHANNELS, height=HEIGHT, width=WIDTH)
        self.assertTrue(torch.equal(via_probing, via_editor))


if __name__ == "__main__":
    unittest.main()
