import unittest

from pc_specific_psd import manifests


class PromptTableTests(unittest.TestCase):
    def test_exact_prompt_and_seed_table(self):
        expected = {
            "p000": ("A photo of a red fox in a snowy forest", (10000, 10100, 10200)),
            "p001": ("A ceramic teapot on a wooden table", (11000, 11100, 11200)),
            "p002": ("A photo of a dog", (12000, 12100, 12200)),
            "p003": ("A photo of a chair", (13000, 13100, 13200)),
        }
        self.assertEqual({p.prompt_id for p in manifests.PROMPTS}, set(expected))
        for prompt in manifests.PROMPTS:
            text, seeds = expected[prompt.prompt_id]
            self.assertEqual(prompt.text, text)
            self.assertEqual(prompt.batch_seeds, seeds)

    def test_prompt_by_id_unknown_raises(self):
        with self.assertRaises(KeyError):
            manifests.prompt_by_id("p999")

    def test_seed_blocks_use_batch_seed_plus_base_index(self):
        prompt = manifests.prompt_by_id("p000")
        blocks = manifests.seed_blocks_for_prompt(prompt)
        self.assertEqual(len(blocks), 3)
        self.assertEqual([b.block_id for b in blocks], ["p000_s000", "p000_s001", "p000_s002"])
        first_block = blocks[0]
        self.assertEqual(first_block.sample_seed(0), 10000)
        self.assertEqual(first_block.sample_seed(3), 10003)
        self.assertEqual(first_block.sample_seed(4), 10004)
        self.assertEqual(first_block.sample_seed(7), 10007)
        with self.assertRaises(ValueError):
            first_block.sample_seed(8)
        with self.assertRaises(ValueError):
            first_block.sample_seed(-1)

    def test_all_seed_blocks_count_and_prompt_block_pairs_match(self):
        self.assertEqual(len(manifests.all_seed_blocks()), 12)
        self.assertEqual(len(manifests.prompt_block_pairs()), manifests.NUM_PROMPT_BLOCK_PAIRS)
        self.assertEqual(manifests.NUM_PROMPT_BLOCK_PAIRS, 12)

    def test_prompt_block_pairs_order_is_deterministic(self):
        pairs_a = manifests.prompt_block_pairs()
        pairs_b = manifests.prompt_block_pairs()
        self.assertEqual(
            [(p.prompt_id, b.block_id) for p, b in pairs_a],
            [(p.prompt_id, b.block_id) for p, b in pairs_b],
        )


class DonorSeedTests(unittest.TestCase):
    def test_deterministic(self):
        a = manifests.donor_seed("p000_s000", base_index=0, donor_index=0)
        b = manifests.donor_seed("p000_s000", base_index=0, donor_index=0)
        self.assertEqual(a, b)

    def test_varies_with_base_index_and_block(self):
        base = manifests.donor_seed("p000_s000", base_index=0, donor_index=0)
        other_base_index = manifests.donor_seed("p000_s000", base_index=1, donor_index=0)
        other_block = manifests.donor_seed("p000_s001", base_index=0, donor_index=0)
        other_donor = manifests.donor_seed("p000_s000", base_index=0, donor_index=1)
        self.assertNotEqual(base, other_base_index)
        self.assertNotEqual(base, other_block)
        self.assertNotEqual(base, other_donor)

    def test_shared_across_groups_by_construction(self):
        # group_id is not a parameter at all: every PC group probed off the same
        # base draw is expected to share this one donor seed (plan §14.2).
        import inspect

        params = list(inspect.signature(manifests.donor_seed).parameters)
        self.assertNotIn("group_id", params)


class PCGroupTests(unittest.TestCase):
    def test_group_sizes_match_plan_4_1(self):
        sizes = {g.group_id: g.size for g in manifests.PC_GROUPS}
        self.assertEqual(sizes, {"B1": 4, "B2": 4, "B3": 4, "B4": 4, "B5": 16, "B6": 68})

    def test_groups_are_disjoint_and_cover_1_to_100(self):
        manifests.validate_pc_groups(manifests.PC_GROUPS, basis_dimension=100)
        covered = set()
        for group in manifests.PC_GROUPS:
            covered |= set(range(group.start_1based, group.end_1based + 1))
        self.assertEqual(covered, set(range(1, 101)))

    def test_python_slice_and_indices_are_zero_based(self):
        group = manifests.pc_group_by_id("B1")
        self.assertEqual(group.python_slice, slice(0, 4))
        self.assertEqual(list(group.indices), [0, 1, 2, 3])

    def test_pc_group_by_id_unknown_raises(self):
        with self.assertRaises(KeyError):
            manifests.pc_group_by_id("B99")

    def test_validate_pc_groups_rejects_overlap(self):
        bad = (manifests.PCGroup("X1", 1, 5), manifests.PCGroup("X2", 4, 8))
        with self.assertRaises(ValueError):
            manifests.validate_pc_groups(bad, basis_dimension=100)

    def test_validate_pc_groups_rejects_out_of_range(self):
        bad = (manifests.PCGroup("X1", 1, 101),)
        with self.assertRaises(ValueError):
            manifests.validate_pc_groups(bad, basis_dimension=100)


class BudgetCalculatorTests(unittest.TestCase):
    def test_probing_image_count_is_84(self):
        self.assertEqual(manifests.probing_image_count(), 84)

    def test_rho020_supplement_is_72(self):
        self.assertEqual(manifests.rho020_supplement_image_count(), 72)

    def test_num_configs_for_candidates(self):
        self.assertEqual(manifests.num_configs_for_candidates(0), 0)
        self.assertEqual(manifests.num_configs_for_candidates(1), 3)
        self.assertEqual(manifests.num_configs_for_candidates(2), 5)
        with self.assertRaises(ValueError):
            manifests.num_configs_for_candidates(3)

    def test_preview_image_count_branches(self):
        self.assertEqual(manifests.preview_image_count(0), 0)
        self.assertEqual(manifests.preview_image_count(1), 36)
        self.assertEqual(manifests.preview_image_count(2), 60)

    def test_full_pilot_image_count_branches(self):
        self.assertEqual(manifests.full_pilot_image_count(0), 0)
        self.assertEqual(manifests.full_pilot_image_count(1), 144)
        self.assertEqual(manifests.full_pilot_image_count(2), 240)

    def test_condition_ids_for_candidates_lengths_and_order(self):
        self.assertEqual(manifests.condition_ids_for_candidates([]), ())
        one = manifests.condition_ids_for_candidates(["B3"])
        self.assertEqual(one, ("reference", "B3_plus", "B3_minus"))
        two = manifests.condition_ids_for_candidates(["B3", "B5"])
        self.assertEqual(two, ("reference", "B3_plus", "B3_minus", "B5_plus", "B5_minus"))
        self.assertEqual(len(one), manifests.num_configs_for_candidates(1))
        self.assertEqual(len(two), manifests.num_configs_for_candidates(2))

    def test_condition_ids_rejects_more_than_two_candidates(self):
        with self.assertRaises(ValueError):
            manifests.condition_ids_for_candidates(["B1", "B2", "B3"])


class RunManifestEntryTests(unittest.TestCase):
    def test_preview_manifest_empty_for_zero_candidates(self):
        self.assertEqual(manifests.build_preview_manifest_entries([]), ())

    def test_preview_manifest_sizes_match_budget_calculator(self):
        one = manifests.build_preview_manifest_entries(["B3"])
        two = manifests.build_preview_manifest_entries(["B3", "B5"])
        self.assertEqual(len(one), manifests.preview_image_count(1))
        self.assertEqual(len(two), manifests.preview_image_count(2))
        self.assertTrue(all(e.base_index == 0 for e in one))
        self.assertEqual({e.condition_id for e in one}, {"reference", "B3_plus", "B3_minus"})

    def test_preview_manifest_image_ids_unique(self):
        entries = manifests.build_preview_manifest_entries(["B3", "B5"])
        self.assertEqual(len({e.image_id for e in entries}), len(entries))

    def test_full_pilot_manifest_empty_for_zero_candidates(self):
        self.assertEqual(manifests.build_full_pilot_manifest_entries([]), ())

    def test_full_pilot_manifest_sizes_match_budget_calculator(self):
        one = manifests.build_full_pilot_manifest_entries(["B3"])
        two = manifests.build_full_pilot_manifest_entries(["B3", "B5"])
        self.assertEqual(len(one), manifests.full_pilot_image_count(1))
        self.assertEqual(len(two), manifests.full_pilot_image_count(2))
        self.assertEqual({e.base_index for e in one}, {4, 5, 6, 7})

    def test_full_pilot_manifest_image_ids_unique(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3", "B5"])
        self.assertEqual(len({e.image_id for e in entries}), len(entries))

    def test_full_pilot_groups_into_4_image_galleries(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3"])
        galleries: dict[tuple[str, str, str], list[int]] = {}
        for e in entries:
            galleries.setdefault((e.condition_id, e.prompt_id, e.block_id), []).append(e.base_index)
        self.assertEqual(len(galleries), 3 * manifests.NUM_PROMPT_BLOCK_PAIRS)
        for base_indices in galleries.values():
            self.assertEqual(sorted(base_indices), [4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
