"""Change 4b: preview-exclusion review operates at condition (group x sign)
granularity, not PC-group granularity -- ``B3_plus`` can be dropped while
``B3_minus`` survives. ``build_full_pilot_manifest_entries_for_conditions``
is the manifest builder that makes that possible: it takes an explicit
condition-id list (already filtered by review), not a candidate-group list.
"""
import unittest

from pc_specific_psd import manifests


class ArbitraryConditionSubsetTests(unittest.TestCase):
    def test_accepts_an_arbitrary_condition_subset_dropping_one_sign_of_one_group(self):
        # A 2-candidate preview run covers reference, B3_plus, B3_minus,
        # B5_plus, B5_minus; review excludes B3_plus and both B5 signs,
        # keeping only reference and B3_minus.
        approved = ["reference", "B3_minus"]
        entries = manifests.build_full_pilot_manifest_entries_for_conditions(approved)
        self.assertEqual({e.condition_id for e in entries}, {"reference", "B3_minus"})
        self.assertNotIn("B3_plus", {e.condition_id for e in entries})
        self.assertNotIn("B5_plus", {e.condition_id for e in entries})
        self.assertNotIn("B5_minus", {e.condition_id for e in entries})

    def test_image_count_matches_len_approved_times_12_times_4_not_the_144_240_ceiling(self):
        approved = ["reference", "B3_minus"]
        entries = manifests.build_full_pilot_manifest_entries_for_conditions(approved)
        expected = len(approved) * manifests.NUM_PROMPT_BLOCK_PAIRS * manifests.NUM_BASE_INDICES_PER_BLOCK
        self.assertEqual(len(entries), expected)
        # Strictly less than the pre-exclusion 2-candidate ceiling (240),
        # since one of the two candidates' signs was excluded.
        self.assertLess(len(entries), manifests.full_pilot_image_count(2))

    def test_full_condition_set_matches_full_pilot_image_count_ceiling(self):
        approved = manifests.condition_ids_for_candidates(["B3", "B5"])
        entries = manifests.build_full_pilot_manifest_entries_for_conditions(approved)
        self.assertEqual(len(entries), manifests.full_pilot_image_count(2))

    def test_reference_is_always_present_even_if_caller_forgets_it(self):
        entries = manifests.build_full_pilot_manifest_entries_for_conditions(["B3_minus"])
        self.assertIn("reference", {e.condition_id for e in entries})

    def test_reference_appears_exactly_once_regardless_of_position_in_input(self):
        entries = manifests.build_full_pilot_manifest_entries_for_conditions(["B3_minus", "reference", "B5_plus"])
        condition_ids = {e.condition_id for e in entries}
        self.assertEqual(condition_ids, {"reference", "B3_minus", "B5_plus"})

    def test_all_conditions_excluded_except_reference_returns_empty_manifest(self):
        self.assertEqual(manifests.build_full_pilot_manifest_entries_for_conditions(["reference"]), ())

    def test_empty_input_returns_empty_manifest(self):
        self.assertEqual(manifests.build_full_pilot_manifest_entries_for_conditions([]), ())

    def test_entries_drawn_from_the_independent_full_pilot_base_index_range(self):
        entries = manifests.build_full_pilot_manifest_entries_for_conditions(["reference", "B3_minus"])
        self.assertEqual({e.base_index for e in entries}, set(manifests.FULL_PILOT_BASE_INDICES))


if __name__ == "__main__":
    unittest.main()
