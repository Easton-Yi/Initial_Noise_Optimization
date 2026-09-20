"""Budget accounting (plan §4.4/§7.1/§14.4): probing (84) + rho=0.20
supplement (72) counts, preview (0/36/60) and full-pilot (0/144/240) counts
branch-selected from select_candidates()'s actual output length, and the
independent-seed guarantee for final evaluation -- full-pilot's base_index
range never overlaps probing/preview's, verified both by inspecting the
manifest and by the runtime ``assert_disjoint_sample_seeds`` check.
"""
import tempfile
import unittest
from pathlib import Path

from pc_specific_psd import manifests, runner
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, freeze_selected_calibration, load_test_config


class ProbingBudgetTests(unittest.TestCase):
    def test_probing_image_count_is_84(self):
        self.assertEqual(manifests.probing_image_count(), 84)

    def test_rho020_supplement_image_count_is_72(self):
        self.assertEqual(manifests.rho020_supplement_image_count(), 72)


class PreviewAndFullPilotBudgetTests(unittest.TestCase):
    def test_preview_image_count_branches_on_candidate_count(self):
        self.assertEqual(manifests.preview_image_count(0), 0)
        self.assertEqual(manifests.preview_image_count(1), 36)
        self.assertEqual(manifests.preview_image_count(2), 60)

    def test_full_pilot_image_count_branches_on_candidate_count(self):
        self.assertEqual(manifests.full_pilot_image_count(0), 0)
        self.assertEqual(manifests.full_pilot_image_count(1), 144)
        self.assertEqual(manifests.full_pilot_image_count(2), 240)

    def test_zero_candidates_builds_no_manifest_entries(self):
        self.assertEqual(manifests.build_preview_manifest_entries([]), ())
        self.assertEqual(manifests.build_full_pilot_manifest_entries([]), ())

    def test_manifest_entry_counts_match_the_declared_budget(self):
        for group_ids, expected_preview, expected_full in (
            (["B5"], 36, 144),
            (["B5", "B6"], 60, 240),
        ):
            self.assertEqual(len(manifests.build_preview_manifest_entries(group_ids)), expected_preview)
            self.assertEqual(len(manifests.build_full_pilot_manifest_entries(group_ids)), expected_full)


class FullPilotUsesIndependentSeedsTests(unittest.TestCase):
    """Final evaluation must never reuse a seed a human already looked at
    during probing/preview review -- full-pilot's base_index range is
    disjoint from both by construction (Change 4a), checked here both
    statically (the manifest's own base_index/sample_seed sets) and via the
    runtime safety net ``generate_full_pilot`` calls automatically.
    """

    def test_preview_and_full_pilot_base_index_sets_are_disjoint(self):
        preview_entries = manifests.build_preview_manifest_entries(["B5"])
        full_entries = manifests.build_full_pilot_manifest_entries(["B5"])
        preview_base_indices = {e.base_index for e in preview_entries}
        full_base_indices = {e.base_index for e in full_entries}
        self.assertEqual(preview_base_indices & full_base_indices, set())

    def test_generate_full_pilot_no_longer_takes_preview_run_dir(self):
        import inspect

        params = inspect.signature(runner.generate_full_pilot).parameters
        self.assertNotIn("preview_run_dir", params)

    def test_full_pilot_runs_without_any_preview_run_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])

            full_adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            full_run_dir = runner.generate_full_pilot(
                loaded, ["B5"], run_id="full_independent", force=False,
                adapter=full_adapter, allow_synthetic_basis=True,
            )
            self.assertEqual(full_run_dir.name, "full_independent")
            self.assertEqual(full_adapter.call_count, 144)
            self.assertEqual(len(list(full_run_dir.rglob("image.png"))), 144)


class DisjointSampleSeedRuntimeCheckTests(unittest.TestCase):
    def test_disjoint_sample_seeds_checked_at_runtime(self):
        preview_entries = manifests.build_preview_manifest_entries(["B5"])
        full_entries = manifests.build_full_pilot_manifest_entries(["B5"])
        # No exception on the real, disjoint manifests.
        manifests.assert_disjoint_sample_seeds(preview_entries, full_entries)

        overlapping = manifests.ConditionDrawEntry(
            preview_entries[0].condition_id, preview_entries[0].prompt_id,
            preview_entries[0].block_id, preview_entries[0].base_index,
        )
        with self.assertRaises(ValueError):
            manifests.assert_disjoint_sample_seeds(preview_entries, full_entries + (overlapping,))

    def test_generate_full_pilot_raises_on_constructed_overlapping_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded = load_test_config(Path(tmp))
            freeze_selected_calibration(loaded, group_ids=["B5"])

            original = manifests.build_full_pilot_manifest_entries_for_conditions
            preview_entries = manifests.build_preview_manifest_entries(["B5"])

            def _overlapping(approved_condition_ids):
                real = original(approved_condition_ids)
                return real[:-1] + (preview_entries[0],)

            full_adapter = FakeAdapterPCA(loaded.model.as_model_config_dict())
            manifests.build_full_pilot_manifest_entries_for_conditions = _overlapping
            try:
                with self.assertRaises(ValueError):
                    runner.generate_full_pilot(
                        loaded, ["B5"], run_id="full_overlap", force=False,
                        approved_condition_ids=["reference", "B5_plus", "B5_minus"],
                        adapter=full_adapter, allow_synthetic_basis=True,
                    )
            finally:
                manifests.build_full_pilot_manifest_entries_for_conditions = original


if __name__ == "__main__":
    unittest.main()
