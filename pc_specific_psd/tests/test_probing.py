import math
import tempfile
import unittest
from pathlib import Path

import torch

from pc_specific_psd import manifests, probing
from pc_specific_psd.basis import PCABasis
from pc_specific_psd.compat_generation import read_jsonl
from pc_specific_psd.tests._runner_test_support import FakeAdapterPCA, load_test_config


def _synthetic_orthonormal_basis(d: int, patch_size: int, channels: int, seed: int = 0) -> PCABasis:
    """Any orthonormal (d, d) matrix works for codec correctness tests;
    eigen-ordering/eigenvalues are irrelevant here, only orthonormality
    matters (rotate_block/encode/decode never use basis.eigenvalues).
    """
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn((d, d), generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(random_matrix)
    return PCABasis(
        components=q.to(torch.float32), eigenvalues=torch.ones(d, dtype=torch.float64), mean=torch.zeros(d, dtype=torch.float64),
        patch_size=patch_size, channels=channels, num_samples=0, metadata={"synthetic": True},
    )


class FairBudgetAngleTests(unittest.TestCase):
    def test_matches_plan_table_for_rho_010(self):
        # plan §14.4 default rho=0.10 table for a 64x64, C=4, p=5 latent
        # (D=16384, N_patch=144): size 4 -> 0.5399, size 16 -> 0.2675, size 68 -> 0.1294 rad.
        d, n_patch = 16384, 144
        self.assertAlmostEqual(probing.fair_budget_angle(0.10, d, n_patch, 4), 0.5399, places=3)
        self.assertAlmostEqual(probing.fair_budget_angle(0.10, d, n_patch, 16), 0.2675, places=3)
        self.assertAlmostEqual(probing.fair_budget_angle(0.10, d, n_patch, 68), 0.1294, places=3)

    def test_rho_zero_gives_exact_identity_angle(self):
        self.assertEqual(probing.fair_budget_angle(0.0, 16384, 144, 4), 0.0)

    def test_theta_in_valid_range(self):
        theta = probing.fair_budget_angle(0.10, 16384, 144, 4)
        self.assertGreaterEqual(theta, 0.0)
        self.assertLessEqual(theta, math.pi / 2)

    def test_infeasible_angle_raises_instead_of_clamping(self):
        with self.assertRaises(ValueError):
            probing.fair_budget_angle(rho=5.0, total_dimension=16384, num_patches=144, group_size=4)

    def test_rejects_negative_rho_and_nonpositive_sizes(self):
        with self.assertRaises(ValueError):
            probing.fair_budget_angle(-0.1, 16384, 144, 4)
        with self.assertRaises(ValueError):
            probing.fair_budget_angle(0.1, 16384, 0, 4)
        with self.assertRaises(ValueError):
            probing.fair_budget_angle(0.1, 16384, 144, 0)


class BuildProbingManifestTests(unittest.TestCase):
    def setUp(self):
        self.entries = probing.build_probing_manifest(channels=4, height=64, width=64, patch_size=5)

    def test_exactly_84_entries(self):
        self.assertEqual(len(self.entries), 84)
        self.assertEqual(len(self.entries), manifests.probing_image_count())

    def test_12_references_and_72_interventions(self):
        references = [e for e in self.entries if e.condition_type == "reference"]
        interventions = [e for e in self.entries if e.condition_type == "intervention"]
        self.assertEqual(len(references), 12)
        self.assertEqual(len(interventions), 72)

    def test_each_block_has_one_reference_and_one_per_group(self):
        by_block: dict[str, list] = {}
        for entry in self.entries:
            by_block.setdefault(entry.block_id, []).append(entry)
        self.assertEqual(len(by_block), 12)
        for block_id, block_entries in by_block.items():
            references = [e for e in block_entries if e.condition_type == "reference"]
            groups = sorted(e.group_id for e in block_entries if e.condition_type == "intervention")
            self.assertEqual(len(references), 1)
            self.assertEqual(groups, [g.group_id for g in manifests.PC_GROUPS])

    def test_donor_shared_across_groups_within_same_base_draw(self):
        block_id = manifests.PROMPTS[0].prompt_id + "_s000"
        donors = {e.donor_seed for e in self.entries if e.block_id == block_id and e.condition_type == "intervention"}
        self.assertEqual(len(donors), 1)

    def test_reference_entries_have_no_group_theta_or_donor(self):
        for entry in self.entries:
            if entry.condition_type == "reference":
                self.assertIsNone(entry.group_id)
                self.assertIsNone(entry.theta)
                self.assertIsNone(entry.donor_seed)

    def test_sample_seed_matches_batch_seed_plus_zero(self):
        for entry in self.entries:
            self.assertEqual(entry.base_index, 0)
            self.assertEqual(entry.sample_seed, entry.batch_seed)

    def test_deterministic_across_calls(self):
        other = probing.build_probing_manifest(channels=4, height=64, width=64, patch_size=5)
        self.assertEqual(self.entries, other)

    def test_image_ids_are_unique(self):
        ids = [e.image_id for e in self.entries]
        self.assertEqual(len(ids), len(set(ids)))


class RealizeProbeTests(unittest.TestCase):
    def setUp(self):
        self.channels, self.height, self.width, self.patch_size = 2, 9, 9, 2
        self.d = self.channels * self.patch_size * self.patch_size
        self.basis = _synthetic_orthonormal_basis(self.d, self.patch_size, self.channels)

    def _make_entry(self, condition_type, group_id=None, theta=None, donor_seed=None):
        return probing.ProbeEntry(
            prompt_id="p000", prompt_text="test", block_id="p000_s000", batch_seed=10000,
            base_index=0, sample_seed=10000, condition_type=condition_type,
            group_id=group_id, theta=theta, donor_seed=donor_seed,
        )

    def test_reference_is_untouched_base_white(self):
        entry = self._make_entry("reference")
        realization = probing.realize_probe(entry, self.basis, channels=self.channels, height=self.height, width=self.width)
        expected = probing.draw_base_latent(entry, channels=self.channels, height=self.height, width=self.width)
        torch.testing.assert_close(realization.output_latent, expected)
        self.assertTrue(realization.complement_and_boundary_unchanged)
        self.assertIsNone(realization.measured_delta_energy)

    def test_theta_zero_intervention_is_exact_identity(self):
        entry = self._make_entry("intervention", group_id="B1", theta=0.0, donor_seed=999)
        realization = probing.realize_probe(entry, self.basis, channels=self.channels, height=self.height, width=self.width)
        base_white = probing.draw_base_latent(entry, channels=self.channels, height=self.height, width=self.width)
        torch.testing.assert_close(realization.output_latent, base_white)
        self.assertAlmostEqual(realization.measured_delta_energy, 0.0, places=6)

    def test_intervention_leaves_complement_and_boundary_unchanged(self):
        group = manifests.pc_group_by_id("B1")
        theta = probing.fair_budget_angle(0.10, self.channels * self.height * self.width, 16, group.size)
        entry = self._make_entry("intervention", group_id="B1", theta=theta, donor_seed=42)
        realization = probing.realize_probe(entry, self.basis, channels=self.channels, height=self.height, width=self.width)
        self.assertTrue(realization.complement_and_boundary_unchanged)

    def test_rotation_matches_closed_form_algebraically(self):
        group = manifests.pc_group_by_id("B2")
        theta = probing.fair_budget_angle(0.10, self.channels * self.height * self.width, 16, group.size)
        entry = self._make_entry("intervention", group_id="B2", theta=theta, donor_seed=7)
        realization = probing.realize_probe(entry, self.basis, channels=self.channels, height=self.height, width=self.width)
        self.assertLess(realization.max_abs_error_vs_closed_form, 1e-5)

    def test_measured_and_theoretical_delta_energy_are_populated_and_finite(self):
        group = manifests.pc_group_by_id("B1")
        theta = probing.fair_budget_angle(0.10, self.channels * self.height * self.width, 16, group.size)
        entry = self._make_entry("intervention", group_id="B1", theta=theta, donor_seed=123)
        realization = probing.realize_probe(entry, self.basis, channels=self.channels, height=self.height, width=self.width)
        self.assertGreater(realization.measured_delta_energy, 0.0)
        self.assertGreater(realization.theoretical_delta_energy, 0.0)
        self.assertTrue(math.isfinite(realization.group_energy_ratio))

    def test_channel_mismatch_raises(self):
        entry = self._make_entry("reference")
        with self.assertRaises(ValueError):
            probing.realize_probe(entry, self.basis, channels=self.channels + 1, height=self.height, width=self.width)

    def test_intervention_missing_fields_raises(self):
        entry = self._make_entry("intervention")  # group_id/theta/donor_seed all None
        with self.assertRaises(ValueError):
            probing.realize_probe(entry, self.basis, channels=self.channels, height=self.height, width=self.width)


class Rho020TriggerTests(unittest.TestCase):
    def _full_annotations(self, clear_change_prompt_ids):
        annotations = []
        for prompt in manifests.PROMPTS:
            for block in manifests.seed_blocks_for_prompt(prompt):
                for group in manifests.PC_GROUPS:
                    ratings = {field: 0 for field in probing.ROLE_FIELDS}
                    if prompt.prompt_id in clear_change_prompt_ids and group.group_id == "B1" and block.block_index == 0:
                        ratings["color"] = 2
                    annotations.append(probing.ProbeAnnotation(prompt.prompt_id, block.block_id, group.group_id, ratings))
        return annotations

    def test_zero_prompts_with_change_permits_supplement(self):
        result = probing.compute_rho020_trigger(self._full_annotations(set()))
        self.assertEqual(result.prompt_count, 0)
        self.assertTrue(result.permitted)

    def test_one_prompt_with_change_permits_supplement(self):
        result = probing.compute_rho020_trigger(self._full_annotations({"p000"}))
        self.assertEqual(result.prompt_count, 1)
        self.assertTrue(result.permitted)

    def test_two_prompts_with_change_blocks_supplement(self):
        result = probing.compute_rho020_trigger(self._full_annotations({"p000", "p001"}))
        self.assertEqual(result.prompt_count, 2)
        self.assertFalse(result.permitted)

    def test_four_prompts_with_change_blocks_supplement(self):
        result = probing.compute_rho020_trigger(self._full_annotations({"p000", "p001", "p002", "p003"}))
        self.assertEqual(result.prompt_count, 4)
        self.assertFalse(result.permitted)

    def test_non_structural_field_counts_too(self):
        # "color" is not a structural field, but rating==2 on it must still count.
        result = probing.compute_rho020_trigger(self._full_annotations({"p000"}))
        self.assertIn("p000", result.prompts_with_clear_change)

    def test_rating_one_does_not_count_as_clear_change(self):
        annotations = []
        for prompt in manifests.PROMPTS:
            for block in manifests.seed_blocks_for_prompt(prompt):
                for group in manifests.PC_GROUPS:
                    ratings = {field: 1 for field in probing.ROLE_FIELDS}
                    annotations.append(probing.ProbeAnnotation(prompt.prompt_id, block.block_id, group.group_id, ratings))
        result = probing.compute_rho020_trigger(annotations)
        self.assertEqual(result.prompt_count, 0)
        self.assertTrue(result.permitted)

    def test_na_does_not_count_as_clear_change(self):
        annotations = []
        for prompt in manifests.PROMPTS:
            for block in manifests.seed_blocks_for_prompt(prompt):
                for group in manifests.PC_GROUPS:
                    ratings = {field: "NA" for field in probing.ROLE_FIELDS}
                    annotations.append(probing.ProbeAnnotation(prompt.prompt_id, block.block_id, group.group_id, ratings))
        result = probing.compute_rho020_trigger(annotations)
        self.assertEqual(result.prompt_count, 0)

    def test_missing_annotations_block_supplement(self):
        incomplete = self._full_annotations(set())[:-1]
        result = probing.compute_rho020_trigger(incomplete)
        self.assertFalse(result.permitted)
        self.assertIn("missing", result.reason)


class Rho020SupplementManifestTests(unittest.TestCase):
    def test_raises_when_not_permitted(self):
        blocked = probing.Rho020TriggerResult(frozenset({"p000", "p001"}), 2, False, "blocked")
        with self.assertRaises(ValueError):
            probing.build_rho020_supplement_manifest(blocked, channels=4, height=64, width=64, patch_size=5)

    def test_builds_72_intervention_only_entries_when_permitted(self):
        permitted = probing.Rho020TriggerResult(frozenset(), 0, True, "ok")
        entries = probing.build_rho020_supplement_manifest(permitted, channels=4, height=64, width=64, patch_size=5)
        self.assertEqual(len(entries), 72)
        self.assertEqual(len(entries), manifests.rho020_supplement_image_count())
        self.assertTrue(all(e.condition_type == "intervention" for e in entries))

    def test_supplement_donor_differs_from_primary_probe_donor(self):
        permitted = probing.Rho020TriggerResult(frozenset(), 0, True, "ok")
        entries = probing.build_rho020_supplement_manifest(permitted, channels=4, height=64, width=64, patch_size=5)
        primary_entries = probing.build_probing_manifest(channels=4, height=64, width=64, patch_size=5)
        by_block_primary = {e.block_id: e.donor_seed for e in primary_entries if e.condition_type == "intervention"}
        for entry in entries:
            self.assertNotEqual(entry.donor_seed, by_block_primary[entry.block_id])


class CandidateRecheckManifestTests(unittest.TestCase):
    def test_empty_candidate_list_raises(self):
        with self.assertRaises(ValueError):
            probing.build_candidate_recheck_manifest([], channels=4, height=64, width=64, patch_size=5)

    def test_single_candidate_produces_12_entries(self):
        entries = probing.build_candidate_recheck_manifest(["B3"], channels=4, height=64, width=64, patch_size=5)
        self.assertEqual(len(entries), 12)
        self.assertTrue(all(e.group_id == "B3" for e in entries))

    def test_two_candidates_produce_24_entries(self):
        entries = probing.build_candidate_recheck_manifest(["B3", "B5"], channels=4, height=64, width=64, patch_size=5)
        self.assertEqual(len(entries), 24)
        self.assertEqual({e.group_id for e in entries}, {"B3", "B5"})

    def test_gated_independently_of_rho020_trigger(self):
        # Even when the rho=0.20 trigger would refuse (>=2 prompts changed),
        # the candidate recheck must still succeed: it shares no gate at all
        # with compute_rho020_trigger.
        blocked_trigger = probing.Rho020TriggerResult(frozenset({"p000", "p001"}), 2, False, "blocked")
        self.assertFalse(blocked_trigger.permitted)
        entries = probing.build_candidate_recheck_manifest(["B1"], channels=4, height=64, width=64, patch_size=5)
        self.assertEqual(len(entries), 12)

    def test_recheck_donor_differs_from_primary_and_supplement_donors(self):
        recheck = probing.build_candidate_recheck_manifest(["B1"], channels=4, height=64, width=64, patch_size=5)
        primary = probing.build_probing_manifest(channels=4, height=64, width=64, patch_size=5)
        primary_donor_by_block = {e.block_id: e.donor_seed for e in primary if e.group_id == "B1"}
        for entry in recheck:
            self.assertNotEqual(entry.donor_seed, primary_donor_by_block[entry.block_id])


class GenerateProbesTests(unittest.TestCase):
    """Orchestration coverage for ``generate_probes``/``ProbingError``/
    ``probe_basis_hash_for`` -- previously exercised only indirectly through
    the ``probe`` CLI command (test_cli.py), never at the module level.
    Reuses ``_runner_test_support``'s synthetic-basis/FakeAdapterPCA fixtures
    (patch_size=5, channels=4, matching its BASIS_DIM=100), mirroring
    test_runner.py::GenerateManifestTests' equivalent coverage for the
    PC-condition (non-probe) generation path.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.config = load_test_config(Path(self._tmp.name))

    def _reference_entry(self):
        return probing.ProbeEntry(
            prompt_id="p000", prompt_text="a test prompt", block_id="p000_s000",
            batch_seed=10000, base_index=0, sample_seed=10000, condition_type="reference",
        )

    def _intervention_entry(self, group_id="B1", donor_seed=999):
        return probing.ProbeEntry(
            prompt_id="p000", prompt_text="a test prompt", block_id="p000_s000",
            batch_seed=10000, base_index=0, sample_seed=10000, condition_type="intervention",
            group_id=group_id, theta=0.2, donor_seed=donor_seed,
        )

    def test_empty_manifest_raises(self):
        adapter = FakeAdapterPCA(self.config.model.as_model_config_dict())
        with self.assertRaises(probing.ProbingError):
            probing.generate_probes(self.config, (), adapter=adapter, allow_synthetic_basis=True)

    def test_probe_basis_hash_for_raises_when_basis_missing(self):
        self.config.resolve_root(self.config.basis.basis_output_path).unlink()
        with self.assertRaises(probing.ProbingError):
            probing.probe_basis_hash_for(self.config)

    def test_generates_expected_sample_records_and_images(self):
        entries = (self._reference_entry(), self._intervention_entry())
        adapter = FakeAdapterPCA(self.config.model.as_model_config_dict())
        run_dir = probing.generate_probes(self.config, entries, adapter=adapter, allow_synthetic_basis=True)
        self.assertEqual(adapter.call_count, 2)

        basis_hash = probing.probe_basis_hash_for(self.config)
        reference_dir = run_dir / "probes" / "reference" / "p000_s000" / "b0"
        intervention_dir = run_dir / "probes" / "B1" / "p000_s000" / "b0"
        self.assertTrue((reference_dir / "image.png").exists())
        self.assertTrue((intervention_dir / "image.png").exists())

        reference_record = read_jsonl(reference_dir / "sample.jsonl")[0]
        self.assertEqual(reference_record["condition_type"], "reference")
        self.assertIsNone(reference_record["group_id"])
        self.assertEqual(reference_record["basis_hash"], basis_hash)

        intervention_record = read_jsonl(intervention_dir / "sample.jsonl")[0]
        self.assertEqual(intervention_record["condition_type"], "intervention")
        self.assertEqual(intervention_record["group_id"], "B1")
        self.assertGreater(intervention_record["measured_delta_energy"], 0.0)

    def test_resume_skips_already_complete_probes(self):
        entries = (self._reference_entry(),)
        adapter = FakeAdapterPCA(self.config.model.as_model_config_dict())
        probing.generate_probes(self.config, entries, run_id="r1", adapter=adapter, allow_synthetic_basis=True)
        self.assertEqual(adapter.call_count, 1)
        probing.generate_probes(self.config, entries, run_id="r1", adapter=adapter, allow_synthetic_basis=True)
        self.assertEqual(adapter.call_count, 1)  # no new call -- the draw is already complete

    def test_incomplete_existing_probe_dir_is_refused_not_overwritten(self):
        entries = (self._reference_entry(),)
        adapter = FakeAdapterPCA(self.config.model.as_model_config_dict())
        run_dir = probing.generate_probes(self.config, entries, run_id="r1", adapter=adapter, allow_synthetic_basis=True)
        sample_dir = run_dir / "probes" / "reference" / "p000_s000" / "b0"
        (sample_dir / "sample.jsonl").unlink()  # simulate a crash mid-write
        with self.assertRaises(probing.ProbingError):
            probing.generate_probes(self.config, entries, run_id="r1", adapter=adapter, allow_synthetic_basis=True)

    def test_adapter_returning_wrong_image_count_raises(self):
        class _WrongCountAdapter(FakeAdapterPCA):
            def generate(self, *args, **kwargs):
                self.call_count += 1
                return []

        entries = (self._reference_entry(),)
        adapter = _WrongCountAdapter(self.config.model.as_model_config_dict())
        with self.assertRaises(probing.ProbingError):
            probing.generate_probes(self.config, entries, adapter=adapter, allow_synthetic_basis=True)


if __name__ == "__main__":
    unittest.main()
