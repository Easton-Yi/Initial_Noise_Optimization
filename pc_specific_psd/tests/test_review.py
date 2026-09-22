import tempfile
import unittest
from pathlib import Path

from pc_specific_psd import manifests, probing, review


def _blank_artifact_block():
    return {field: "absent" for field in review.ARTIFACT_FIELDS}


def _fill_probe_row(blind_row, **overrides):
    filled = dict(blind_row)
    filled.update({field: 0 for field in review.ROLE_FIELDS})
    filled["shape_change_kind"] = "not_applicable"
    filled["prompt_plausibility_before"] = "yes"
    filled["prompt_plausibility_after"] = "yes"
    filled["artifacts_before"] = _blank_artifact_block()
    filled["artifacts_after"] = _blank_artifact_block()
    filled["valid_structural_change"] = "no"
    filled["evidence"] = "evidence text"
    filled["confidence"] = "clear"
    filled.update(overrides)
    return filled


def _fill_preview_row(blind_row, **overrides):
    filled = dict(blind_row)
    filled["prompt_plausibility"] = "yes"
    filled["artifacts"] = _blank_artifact_block()
    filled["quality"] = "good"
    filled["evidence"] = "evidence text"
    filled["confidence"] = "clear"
    filled.update(overrides)
    return filled


def _fill_gallery_row(blind_row, **overrides):
    filled = dict(blind_row)
    ratings = {}
    for label in blind_row["blind_labels"]:
        ratings[label] = {
            "per_image_artifacts": [_blank_artifact_block() for _ in range(manifests.NUM_BASE_INDICES_PER_BLOCK)],
            "plausibility": "yes",
            "diversity_impression": "medium",
        }
    filled["ratings"] = ratings
    filled["preference"] = blind_row["blind_labels"][0]
    filled["evidence"] = "evidence text"
    filled["confidence"] = "clear"
    filled.update(overrides)
    return filled


class EnumConsistencyTests(unittest.TestCase):
    def test_structural_and_appearance_partition_role_fields(self):
        self.assertEqual(set(review.STRUCTURAL_ROLE_FIELDS) | set(review.APPEARANCE_ROLE_FIELDS), set(review.ROLE_FIELDS))
        self.assertEqual(set(review.STRUCTURAL_ROLE_FIELDS) & set(review.APPEARANCE_ROLE_FIELDS), set())

    def test_role_fields_match_probing_module(self):
        self.assertEqual(review.ROLE_FIELDS, probing.ROLE_FIELDS)


class ExportProbeReviewTests(unittest.TestCase):
    def setUp(self):
        self.manifest = probing.build_probing_manifest(channels=4, height=16, width=16, patch_size=4)
        self.interventions = [e for e in self.manifest if e.condition_type == "intervention"]

    def test_export_empty_raises(self):
        with self.assertRaises(ValueError):
            review.export_probe_review([])

    def test_export_produces_one_row_per_intervention(self):
        blind_rows, mapping = review.export_probe_review(self.interventions)
        self.assertEqual(len(blind_rows), len(self.interventions))
        self.assertEqual(len(mapping), len(self.interventions))

    def test_blind_rows_never_leak_group_identity(self):
        blind_rows, _ = review.export_probe_review(self.interventions)
        for row in blind_rows:
            self.assertNotIn("group_id", row)
            self.assertNotIn("rho", row)
            self.assertNotIn("donor_seed", row)

    def test_pair_ids_unique(self):
        blind_rows, mapping = review.export_probe_review(self.interventions)
        self.assertEqual(len({r["pair_id"] for r in blind_rows}), len(blind_rows))
        self.assertEqual(len({m.pair_id for m in mapping}), len(mapping))

    def test_mapping_pair_ids_match_blind_row_pair_ids(self):
        blind_rows, mapping = review.export_probe_review(self.interventions)
        self.assertEqual({r["pair_id"] for r in blind_rows}, {m.pair_id for m in mapping})

    def test_export_is_deterministic_across_calls(self):
        rows_a, mapping_a = review.export_probe_review(self.interventions)
        rows_b, mapping_b = review.export_probe_review(self.interventions)
        self.assertEqual([r["pair_id"] for r in rows_a], [r["pair_id"] for r in rows_b])
        self.assertEqual(mapping_a, mapping_b)

    def test_row_order_is_shuffled_relative_to_input(self):
        blind_rows, _ = review.export_probe_review(self.interventions)
        input_order = [review._probe_pair_id(e) for e in self.interventions]
        self.assertNotEqual([r["pair_id"] for r in blind_rows], input_order)
        self.assertEqual(set(r["pair_id"] for r in blind_rows), set(input_order))

    def test_blank_template_fields_all_none(self):
        blind_rows, _ = review.export_probe_review(self.interventions)
        row = blind_rows[0]
        for field in review.ROLE_FIELDS:
            self.assertIsNone(row[field])
        self.assertIsNone(row["shape_change_kind"])
        self.assertIsNone(row["valid_structural_change"])
        for field in review.ARTIFACT_FIELDS:
            self.assertIsNone(row["artifacts_before"][field])
            self.assertIsNone(row["artifacts_after"][field])


class IngestProbeReviewTests(unittest.TestCase):
    def setUp(self):
        self.manifest = probing.build_probing_manifest(channels=4, height=16, width=16, patch_size=4)
        self.interventions = [e for e in self.manifest if e.condition_type == "intervention"]
        self.blind_rows, self.mapping = review.export_probe_review(self.interventions)
        self.expected_pair_ids = [r["pair_id"] for r in self.blind_rows]

    def test_roundtrip_happy_path(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        rows = review.ingest_probe_review(filled, self.expected_pair_ids)
        self.assertEqual(len(rows), len(self.blind_rows))
        self.assertEqual([r.pair_id for r in rows], self.expected_pair_ids)

    def test_missing_required_field_raises(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        del filled[0]["evidence"]
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)

    def test_none_value_treated_as_missing(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        filled[0]["confidence"] = None
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)

    def test_invalid_enum_value_raises(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        filled[0]["shape_change_kind"] = "bogus"
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)

    def test_invalid_role_rating_raises(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        filled[0]["layout"] = 3
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)

    def test_role_rating_na_is_accepted_and_distinct_from_zero(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        filled[0]["layout"] = "NA"
        rows = review.ingest_probe_review(filled, self.expected_pair_ids)
        by_pair = {r.pair_id: r for r in rows}
        self.assertEqual(by_pair[filled[0]["pair_id"]].role_ratings["layout"], "NA")
        self.assertNotEqual(by_pair[filled[0]["pair_id"]].role_ratings["layout"], 0)

    def test_unexpected_pair_id_raises(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        filled[0]["pair_id"] = "not_a_real_pair_id"
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)

    def test_duplicate_row_raises(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        filled.append(dict(filled[0]))
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)

    def test_incomplete_set_raises(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows][:-1]
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)

    def test_empty_evidence_string_raises(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]
        filled[0]["evidence"] = "   "
        with self.assertRaises(ValueError):
            review.ingest_probe_review(filled, self.expected_pair_ids)


class ToProbeAnnotationAndTriggerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = probing.build_probing_manifest(channels=4, height=16, width=16, patch_size=4)
        self.interventions = [e for e in self.manifest if e.condition_type == "intervention"]
        self.blind_rows, self.mapping = review.export_probe_review(self.interventions)
        self.expected_pair_ids = [r["pair_id"] for r in self.blind_rows]
        self.group_by_pair = {m.pair_id: m.group_id for m in self.mapping}

    def test_to_probe_annotation_carries_group_id_and_ratings(self):
        filled = [_fill_probe_row(r, layout=2) for r in self.blind_rows]
        rows = review.ingest_probe_review(filled, self.expected_pair_ids)
        row = rows[0]
        group_id = self.group_by_pair[row.pair_id]
        annotation = row.to_probe_annotation(group_id)
        self.assertEqual(annotation.prompt_id, row.prompt_id)
        self.assertEqual(annotation.block_id, row.block_id)
        self.assertEqual(annotation.group_id, group_id)
        self.assertEqual(annotation.role_ratings["layout"], 2)

    def test_compute_rho020_trigger_matches_direct_probing_call(self):
        filled = [_fill_probe_row(r, layout=2) for r in self.blind_rows]
        rows = review.ingest_probe_review(filled, self.expected_pair_ids)
        via_review = review.compute_rho020_trigger(rows, self.mapping)

        annotations = tuple(row.to_probe_annotation(self.group_by_pair[row.pair_id]) for row in rows)
        via_probing = probing.compute_rho020_trigger(annotations)

        self.assertEqual(via_review, via_probing)

    def test_compute_rho020_trigger_permitted_when_no_clear_changes(self):
        filled = [_fill_probe_row(r) for r in self.blind_rows]  # all ratings 0
        rows = review.ingest_probe_review(filled, self.expected_pair_ids)
        result = review.compute_rho020_trigger(rows, self.mapping)
        self.assertTrue(result.permitted)
        self.assertEqual(result.prompt_count, 0)


def _make_probe_fixture(pass_spec, groups=("B1", "B2", "B3")):
    """Builds synthetic (rows, mapping) directly -- one row per (prompt, seed
    block, group) triple, matching the probing manifest's real shape but
    without needing a basis/codec. ``pass_spec`` maps
    ``(group_id, role_field) -> set of prompt_ids`` that should locally pass
    (>=2 of 3 blocks qualify) for that group/field; every other combination
    gets a non-qualifying rating of 0.
    """
    rows = []
    mapping = []
    counter = 0
    for prompt in manifests.PROMPTS:
        blocks = manifests.seed_blocks_for_prompt(prompt)
        for group_id in groups:
            for block_index, block in enumerate(blocks):
                pair_id = f"pair_{counter}"
                counter += 1
                role_ratings = {}
                for field in review.ROLE_FIELDS:
                    passing_prompts = pass_spec.get((group_id, field), set())
                    qualifies = prompt.prompt_id in passing_prompts and block_index < 2
                    role_ratings[field] = review.CLEAR_CHANGE_RATING if qualifies else 0
                rows.append(review.ProbeReviewRow(
                    pair_id=pair_id, prompt_id=prompt.prompt_id, block_id=block.block_id,
                    role_ratings=role_ratings, shape_change_kind="not_applicable",
                    prompt_plausibility_before="yes", prompt_plausibility_after="yes",
                    artifacts_before=_blank_artifact_block(), artifacts_after=_blank_artifact_block(),
                    valid_structural_change="yes", evidence="e", confidence="clear",
                ))
                mapping.append(review.ProbeReviewMappingEntry(
                    pair_id=pair_id, prompt_id=prompt.prompt_id, block_id=block.block_id,
                    group_id=group_id, donor_seed=0, rho=None,
                    reference_image_id="ref", intervention_image_id="int",
                ))
    return tuple(rows), tuple(mapping)


class SelectCandidatesTests(unittest.TestCase):
    def test_empty_pool_returns_no_candidates(self):
        rows, mapping = _make_probe_fixture(pass_spec={})
        result = review.select_candidates(rows, mapping, groups=manifests.PC_GROUPS, prompts=manifests.PROMPTS)
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.reserve, ())

    def test_single_nominated_group_is_selected(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows, mapping = _make_probe_fixture(pass_spec={("B1", "layout"): all_prompts})
        result = review.select_candidates(rows, mapping)
        self.assertEqual(result.candidates, ("B1",))
        self.assertEqual(result.evidence_by_group["B1"].coverage, 4)

    def test_structural_candidate_prioritized_over_appearance(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        two_prompts = {"p000", "p001"}
        # B2 (appearance-only) has higher coverage than B1 (structural), but
        # B1 must still take the first slot per the plan's "at least one
        # structural candidate first" rule.
        rows, mapping = _make_probe_fixture(pass_spec={
            ("B1", "pose"): two_prompts,
            ("B2", "color"): all_prompts,
        })
        result = review.select_candidates(rows, mapping)
        self.assertEqual(result.candidates[0], "B1")
        self.assertIn("B2", result.candidates)

    def test_max_two_candidates(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows, mapping = _make_probe_fixture(pass_spec={
            ("B1", "layout"): all_prompts,
            ("B2", "pose"): all_prompts,
            ("B3", "shape"): all_prompts,
        }, groups=("B1", "B2", "B3"))
        result = review.select_candidates(rows, mapping)
        self.assertEqual(len(result.candidates), 2)

    def test_tie_broken_by_declared_pc_groups_order(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        # B2 and B3 tie exactly on coverage and valid_change_count; PC_GROUPS
        # declares B2 before B3, so B2 must win the (single) structural slot.
        rows, mapping = _make_probe_fixture(pass_spec={
            ("B2", "layout"): all_prompts,
            ("B3", "layout"): all_prompts,
        }, groups=("B2", "B3"))
        result = review.select_candidates(rows, mapping, groups=(manifests.pc_group_by_id("B2"), manifests.pc_group_by_id("B3")))
        self.assertEqual(result.candidates, ("B2", "B3"))

    def test_appearance_only_pool_used_when_no_structural_pool(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows, mapping = _make_probe_fixture(pass_spec={("B1", "color"): all_prompts})
        result = review.select_candidates(rows, mapping)
        self.assertEqual(result.candidates, ("B1",))

    def test_recurrence_below_threshold_is_not_nominated_but_reserved(self):
        one_prompt = {"p000"}
        rows, mapping = _make_probe_fixture(pass_spec={("B1", "layout"): one_prompt})
        result = review.select_candidates(rows, mapping)
        self.assertEqual(result.candidates, ())
        self.assertIn("B1", result.reserve)

    def test_valid_structural_change_no_blocks_structural_nomination(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows, mapping = _make_probe_fixture(pass_spec={("B1", "layout"): all_prompts})
        # Flip valid_structural_change to "no" on every row -- the structural
        # field must stop qualifying even though the raw rating is still 2.
        import dataclasses
        rows = tuple(dataclasses.replace(r, valid_structural_change="no") for r in rows)
        result = review.select_candidates(rows, mapping)
        self.assertNotIn("B1", result.candidates)
        self.assertNotIn("B1", result.reserve)

    def test_valid_structural_change_does_not_gate_appearance_fields(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        rows, mapping = _make_probe_fixture(pass_spec={("B1", "color"): all_prompts})
        import dataclasses
        # Even with valid_structural_change="no" throughout, a pure-color
        # nomination must still succeed: that field has no analogous validity
        # gate in the plan's schema.
        rows = tuple(dataclasses.replace(r, valid_structural_change="no") for r in rows)
        result = review.select_candidates(rows, mapping)
        self.assertEqual(result.candidates, ("B1",))

    def test_reserve_excludes_selected_candidates(self):
        all_prompts = {p.prompt_id for p in manifests.PROMPTS}
        one_prompt = {"p000"}
        rows, mapping = _make_probe_fixture(pass_spec={
            ("B1", "layout"): all_prompts,
            ("B2", "pose"): one_prompt,
        }, groups=("B1", "B2"))
        result = review.select_candidates(rows, mapping)
        self.assertIn("B1", result.candidates)
        self.assertNotIn("B1", result.reserve)
        self.assertIn("B2", result.reserve)


class ExportPreviewReviewTests(unittest.TestCase):
    def test_export_empty_raises(self):
        with self.assertRaises(ValueError):
            review.export_preview_review(())

    def test_sizing_matches_manifest_for_one_candidate(self):
        entries = manifests.build_preview_manifest_entries(["B3"])
        blind_rows, mapping = review.export_preview_review(entries)
        self.assertEqual(len(blind_rows), 36)
        self.assertEqual(len(mapping), 36)

    def test_sizing_matches_manifest_for_two_candidates(self):
        entries = manifests.build_preview_manifest_entries(["B3", "B5"])
        blind_rows, mapping = review.export_preview_review(entries)
        self.assertEqual(len(blind_rows), 60)
        self.assertEqual(len(mapping), 60)

    def test_no_condition_id_leak(self):
        entries = manifests.build_preview_manifest_entries(["B3", "B5"])
        blind_rows, _ = review.export_preview_review(entries)
        for row in blind_rows:
            self.assertNotIn("condition_id", row)
            self.assertNotIn("prompt_id", row)
            self.assertNotIn("block_id", row)

    def test_no_diversity_or_gallery_level_field(self):
        entries = manifests.build_preview_manifest_entries(["B3"])
        blind_rows, _ = review.export_preview_review(entries)
        row = blind_rows[0]
        self.assertNotIn("diversity_impression", row)
        self.assertNotIn("preference", row)

    def test_image_ids_unique(self):
        entries = manifests.build_preview_manifest_entries(["B3", "B5"])
        blind_rows, mapping = review.export_preview_review(entries)
        self.assertEqual(len({r["image_id"] for r in blind_rows}), len(blind_rows))
        self.assertEqual(len({m.image_id for m in mapping}), len(mapping))


class IngestPreviewReviewTests(unittest.TestCase):
    def setUp(self):
        self.entries = manifests.build_preview_manifest_entries(["B3", "B5"])
        self.blind_rows, self.mapping = review.export_preview_review(self.entries)
        self.expected_ids = [r["image_id"] for r in self.blind_rows]

    def test_roundtrip_happy_path(self):
        filled = [_fill_preview_row(r) for r in self.blind_rows]
        rows = review.ingest_preview_review(filled, self.expected_ids)
        self.assertEqual(len(rows), len(self.blind_rows))

    def test_condition_id_field_rejected(self):
        filled = [_fill_preview_row(r) for r in self.blind_rows]
        filled[0]["condition_id"] = "B3_plus"
        with self.assertRaises(ValueError):
            review.ingest_preview_review(filled, self.expected_ids)

    def test_missing_field_raises(self):
        filled = [_fill_preview_row(r) for r in self.blind_rows]
        del filled[0]["quality"]
        with self.assertRaises(ValueError):
            review.ingest_preview_review(filled, self.expected_ids)

    def test_invalid_enum_raises(self):
        filled = [_fill_preview_row(r) for r in self.blind_rows]
        filled[0]["quality"] = "excellent"
        with self.assertRaises(ValueError):
            review.ingest_preview_review(filled, self.expected_ids)

    def test_incomplete_set_raises(self):
        filled = [_fill_preview_row(r) for r in self.blind_rows][:-1]
        with self.assertRaises(ValueError):
            review.ingest_preview_review(filled, self.expected_ids)

    def test_unexpected_image_id_raises(self):
        filled = [_fill_preview_row(r) for r in self.blind_rows]
        filled[0]["image_id"] = "not_a_real_image_id"
        with self.assertRaises(ValueError):
            review.ingest_preview_review(filled, self.expected_ids)


class ExportGalleryReviewTests(unittest.TestCase):
    def test_export_empty_raises(self):
        with self.assertRaises(ValueError):
            review.export_gallery_review(())

    def test_groups_by_prompt_and_block(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3"])
        blind_rows, mapping = review.export_gallery_review(entries)
        self.assertEqual(len(blind_rows), manifests.NUM_PROMPT_BLOCK_PAIRS)

    def test_blind_labels_match_condition_count_one_candidate(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3"])
        blind_rows, _ = review.export_gallery_review(entries)
        for row in blind_rows:
            self.assertEqual(row["blind_labels"], ["A", "B", "C"])

    def test_blind_labels_match_condition_count_two_candidates(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3", "B5"])
        blind_rows, _ = review.export_gallery_review(entries)
        for row in blind_rows:
            self.assertEqual(row["blind_labels"], ["A", "B", "C", "D", "E"])

    def test_no_condition_identity_leak(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3", "B5"])
        blind_rows, _ = review.export_gallery_review(entries)
        for row in blind_rows:
            self.assertNotIn("condition_id", row)
            self.assertNotIn("prompt_id", row)
            self.assertNotIn("block_id", row)

    def test_mapping_covers_every_condition_with_four_images(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3", "B5"])
        _, mapping = review.export_gallery_review(entries)
        self.assertEqual(len(mapping), manifests.NUM_PROMPT_BLOCK_PAIRS * 5)
        for m in mapping:
            self.assertEqual(len(m.image_ids), manifests.NUM_BASE_INDICES_PER_BLOCK)

    def test_group_keys_unique(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3", "B5"])
        blind_rows, _ = review.export_gallery_review(entries)
        self.assertEqual(len({r["group_key"] for r in blind_rows}), len(blind_rows))

    def test_export_is_deterministic_across_calls(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3"])
        rows_a, mapping_a = review.export_gallery_review(entries)
        rows_b, mapping_b = review.export_gallery_review(entries)
        self.assertEqual([r["group_key"] for r in rows_a], [r["group_key"] for r in rows_b])
        self.assertEqual(mapping_a, mapping_b)


class IngestGalleryReviewTests(unittest.TestCase):
    def setUp(self):
        self.entries = manifests.build_full_pilot_manifest_entries(["B3", "B5"])
        self.blind_rows, self.mapping = review.export_gallery_review(self.entries)
        self.expected_group_keys = [r["group_key"] for r in self.blind_rows]
        self.blind_labels_by_group = {r["group_key"]: r["blind_labels"] for r in self.blind_rows}

    def test_roundtrip_happy_path(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        rows = review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)
        self.assertEqual(len(rows), len(self.blind_rows))

    def test_condition_identity_field_rejected(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        filled[0]["condition_id"] = "reference"
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)

    def test_preference_must_be_a_blind_label_or_tie_or_none(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        filled[0]["preference"] = "Z"
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)

    def test_preference_accepts_tie_and_none(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        filled[0]["preference"] = "tie"
        filled[1]["preference"] = "none"
        rows = review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)
        by_key = {r.group_key: r for r in rows}
        self.assertEqual(by_key[filled[0]["group_key"]].preference, "tie")
        self.assertEqual(by_key[filled[1]["group_key"]].preference, "none")

    def test_ratings_must_cover_exactly_the_blind_labels(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        del filled[0]["ratings"][filled[0]["blind_labels"][0]]
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)

    def test_per_image_artifacts_wrong_length_raises(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        label = filled[0]["blind_labels"][0]
        filled[0]["ratings"][label]["per_image_artifacts"] = filled[0]["ratings"][label]["per_image_artifacts"][:2]
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)

    def test_invalid_diversity_enum_raises(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        label = filled[0]["blind_labels"][0]
        filled[0]["ratings"][label]["diversity_impression"] = "extreme"
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)

    def test_incomplete_set_raises(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows][:-1]
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)

    def test_unexpected_group_key_raises(self):
        filled = [_fill_gallery_row(r) for r in self.blind_rows]
        filled[0]["group_key"] = "not_a_real_group_key"
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(filled, self.expected_group_keys, self.blind_labels_by_group)


class CrossTrackSchemaSeparationTests(unittest.TestCase):
    def test_preview_loader_rejects_gallery_shaped_row(self):
        entries = manifests.build_full_pilot_manifest_entries(["B3"])
        gallery_blind_rows, _ = review.export_gallery_review(entries)
        with self.assertRaises(ValueError):
            review.ingest_preview_review(gallery_blind_rows, [r["group_key"] for r in gallery_blind_rows])

    def test_gallery_loader_rejects_preview_shaped_row(self):
        entries = manifests.build_preview_manifest_entries(["B3"])
        preview_blind_rows, _ = review.export_preview_review(entries)
        with self.assertRaises(ValueError):
            review.ingest_gallery_review(preview_blind_rows, [r["image_id"] for r in preview_blind_rows], {})

    def test_probe_loader_rejects_preview_shaped_row(self):
        preview_entries = manifests.build_preview_manifest_entries(["B3"])
        preview_blind_rows, _ = review.export_preview_review(preview_entries)
        with self.assertRaises(ValueError):
            review.ingest_probe_review(preview_blind_rows, [r["image_id"] for r in preview_blind_rows])


class EffectPreviewGridTests(unittest.TestCase):
    def test_grid_has_paired_rows_and_reference_plus_condition_columns(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "grid.png"
            review.export_effect_preview_grid(Path(tmp) / "run", ["dose_a", "dose_b"], output)
            self.assertTrue(output.exists())
            with Image.open(output) as grid:
                self.assertGreater(grid.width, 300)
                self.assertGreater(grid.height, 1500)
                self.assertGreater(grid.height, grid.width)


if __name__ == "__main__":
    unittest.main()
