"""Phase C: the three independent, blind, human-annotation review tracks
(plan §4.4/§7.1/§8/§14.5).

Probe review, preview review, and gallery review share one shape -- a human
fills in a template of blank fields, code only validates the filled-in file
and applies fixed, deterministic rules to it -- but each has its own
export/ingest schema because its review *unit* differs (a single-image-pair,
a single image, or a 4-image gallery) and none may be conflated:

- *Probe review*: 72 (prompt, seed-block, PC-group) pairs, each comparing the
  block's one reference image against that group's rotated intervention
  image. Feeds ``select_candidates()`` and (via ``compute_rho020_trigger``,
  re-exported from ``probing.py``) the rho=0.20 uniform-supplement gate.
- *Preview review*: single condition-blinded images, sized by the actual
  preview-run manifest (0/36/60, branching on how many candidates
  ``select_candidates()`` returned).
- *Gallery review*: 4-image galleries, grouped and condition-blinded per
  (prompt, seed-block) so a reviewer can compare all conditions for that draw
  side by side; sized by the actual full-pilot manifest (0/144-in-3-configs/
  240-in-5-configs).

Every export function returns its blind template and its un-exported
identity mapping as plain in-memory structures -- writing them to disk is the
CLI's job (not yet implemented), mirroring how ``probing.py`` returns
dataclasses rather than touching the filesystem itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

from pc_specific_psd import manifests
from pc_specific_psd.compat_generation import sha256_text
from pc_specific_psd.probing import (
    ROLE_FIELDS,
    ProbeAnnotation,
    ProbeEntry,
    Rho020TriggerResult,
)
from pc_specific_psd.probing import compute_rho020_trigger as _compute_rho020_trigger

_MISSING = object()  # sentinel: distinguishes "key absent" from any legitimate value, including None-like "NA"

STRUCTURAL_ROLE_FIELDS: tuple[str, ...] = ("layout", "pose", "shape")
APPEARANCE_ROLE_FIELDS: tuple[str, ...] = ("color", "light", "texture")
assert set(STRUCTURAL_ROLE_FIELDS) | set(APPEARANCE_ROLE_FIELDS) == set(ROLE_FIELDS)

ROLE_RATING_VALUES = (0, 1, 2, "NA")
SHAPE_CHANGE_KINDS = ("reasonable_change", "distortion", "not_applicable")
YES_NO_UNCERTAIN = ("yes", "no", "uncertain")
ARTIFACT_LEVELS = ("absent", "present", "uncertain", "NA")
ARTIFACT_FIELDS = ("extra_object", "incorrect_repetition", "subject_deformation")
CONFIDENCE_LEVELS = ("clear", "uncertain")

CLEAR_CHANGE_RATING = 2
LOCAL_PASS_MIN_BASE_DRAWS = 2  # of 3 base draws within one prompt (plan §14.5)
LOCAL_PASS_TOTAL_BASE_DRAWS = 3
RECURRENCE_MIN_PROMPTS = 2  # of 4 prompts (plan §14.5)
MAX_CANDIDATES = 2

PAIR_ID_NAMESPACE = "pc_probe_review_pair_v1"
PROBE_REVIEW_SHUFFLE_SEED = 20260830  # distinct from PROBE_DONOR_MASTER_SEED (20260829)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_enum(value, allowed: Sequence, field_name: str, row_id: str) -> None:
    if value not in allowed:
        raise ValueError(f"{row_id}: field {field_name!r} has invalid value {value!r}, expected one of {allowed}")


def _get_required(raw: Mapping, field_name: str, row_id: str):
    value = raw.get(field_name, _MISSING)
    if value is _MISSING or value is None:
        raise ValueError(f"{row_id}: required field {field_name!r} is missing")
    return value


def _validate_role_ratings(raw: Mapping, row_id: str) -> dict:
    ratings = {}
    for field in ROLE_FIELDS:
        value = _get_required(raw, field, row_id)
        _validate_enum(value, ROLE_RATING_VALUES, field, row_id)
        ratings[field] = value
    return ratings


def _validate_artifact_block(raw: Mapping, block_name: str, row_id: str) -> dict:
    block = _get_required(raw, block_name, row_id)
    _require(isinstance(block, Mapping), f"{row_id}: field {block_name!r} must be a mapping of artifact type -> level")
    result = {}
    for field in ARTIFACT_FIELDS:
        value = _get_required(block, field, f"{row_id}.{block_name}")
        _validate_enum(value, ARTIFACT_LEVELS, field, f"{row_id}.{block_name}")
        result[field] = value
    return result


# == Track 1: probe review ====================================================


@dataclass(frozen=True)
class ProbeReviewRow:
    """One blind-reviewed (reference, intervention) pair -- the full §14.5
    field table. Unblinded identity (``group_id``/``rho``/``donor_seed``) is
    looked up from the separate mapping ``export_probe_review`` returns, not
    carried on this row.
    """
    pair_id: str
    prompt_id: str
    block_id: str
    role_ratings: dict  # ROLE_FIELDS -> 0 | 1 | 2 | "NA"
    shape_change_kind: str  # SHAPE_CHANGE_KINDS
    prompt_plausibility_before: str
    prompt_plausibility_after: str
    artifacts_before: dict  # ARTIFACT_FIELDS -> ARTIFACT_LEVELS
    artifacts_after: dict
    valid_structural_change: str  # YES_NO_UNCERTAIN
    evidence: str
    confidence: str  # CONFIDENCE_LEVELS

    def to_probe_annotation(self, group_id: str) -> ProbeAnnotation:
        """Reduces this row to the shape ``probing.compute_rho020_trigger``
        consumes -- the rho=0.20 trigger only needs the six role ratings.
        ``group_id`` must come from this row's ``ProbeReviewMappingEntry``
        (looked up by ``pair_id``): a ``ProbeReviewRow`` is intentionally
        blind to its own group identity, so unblinding it is the caller's
        responsibility, not something this method can infer on its own.
        """
        return ProbeAnnotation(prompt_id=self.prompt_id, block_id=self.block_id, group_id=group_id, role_ratings=self.role_ratings)


@dataclass(frozen=True)
class ProbeReviewMappingEntry:
    """The un-exported (pair_id -> real identity) record. Never shown to the
    blind reviewer; only used to unblind results after ingestion.
    """
    pair_id: str
    prompt_id: str
    block_id: str
    group_id: str
    donor_seed: int
    rho: Optional[float]
    reference_image_id: str
    intervention_image_id: str


def _probe_pair_id(entry: ProbeEntry) -> str:
    digest = sha256_text(PAIR_ID_NAMESPACE, entry.prompt_id, entry.block_id, entry.group_id, entry.donor_seed, entry.theta)
    return f"pair_{digest[:16]}"


def export_probe_review(intervention_entries: Sequence[ProbeEntry]) -> tuple[tuple[dict, ...], tuple[ProbeReviewMappingEntry, ...]]:
    """Builds the blind template (one blank row per intervention entry) and
    the separate, un-exported identity mapping. ``intervention_entries``
    should be ``probing.build_probing_manifest()``'s 72 intervention entries
    (references are reused as the "before" side of every pair, so they don't
    get their own row). Row order is deterministically shuffled so a
    reviewer cannot infer group identity from a fixed per-block position.
    """
    entries = [e for e in intervention_entries if e.condition_type == "intervention"]
    _require(len(entries) > 0, "export_probe_review requires at least one intervention entry")

    mapping: list[ProbeReviewMappingEntry] = []
    blind_rows: list[dict] = []
    seen_pair_ids: set[str] = set()
    for entry in entries:
        pair_id = _probe_pair_id(entry)
        _require(pair_id not in seen_pair_ids, f"duplicate pair_id {pair_id} for entry {entry.image_id}")
        seen_pair_ids.add(pair_id)
        reference_image_id = f"{entry.block_id}_b{entry.base_index}_reference"
        mapping.append(ProbeReviewMappingEntry(
            pair_id=pair_id, prompt_id=entry.prompt_id, block_id=entry.block_id,
            group_id=entry.group_id, donor_seed=entry.donor_seed, rho=None,
            reference_image_id=reference_image_id, intervention_image_id=entry.image_id,
        ))
        blind_rows.append({
            "pair_id": pair_id,
            "prompt_id": entry.prompt_id,
            "block_id": entry.block_id,
            "prompt_text": entry.prompt_text,
            **{field: None for field in ROLE_FIELDS},
            "shape_change_kind": None,
            "prompt_plausibility_before": None,
            "prompt_plausibility_after": None,
            "artifacts_before": {field: None for field in ARTIFACT_FIELDS},
            "artifacts_after": {field: None for field in ARTIFACT_FIELDS},
            "valid_structural_change": None,
            "evidence": None,
            "confidence": None,
        })

    import random
    random.Random(PROBE_REVIEW_SHUFFLE_SEED).shuffle(blind_rows)

    return tuple(blind_rows), tuple(mapping)


def ingest_probe_review(raw_rows: Sequence[Mapping], expected_pair_ids: Sequence[str]) -> tuple[ProbeReviewRow, ...]:
    """Validates and parses a filled-in probe-review file. Every field is
    required and enum-checked; ``None``/absent is never silently coerced to a
    default value (e.g. "0" or "absent") -- both raise.
    """
    expected = set(expected_pair_ids)
    seen: dict[str, ProbeReviewRow] = {}
    for raw in raw_rows:
        pair_id = _get_required(raw, "pair_id", "<row>")
        row_id = f"pair {pair_id}"
        _require(pair_id in expected, f"{row_id}: not one of the expected pair_ids for this run")
        _require(pair_id not in seen, f"{row_id}: duplicate row")
        prompt_id = _get_required(raw, "prompt_id", row_id)
        block_id = _get_required(raw, "block_id", row_id)

        role_ratings = _validate_role_ratings(raw, row_id)
        shape_change_kind = _get_required(raw, "shape_change_kind", row_id)
        _validate_enum(shape_change_kind, SHAPE_CHANGE_KINDS, "shape_change_kind", row_id)

        plausibility_before = _get_required(raw, "prompt_plausibility_before", row_id)
        _validate_enum(plausibility_before, YES_NO_UNCERTAIN, "prompt_plausibility_before", row_id)
        plausibility_after = _get_required(raw, "prompt_plausibility_after", row_id)
        _validate_enum(plausibility_after, YES_NO_UNCERTAIN, "prompt_plausibility_after", row_id)

        artifacts_before = _validate_artifact_block(raw, "artifacts_before", row_id)
        artifacts_after = _validate_artifact_block(raw, "artifacts_after", row_id)

        valid_structural_change = _get_required(raw, "valid_structural_change", row_id)
        _validate_enum(valid_structural_change, YES_NO_UNCERTAIN, "valid_structural_change", row_id)

        evidence = _get_required(raw, "evidence", row_id)
        _require(isinstance(evidence, str) and evidence.strip() != "", f"{row_id}: evidence must be a non-empty string")

        confidence = _get_required(raw, "confidence", row_id)
        _validate_enum(confidence, CONFIDENCE_LEVELS, "confidence", row_id)

        seen[pair_id] = ProbeReviewRow(
            pair_id=pair_id, prompt_id=prompt_id, block_id=block_id,
            role_ratings=role_ratings, shape_change_kind=shape_change_kind,
            prompt_plausibility_before=plausibility_before, prompt_plausibility_after=plausibility_after,
            artifacts_before=artifacts_before, artifacts_after=artifacts_after,
            valid_structural_change=valid_structural_change, evidence=evidence, confidence=confidence,
        )

    missing = expected - set(seen)
    _require(not missing, f"{len(missing)} probe-review row(s) missing: {sorted(missing)[:5]}...")
    return tuple(seen[pid] for pid in expected_pair_ids)


def compute_rho020_trigger(
    rows: Sequence[ProbeReviewRow],
    mapping: Sequence[ProbeReviewMappingEntry],
    **kwargs,
) -> Rho020TriggerResult:
    """Re-exports ``probing.compute_rho020_trigger`` (plan: "review.py also
    exposes compute_rho020_trigger(), used by probing.py") over ingested
    ``ProbeReviewRow`` data, unblinding each row's ``group_id`` via
    ``mapping`` rather than duplicating the trigger's counting logic here.
    """
    group_by_pair = {m.pair_id: m.group_id for m in mapping}
    annotations = tuple(row.to_probe_annotation(group_by_pair[row.pair_id]) for row in rows)
    return _compute_rho020_trigger(annotations, **kwargs)


# -- select_candidates(): the §14.5 nomination rule ---------------------------


@dataclass(frozen=True)
class GroupNominationEvidence:
    group_id: str
    nominated_role_fields: tuple[str, ...]  # role fields reaching >=2/4 prompts recurrence
    locally_passing_role_fields: tuple[str, ...]  # role fields reaching >=1/4 prompts (weaker, for reserve)
    coverage: int  # distinct prompts where >=1 nominated role field locally passed
    valid_change_count: int  # raw tally of qualifying (prompt, block, role_field) occurrences, nominated fields only


@dataclass(frozen=True)
class CandidateSelectionResult:
    candidates: tuple[str, ...]  # 0, 1, or 2 PC group ids, in priority order
    reserve: tuple[str, ...]  # groups with partial evidence, not selected, in priority order
    evidence_by_group: dict  # group_id -> GroupNominationEvidence, for provenance/debugging


def _role_qualifies(row: ProbeReviewRow, role_field: str) -> bool:
    """A role field counts as "clear, valid change" on one row. For
    structural fields (layout/pose/shape) this additionally requires
    ``valid_structural_change == "yes"``, since the plan defines that field
    precisely to exclude error/distortion-driven "changes" from counting as
    genuine structural evidence. Appearance fields (color/light/texture)
    have no analogous validity field in the plan's schema, so a bare
    ``rating == 2`` is used for them (plan §14.5: "颜色、光照、纹理...各自0/1/2").
    """
    if row.role_ratings.get(role_field) != CLEAR_CHANGE_RATING:
        return False
    if role_field in STRUCTURAL_ROLE_FIELDS:
        return row.valid_structural_change == "yes"
    return True


def select_candidates(
    rows: Sequence[ProbeReviewRow],
    mapping: Sequence[ProbeReviewMappingEntry],
    *,
    groups: tuple[manifests.PCGroup, ...] = manifests.PC_GROUPS,
    prompts: tuple[manifests.Prompt, ...] = manifests.PROMPTS,
) -> CandidateSelectionResult:
    """The plan §14.5 nomination rule, applied automatically once
    ``ingest_probe_review`` has succeeded (no separate manual-approval step
    in code): a role field is "locally passing" for a group in one prompt
    when it qualifies (see ``_role_qualifies``) in >= 2 of that prompt's 3
    base draws; it is "nominated" when it locally passes in >= 2 of the 4
    prompts (the recurring *category*, not a consistent direction). A group
    enters the candidate pool if it has >=1 nominated role field.

    Ranking within the pool is (coverage desc, valid_change_count desc,
    declared PC_GROUPS order asc) -- the plan's own "engineering rule, not a
    significance test" framing licenses this as one reasonable, deterministic
    tie-break rather than a stricter statistical procedure. Per the plan's
    explicit preference ("优先至少一个布局/姿态/形状候选"), the top slot is filled
    from the structural-capable pool when non-empty; the second slot may come
    from either pool. The plan's softer "作用互补优于两个几乎相同的组" preference
    is a qualitative note, not a computable similarity metric, and is
    intentionally left as a human judgment call at candidate confirmation
    time rather than encoded here.
    """
    group_by_pair = {m.pair_id: m.group_id for m in mapping}
    declared_order = {g.group_id: i for i, g in enumerate(groups)}

    by_group_role_prompt_count: dict[str, dict[str, dict[str, int]]] = {
        g.group_id: {field: {p.prompt_id: 0 for p in prompts} for field in ROLE_FIELDS} for g in groups
    }
    for row in rows:
        group_id = group_by_pair.get(row.pair_id)
        if group_id is None or group_id not in by_group_role_prompt_count:
            continue
        for field in ROLE_FIELDS:
            if _role_qualifies(row, field):
                by_group_role_prompt_count[group_id][field][row.prompt_id] += 1

    evidence_by_group: dict[str, GroupNominationEvidence] = {}
    for group in groups:
        role_prompts_passing: dict[str, set] = {field: set() for field in ROLE_FIELDS}
        for field in ROLE_FIELDS:
            for prompt in prompts:
                if by_group_role_prompt_count[group.group_id][field][prompt.prompt_id] >= LOCAL_PASS_MIN_BASE_DRAWS:
                    role_prompts_passing[field].add(prompt.prompt_id)

        nominated_fields = tuple(f for f in ROLE_FIELDS if len(role_prompts_passing[f]) >= RECURRENCE_MIN_PROMPTS)
        locally_passing_fields = tuple(f for f in ROLE_FIELDS if len(role_prompts_passing[f]) >= 1)

        covered_prompts: set = set()
        for f in nominated_fields:
            covered_prompts |= role_prompts_passing[f]
        valid_change_count = sum(
            by_group_role_prompt_count[group.group_id][f][p.prompt_id]
            for f in nominated_fields
            for p in prompts
        )

        evidence_by_group[group.group_id] = GroupNominationEvidence(
            group_id=group.group_id, nominated_role_fields=nominated_fields,
            locally_passing_role_fields=locally_passing_fields,
            coverage=len(covered_prompts), valid_change_count=valid_change_count,
        )

    candidates, reserve = _rank_and_select_candidates(evidence_by_group, declared_order)
    return CandidateSelectionResult(candidates=candidates, reserve=reserve, evidence_by_group=evidence_by_group)


def _rank_and_select_candidates(
    evidence_by_group: Mapping[str, GroupNominationEvidence], declared_order: Mapping[str, int],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The ranking/tie-break/candidate/reserve selection tail shared by
    ``select_candidates`` and ``select_candidates_from_group_review`` (Change
    2) -- both build the same ``GroupNominationEvidence`` shape from their own
    review track, then apply this one identical rule to it, so the two entry
    points can never silently drift apart on ranking behavior.
    """
    pool = [e for e in evidence_by_group.values() if e.nominated_role_fields]

    def rank_key(e: GroupNominationEvidence) -> tuple:
        return (-e.coverage, -e.valid_change_count, declared_order[e.group_id])

    structural_pool = sorted(
        (e for e in pool if any(f in STRUCTURAL_ROLE_FIELDS for f in e.nominated_role_fields)), key=rank_key
    )
    appearance_only_pool = sorted(
        (e for e in pool if not any(f in STRUCTURAL_ROLE_FIELDS for f in e.nominated_role_fields)), key=rank_key
    )

    candidates: list[str] = []
    if structural_pool:
        candidates.append(structural_pool[0].group_id)
    elif appearance_only_pool:
        candidates.append(appearance_only_pool[0].group_id)

    remaining_pool = sorted((e for e in pool if e.group_id not in candidates), key=rank_key)
    if remaining_pool and len(candidates) < MAX_CANDIDATES:
        candidates.append(remaining_pool[0].group_id)
    candidates = candidates[:MAX_CANDIDATES]

    reserve_evidence = sorted(
        (e for e in evidence_by_group.values() if e.group_id not in candidates and e.locally_passing_role_fields),
        key=rank_key,
    )
    reserve = tuple(e.group_id for e in reserve_evidence)
    return tuple(candidates), reserve


# == Track 2: preview review ===================================================


QUALITY_LEVELS = ("good", "acceptable", "poor")
PREVIEW_IMAGE_ID_NAMESPACE = "pc_preview_review_image_v1"


@dataclass(frozen=True)
class PreviewReviewRow:
    image_id: str  # anonymized -- does not embed condition_id
    prompt_plausibility: str  # YES_NO_UNCERTAIN
    artifacts: dict  # ARTIFACT_FIELDS -> ARTIFACT_LEVELS (single image, no before/after)
    quality: str  # QUALITY_LEVELS
    evidence: str
    confidence: str  # CONFIDENCE_LEVELS


@dataclass(frozen=True)
class PreviewReviewMappingEntry:
    image_id: str
    condition_id: str
    prompt_id: str
    block_id: str


def export_preview_review(entries: Sequence[manifests.ConditionDrawEntry]) -> tuple[tuple[dict, ...], tuple[PreviewReviewMappingEntry, ...]]:
    """Reads the actual preview-run manifest's length to size the package --
    refuses outright when it is empty (zero candidates were selected, so
    there is no preview run to review at all).
    """
    _require(len(entries) > 0, "no preview manifest entries: select_candidates() returned 0 candidates, there is nothing to review")

    mapping: list[PreviewReviewMappingEntry] = []
    blind_rows: list[dict] = []
    seen_ids: set[str] = set()
    for entry in entries:
        digest = sha256_text(PREVIEW_IMAGE_ID_NAMESPACE, entry.prompt_id, entry.block_id, entry.condition_id, entry.base_index)
        image_id = f"pv_{digest[:16]}"
        _require(image_id not in seen_ids, f"duplicate preview image_id {image_id}")
        seen_ids.add(image_id)
        mapping.append(PreviewReviewMappingEntry(image_id=image_id, condition_id=entry.condition_id, prompt_id=entry.prompt_id, block_id=entry.block_id))
        blind_rows.append({
            "image_id": image_id,
            "prompt_plausibility": None,
            "artifacts": {field: None for field in ARTIFACT_FIELDS},
            "quality": None,
            "evidence": None,
            "confidence": None,
        })

    import random
    random.Random(PROBE_REVIEW_SHUFFLE_SEED).shuffle(blind_rows)

    return tuple(blind_rows), tuple(mapping)


def ingest_preview_review(raw_rows: Sequence[Mapping], expected_image_ids: Sequence[str]) -> tuple[PreviewReviewRow, ...]:
    expected = set(expected_image_ids)
    seen: dict[str, PreviewReviewRow] = {}
    for raw in raw_rows:
        image_id = _get_required(raw, "image_id", "<row>")
        row_id = f"image {image_id}"
        _require(image_id in expected, f"{row_id}: not one of the expected image_ids for this run")
        _require(image_id not in seen, f"{row_id}: duplicate row")
        _require("condition_id" not in raw, f"{row_id}: preview review must not carry condition_id (blinded field)")

        plausibility = _get_required(raw, "prompt_plausibility", row_id)
        _validate_enum(plausibility, YES_NO_UNCERTAIN, "prompt_plausibility", row_id)
        artifacts = _validate_artifact_block(raw, "artifacts", row_id)
        quality = _get_required(raw, "quality", row_id)
        _validate_enum(quality, QUALITY_LEVELS, "quality", row_id)
        evidence = _get_required(raw, "evidence", row_id)
        _require(isinstance(evidence, str) and evidence.strip() != "", f"{row_id}: evidence must be a non-empty string")
        confidence = _get_required(raw, "confidence", row_id)
        _validate_enum(confidence, CONFIDENCE_LEVELS, "confidence", row_id)

        seen[image_id] = PreviewReviewRow(
            image_id=image_id, prompt_plausibility=plausibility, artifacts=artifacts,
            quality=quality, evidence=evidence, confidence=confidence,
        )

    missing = expected - set(seen)
    _require(not missing, f"{len(missing)} preview-review row(s) missing: {sorted(missing)[:5]}...")
    return tuple(seen[iid] for iid in expected_image_ids)


EXPECTED_RMS_CHANGE_FIELDS: tuple[str, ...] = (
    "composition",
    "position",
    "pose",
    "shape",
    "color_lighting",
    "texture",
    "structural_breakage",
    "extra_or_repeated_objects",
)


@dataclass(frozen=True)
class ExpectedRMSPreviewReviewRow:
    image_id: str
    change_ratings: dict
    prompt_plausibility: str
    artifacts: dict
    quality: str
    evidence: str
    change_evidence: str
    confidence: str


def export_expected_rms_preview_review(
    entries: Sequence[manifests.ConditionDrawEntry],
) -> tuple[tuple[dict, ...], tuple[PreviewReviewMappingEntry, ...]]:
    """Export the expected-RMS review schema without changing legacy previews."""
    rows, mapping = export_preview_review(entries)
    expanded = []
    for row in rows:
        expanded.append({
            **row,
            **{field: None for field in EXPECTED_RMS_CHANGE_FIELDS},
            "change_evidence": None,
        })
    return tuple(expanded), mapping


def ingest_expected_rms_preview_review(
    raw_rows: Sequence[Mapping],
    expected_image_ids: Sequence[str],
) -> tuple[ExpectedRMSPreviewReviewRow | PreviewReviewRow, ...]:
    """Read the expanded schema, while accepting pre-expansion review files.

    A file with none of the expected-RMS fields is treated as a legacy preview
    review. Mixed or partially populated expanded schemas fail validation.
    """
    any_expanded = any(
        any(field in raw for field in EXPECTED_RMS_CHANGE_FIELDS)
        or "change_evidence" in raw
        for raw in raw_rows
    )
    base_rows = ingest_preview_review(raw_rows, expected_image_ids)
    if not any_expanded:
        return base_rows
    raw_by_id = {raw.get("image_id"): raw for raw in raw_rows}
    result = []
    for base in base_rows:
        raw = raw_by_id[base.image_id]
        ratings = {}
        for field in EXPECTED_RMS_CHANGE_FIELDS:
            value = _get_required(raw, field, f"image {base.image_id}")
            _validate_enum(value, ROLE_RATING_VALUES, field, f"image {base.image_id}")
            ratings[field] = value
        change_evidence = _get_required(raw, "change_evidence", f"image {base.image_id}")
        _require(
            isinstance(change_evidence, str) and change_evidence.strip() != "",
            f"image {base.image_id}: change_evidence must be a non-empty string",
        )
        result.append(ExpectedRMSPreviewReviewRow(
            image_id=base.image_id,
            change_ratings=ratings,
            prompt_plausibility=base.prompt_plausibility,
            artifacts=base.artifacts,
            quality=base.quality,
            evidence=base.evidence,
            change_evidence=change_evidence,
            confidence=base.confidence,
        ))
    return tuple(result)


# == Track 3: gallery review ====================================================


DIVERSITY_LEVELS = ("low", "medium", "high")
_BLIND_LABELS = tuple("ABCDE")  # up to 5 configs (Reference/A+/A-/B+/B-)


@dataclass(frozen=True)
class GalleryRatings:
    per_image_artifacts: tuple[dict, ...]  # length 4, each ARTIFACT_FIELDS -> ARTIFACT_LEVELS
    plausibility: str  # YES_NO_UNCERTAIN, per gallery
    diversity_impression: str  # DIVERSITY_LEVELS, per gallery


@dataclass(frozen=True)
class GalleryReviewRow:
    group_key: str  # anonymized identity for one (prompt_id, block_id)
    blind_labels: tuple[str, ...]  # this group's condition count -> labels actually used, e.g. ("A","B","C")
    ratings: dict  # blind_label -> GalleryRatings
    preference: str  # one of blind_labels, or "tie", or "none"
    evidence: str
    confidence: str  # CONFIDENCE_LEVELS


@dataclass(frozen=True)
class GalleryReviewMappingEntry:
    group_key: str
    prompt_id: str
    block_id: str
    blind_label: str
    condition_id: str
    image_ids: tuple[str, ...]  # the 4 image_ids behind this blind_label, base_index order


def _group_key(prompt_id: str, block_id: str) -> str:
    digest = sha256_text("pc_gallery_review_group_v1", prompt_id, block_id)
    return f"grp_{digest[:16]}"


def export_gallery_review(entries: Sequence[manifests.ConditionDrawEntry]) -> tuple[tuple[dict, ...], tuple[GalleryReviewMappingEntry, ...]]:
    """Reads the actual full-pilot manifest's length to size the package --
    refuses outright when empty (zero candidates, no pilot to review). Groups
    entries by (prompt_id, block_id) so every condition's 4-image gallery for
    that draw is reviewed side by side, with condition identity blinded to
    labels A/B/C/... in declared-condition order (deterministic, not
    randomized -- randomizing which physical label a condition gets per group
    would need a per-group shuffle key; declared order is simpler and the
    condition identity itself, not its position, is what the review must
    hide, since ``mapping`` -- never exported to the reviewer -- is what
    resolves labels back to conditions).
    """
    _require(len(entries) > 0, "no full-pilot manifest entries: select_candidates() returned 0 candidates, there is nothing to review")

    galleries: dict[tuple[str, str, str], dict[int, manifests.ConditionDrawEntry]] = {}
    for entry in entries:
        key = (entry.condition_id, entry.prompt_id, entry.block_id)
        galleries.setdefault(key, {})[entry.base_index] = entry

    groups: dict[tuple[str, str], dict[str, list[manifests.ConditionDrawEntry]]] = {}
    for (condition_id, prompt_id, block_id), by_base_index in galleries.items():
        _require(set(by_base_index) == set(manifests.FULL_PILOT_BASE_INDICES), f"incomplete gallery for {condition_id}/{prompt_id}/{block_id}")
        ordered = [by_base_index[i] for i in manifests.FULL_PILOT_BASE_INDICES]
        groups.setdefault((prompt_id, block_id), {})[condition_id] = ordered

    mapping: list[GalleryReviewMappingEntry] = []
    blind_rows: list[dict] = []
    for (prompt_id, block_id), conditions_in_group in sorted(groups.items()):
        # Blind labels follow first-appearance order in `entries`, which already reflects
        # `manifests.condition_ids_for_candidates`'s fixed declared order (reference, then
        # each candidate's +/- pair) -- not alphabetical, so "reference" isn't privileged.
        seen_order = []
        for entry in entries:
            if entry.prompt_id == prompt_id and entry.block_id == block_id and entry.condition_id not in seen_order:
                seen_order.append(entry.condition_id)
        _require(len(seen_order) <= len(_BLIND_LABELS), f"too many conditions ({len(seen_order)}) to blind-label for {prompt_id}/{block_id}")

        group_key = _group_key(prompt_id, block_id)
        blind_labels = tuple(_BLIND_LABELS[: len(seen_order)])
        for label, condition_id in zip(blind_labels, seen_order):
            gallery_entries = conditions_in_group[condition_id]
            mapping.append(GalleryReviewMappingEntry(
                group_key=group_key, prompt_id=prompt_id, block_id=block_id,
                blind_label=label, condition_id=condition_id,
                image_ids=tuple(e.image_id for e in gallery_entries),
            ))

        blind_rows.append({
            "group_key": group_key,
            "blind_labels": list(blind_labels),
            "ratings": {
                label: {
                    "per_image_artifacts": [{field: None for field in ARTIFACT_FIELDS} for _ in range(manifests.NUM_BASE_INDICES_PER_BLOCK)],
                    "plausibility": None,
                    "diversity_impression": None,
                }
                for label in blind_labels
            },
            "preference": None,
            "evidence": None,
            "confidence": None,
        })

    import random
    random.Random(PROBE_REVIEW_SHUFFLE_SEED).shuffle(blind_rows)

    return tuple(blind_rows), tuple(mapping)


def _validate_gallery_ratings(raw: Mapping, blind_label: str, row_id: str) -> GalleryRatings:
    per_image_raw = _get_required(raw, "per_image_artifacts", f"{row_id}.{blind_label}")
    _require(isinstance(per_image_raw, Sequence) and len(per_image_raw) == manifests.NUM_BASE_INDICES_PER_BLOCK,
             f"{row_id}.{blind_label}: per_image_artifacts must have exactly {manifests.NUM_BASE_INDICES_PER_BLOCK} entries")
    per_image = tuple(
        _validate_artifact_block({"artifacts": img}, "artifacts", f"{row_id}.{blind_label}[{i}]")
        for i, img in enumerate(per_image_raw)
    )
    plausibility = _get_required(raw, "plausibility", f"{row_id}.{blind_label}")
    _validate_enum(plausibility, YES_NO_UNCERTAIN, "plausibility", f"{row_id}.{blind_label}")
    diversity = _get_required(raw, "diversity_impression", f"{row_id}.{blind_label}")
    _validate_enum(diversity, DIVERSITY_LEVELS, "diversity_impression", f"{row_id}.{blind_label}")
    return GalleryRatings(per_image_artifacts=per_image, plausibility=plausibility, diversity_impression=diversity)


def ingest_gallery_review(raw_rows: Sequence[Mapping], expected_group_keys: Sequence[str], blind_labels_by_group: Mapping[str, Sequence[str]]) -> tuple[GalleryReviewRow, ...]:
    expected = set(expected_group_keys)
    seen: dict[str, GalleryReviewRow] = {}
    for raw in raw_rows:
        group_key = _get_required(raw, "group_key", "<row>")
        row_id = f"group {group_key}"
        _require(group_key in expected, f"{row_id}: not one of the expected group_keys for this run")
        _require(group_key not in seen, f"{row_id}: duplicate row")
        _require("condition_id" not in raw and "condition_ids" not in raw, f"{row_id}: gallery review must not carry condition identity (blinded field)")

        expected_labels = tuple(blind_labels_by_group[group_key])
        ratings_raw = _get_required(raw, "ratings", row_id)
        _require(set(ratings_raw) == set(expected_labels), f"{row_id}: ratings must cover exactly the blind labels {expected_labels}")
        ratings = {label: _validate_gallery_ratings(ratings_raw[label], label, row_id) for label in expected_labels}

        preference = _get_required(raw, "preference", row_id)
        _validate_enum(preference, expected_labels + ("tie", "none"), "preference", row_id)

        evidence = _get_required(raw, "evidence", row_id)
        _require(isinstance(evidence, str) and evidence.strip() != "", f"{row_id}: evidence must be a non-empty string")
        confidence = _get_required(raw, "confidence", row_id)
        _validate_enum(confidence, CONFIDENCE_LEVELS, "confidence", row_id)

        seen[group_key] = GalleryReviewRow(
            group_key=group_key, blind_labels=expected_labels, ratings=ratings,
            preference=preference, evidence=evidence, confidence=confidence,
        )

    missing = expected - set(seen)
    _require(not missing, f"{len(missing)} gallery-review row(s) missing: {sorted(missing)[:5]}...")
    return tuple(seen[key] for key in expected_group_keys)


# == Consolidated review artifacts (plan Change 2) ============================
#
# Unlike Tracks 1-3 above, these two artifacts are not blind: the plan never
# describes a blinding step for them, and each is a post-hoc, group/condition-
# level *consolidation* view for a reviewer who already knows which group or
# condition they are looking at (the blind, per-pair/per-image detail views
# above remain available, unmodified, for anyone who wants the full table).
# They are also the only review artifacts that write to disk themselves --
# ``workflow.py`` calls them directly, with no CLI subcommand in between, so
# the export/ingest functions do the CSV + contact-sheet I/O inline.

_GROUP_REVIEW_CAVEAT = "这是多 seed 支持的探索性筛选，不额外宣称做过独立的候选确认实验."

_CONTACT_SHEET_CELL_SIZE = (128, 128)
_CONTACT_SHEET_MARGIN = 8
_CONTACT_SHEET_LABEL_HEIGHT = 16


def _build_contact_sheet(cells: Sequence[Sequence[tuple[str, Optional[Path]]]]):
    """A grid of labeled thumbnails, one row per ``cells`` entry. Each cell is
    ``(label, image_path)``; ``image_path=None`` renders a blank gray
    placeholder (used when the referenced sample hasn't been generated,
    which should not normally happen for a run this function is called on,
    but is handled rather than crashing the export).
    """
    from PIL import Image, ImageDraw

    num_cols = max((len(row) for row in cells), default=0)
    num_rows = len(cells)
    cell_w, cell_h = _CONTACT_SHEET_CELL_SIZE
    cell_full_h = cell_h + _CONTACT_SHEET_LABEL_HEIGHT
    sheet_w = _CONTACT_SHEET_MARGIN + num_cols * (cell_w + _CONTACT_SHEET_MARGIN)
    sheet_h = _CONTACT_SHEET_MARGIN + num_rows * (cell_full_h + _CONTACT_SHEET_MARGIN)
    sheet = Image.new("RGB", (max(sheet_w, 1), max(sheet_h, 1)), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    for row_index, row in enumerate(cells):
        for col_index, (label, image_path) in enumerate(row):
            x = _CONTACT_SHEET_MARGIN + col_index * (cell_w + _CONTACT_SHEET_MARGIN)
            y = _CONTACT_SHEET_MARGIN + row_index * (cell_full_h + _CONTACT_SHEET_MARGIN)
            if image_path is not None and Path(image_path).exists():
                with Image.open(image_path) as source:
                    thumb = source.convert("RGB").copy()
                thumb.thumbnail(_CONTACT_SHEET_CELL_SIZE)
                sheet.paste(thumb, (x, y))
            else:
                draw.rectangle([x, y, x + cell_w, y + cell_h], fill=(200, 200, 200))
            draw.text((x, y + cell_h), label, fill=(0, 0, 0))

    return sheet


def export_effect_preview_grid(
    preview_run_dir: Path,
    condition_ids: Sequence[str],
    output_path: Path,
    *,
    condition_labels: Mapping[str, str] | None = None,
) -> Path:
    """Write the v2 comparison grid with one paired prompt/seed per row.

    The reference is always the first column; each following column is one
    unique frozen formal-operator condition for that exact base-noise draw.
    """
    ordered_conditions = ("reference", *tuple(dict.fromkeys(condition_ids)))
    cells = []
    for prompt, block in manifests.prompt_block_pairs():
        cells.append([
            (
                f"{prompt.prompt_id}/{block.block_id}/" + (
                    condition_labels.get(condition_id, condition_id) if condition_labels else condition_id
                ),
                Path(preview_run_dir) / "generations" / condition_id / block.block_id / "b0" / "image.png",
            )
            for condition_id in ordered_conditions
        ])
    sheet = _build_contact_sheet(cells)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path


# -- Touchpoint 1: probe-group review -----------------------------------------


@dataclass(frozen=True)
class ProbeGroupReviewRow:
    """One (group, prompt) aggregate, filled in against the group's
    contact-sheet PNG (4 prompts x 3 base draws for that group). Every field
    is ``None`` until answered -- see the module-level tri-state contract.
    """
    group_id: str
    prompt_id: str
    draws_with_change: Optional[int]  # 0-3 base draws showing a clear, attributable change
    main_change_type: Optional[str]  # one of ROLE_FIELDS, or "none"
    artifact_recurring: Optional[bool]  # True if the same artifact/distortion recurs across draws


_GROUP_REVIEW_CHANGE_TYPES = ROLE_FIELDS + ("none",)


def export_probe_group_review(
    config,
    run_dir: Path,
    *,
    groups: tuple[manifests.PCGroup, ...] = manifests.PC_GROUPS,
    prompts: tuple[manifests.Prompt, ...] = manifests.PROMPTS,
    output_dir: Optional[Path] = None,
) -> dict:
    """Builds one row per (group, prompt) -- len(groups) x len(prompts) rows
    -- from the same 84 probe images ``export_probe_review`` reads, aggregated
    to group level, and one contact-sheet PNG per group (its images across
    every prompt x base draw). Writes ``probe_group_review.csv`` plus the
    contact sheets to ``output_dir`` (default: alongside ``config``'s file).

    {caveat}
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else Path(config.config_path).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "probe_group_review.csv"
    contact_sheet_paths: dict[str, Path] = {}

    import csv as csv_module

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=["group_id", "prompt_id", "draws_with_change", "main_change_type", "artifact_recurring"])
        writer.writeheader()
        for group in groups:
            for prompt in prompts:
                writer.writerow({"group_id": group.group_id, "prompt_id": prompt.prompt_id, "draws_with_change": "", "main_change_type": "", "artifact_recurring": ""})

    for group in groups:
        rows = []
        for prompt in prompts:
            row_cells = []
            for block in manifests.seed_blocks_for_prompt(prompt):
                ref_path = run_dir / "probes" / "reference" / block.block_id / "b0" / "image.png"
                group_path = run_dir / "probes" / group.group_id / block.block_id / "b0" / "image.png"
                row_cells.append((f"{prompt.prompt_id}/{block.block_id}/ref", ref_path))
                row_cells.append((f"{prompt.prompt_id}/{block.block_id}/{group.group_id}", group_path))
            rows.append(row_cells)
        sheet = _build_contact_sheet(rows)
        sheet_path = output_dir / f"probe_group_review_{group.group_id}.png"
        sheet.save(sheet_path, format="PNG")
        contact_sheet_paths[group.group_id] = sheet_path

    return {"csv_path": csv_path, "contact_sheet_paths": contact_sheet_paths}


export_probe_group_review.__doc__ = export_probe_group_review.__doc__.format(caveat=_GROUP_REVIEW_CAVEAT)


@dataclass(frozen=True)
class ProbeGroupReviewIngestResult:
    status: str  # "missing" | "incomplete" | "complete"
    rows: tuple[ProbeGroupReviewRow, ...] = ()
    incomplete_keys: tuple[str, ...] = ()  # "<group_id>/<prompt_id>" entries still unanswered


def ingest_probe_group_review(
    path: Path,
    *,
    groups: tuple[manifests.PCGroup, ...] = manifests.PC_GROUPS,
    prompts: tuple[manifests.Prompt, ...] = manifests.PROMPTS,
) -> ProbeGroupReviewIngestResult:
    """Three-state ingest (module-level contract): missing file, present but
    incomplete (one or more rows still blank), or present and fully filled.
    """
    import csv as csv_module

    path = Path(path)
    if not path.exists():
        return ProbeGroupReviewIngestResult(status="missing")

    expected_keys = [f"{g.group_id}/{p.prompt_id}" for g in groups for p in prompts]
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv_module.DictReader(handle))

    seen: dict[str, Mapping] = {}
    for raw in raw_rows:
        key = f"{raw.get('group_id')}/{raw.get('prompt_id')}"
        _require(key in expected_keys, f"unexpected (group_id, prompt_id) row: {key!r}")
        _require(key not in seen, f"duplicate row for {key!r}")
        seen[key] = raw

    missing_keys = [key for key in expected_keys if key not in seen]
    _require(not missing_keys, f"probe-group-review file is missing rows for: {missing_keys[:5]}...")

    incomplete: list[str] = []
    rows: list[ProbeGroupReviewRow] = []
    for key in expected_keys:
        raw = seen[key]
        group_id, prompt_id = key.split("/", 1)
        draws_raw = (raw.get("draws_with_change") or "").strip()
        change_type_raw = (raw.get("main_change_type") or "").strip()
        recurring_raw = (raw.get("artifact_recurring") or "").strip().lower()
        if draws_raw == "" or change_type_raw == "" or recurring_raw == "":
            incomplete.append(key)
            continue
        _require(draws_raw.isdigit() and int(draws_raw) in (0, 1, 2, 3), f"{key}: draws_with_change must be 0-3, got {draws_raw!r}")
        _validate_enum(change_type_raw, _GROUP_REVIEW_CHANGE_TYPES, "main_change_type", key)
        _require(recurring_raw in ("true", "false"), f"{key}: artifact_recurring must be true/false, got {recurring_raw!r}")
        rows.append(ProbeGroupReviewRow(
            group_id=group_id, prompt_id=prompt_id, draws_with_change=int(draws_raw),
            main_change_type=change_type_raw, artifact_recurring=recurring_raw == "true",
        ))

    if incomplete:
        return ProbeGroupReviewIngestResult(status="incomplete", incomplete_keys=tuple(incomplete))
    return ProbeGroupReviewIngestResult(status="complete", rows=tuple(rows))


def _group_review_row_qualifies(row: ProbeGroupReviewRow) -> bool:
    """Mirrors ``_role_qualifies``: a row counts as a locally-passing draw
    only when it names a real role field (not ``"none"``/unanswered), meets
    the same-as-Track-1 minimum base-draw count, and the recurring artifact
    flag -- this track's closest analogue to ``valid_structural_change`` --
    does not disqualify it.
    """
    if row.main_change_type is None or row.main_change_type == "none":
        return False
    if row.draws_with_change is None or row.draws_with_change < LOCAL_PASS_MIN_BASE_DRAWS:
        return False
    return not row.artifact_recurring


def select_candidates_from_group_review(
    rows: Sequence[ProbeGroupReviewRow],
    *,
    groups: tuple[manifests.PCGroup, ...] = manifests.PC_GROUPS,
    prompts: tuple[manifests.Prompt, ...] = manifests.PROMPTS,
) -> CandidateSelectionResult:
    """The same nomination/ranking rule as ``select_candidates`` (shared via
    ``_rank_and_select_candidates``), read off the coarser
    ``ProbeGroupReviewRow`` shape instead of the 72-row detailed track.

    {caveat}
    """
    declared_order = {g.group_id: i for i, g in enumerate(groups)}
    row_by_key = {(r.group_id, r.prompt_id): r for r in rows}

    evidence_by_group: dict[str, GroupNominationEvidence] = {}
    for group in groups:
        role_prompts_passing: dict[str, set] = {field: set() for field in ROLE_FIELDS}
        valid_change_counts: dict[str, int] = {field: 0 for field in ROLE_FIELDS}
        for prompt in prompts:
            row = row_by_key.get((group.group_id, prompt.prompt_id))
            if row is None or not _group_review_row_qualifies(row):
                continue
            field = row.main_change_type
            role_prompts_passing[field].add(prompt.prompt_id)
            valid_change_counts[field] += row.draws_with_change

        nominated_fields = tuple(f for f in ROLE_FIELDS if len(role_prompts_passing[f]) >= RECURRENCE_MIN_PROMPTS)
        locally_passing_fields = tuple(f for f in ROLE_FIELDS if len(role_prompts_passing[f]) >= 1)

        covered_prompts: set = set()
        for f in nominated_fields:
            covered_prompts |= role_prompts_passing[f]
        valid_change_count = sum(valid_change_counts[f] for f in nominated_fields)

        evidence_by_group[group.group_id] = GroupNominationEvidence(
            group_id=group.group_id, nominated_role_fields=nominated_fields,
            locally_passing_role_fields=locally_passing_fields,
            coverage=len(covered_prompts), valid_change_count=valid_change_count,
        )

    candidates, reserve = _rank_and_select_candidates(evidence_by_group, declared_order)
    return CandidateSelectionResult(candidates=candidates, reserve=reserve, evidence_by_group=evidence_by_group)


select_candidates_from_group_review.__doc__ = select_candidates_from_group_review.__doc__.format(caveat=_GROUP_REVIEW_CAVEAT)


# -- Touchpoint 2: preview exclusion review -----------------------------------


@dataclass(frozen=True)
class PreviewExclusionRow:
    """One row per condition covered by the preview run. ``change_type`` is
    only meaningful when ``excluded=True``; both fields are ``None`` until
    answered.
    """
    condition_id: str
    excluded: Optional[bool]
    change_type: Optional[str]


def export_preview_exclusion_review(
    config,
    preview_run_dir: Path,
    candidate_group_ids: Sequence[str],
    *,
    output_dir: Optional[Path] = None,
) -> dict:
    """One row per condition the preview run covers (``reference`` plus each
    candidate group's +/- pair), asking only whether that condition shows
    obvious structural damage / extraneous objects / severe texture-color
    anomaly -- never a quality/preference score -- plus one contact-sheet PNG
    per condition (its images across the 12 (prompt, seed-block) draws).
    Writes ``preview_exclusion_review.csv`` plus the contact sheets to
    ``output_dir`` (default: alongside ``config``'s file).
    """
    preview_run_dir = Path(preview_run_dir)
    condition_ids = manifests.condition_ids_for_candidates(candidate_group_ids)
    _require(len(condition_ids) > 0, "no candidates: nothing to review")
    output_dir = Path(output_dir) if output_dir else Path(config.config_path).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "preview_exclusion_review.csv"
    contact_sheet_paths: dict[str, Path] = {}

    import csv as csv_module

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=["condition_id", "excluded", "change_type"])
        writer.writeheader()
        for condition_id in condition_ids:
            writer.writerow({"condition_id": condition_id, "excluded": "", "change_type": ""})

    for condition_id in condition_ids:
        row_cells = [
            (f"{prompt.prompt_id}/{block.block_id}", preview_run_dir / "generations" / condition_id / block.block_id / "b0" / "image.png")
            for prompt, block in manifests.prompt_block_pairs()
        ]
        wrap = 4
        cells = [row_cells[i:i + wrap] for i in range(0, len(row_cells), wrap)]
        sheet = _build_contact_sheet(cells)
        sheet_path = output_dir / f"preview_exclusion_review_{condition_id}.png"
        sheet.save(sheet_path, format="PNG")
        contact_sheet_paths[condition_id] = sheet_path

    return {"csv_path": csv_path, "contact_sheet_paths": contact_sheet_paths}


@dataclass(frozen=True)
class PreviewExclusionIngestResult:
    status: str  # "missing" | "incomplete" | "complete"
    rows: tuple[PreviewExclusionRow, ...] = ()
    incomplete_condition_ids: tuple[str, ...] = ()


def ingest_preview_exclusion_review(path: Path, candidate_group_ids: Sequence[str]) -> PreviewExclusionIngestResult:
    """Three-state ingest (module-level contract): missing file, present but
    incomplete, or present and fully filled.
    """
    import csv as csv_module

    path = Path(path)
    if not path.exists():
        return PreviewExclusionIngestResult(status="missing")

    expected_ids = list(manifests.condition_ids_for_candidates(candidate_group_ids))
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv_module.DictReader(handle))

    seen: dict[str, Mapping] = {}
    for raw in raw_rows:
        condition_id = raw.get("condition_id")
        _require(condition_id in expected_ids, f"unexpected condition_id row: {condition_id!r}")
        _require(condition_id not in seen, f"duplicate row for condition_id {condition_id!r}")
        seen[condition_id] = raw

    missing = [cid for cid in expected_ids if cid not in seen]
    _require(not missing, f"preview-exclusion-review file is missing rows for: {missing[:5]}...")

    incomplete: list[str] = []
    rows: list[PreviewExclusionRow] = []
    for condition_id in expected_ids:
        raw = seen[condition_id]
        excluded_raw = (raw.get("excluded") or "").strip().lower()
        if excluded_raw == "":
            incomplete.append(condition_id)
            continue
        _require(excluded_raw in ("true", "false"), f"{condition_id}: excluded must be true/false, got {excluded_raw!r}")
        excluded = excluded_raw == "true"
        change_type_raw = (raw.get("change_type") or "").strip()
        if excluded and change_type_raw == "":
            incomplete.append(condition_id)
            continue
        rows.append(PreviewExclusionRow(
            condition_id=condition_id, excluded=excluded,
            change_type=change_type_raw if change_type_raw else None,
        ))

    if incomplete:
        return PreviewExclusionIngestResult(status="incomplete", incomplete_condition_ids=tuple(incomplete))
    return PreviewExclusionIngestResult(status="complete", rows=tuple(rows))


def approved_condition_ids(rows: Sequence[PreviewExclusionRow]) -> tuple[str, ...]:
    """Condition ids not marked excluded, in the order they appear in
    ``rows``, always including ``"reference"`` regardless of its own row's
    value -- a ``reference`` row marked ``excluded=True`` is a hard-stop
    condition for the caller (``workflow.py``) to detect and raise on
    separately, not something this function silently special-cases away.
    """
    approved = [row.condition_id for row in rows if row.excluded is False]
    if "reference" not in approved:
        approved = ["reference"] + approved
    return tuple(approved)
    return tuple(seen[gk] for gk in expected_group_keys)
