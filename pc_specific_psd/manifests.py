"""Locked prompt/seed-block table (plan §14.2), PC-group partition (plan §4.1),
and budget calculators (plan §4.4/§7.1). Nothing here depends on model/basis
code, so every downstream module (probing, review, runner) can import this
table without pulling in any heavy dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pc_specific_psd.compat_generation import derived_seed

PROBE_DONOR_MASTER_SEED = 20260829
PROBE_DONOR_SEED_NAMESPACE = "pc_probe_donor_v1"

NUM_BASE_INDICES_PER_BLOCK = 4  # base_index in [0, 4): the formal 4-image galleries
PROBING_BASE_INDEX = 0          # role probing (plan §4.4) only ever uses base_index=0 per block
FULL_PILOT_BASE_INDEX_OFFSET = NUM_BASE_INDICES_PER_BLOCK  # full-pilot draws base_index in [4, 8):
# independent of every base_index probing/preview ever renders (both only ever use 0..3),
# so full-pilot's sample_seed values can never collide with an already-reviewed draw.


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str
    batch_seeds: tuple[int, int, int]  # one per seed block s000/s001/s002


PROMPTS: tuple[Prompt, ...] = (
    Prompt("p000", "A photo of a red fox in a snowy forest", (10000, 10100, 10200)),
    Prompt("p001", "A ceramic teapot on a wooden table", (11000, 11100, 11200)),
    Prompt("p002", "A photo of a dog", (12000, 12100, 12200)),
    Prompt("p003", "A photo of a chair", (13000, 13100, 13200)),
)

_PROMPTS_BY_ID = {p.prompt_id: p for p in PROMPTS}


def prompt_by_id(prompt_id: str) -> Prompt:
    try:
        return _PROMPTS_BY_ID[prompt_id]
    except KeyError as exc:
        raise KeyError(f"unknown prompt_id {prompt_id!r}") from exc


@dataclass(frozen=True)
class SeedBlock:
    prompt_id: str
    block_index: int  # 0, 1, or 2 -> s000/s001/s002
    batch_seed: int

    @property
    def block_id(self) -> str:
        return f"{self.prompt_id}_s{self.block_index:03d}"

    def sample_seed(self, base_index: int) -> int:
        upper = 2 * NUM_BASE_INDICES_PER_BLOCK
        if not 0 <= base_index < upper:
            raise ValueError(f"base_index must be in [0, {upper}), got {base_index}")
        return self.batch_seed + base_index


def seed_blocks_for_prompt(prompt: Prompt) -> tuple[SeedBlock, ...]:
    return tuple(SeedBlock(prompt.prompt_id, i, seed) for i, seed in enumerate(prompt.batch_seeds))


def all_seed_blocks() -> tuple[SeedBlock, ...]:
    blocks: list[SeedBlock] = []
    for prompt in PROMPTS:
        blocks.extend(seed_blocks_for_prompt(prompt))
    return tuple(blocks)


def prompt_block_pairs() -> tuple[tuple[Prompt, SeedBlock], ...]:
    """The 12 (prompt, seed-block) pairs, in a fixed deterministic order
    (prompt order, then block order) -- this is the unit both the preview and
    full-pilot manifests iterate over (plan §7.1).
    """
    pairs: list[tuple[Prompt, SeedBlock]] = []
    for prompt in PROMPTS:
        for block in seed_blocks_for_prompt(prompt):
            pairs.append((prompt, block))
    return tuple(pairs)


NUM_PROMPT_BLOCK_PAIRS = len(PROMPTS) * 3  # 12


def donor_seed(block_id: str, base_index: int, donor_index: int = 0) -> int:
    """``derived_seed(20260829, block_id, "pc_probe_donor_v1", base_index, donor_index)``
    per plan §14.2. Deliberately excludes ``group_id`` -- all PC groups being
    probed off the same base draw share one donor tensor.
    """
    return derived_seed(PROBE_DONOR_MASTER_SEED, block_id, PROBE_DONOR_SEED_NAMESPACE, base_index, donor_index)


@dataclass(frozen=True)
class PCGroup:
    group_id: str
    start_1based: int  # inclusive
    end_1based: int  # inclusive

    @property
    def size(self) -> int:
        return self.end_1based - self.start_1based + 1

    @property
    def python_slice(self) -> slice:
        """0-based half-open slice, directly usable against basis component columns."""
        return slice(self.start_1based - 1, self.end_1based)

    @property
    def indices(self) -> range:
        return range(self.start_1based - 1, self.end_1based)


PC_GROUPS: tuple[PCGroup, ...] = (
    PCGroup("B1", 1, 4),
    PCGroup("B2", 5, 8),
    PCGroup("B3", 9, 12),
    PCGroup("B4", 13, 16),
    PCGroup("B5", 17, 32),
    PCGroup("B6", 33, 100),
)

_PC_GROUPS_BY_ID = {g.group_id: g for g in PC_GROUPS}


def pc_group_by_id(group_id: str) -> PCGroup:
    try:
        return _PC_GROUPS_BY_ID[group_id]
    except KeyError as exc:
        raise KeyError(f"unknown group_id {group_id!r}") from exc


def validate_pc_groups(groups: Sequence[PCGroup] = PC_GROUPS, *, basis_dimension: int = 100) -> None:
    """Config-load-time sanity check: groups are 1-based, within
    ``[1, basis_dimension]``, and pairwise disjoint. Not a per-call hot path.
    """
    covered: set[int] = set()
    for group in groups:
        if group.start_1based < 1 or group.end_1based > basis_dimension or group.start_1based > group.end_1based:
            raise ValueError(
                f"PC group {group.group_id} range [{group.start_1based}, {group.end_1based}] "
                f"invalid for basis dimension {basis_dimension}"
            )
        span = set(range(group.start_1based, group.end_1based + 1))
        overlap = covered & span
        if overlap:
            raise ValueError(f"PC group {group.group_id} overlaps previously declared groups at indices {sorted(overlap)}")
        covered |= span


# -- Budget calculators (plan §4.4/§7.1) --------------------------------------

_NUM_BASE_DRAWS_PER_PROMPT_FOR_PROBING = 3  # base_index=0 from each of the 3 seed blocks


def probing_image_count(num_prompts: int = len(PROMPTS), num_groups: int = len(PC_GROUPS)) -> int:
    """4 prompts x 3 base draws x 6 groups (intervention) + 4 prompts x 3 base
    draws (reference) = 84 (plan §4.4).
    """
    return (
        num_prompts * _NUM_BASE_DRAWS_PER_PROMPT_FOR_PROBING * num_groups
        + num_prompts * _NUM_BASE_DRAWS_PER_PROMPT_FOR_PROBING
    )


def rho020_supplement_image_count(num_prompts: int = len(PROMPTS), num_groups: int = len(PC_GROUPS)) -> int:
    """All six groups x 4 prompts x 3 base draws = 72; the 12 existing
    per-prompt references are reused, not regenerated (plan §14.4).
    """
    return num_prompts * _NUM_BASE_DRAWS_PER_PROMPT_FOR_PROBING * num_groups


def num_configs_for_candidates(num_candidates: int) -> int:
    """0 -> 0 (no manifest at all); 1 -> 3 (Reference/cand+/cand-);
    2 -> 5 (Reference/A+/A-/B+/B-). ``select_candidates()`` must never return
    anything else.
    """
    if num_candidates == 0:
        return 0
    if num_candidates == 1:
        return 3
    if num_candidates == 2:
        return 5
    raise ValueError(f"select_candidates() must return 0, 1, or 2 candidates, got {num_candidates}")


def condition_ids_for_candidates(candidate_group_ids: Sequence[str]) -> tuple[str, ...]:
    """Reference plus a declared-order +/- pair per candidate PC group.
    Length always matches ``num_configs_for_candidates(len(candidate_group_ids))``.
    """
    if len(candidate_group_ids) == 0:
        return ()
    conditions = ["reference"]
    for group_id in candidate_group_ids:
        conditions.append(f"{group_id}_plus")
        conditions.append(f"{group_id}_minus")
    if len(conditions) != num_configs_for_candidates(len(candidate_group_ids)):
        raise ValueError(f"select_candidates() must return 0, 1, or 2 candidates, got {len(candidate_group_ids)}")
    return tuple(conditions)


def preview_image_count(num_candidates: int) -> int:
    """Single-image (``base_index=0`` only) draws across the 12 (prompt,
    seed-block) pairs: 0 -> 0, 1 -> 36 (3x12), 2 -> 60 (5x12).
    """
    return num_configs_for_candidates(num_candidates) * NUM_PROMPT_BLOCK_PAIRS


def full_pilot_image_count(num_candidates: int) -> int:
    """4-image galleries across the same 12 (prompt, seed-block) pairs:
    0 -> 0, 1 -> 144 (3x12x4), 2 -> 240 (5x12x4).
    """
    return num_configs_for_candidates(num_candidates) * NUM_PROMPT_BLOCK_PAIRS * NUM_BASE_INDICES_PER_BLOCK


# -- Preview/full-pilot run manifest entries ----------------------------------
#
# ``runner.py`` (not yet implemented) is the module that will eventually
# generate images against these manifests, but the *shape* of "the actual
# immutable run manifest" is needed now by review.py's preview/gallery review
# tracks, which must size and structure their review packages from the real
# manifest rather than a hardcoded count. Defining it once here -- reusing the
# same ``condition_ids_for_candidates``/``prompt_block_pairs`` primitives the
# budget calculators above already use -- means runner.py can adopt this exact
# entry shape later instead of inventing a second, incompatible one.


@dataclass(frozen=True)
class ConditionDrawEntry:
    condition_id: str
    prompt_id: str
    block_id: str
    base_index: int

    @property
    def image_id(self) -> str:
        return f"{self.condition_id}_{self.block_id}_b{self.base_index}"


def build_preview_manifest_entries(candidate_group_ids: Sequence[str]) -> tuple[ConditionDrawEntry, ...]:
    """Zero candidates -> empty manifest (no preview run at all); one/two
    candidates -> single-image (``base_index=0`` only) draws across all
    conditions x the 12 (prompt, seed-block) pairs (plan §7.1).
    """
    condition_ids = condition_ids_for_candidates(candidate_group_ids)
    if not condition_ids:
        return ()
    entries = [
        ConditionDrawEntry(condition_id, prompt.prompt_id, block.block_id, PROBING_BASE_INDEX)
        for condition_id in condition_ids
        for prompt, block in prompt_block_pairs()
    ]
    expected = preview_image_count(len(candidate_group_ids))
    if len(entries) != expected:
        raise AssertionError(f"preview manifest built {len(entries)} entries, expected {expected}")
    return tuple(entries)


FULL_PILOT_BASE_INDICES = range(FULL_PILOT_BASE_INDEX_OFFSET, FULL_PILOT_BASE_INDEX_OFFSET + NUM_BASE_INDICES_PER_BLOCK)


def build_full_pilot_manifest_entries(candidate_group_ids: Sequence[str]) -> tuple[ConditionDrawEntry, ...]:
    """Zero candidates -> empty manifest (no full-pilot run at all); one/two
    candidates -> 4-image galleries across all conditions x the 12 (prompt,
    seed-block) pairs, drawn at ``base_index`` in ``FULL_PILOT_BASE_INDICES``
    -- a range never touched by probing or preview -- so every full-pilot
    sample seed is independent of any image a human has already reviewed
    (plan §7.1/§8).
    """
    condition_ids = condition_ids_for_candidates(candidate_group_ids)
    if not condition_ids:
        return ()
    entries = [
        ConditionDrawEntry(condition_id, prompt.prompt_id, block.block_id, base_index)
        for condition_id in condition_ids
        for prompt, block in prompt_block_pairs()
        for base_index in FULL_PILOT_BASE_INDICES
    ]
    expected = full_pilot_image_count(len(candidate_group_ids))
    if len(entries) != expected:
        raise AssertionError(f"full-pilot manifest built {len(entries)} entries, expected {expected}")
    return tuple(entries)


def build_full_pilot_manifest_entries_for_conditions(approved_condition_ids: Sequence[str]) -> tuple[ConditionDrawEntry, ...]:
    """Full-pilot manifest over an explicit, already-exclusion-filtered set of
    condition ids (e.g. ``["reference", "B3_minus"]``), rather than a
    candidate-group list -- lets preview-exclusion review (Change 4b) drop a
    single sign of a group without touching the other. Empty input, or input
    that reduces to only ``"reference"``, returns an empty manifest -- a
    reference-only batch has nothing to compare against and is not generated.
    """
    real_conditions = [c for c in approved_condition_ids if c != "reference"]
    if not real_conditions:
        return ()
    condition_ids = ["reference"] + real_conditions
    entries = [
        ConditionDrawEntry(condition_id, prompt.prompt_id, block.block_id, base_index)
        for condition_id in condition_ids
        for prompt, block in prompt_block_pairs()
        for base_index in FULL_PILOT_BASE_INDICES
    ]
    expected = len(condition_ids) * NUM_PROMPT_BLOCK_PAIRS * NUM_BASE_INDICES_PER_BLOCK
    if len(entries) != expected:
        raise AssertionError(f"full-pilot manifest built {len(entries)} entries, expected {expected}")
    return tuple(entries)


_SEED_BLOCKS_BY_ID = {block.block_id: block for block in all_seed_blocks()}


def sample_seed_for_entry(entry: ConditionDrawEntry) -> int:
    return _SEED_BLOCKS_BY_ID[entry.block_id].sample_seed(entry.base_index)


def assert_disjoint_sample_seeds(
    entries_a: Sequence[ConditionDrawEntry], entries_b: Sequence[ConditionDrawEntry]
) -> None:
    """Raise if any entry in ``entries_a`` and any entry in ``entries_b``
    resolve to the same actual ``SeedBlock.sample_seed`` value. A disjoint
    ``base_index`` range only guarantees disjoint seeds if every block's
    ``batch_seed`` is spaced widely enough -- true today by construction, but
    this is the runtime safety net rather than an assumption baked silently
    into the base_index ranges alone.
    """
    seeds_a = {sample_seed_for_entry(entry) for entry in entries_a}
    seeds_b = {sample_seed_for_entry(entry) for entry in entries_b}
    overlap = seeds_a & seeds_b
    if overlap:
        raise ValueError(f"sample seeds overlap between the two entry sets: {sorted(overlap)}")
