# Implementation Decisions

This document records how the plan (`PCA-Specific-PSD-Editing-Plan.md`,
`PC-Specific-PSD-Implementation-Prompt.md`) was actually mapped onto code:
the real interface mapping, every place the implementation deviates from the
plan's literal text, the import-compatibility design, RNG/seed conventions,
coordinate/FFT conventions, and the status of the calibration defaults
shipped in `configs/sdxl_turbo_pca_v1.yaml`. Nothing here changes the
scientific core (basis definition, probing rotation, PSD-preserving edit,
calibration procedure) described in the plan — deviations below are packaging
and data-representation choices, not algorithm changes.

## Actual interface mapping (plan concept → code)

| Plan concept | Code |
| --- | --- |
| Shared PCA basis over clean latent patches | `basis.py`: `sample_random_patches`, `StreamingMeanCovariance`, `build_pca_basis`, `save_basis`/`load_basis` |
| Non-overlapping probing codec, block rotation | `patch_codec.NonOverlapCodec`, `probing.py` |
| Stride-1 overlapping PSD editor | `patch_codec.OverlapCodec`, `psd_editor.py` |
| Three human-annotation review tracks | `review.py` (`export_probe_review`/`ingest_probe_review`, `export_preview_review`/`ingest_preview_review`, `export_gallery_review`/`ingest_gallery_review`) |
| Radial PSD estimator, transfer-matrix diagnostics | `spectral.py` |
| Finite-candidate-set calibration | `calibration.py` |
| PC-condition schema, resume, independent full-pilot seeds | `runner.py` |
| Metric wrapping (bypassing `evaluate_run`'s fixed alpha/gamma) | `metrics.py` (+ `compat_metrics.py`) |
| Block/bootstrap analysis, no-interpolation guard | `analysis.py` |
| `python -m pc_specific_psd <command>` | `cli.py`, `__main__.py` |
| Config schema + per-command gating | `config.py` (`COMMAND_TIERS`, `resolve_config`) |
| Locked prompt/seed/PC-group tables, budget calculators | `manifests.py` |
| `SDXLTurboAdapter` + revision enforcement + mandatory paired generator | `adapters.SDXLTurboAdapterPCA` |

## Deviations from the plan's literal text

### 1. `manifests.py` replaces the planned `manifests/*.jsonl`/`*.json` data files

The plan's module layout names three external data files:
`manifests/prompts.jsonl`, `manifests/blocks_pca.jsonl`,
`manifests/pc_groups_v1.json`. The implementation instead hardcodes this data
as versioned, frozen Python constants directly in `manifests.py`:
`PROMPTS: tuple[Prompt, ...]` (4 entries) and `PC_GROUPS: tuple[PCGroup,
...]` (6 entries), plus derived helpers (`seed_blocks_for_prompt`,
`donor_seed`, the budget calculators, `build_preview_manifest_entries`/
`build_full_pilot_manifest_entries`).

**Why**: this data is small, fixed for the life of the study (§14.2's table
is explicitly "locked"), and needs to be validated at import time (e.g.
`validate_pc_groups` asserting the six groups exactly partition the
100-dimensional basis with no gap/overlap). Keeping it as Python constants
means that validation runs automatically on every import and is covered by
ordinary unit tests, rather than requiring a separate schema-validation step
for hand-edited JSON/JSONL files that could drift from the code that
consumes them. The `pc_specific_psd/manifests/` directory still exists (kept
empty) as the on-disk location for anything genuinely generated per-run
(review exports, candidate files, run manifests) — see "What actually lives
under `manifests/`" below.

**Consequence**: there is no `manifests/prompts.jsonl` to hand-edit if the
prompt or seed table ever needs to change; a code change to `manifests.py`
(and its tests) is required instead. This is intentional — the table is
locked by the plan, not meant to be tunable.

### 2. `configs/sdxl_turbo_pca_v1.yaml` replaces `noise_init`'s config **by value**, not by reference

The plan says `model`/`generation` should be reused from `noise_init`
"by value". The production config actually copies these fields from
`noise_init/configs/sdxl_turbo_full.yaml` (sha256
`99880bda712dc331e3dfe4ad117dfe6fcc00377dd54f4eb53636efb6704d6b50` at the
time of copying, recorded in the config's header comment) rather than
loading or `!include`-ing that file at runtime.

**Why**: `noise_init/` must never be modified, and a runtime include would
create a hidden coupling where a future edit to `noise_init`'s config
silently changes this package's behavior without a corresponding version
bump or test run here. A byte-identical, hash-stamped copy is auditable and
reproducible independent of `noise_init`'s own config file's future edits.

**Consequence**: if `noise_init/configs/sdxl_turbo_full.yaml` changes
(e.g. a different VAE checkpoint), `configs/sdxl_turbo_pca_v1.yaml` must be
updated by hand and its header hash comment refreshed — this is a manual
step, not automatic.

### 3. `psd.groups` declares all six PC groups up front, with placeholder candidate lists

The plan's calibration section (`_full_tier_missing` in `config.py`) requires
every **declared** PC group in `psd.groups` to have a frozen τ selection
before any `full`-tier command (`generate-psd`, metrics, analyze, …)
resolves. Since which 1-2 groups become real study candidates is only known
after real probing and human annotation (not yet run — see
`IMPLEMENTATION_REPORT.md`), the shipped production config declares **all
six** groups (`B1`–`B6`) with the same placeholder-but-structurally-valid
`tau_plus_candidates`/`tau_minus_candidates` (`[0.5, 1.0, 2.0]` /
`[-0.5, -1.0, -2.0]`) and `target_rms: 1.0`, and three placeholder
`gate_candidates` (`(r_s, beta)` pairs `(4,2)`, `(8,2)`, `(8,4)`).

**Why**: this lets the config pass `validate-config`/`build-basis`/`probe`/
`calibrate --dry-run` immediately (all verified this pass — see
`IMPLEMENTATION_REPORT.md`) without waiting on human annotation, while being
honest that no real measurement backs these specific numbers yet.

**Status**: these are **first-pass placeholder defaults, not plan-mandated
or measured values** — grepping the plan text found no explicit numeric
defaults for gate `(r_s, beta)` or τ candidate lists (only the functional
forms of `w(r)`/`t_B(r; tau)` in §5.3 and `rho=0.10` as the probing default,
which *is* used verbatim). Once real probing + annotation identifies 1-2
real candidate groups, this file should be edited to declare only those
groups (or the placeholder groups' candidate lists should be replaced with
values chosen for a genuine reason), and `calibrate` (not this file) is what
actually freezes the selected gate/τ/correction into
`psd.calibration_result_path`. Until `calibrate` has run for real, every
value in `psd.*` other than `protocol`, `num_bins`, `rho`, and the bank
sizes/seeds is a candidate to choose from, not a result.

### 4. `basis.dataset_manifest_path: null` in the production config

The plan expects a real dataset manifest path. This sandbox has no local
ImageNet/COCO-style dataset, so the production config ships with
`dataset_manifest_path: null`, which `build-basis --dry-run` correctly
reports as `ready: false`. This is not a deviation from the plan's schema
(the field is documented as optional in `config.py`,
`_BASIS_OPTIONAL_FIELDS`) but is called out here because it is the single
field that gates all real generation work — see
`IMPLEMENTATION_REPORT.md`.

## Import-compatibility design

`noise_init/` uses flat, same-directory imports (`from io_utils import ...`),
not a regular installable package, and must never be modified. Two
independent shims handle this:

- `compat_generation.py` — imports `io_utils`, `noise_methods`,
  `model_adapters`; re-exports `NoiseBatch`, `sample_noise_batch`,
  `load_or_create_noise_batch`, `normalize`, `pink`, `pink_filter`,
  `same_phase_floor`, `independent_white`, `radial_frequency_grid`,
  `GenerationConfig`, `T2IModelAdapter`, `build_adapter`,
  `SDXLTurboAdapter`, `derived_seed`, `tensor_hash`, `sha256_text`,
  `canonical_json`, `read_json`/`write_json`/`read_jsonl`/`write_jsonl`,
  `ensure_immutable_run`, `environment_record`, `file_hash`.
- `compat_metrics.py` — imports only `metric_runner`; re-exports
  `MetricRunner`. Imported **lazily**, inside the function in `metrics.py`
  that needs it — never at another module's top level.

Both call a shared private helper, `_compat_common.load_noise_init_module`,
which resolves `NOISE_INIT_ROOT = pc_specific_psd/../noise_init`, temporarily
inserts it into `sys.path`, sets `sys.dont_write_bytecode = True` for the
duration (so importing never creates a `noise_init/__pycache__`), imports the
named module, asserts the loaded module's `__file__` actually resolves inside
`NOISE_INIT_ROOT` (raising `RuntimeError` otherwise — catches accidental
shadowing by a same-named module elsewhere on `sys.path`), then restores
`sys.path`/bytecode state.

`compat_generation.py` and `compat_metrics.py` never import each other —
verified structurally (not just by convention) by
`tests/test_compat_independence.py` via `ast` inspection of each module's own
import statements, plus a repo-wide scan asserting no other module in the
package imports `compat_metrics` at module top level. This is why
`validate-config`, `build-basis`, `probe`, and `generate-psd --dry-run` never
require the metrics venv's `transformers==4.45.2` pin to be importable, and
`metrics` never requires the generation venv's `diffusers`/
`transformers==4.56.2` stack.

`adapters.SDXLTurboAdapterPCA` subclasses `noise_init`'s `SDXLTurboAdapter`
rather than reimplementing pipeline construction, because the base adapter
already handles model loading; it overrides only what the base adapter gets
wrong for this study (dropping `revision`, never passing a `generator`).

## RNG / seed conventions

No new global seed derivation scheme is introduced; every seed used here
composes `noise_init`'s existing `derived_seed`/`sample_noise_batch`
machinery:

- **Base draw / `base_white`**: for a `(prompt_id, block_id, base_index)`
  draw, `sample_seed = block.batch_seed + base_index`
  (`manifests.SeedBlock.sample_seed`). `base_white` is
  `sample_noise_batch(sample_seed, block_id, shape,
  batch_seed=sample_seed).base_white` — the ordinary base-draw path used for
  every other base draw in `noise_init`, not a PCA-specific mechanism
  (`psd_editor.base_white_for_draw`, `probing.draw_base_latent`). Every
  condition compared for the same draw (Reference/A+/A-/B+/B-, or
  Reference/cand+/cand-) reuses this identical tensor —
  `tests/test_base_white_pairing.py` and
  `test_runner.RenderConditionLatentTests` assert byte-identical reuse across
  conditions sharing a draw, and different values across different draws.
  A single fixed `(prompt_id, batch_seed, base_index)` triple is used only as
  a test fixture, never treated as shared production noise.
- **Donor noise for probing rotation**: `donor_seed(block_id, base_index,
  donor_index=0) = derived_seed(20260829, block_id, "pc_probe_donor_v1",
  base_index, donor_index)`. Deliberately excludes `group_id` — all PC groups
  probed off the same base draw share one donor tensor, per plan §14.2.
- **Paired scheduler generator**: `adapters.SDXLTurboAdapterPCA.generate`
  constructs a fresh `torch.Generator(device)` seeded by
  `derived_seed(seed, "sdxl_pipeline_generator", pair_key)` on every call —
  never cached or reused across calls — so every member of one paired
  comparison gets identical scheduler randomness and only the injected noise
  differs.
- **Calibration/validation banks**: `cli._calibration_bank`/`_validation_bank`
  draw from `torch.Generator("cpu").manual_seed(cfg.psd.calibration_bank_seed
  / validation_bank_seed)` — plain, config-declared seeds, independent of the
  `derived_seed` namespace used for generation, since these banks are pure
  noise-only calibration inputs with no prompt/model involved.
- **Analysis bootstrap**: matches `noise_init.analysis`'s recipe (2000
  replicates, seed 20260829, 95% CI) — reused, not reinvented.

## Coordinate / FFT convention

`spectral.py` builds its own full-FFT or conjugate-weighted rFFT annular
radial binning (`radial_bin_index`), explicitly **not** reusing
`noise_init.noise_statistics.psd_mean`/`psd_high_frequency_mean`, which the
plan (§14.6) and this package's docstrings both note are not true
annular/Hermitian-weighted PSD measures. Frequency axes use
`torch.fft.fftfreq`/`torch.fft.rfftfreq` scaled by the corresponding
dimension size (so bin edges are in integer-cycle units, matching
`noise_init`'s own convention); radius is Euclidean distance in that
integer-cycle frequency space, linearly binned from 0 to the maximum radius
present into `num_bins` annuli. `rfft`-domain sums use conjugate weight 2 for
every column except DC (and Nyquist, when the width is even) to correctly
recover the full-plane annular energy from the omitted conjugate-symmetric
half — checked against a direct full-FFT computation in
`tests/test_spectral.py::RadialPSDParsevalTests::test_full_fft2_agrees_with_rfft_based_computation`.

## Frozen same-phase reference

`psd_editor.SAME_PHASE_ALPHA = 0.9` and `psd_editor.SAME_PHASE_GAMMA = 0.05`
are module-level constants, not config fields; `config.py` has no YAML key
for them and any attempt to supply one is rejected
(`tests/test_frozen_reference.py`). They are still echoed into resolved
config/provenance output for transparency, per the plan's requirement that
frozen constants remain visible even though they aren't user-configurable.

The `operator_clean` reference and every candidate now use the frozen scale
profile `expected_unit_rms_rfft_v1`. The unscaled response remains
`sqrt((1-gamma) * H_alpha(r)^2 + gamma)`, but is multiplied once by the
reciprocal of its analytic expected RMS. The expectation is computed with
Parseval's theorem over the rFFT half-spectrum: DC has conjugate weight 1,
the Nyquist column has weight 1 only for even widths, all other positive
frequency columns have weight 2, and the denominator is `height * width`.
For the production 64×64 latent this raw multiplier is approximately
0.2367667109 and the derived scale is approximately 4.2235667177; neither
number is hard-coded in production code.

This is one deterministic scalar for a fixed `(height, width, alpha, gamma)`,
not a statistic of an individual latent. It therefore preserves linearity,
Gaussianity, phase, and the same-phase relative spectral shape. A per-sample
or per-channel normalization would be nonlinear and realization-dependent,
so `operator_clean` continues not to call `normalize()`. The scale is applied
inside the shared reference response only: `apply_psd_edit_tau_zero()` uses
it for the reference and `apply_psd_edit()` uses it for candidates, while
calibration and runner add no second scale.

Calibration registries persist this profile. A legacy registry with no
profile is loadable for diagnosis but is stale for full/generation commands
and must be regenerated; smoke-check cache keys also include the profile.
This RMS correction is independent of the earlier geometry correction:
SDXL-Turbo output remains 512×512, while probing, calibration, smoke checks,
preview, and full-pilot operators continue to use 4×64×64 latents.

## Calibration-default rationale and status

`calibration.py` implements the plan's exact procedure: walk `(r_s, beta)`
gate candidates in declared order, reject any that violate the configured
PSD tolerance / correction-gain bound / transfer condition-number threshold
after up to `MAX_CALIBRATION_ITERATIONS = 3` offline refinement rounds
against the **calibration bank only**; among survivors, pick each group's
τ+/τ- independently by closest calibration-bank RMS to `target_rms`, ties
broken by declared order; freeze the result and consult the **validation
bank exactly once** as an accept/reject gate (no automatic fallback to
another candidate on failure). `τ=0` is defined, not measured: the
correction array is exactly `1.0` for every bin by construction
(`tests/test_tau_zero_correction.py`).

None of the numeric fields in the production config's `psd` section
(`num_bins: 32`, `psd_tolerance: 0.05`, `correction_gain_bound: 4.0`,
`condition_number_threshold: 50.0`, bank sizes `256`/`128`) come from the
plan text as mandated defaults — the plan specifies the *procedure*, not
these specific numbers. They are first-pass, reasonable-order-of-magnitude
choices made for this pass so the config is structurally complete and
`calibrate --dry-run` can run; they have **not** been validated against real
noise banks (`calibrate` itself has not been run for real — see
`IMPLEMENTATION_REPORT.md`), and should be revisited once real
calibration-bank measurements are available. `probing.rho = 0.10` is the one
numeric default taken directly from the plan (its documented default value).

## Unverified-adapter (FLUX) guard

Plan §2.3/§10/§11 requires that if a FLUX codec/basis has not been confirmed,
the CLI must explicitly mark it `not_verified` and refuse to run a formal PCA
experiment against it, rather than silently reusing SDXL's basis/PC numbering
or guessing τ values for it ("FLUX如未完成codec确认，应在CLI明确标
not_verified，拒绝正式PCA实验；不能复制SDXL basis/PC编号或猜C值"). Before this
pass, `config.py` did not validate `model.adapter` at all, and `runner.py`/
`cli.py` unconditionally constructed `SDXLTurboAdapterPCA` regardless of its
value — so a config declaring `adapter: flux2klein` would silently run using
the SDXL-fit basis and PC groups instead of being refused.

**Fix**: `config.py` defines `_SUPPORTED_ADAPTERS = ("sdxl_turbo",)`, and
`_build_model_config` raises `ConfigError` (message contains the literal
string `not_verified`) for any other `model.adapter` value. Because this
check runs inside `load_config`/`resolve_config`, which every command calls
before its own per-command tier validation, and `cli.py`'s `main()` catches
`ConfigError` unconditionally (`error: {exc}`, exit 1) regardless of
`--dry-run`, the refusal fires for every command — `validate-config`,
`build-basis --dry-run`, `probe --dry-run`, and a real `generate-psd` call
all refuse identically before doing any adapter-specific work. Verified by
`tests/test_config.py::test_model_adapter_flux2klein_is_refused_as_not_verified`,
`test_model_adapter_unknown_value_is_refused_as_not_verified`, and
`tests/test_cli.py::UnverifiedAdapterRefusalTests` (4 methods spanning those
command tiers).

**Why SDXL-Turbo specifically, and why refuse rather than warn**: the basis
(`d = C·p² = 4·5² = 100`), the six PC groups' 1-based index partition, and
both patch codecs in this package are fit and dimensioned against
SDXL-Turbo's VAE latent space only (`patch_size=5`, `channels=4`). There is
no FLUX basis, no FLUX PC group partition, and no validated patch geometry
for FLUX's (differently-shaped) latent space in this codebase —
`compat_generation.py` re-exports `Flux2KleinAdapter` for a possible future
pass, but nothing here ever instantiates or dispatches to it. A warning would
still let a mismatched basis run silently; refusing at config-load time is
the only way to satisfy the plan's "拒绝正式PCA实验" (refuse the formal PCA
experiment) requirement literally. This satisfies acceptance ID `T03`
(see `IMPLEMENTATION_CHECKLIST.md`).

## Probe-review ingestion: no dedicated `ingest-probe-review` command

The plan's required-command list (§13/module layout) does not include a
top-level `ingest-probe-review` command, and this pass does not add one.
Instead, every place that needs completed probe-review annotations reads and
validates them through one shared function, `review.ingest_probe_review()`,
invoked from three call sites — each through a shared CLI helper,
`cli._load_and_ingest_probe_review`:

1. `select-candidates --review-file <path> [--mapping-file <path>]` — the
   primary and most common entry point; ingestion happens first, and the
   deterministic nomination rule only runs against the ingested, validated
   data.
2. `probe --rho020-followup --review-file <path>` — ingests the same file
   shape to compute `probing.compute_rho020_trigger()`'s prompt-count check
   before permitting the ρ=0.20 uniform-supplement manifest.
3. The candidate/reserve recheck manifest builder in `probing.py` — ingests
   the file to confirm which group(s) `select_candidates()` actually
   nominated before building the recheck draws.

All three call sites refuse — before running the nomination rule, the
ρ=0.20 trigger check, or the recheck builder, respectively — when the
review file is missing, malformed, or fails schema validation; none
silently proceeds on unvalidated or partially-read annotation data. This is
exercised directly by
`tests/test_cli.py::SelectCandidatesReviewFileIngestionRefusalTests`
(`test_select_candidates_refuses_on_missing_review_file`,
`test_select_candidates_refuses_on_malformed_review_file`), and documented
for users in `pc_specific_psd/README.md` ("`select-candidates` is the single
entry point for probe-review evidence — there is no separate top-level
`ingest-probe-review` command"). This satisfies acceptance ID `B06`.

**Why one shared ingestion path instead of a standalone command**: the plan
frames ingestion as a step inside whichever downstream operation needs the
annotation data, not as an independent action a user would run on its own —
there is nothing useful to do with an ingested-but-otherwise-unused review
file. Centralizing the loader in `review.ingest_probe_review()` and reusing
it from all three call sites means the missing/malformed/partial-field
validation logic is written and tested once, rather than three times with a
risk of drift between them.

## Workflow consolidation pass (single-entry-point orchestrator)

A later, separate pass adds `workflow.py` and the `workflow` CLI command.
Unlike the sections above, this is not mapping the editing plan's science
onto code — the algorithm (PCA basis, rotation, PSD editor, calibration) is
completely unchanged. It addresses a purely operational problem observed
after the rest of this package was already implemented and tested: running
the pipeline for real requires ~8 commands invoked by hand in order, with
JSON paths hand-copied between them and the human reviewer reading a flat
72-row pairwise table to make what is really a per-PC-group decision. The
fix is additive orchestration, not a redesign — no new experimental module,
no new metric dependency, no new human-review round, no algorithm change.
Five decisions were made in doing this:

1. **One orchestrator, reusing existing functions, never reimplementing
   them.** `workflow.run_workflow()` is a resumable 14-stage state machine
   that calls the same `config`/`basis`/`probing`/`calibration`/`runner`/
   `metrics`/`analysis` functions each standalone command already calls —
   it does not duplicate their logic. Progress persists to
   `<config_dir>/<run.name>_workflow_state.json`; re-running the same
   command resumes from the first incomplete stage. All 15 pre-existing
   commands stay independently runnable and byte-for-byte unmodified in
   their own schemas (see README's "The `workflow` command" section).

2. **Two new consolidated review artifacts sit *on top of*, not instead of,
   the existing three tracks.** `ProbeGroupReviewRow`/`PreviewExclusionRow`
   (`review.py`) are coarser aggregate views built from the same underlying
   84 probe images / preview-run manifest the original `ProbeReviewRow`/
   `PreviewReviewRow` tracks already use — a per-group and per-condition
   roll-up, respectively, each with a rendered PNG contact sheet, instead of
   the human reading raw per-image rows. `workflow` never touches
   `gallery-review` (its diversity/preference fields are exactly what
   could bias an automated pipeline toward "picking a winner by richness"
   rather than a rejection-for-cause criterion); `analysis.analyze_run()`
   already reports that axis as `incomplete`, not omitted, when it isn't
   ingested — no new code was needed to skip it safely.

3. **Group-level and condition-level decisions are kept deliberately
   separate, meeting only at the full-pilot manifest.** Calibration stores
   `(tau_plus, tau_minus)` per PC group — there is no way to calibrate only
   one sign — so `select_candidates()`/`candidates.json` stay group-level,
   and both signs of a selected group are always calibrated and always
   previewed together. But the preview-exclusion review is per-*condition*
   (`B5_plus`/`B5_minus` are separate rows), because a human must be able
   to keep `B5_minus` while dropping `B5_plus` for artifacts — a group-level
   exclusion cannot express that. `manifests.build_full_pilot_manifest_entries_for_conditions`
   is the new function that reads the finer, approved-conditions list; the
   original group-based `build_full_pilot_manifest_entries` is kept,
   unmodified, for standalone use.

4. **Full-pilot seeds are now independent of every seed a human has already
   looked at.** Probing and preview both only ever draw `base_index 0-3`;
   before this pass, `generate_full_pilot`'s `preview_run_dir` reuse path
   copied preview-run image bytes into the full pilot for `base_index==0`,
   so an already-screened image could end up counted in final statistics.
   The fix shifts the full-pilot draw range to a disjoint block
   (`base_index 4-7`, `FULL_PILOT_BASE_INDEX_OFFSET`) and deletes the reuse
   branch entirely — this is checked at runtime, not just assumed from the
   base_index ranges being disjoint on paper:
   `manifests.assert_disjoint_sample_seeds` recomputes the actual
   `SeedBlock.sample_seed` for every probing/preview entry and every
   full-pilot entry and raises if the sets intersect, since a disjoint
   `base_index` range only guarantees disjoint seeds while every block's
   `batch_seed` stays spaced widely enough (true today by construction, but
   not something to silently assume stays true forever). The image-count
   ceiling (144/240) is unchanged — same draw count, non-overlapping seeds.

5. **The smoke check is calibration-independent by construction, because it
   must run before probing — before any calibration result exists.**
   `adapters.smoke_check()` never touches `SELECTED`/frozen corrections
   (which would raise pre-calibration); it instead builds the τ=0 latent via
   `psd_editor.apply_psd_edit_tau_zero` (needs only basis/codec, no PC group,
   no calibration), confirms the exact same tensor was what got passed to
   `adapter.generate()` (via the existing tensor-hash provenance fields, no
   new instrumentation), and compares it against the canonical
   analytically scaled `same_phase_floor` ground truth
   (`compat_generation`'s re-export of the
   independent reference implementation, not a second internal derivation of
   the same shortcut checked against itself), and verifies analytic response
   energy is one. Its cache key is
   `(config_hash, basis_hash, reference_scale_profile)` — deliberately excluding
   `calibration_hash` — so the same cached pass covers both the pre-probing
   and pre-full-pilot call sites whenever config/basis haven't changed,
   without ever depending on calibration having happened.

6. **Venv switching re-invokes the existing CLI as a subprocess rather than
   adding a second execution mechanism.** `--generation-python`/
   `--metrics-python` (default: `sys.executable`) make `workflow` call
   `python -m pc_specific_psd <that stage's own command>` via
   `subprocess.run` when the configured interpreter differs from the one
   currently running, and call the same code in-process otherwise — no new
   IPC/RPC layer, and no logic duplicated between the in-process and
   subprocess paths. The full-pilot dispatch always also passes
   `--approved-conditions-file` explicitly, even across a venv switch: the
   child process only knows the group-level `candidates.json` otherwise, and
   would silently regenerate an excluded sign (e.g. `B5_plus`) that the
   parent workflow believed was excluded.

7. **One selected-candidate set now drives validation, calibration, and
   generation together**, instead of three independent filters that could
   drift apart. `config.validate_for_command`/`resolve_config`/
   `_full_tier_missing` gained an optional `candidate_group_ids` parameter
   (default `None` preserves today's "check every declared group" behavior
   for every existing standalone invocation); `calibrate` now accepts the
   same `--candidates-file` flag `generate-psd`/`metrics`/`analyze` already
   had, and filters `group_specs` by it. `workflow` always passes its own
   explicit, stage-stamped candidates file to every stage — never a bare
   default-path lookup — so a stale `candidates.json` from a previous run
   can't be picked up by accident.

See `README.md`'s "The `workflow` command" section for the user-facing
stage sequence, review-stop behavior, and flags, and
`IMPLEMENTATION_CHECKLIST.md`'s note on why `workflow` is treated as a
CLI-command extension rather than a new §13 acceptance ID.

## What actually lives under `manifests/`

`pc_specific_psd/manifests/` exists as an empty directory. It is the intended
on-disk destination for generated, per-run artifacts (review export/mapping
files, `select-candidates` output, run manifests) once real commands are
actually executed — none of those commands have been run against real data
in this pass (see decision #1 above and `IMPLEMENTATION_REPORT.md`), so the
directory is currently empty. This is separate from `manifests.py`'s locked,
versioned constants, which are code, not generated data.
