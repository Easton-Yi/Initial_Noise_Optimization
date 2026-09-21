# pc_specific_psd

PCA-specific PSD noise-editing pipeline for SDXL-Turbo. Implements
`docs/PCA-Specific-PSD-Editing-Plan.md` / `docs/PC-Specific-PSD-Implementation-Prompt.md`:
build a shared PCA basis over clean SDXL-Turbo VAE latent patches, run a
non-overlapping "probing" intervention to find structure/appearance-sensitive
PC groups via human-annotated paired generations, build a stride-1
overlapping PSD editor that reassigns spectral power between a candidate PC
group and its complement while preserving the reference radial PSD, calibrate
it on noise alone, then run a bounded Reference/A+/A-/B+/B- Q/D/artifact
study.

Everything in this package lives under `pc_specific_psd/`. It never modifies
`noise_init/`; it only reads/imports from it (see "Environment" below and
`docs/IMPLEMENTATION_DECISIONS.md`).

**Status of this pass**: the full software stack is implemented and tested
on CPU with synthetic data. No real PCA basis, no real SDXL-Turbo generation,
and no human annotation have been produced yet — see
`docs/IMPLEMENTATION_REPORT.md` for exactly how far each stage got and the
next commands to run on a machine with a dataset and a GPU.

## Environment

This package reuses `noise_init`'s two existing virtual environments rather
than declaring its own dependencies — there is no `pc_specific_psd`-specific
`requirements*.txt`.

- **Generation venv** (`noise_init/requirements.txt` or
  `noise_init/requirements_generation.txt`, depending on which GPU you're
  running on — the `_generation` variant pins `torch==2.11.0` for Blackwell
  cards such as the RTX Pro 4000): needed for `build-basis`, `probe`,
  `generate-psd`, and any command whose `--dry-run` is *not* passed. Provides
  `diffusers`/`transformers==4.56.2`.
- **Metrics venv** (`noise_init/requirements_metrics.txt`, pinned to
  `transformers==4.45.2` for HPSv3): needed only for the `metrics` command.

`pc_specific_psd/compat_generation.py` and `pc_specific_psd/compat_metrics.py`
are the only two modules that import anything from `noise_init/`, and they
are deliberately independent of each other (`tests/test_compat_independence.py`
asserts this via `ast` inspection of their import graphs, and that
`compat_metrics` is only ever imported lazily, inside `metrics.py`'s own
functions, never at another module's top level). This is why
`validate-config`, `build-basis`, `probe`, and `generate-psd --dry-run` never
require the metrics venv's conflicting `transformers` pin to be importable,
and `metrics` never requires the generation venv's `diffusers` stack.

Both shims load `noise_init`'s modules (`io_utils`, `noise_methods`,
`model_adapters`, `metric_runner`) via a shared private helper,
`_compat_common.load_noise_init_module`, which temporarily puts
`noise_init/` on `sys.path` (it uses flat same-directory imports internally),
suppresses bytecode writing so no `noise_init/__pycache__` is created, and
asserts each loaded module's `__file__` actually resolves inside
`noise_init/` before returning it. `noise_init/` itself is never modified,
moved, or reformatted by anything in this package.

No `pc_specific_psd`-specific install step is required beyond having one of
the two `noise_init` venvs active — from the repo root:

```bash
python3 -m unittest discover -s pc_specific_psd/tests -v
```

runs the full CPU/synthetic test suite (500 tests as of this pass) with only
`torch`/`pyyaml` needed (both already in the base `noise_init` venv); it does
not touch a GPU, download a model, or import `compat_metrics`.

## Input data format

### Config file

A single YAML file (see `configs/sdxl_turbo_pca_v1.yaml` for a filled-in
example) with six top-level sections: `run`, `model`, `generation`, `basis`,
`probing`, `psd`. Every command loads it through `config.resolve_config(path,
command)`, which validates only the fields that command's **tier** actually
needs (`config.COMMAND_TIERS`):

| Tier | Commands | Requires |
| --- | --- | --- |
| `basis` | `validate-config`, `build-basis`, `inspect-basis`, `workflow` | `model`, `generation`, `basis` sections only |
| `probing` | `probe`, `export-review`, `select-candidates` | + `probing.rho`, the PC group partition — **no** PSD/calibration field |
| `calibration` | `calibrate`, `validate-noise` | + the calibration protocol/bank-size/candidate-list fields to be *declared* (not yet selected) |
| `full` | `generate-psd`, `export-preview-review`, `ingest-preview-review`, `export-gallery-review`, `ingest-gallery-review`, `metrics`, `analyze` | + every declared PC group in `psd.groups` must already have a *frozen* τ selection (i.e. `calibrate` has run and written `psd.calibration_result_path`) |

`workflow` is deliberately gated at the loosest (`basis`) tier only at startup —
it re-validates the config against each stage's own (stricter) tier itself,
right before running that stage, exactly as if that stage's standalone
command had been invoked directly.

A command whose tier isn't satisfied refuses immediately with the exact list
of missing fields — it never fails deeper inside another module. All
relative paths in the config (`basis_output_path`, `calibration_result_path`,
`outputs_root`, …) resolve against the **config file's own directory**, never
the process's current working directory (`tests/test_cli_dry_run.py`).

`model.adapter` only accepts `"sdxl_turbo"`. Every other value, including
`"flux2klein"`, is refused at config-load time with a `ConfigError` whose
message contains `not_verified` — before any command-specific work runs, and
even under `--dry-run` (`config.py::_build_model_config`,
`_SUPPORTED_ADAPTERS`). This package's basis (`d = C·p² = 4·5² = 100`), its
six PC groups, and both patch codecs are fit and validated against
SDXL-Turbo's VAE latent space only; none of that is transferable to FLUX (or
any other adapter) without a new basis, new PC groups, and full
revalidation, so an unverified adapter is refused outright rather than
silently run against a mismatched basis. See
`docs/IMPLEMENTATION_DECISIONS.md`'s "Unverified-adapter (FLUX) guard"
section for the full rationale.

### Dataset manifest (for real `build-basis`)

`basis.dataset_manifest_path` points at a JSON file:

```json
{
  "source": "imagenet_2000_v1",
  "entries": [
    {"image_id": "im000001", "path": "/abs/or/relative/path.png", "sha256": "..."},
    ...
  ]
}
```

`source` records which real dataset this came from and is never silently
swapped after construction. This pass ships `dataset_manifest_path: null` in
the production config — no real dataset is available in this sandbox, so
real basis construction is blocked until a manifest is supplied (it is not a
scheduling gap: `build-basis` structurally cannot proceed without this
path).

#### Preparing a manifest with `prepare_coco_data.sh`

`prepare_coco_data.sh` builds exactly this manifest from a deterministic
subset of the official COCO 2017 training set: it downloads the official
annotations, seeded-shuffles and takes the first 2000 image records
(`SEED=20260829`, `COUNT=2000`, both fixed in the script), downloads those
images, verifies every one is readable, hashes it with SHA-256, and writes
the manifest to `pc_specific_psd/manifests/coco_train2017_2000_seed20260829.json`.
Re-running it resumes (annotation download and image download both use
`wget --continue`; already-verified images are not re-downloaded).

On the GPU machine, from the repo root:

```bash
mkdir -p /workspace/datasets/coco_2000

bash pc_specific_psd/prepare_coco_data.sh \
  /workspace/datasets/coco_2000 \
  /workspace/Initial_Noise_Optimization
```

Requires `wget`, `unzip`, `sha256sum`, and Pillow in the active Python
environment; the script checks for all four up front and fails fast with an
install hint if any is missing. Point the config's
`basis.dataset_manifest_path` at the manifest path it prints once it
completes, then continue with `validate-config` and `build-basis` (the
script itself prints these two next-step commands on success).

### Prompts and PC groups

Both are hardcoded, versioned Python constants in `manifests.py` (not
external data files — see the deviation recorded in
`docs/IMPLEMENTATION_DECISIONS.md`):

- `manifests.PROMPTS`: 4 prompts (`p000`–`p003`), each with 3 seed blocks
  (`batch_seed` triples `10000/10100/10200`, `11000/11100/11200`,
  `12000/12100/12200`, `13000/13100/13200`).
- `manifests.PC_GROUPS`: 6 groups spanning the full 100-dimensional basis
  (`patch_size=5, channels=4`): `B1` (1-4), `B2` (5-8), `B3` (9-12), `B4`
  (13-16), `B5` (17-32), `B6` (33-100).

### Review/annotation files

Three independent human-annotation tracks (probe / preview / full-gallery),
each with its own export/ingest schema (`review.py`). Every `export-*`
command writes a blinded template (condition/pair identity anonymized into a
separate mapping file) with required fields left blank; a human fills those
in; the matching `ingest-*` (or `select-candidates`/`probe
--rho020-followup`, which ingest inline) validates the completed file and
refuses — before running any downstream rule — if it's missing, malformed, or
fails schema validation. No command ever invents, guesses, or partially
auto-fills an annotation value.

`workflow` (below) does not use any of these three tracks directly. Instead
it drives two additional, coarser artifacts in `review.py` that sit on top of
the same underlying data, for its two review stops:

- **Probe-group review** (`ProbeGroupReviewRow`, `export_probe_group_review`/
  `ingest_probe_group_review`/`select_candidates_from_group_review`): rolls
  the same 84 probe images `export-review` already renders up to one row per
  `(PC group, prompt)` — 6 groups × 4 prompts = 24 rows — plus one PNG
  contact-sheet per group, so a human judges "did this group show a
  structural change" at a glance instead of reading all 72 pairwise rows.
  Applies the identical nomination rule as `select_candidates()` (≥2 valid
  draws per qualifying prompt, ≥2 qualifying prompts, max 2 groups). Every
  export/report built from it repeats the caveat: this is a multi-seed
  exploratory screen, not a claim that an independent confirmation experiment
  was run.
- **Preview-exclusion review** (`PreviewExclusionRow`,
  `export_preview_exclusion_review`/`ingest_preview_exclusion_review`/
  `approved_condition_ids`): one row per rendered condition (`reference`,
  `B5_plus`, `B5_minus`, ...), plus one contact-sheet PNG per condition,
  asking only "does this condition show obvious structural damage / an
  extraneous object / a severe texture-color anomaly — yes/no, and what
  kind" — never a quality/preference score. `reference` is always kept in
  the approved set; if a human marks `reference` itself excluded, `workflow`
  treats that as a hard stop for manual investigation rather than a routine
  exclusion, since a broken reference invalidates every paired comparison.

Both are tri-state on every required field (`None` until answered, never
defaulted to `false`), and both distinguish "file missing" from "file present
but incomplete" from "file present and complete" — an unfilled template is
never accidentally ingestible. `workflow`'s state file also hashes each
review file at the moment it's consumed; if the file changes afterward
(e.g. a human re-edits it after `candidates.json` was already derived from
it), resuming `workflow` stops with a conflict error instead of silently
mixing old and new decisions.

The original detailed probe/preview tracks, and the full `export-gallery-review`/
`ingest-gallery-review` track, remain fully usable standalone and are never
touched by `workflow` — in particular, gallery-review's diversity/preference
fields are deliberately never consulted by the automated path, to avoid
picking a "winner" by richness rather than absence-of-damage.

## Commands

All commands accept `--config <path>` (required), `--run-id <id>` (override
the run/manifest id), and `--dry-run` (report planned actions — image counts,
required inputs, unresolved config fields — without touching disk, a model,
or either compat shim's heavy dependencies).

```
validate-config   --config CFG [--run-id ID] [--dry-run]
build-basis       --config CFG [--run-id ID] [--dry-run]
inspect-basis     --config CFG [--dry-run] [--allow-synthetic-basis]

probe             --config CFG [--dry-run] [--allow-synthetic-basis] [--force]
                  [--rho020-followup | --candidate-recheck]
                  [--group-ids B1,B2,...]              # with --candidate-recheck
                  [--review-file PATH --mapping-file PATH]  # with --rho020-followup

export-review     --config CFG [--dry-run]
                  [--review-output PATH] [--mapping-output PATH]

select-candidates --config CFG [--dry-run]
                  [--review-file PATH] [--mapping-file PATH] [--output PATH]

calibrate         --config CFG [--dry-run] [--allow-synthetic-basis] [--candidates-file PATH]
validate-noise    --config CFG [--dry-run] [--allow-synthetic-basis]

generate-psd      --config CFG [--dry-run] [--allow-synthetic-basis] [--force]
                  --stage {preview,full} [--candidates-file PATH]
                  [--approved-conditions-file PATH]   # --stage full only

export-preview-review  --config CFG [--dry-run] [--candidates-file PATH]
                        [--review-output PATH] [--mapping-output PATH]
ingest-preview-review  --config CFG [--dry-run] [--candidates-file PATH]
                        [--review-file PATH] [--mapping-file PATH] [--output PATH]

export-gallery-review  --config CFG [--dry-run] [--candidates-file PATH]
                        [--review-output PATH] [--mapping-output PATH]
ingest-gallery-review  --config CFG [--dry-run] [--candidates-file PATH]
                        [--review-file PATH] [--mapping-file PATH] [--output PATH]

metrics           --config CFG [--dry-run] [--force] [--device cpu]
analyze           --config CFG [--dry-run]
                  [--gallery-review-file PATH] [--gallery-mapping-file PATH]

workflow          --config CFG [--run-id ID] [--dry-run] [--force]
                  [--allow-synthetic-basis] [--candidates-file PATH]
                  [--generation-python PATH] [--metrics-python PATH]
                  [--skip-smoke-check]
                  [--probe-group-review-file PATH] [--preview-exclusion-file PATH]
```

`--rho020-followup` and `--candidate-recheck` are mutually exclusive on
`probe`. `select-candidates` is the single entry point for probe-review
evidence — there is no separate top-level `ingest-probe-review` command.
`--preview-run-dir` no longer exists on `generate-psd`: full-pilot draws now
come from an independent seed range (see "Budgets" below) rather than ever
reusing preview-run image bytes, so there is nothing left to point it at.

### The `workflow` command (single entry point)

`workflow` chains the 15 commands above end-to-end through their own
existing functions — it introduces no new algorithm, calibration, or
generation logic, only orchestration. It is resumable: progress is persisted
to `<config_dir>/<run.name>_workflow_state.json`, and re-running the same
`workflow` invocation continues from the first incomplete stage rather than
restarting or repeating finished ones.

Stage sequence: `validate-config` → `build-basis` → smoke check → `probe` →
**stop for review 1** → `select-candidates` → `calibrate` → preview
generation → **stop for review 2** → smoke check (reused from the first
pass if config/basis are unchanged) → full-pilot generation → `metrics` →
`analyze`.

- **Review 1** uses the probe-group-review artifact (above). If it doesn't
  exist yet, `workflow` writes its template + contact sheets, prints the
  exact path and the exact command to resume, and exits with status
  `awaiting_review_1`. If it exists but has unanswered fields, it reports
  exactly which rows are incomplete rather than treating the file as done.
  **If nomination yields zero candidates**, `workflow` stops there (status
  `no_candidates`, printed as "本轮无候选") — no calibration, preview, or
  further stage runs, since there is nothing left to compare against
  reference.
- **Review 2** uses the preview-exclusion-review artifact the same way
  (`awaiting_review_2`). **If every edited condition is excluded** (only
  `reference` survives), `workflow` stops there (status
  `no_approved_conditions`, printed as "本轮无通过预览的编辑条件") — no
  full-pilot, metrics, or analyze stage runs, since a reference-only batch
  has nothing to compare against and is not generated just to hit a fixed
  image count. If `reference` itself was marked excluded, `workflow` raises
  instead of proceeding — that needs manual investigation, not an automated
  decision.
- A smoke check (§ "Real-model smoke check" — none of this needs
  calibration to exist) runs before both real-generation stages (`probe`,
  full-pilot), reusing a cached pass keyed by `(config_hash, basis_hash)`
  when config/basis haven't changed between them. Pass `--skip-smoke-check`
  to disable it for CPU/test runs.
- `--generation-python`/`--metrics-python` (default: the interpreter running
  `workflow` itself) let the build/probe/preview/full-pilot stages and the
  metrics stage run in their own separate venvs — when a stage's configured
  interpreter differs from the current one, `workflow` shells out to
  `python -m pc_specific_psd <that stage's own command>` instead of calling
  it in-process, reusing the existing CLI rather than adding a second
  execution path. The full-pilot dispatch always also passes
  `--approved-conditions-file`, so an excluded sign (e.g. `B5_plus`) can't be
  silently regenerated by a child interpreter that only knows the
  group-level candidate list.
- `--probe-group-review-file`/`--preview-exclusion-file` override the
  directory the two review templates are written to/read from (they default
  to alongside `--config`); each is always a fixed filename
  (`probe_group_review.csv` / `preview_exclusion_review.csv`) inside that
  directory, not an arbitrary full path.

Every command `workflow` drives remains fully usable on its own for
step-by-step or debugging use — `workflow` is additive, not a replacement.

### Verified command sequence (this pass, against `configs/sdxl_turbo_pca_v1.yaml`)

```bash
python3 -m pc_specific_psd validate-config --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml
python3 -m pc_specific_psd build-basis      --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run   # ready: false (no dataset)
python3 -m pc_specific_psd inspect-basis    --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run   # ready: false
python3 -m pc_specific_psd probe            --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run   # image_count: 84 (succeeds despite psd/calibration fields being unresolved)
python3 -m pc_specific_psd export-review    --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run   # pair_count: 84
python3 -m pc_specific_psd calibrate        --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run   # gate_candidate_count: 3, group_ids: [B1..B6]
python3 -m pc_specific_psd generate-psd     --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run --stage preview  # refuses: psd.calibration_result_path not resolved
python3 -m pc_specific_psd workflow         --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run   # reports the full planned stage sequence and current resume point
```

None of these writes anything to disk (verified by `find`-ing the referenced
output directories before and after).

## Budgets

Both the probing manifest and the preview/full-pilot manifests are computed
by `manifests.py`, never hardcoded, and branch on `select_candidates()`'s
actual output length:

| Stage | 0 candidates | 1 candidate | 2 candidates |
| --- | --- | --- | --- |
| Probing (fixed, precedes any candidate selection) | 84 images (`manifests.probing_image_count()`) | — | — |
| ρ=0.20 supplement (conditional follow-up) | 72 images (`manifests.rho020_supplement_image_count()`), only when triggered | — | — |
| Preview | no manifest | 36 images (3 configs × 12 draws) | 60 images (5 configs × 12 draws) |
| Full pilot | no manifest | 144 images (3 configs × 12 draws × 4-image galleries) | 240 images (5 configs × 12 draws × 4-image galleries) |

`manifests.num_configs_for_candidates(n)` raises for any `n` other than 0, 1,
2 — a third candidate is out of scope for this study.

The figures above are the full-pilot budget **ceiling** — the count when
every declared condition survives review 2. Full-pilot draws come from an
independent `base_index` range (`4`–`7`) that probing/preview never touch
(they only ever use `0`–`3`), so final-evaluation images are guaranteed to be
seeds no human has already looked at during preview screening — no
preview-image reuse happens any more. When `workflow`'s preview-exclusion
review drops one or more conditions (e.g. keeping `B5_minus` but excluding
`B5_plus`), the *executed* full-pilot image count shrinks below the ceiling
(`len(approved_condition_ids) * 12 * 4`, always still including `reference`)
rather than backfilling with another condition to hit the fixed number —
`workflow`'s final report states both the executed count and the
pre-exclusion ceiling side by side.

## Resuming a run

Generation commands go through `compat_generation.ensure_immutable_run`
(reused from `noise_init.io_utils` by value, not reimplemented): a run
directory records a hash of its own config/manifest, and re-invoking the same
command against an unchanged config resumes/reuses that run rather than
regenerating; a changed config under the same run id is rejected rather than
silently overwritten. Pass `--force` to recompute/regenerate even when a
complete resume point exists.

## What's real and what's not (this pass)

Implemented and CPU/synthetic-tested: basis construction and codecs (against
synthetic latents), the 84-image probing manifest and non-overlap codec, all
three review-track schemas and the candidate nomination rule, the stride-1
overlapping PSD editor and its τ=0 identity path, spectral/PSD estimation,
calibration's finite-candidate-set selection procedure, the runner's
preview/full budget branching and independent full-pilot seeds, metrics
wrapping, paired-bootstrap analysis, and the full CLI including per-command
config-tier validation and
`--dry-run` for every command. Also implemented and CPU/synthetic-tested this
round: the `workflow` orchestrator's full 14-stage state machine end-to-end
(both review stops and their incomplete/complete/conflict handling, both
no-candidates/no-approved-conditions terminal states, the reference-excluded
hard stop, the calibration-independent smoke check with its
`(config_hash, basis_hash, reference_scale_profile)` cache, and cross-venv dispatch — all exercised
with a fake adapter/metric runner standing in for real model calls).

The frozen same-phase reference remains α=0.9, γ=0.05, but
`operator_clean` now applies the analytic `expected_unit_rms_rfft_v1` scale
to both reference and candidates exactly once. It derives expected RMS from
the rFFT response using Hermitian column weights (DC/Nyquist 1, interior 2)
and `height * width` normalization. Because this is a fixed size-dependent
scalar rather than per-sample normalization, the operator stays linear,
Gaussian, phase-preserving, and spectrally shape-preserving. Calibration
registries and smoke-cache entries created without this profile are stale and
must be regenerated. This change does not alter the separate size contract:
SDXL-Turbo still produces 512×512 images from 4×64×64 initial latents.

**Not run** in this pass, and requiring resources outside this sandbox:

- Real PCA basis construction — needs a real image dataset manifest (no
  ImageNet/COCO copy or cached SDXL-Turbo/VAE weights are present here).
- Real SDXL-Turbo/VAE download and GPU generation — the probing run, preview
  run, and full pilot are all deferred to a separate RTX Pro 4000/4090/5090
  machine (this sandbox has an RTX 5060 8GB and no cached weights).
- All three rounds of human annotation (probe / preview / full-gallery
  review), and the two consolidated `workflow` review artifacts built on top
  of them — none of these can be automated by design; see
  `docs/PCA-Specific-PSD-Editing-Plan.md`'s human-gate clarification.
- Calibration, candidate selection, and metrics/analysis over real data,
  which all depend on the above.
- A real (non-fake-adapter) smoke check against actual SDXL-Turbo weights,
  and a full end-to-end `workflow` run against real data on a GPU machine —
  the exact command to run once a dataset manifest and GPU access are
  available is `python3 -m pc_specific_psd workflow --config
  pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml`.

See `docs/IMPLEMENTATION_REPORT.md` for the acceptance-ID-level breakdown and
the exact next commands once a dataset path and GPU access are available.
