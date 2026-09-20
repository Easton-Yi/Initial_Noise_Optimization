# Implementation Report

## Scope of files touched

Every file added or modified this pass lives under `pc_specific_psd/`.
Nothing under `noise_init/`, the repo root, or any other directory was
modified, moved, deleted, or reformatted (confirmed via `git status --short`
at the repo root after every change: the only line ever present is
`?? pc_specific_psd/`).

New content under `pc_specific_psd/`:

- Package modules: `__init__.py`, `__main__.py`, `_compat_common.py`,
  `compat_generation.py`, `compat_metrics.py`, `config.py`, `manifests.py`,
  `patch_codec.py`, `basis.py`, `probing.py`, `review.py`, `psd_editor.py`,
  `spectral.py`, `calibration.py`, `adapters.py`, `runner.py`, `metrics.py`,
  `analysis.py`, `cli.py`, `workflow.py` (a later, separate pass — see
  `docs/IMPLEMENTATION_DECISIONS.md`'s "Workflow consolidation" section; it
  orchestrates the other modules' existing functions and adds no new
  algorithm).
- `configs/sdxl_turbo_pca_v1.yaml` — the production config (§ below).
- `manifests/` — empty; the intended destination for generated per-run
  artifacts (see `docs/IMPLEMENTATION_DECISIONS.md` decision #1). Nothing
  has been written there yet because no real command has been run against
  real data.
- `tests/` — 31 test files, 500 tests total (see "Test commands and
  results" below).
- `docs/PC-Specific-PSD-Implementation-Prompt.md`,
  `docs/PCA-Specific-PSD-Editing-Plan.md` (pre-existing planning docs, not
  modified this pass) plus the four newly-written deliverables:
  `README.md` (package root), `docs/IMPLEMENTATION_DECISIONS.md`,
  `docs/IMPLEMENTATION_CHECKLIST.md`, `docs/IMPLEMENTATION_REPORT.md` (this
  file).

## Test commands and results

```bash
python3 -m unittest discover -s pc_specific_psd/tests -v
```

Result: **500 tests, all passing**, no errors, no skips (~74s on CPU as of
this update; ~51s/422 tests when this report was first written). This is the
complete test suite for the package — every algorithmic module (`basis`,
`patch_codec`, `probing`, `review`, `psd_editor`, `spectral`, `calibration`,
`runner`, `metrics`, `analysis`, `config`, `manifests`, `adapters`, `cli`,
`workflow`) has a dedicated test file, plus cross-cutting tests for
cwd-independence, import-graph independence, budget accounting, calibration
gating, the τ=0 identity path, base-white pairing, the frozen same-phase
reference, the review-ingestion refusal path through the real CLI entry
point, the unverified-adapter (FLUX) refusal guard (2 tests in
`test_config.py` and 4 in `test_cli.py::UnverifiedAdapterRefusalTests`), and
(added by the later workflow-consolidation pass) `workflow`'s full stage
sequencing/resumability/review-conflict handling
(`tests/test_workflow.py`), its cross-venv approved-conditions argv contract
(`tests/test_workflow_cross_venv_exclusion.py`), the two new consolidated
review artifacts (`tests/test_review_group_consolidation.py`,
`tests/test_preview_exclusion_review.py`), candidate-filtered calibration
(`tests/test_calibrate_candidates_filter.py`), per-condition full-pilot
exclusion (`tests/test_full_pilot_condition_exclusion.py`), and the
calibration-independent smoke check (`tests/test_smoke_check.py`).

The production config was additionally exercised directly against the real
CLI (not just test fixtures), confirming its dry-run behavior matches the
plan's specified acceptance criteria exactly:

```bash
python3 -m pc_specific_psd validate-config --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml
# -> ok

python3 -m pc_specific_psd build-basis --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run
# -> {"status": "dry_run", "ready": false, ...}   (no dataset_manifest_path)

python3 -m pc_specific_psd inspect-basis --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run
# -> {"status": "dry_run", "ready": false, ...}

python3 -m pc_specific_psd probe --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run
# -> {"status": "dry_run", "image_count": 84, ...}   (succeeds with psd/calibration fields unresolved)

python3 -m pc_specific_psd export-review --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run
# -> {"status": "dry_run", "pair_count": 84, ...}

python3 -m pc_specific_psd calibrate --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run
# -> {"status": "dry_run", "gate_candidate_count": 3, "group_ids": ["B1","B2","B3","B4","B5","B6"], ...}

python3 -m pc_specific_psd generate-psd --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run --stage preview
# -> refuses: psd.calibration_result_path (and other needs_calibration fields) not resolved

python3 -m pc_specific_psd workflow --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --dry-run
# -> reports the full 14-stage planned sequence and current resume point, touching nothing
```

`find pc_specific_psd/configs pc_specific_psd/basis pc_specific_psd/outputs
pc_specific_psd/calibration -type f` before and after this sequence
confirmed only the config YAML itself exists on disk — every `--dry-run`
invocation wrote nothing, including no `basis/`, `outputs/`, or
`calibration/` directory (the paths referenced by the config but never
created).

See `docs/IMPLEMENTATION_CHECKLIST.md` for the ID-by-ID mapping of this
evidence onto the Implementation Prompt's §13 acceptance table, which
enumerates exactly **39** IDs (`S01`-`T03`/`D01`-`D02`). This pass also adds
4 CLI commands beyond the plan's originally-required 11
(`export-preview-review`, `ingest-preview-review`, `export-gallery-review`,
`ingest-gallery-review`); these are reported in the checklist as
**CLI-command extensions**, not as additional §13 IDs — they do not add rows
to the acceptance table.

## How far CPU/mock vs. GPU/real experiments got

**CPU/synthetic — complete for this pass:**

- Basis construction and both patch codecs, tested against synthetic
  latent tensors (not real VAE output).
- The full 84-image probing manifest, non-overlap rotation, fair-budget
  angle computation, and measured-vs-theoretical statistics recording.
- All three review-track schemas (probe/preview/gallery), the candidate
  nomination rule, the ρ=0.20 trigger, and the candidate-recheck gate.
- The stride-1 overlapping PSD editor, including the τ=0 identity path
  verified through both the scalar shortcut and the forced full-codec path.
- Radial PSD/transfer-matrix diagnostics (`spectral.py`).
- Calibration's finite-candidate-set selection procedure against synthetic
  Gaussian noise banks, and the legacy-matched/operator-clean protocol
  separation.
- The runner's PC-condition schema, resume/immutability behavior,
  independent full-pilot seeds (never reusing a preview-reviewed draw), and
  preview/full budget branching — all exercised with a fake adapter
  (`FakeAdapterPCA`) standing in for real generation.
- Metrics wrapping and paired-bootstrap/no-interpolation analysis, both
  exercised with a fake metric runner (`FakeMetricRunner`) standing in for
  real HPSv3/CLIP/DreamSim/LPIPS/Vendi calls.
- The full CLI (all 15 original commands, plus the later `workflow`
  orchestrator command — 16 total), per-command config-tier validation, and
  `--dry-run` reporting for every command, validated both by the test suite
  and by hand against the real production config.
- `workflow`'s full state machine (both human-review stops and their
  incomplete/complete/conflict handling, both no-candidates/
  no-approved-conditions terminal states, the reference-excluded hard stop,
  the calibration-independent smoke check and its cache, and cross-venv
  dispatch), exercised end-to-end with a fake adapter/metric runner standing
  in for real generation/metrics calls — see
  `docs/IMPLEMENTATION_DECISIONS.md`'s "Workflow consolidation" section.

**GPU/real — not started, by design (this sandbox has an RTX 5060 8GB with
no cached SDXL-Turbo/VAE weights and no local image dataset; real generation
is deferred to a separate RTX Pro 4000/4090/5090 machine):**

- No real PCA basis has been built — `basis.dataset_manifest_path` is
  `null` in the production config.
- No real SDXL-Turbo/VAE weights have been downloaded or run.
- No probing images have been generated.
- No human annotation has been produced for any of the three review tracks
  (this is expected — it is the plan's explicit hard human-annotation gate,
  not an oversight).
- No candidate selection, calibration, preview generation, full-pilot
  generation, real metric computation, or real analysis has been run.

## Remaining data / calibration dependencies

The later workflow-consolidation pass (see
`docs/IMPLEMENTATION_DECISIONS.md`) automates steps 2-7 below into one
resumable command, `python3 -m pc_specific_psd workflow --config
pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml`, stopping only at the two
human-review touchpoints (using the two new consolidated review artifacts,
not the detailed per-image ones referenced below) and printing the exact
resume command each time. The manual, per-command sequence below remains
fully valid and is what `workflow` itself calls internally at each stage; it
is documented here for step-by-step/debugging use or for anyone who wants
the full detailed review tracks instead of the consolidated ones.

In order, the next real steps and their exact inputs:

1. **Dataset manifest.** Supply a JSON file (`source` + `entries: [{image_id,
   path, sha256}, ...]`, ≥2000 images per plan §3) and set
   `basis.dataset_manifest_path` in `configs/sdxl_turbo_pca_v1.yaml` to its
   path. No auto-download will happen; this requires an explicit path from
   the user.
2. **GPU + SDXL-Turbo/VAE weights.** On the RTX Pro 4000/4090/5090 machine,
   with the generation venv active (`noise_init/requirements.txt` or
   `requirements_generation.txt`), run:
   ```bash
   python3 -m pc_specific_psd build-basis --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml
   python3 -m pc_specific_psd probe --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml
   ```
   `probe` produces the 84-image probing manifest (`manifests.probing_image_count() == 84`).
3. **Human annotation, round 1 (probe review).**
   ```bash
   python3 -m pc_specific_psd export-review --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml
   ```
   A person fills in the blind paired-role template. Then:
   ```bash
   python3 -m pc_specific_psd select-candidates --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml \
       --review-file <completed_review.json> --mapping-file <mapping.json>
   ```
   This may nominate 0, 1, or 2 PC groups; if the review shows insufficient
   role-change evidence (`compute_rho020_trigger` count `< 2`), a one-time
   `probe --rho020-followup` (72 more images) may run first.
4. **Calibration.**
   ```bash
   python3 -m pc_specific_psd calibrate --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml
   ```
   Freezes a single gate/τ/correction configuration per candidate group into
   `psd.calibration_result_path`, validated once against the independent
   validation bank.
5. **Preview generation + review, round 2.**
   ```bash
   python3 -m pc_specific_psd generate-psd --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml \
       --stage preview --candidates-file <select-candidates output>
   python3 -m pc_specific_psd export-preview-review --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml \
       --candidates-file <...>
   ```
   Budget: 36 images for 1 candidate, 60 for 2 (§ README "Budgets"). A person
   fills in the blind single-image template; ingest with
   `ingest-preview-review`.
6. **Full pilot + gallery review, round 3.**
   ```bash
   python3 -m pc_specific_psd generate-psd --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml \
       --stage full --candidates-file <...>
   python3 -m pc_specific_psd export-gallery-review --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml \
       --candidates-file <...>
   ```
   Budget: 144 images for 1 candidate, 240 for 2 (this is now a ceiling, not
   a fixed count — `workflow`'s per-condition exclusion, § README "Budgets",
   can execute fewer). Full-pilot draws use an independent seed range
   (`base_index 4-7`) never touched by probing/preview, so no image reviewed
   during preview screening is reused here (`--preview-run-dir` no longer
   exists). A person fills in the blind 4-image-gallery template; ingest
   with `ingest-gallery-review`.
7. **Metrics + analysis.** On the metrics venv
   (`noise_init/requirements_metrics.txt`):
   ```bash
   python3 -m pc_specific_psd metrics --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml --device cuda
   python3 -m pc_specific_psd analyze --config pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml \
       --gallery-review-file <...> --gallery-mapping-file <...>
   ```

Each of steps 2-7 is currently `BLOCKED` on the step before it, and all of
them are `BLOCKED` on step 1 (the dataset manifest) and access to a GPU
machine with SDXL-Turbo weights. None of the placeholder numeric values in
`configs/sdxl_turbo_pca_v1.yaml`'s `psd` section (gate candidates, τ
candidates, tolerances, bank sizes — see
`docs/IMPLEMENTATION_DECISIONS.md` decision #3) have been validated against
real noise; `calibrate` (step 4) is what will actually select and freeze
real values from among them, or reveal that the candidate lists need
revising first.

## Summary for the user

This pass involves three genuinely different questions, which are kept
strictly separate below rather than blended into one "percent complete"
number: whether the code that was written is correct, whether any of it has
actually been executed against real data, and whether PCA editing has been
shown to help. The answers are, respectively: yes (on CPU/synthetic
evidence), no (nothing beyond CLI wiring has touched a GPU or real weights),
and unknown (no result exists to report either way yet).

### 1. Implementation correctness (CPU/synthetic-tested, this pass)

- **Implemented entry points**: all 16 CLI commands
  (`validate-config, build-basis, inspect-basis, probe, export-review,
  select-candidates, calibrate, validate-noise, generate-psd,
  export-preview-review, ingest-preview-review, export-gallery-review,
  ingest-gallery-review, metrics, analyze, workflow`), each with `--dry-run`.
  11 of these are the plan's originally-required commands; the 4
  `*-preview-review`/`*-gallery-review` commands and the later `workflow`
  orchestrator are CLI-command extensions (the first four give the preview
  and full-gallery review tracks their own export/ingest commands;
  `workflow` chains all the others end-to-end via their own functions) —
  none are additional §13 acceptance IDs.
- **Mapping to the plan**: see `docs/IMPLEMENTATION_DECISIONS.md`'s
  interface-mapping table, its 4 documented deviations (manifests.py instead
  of external data files; config copied by value with a recorded hash
  instead of included by reference; all six PC groups pre-declared with
  placeholder candidate sets; `dataset_manifest_path: null`), and its two
  additional sections on the unverified-adapter (FLUX) guard and the
  probe-review single-ingestion-path design.
- **Acceptance IDs**: the Implementation Prompt's §13 table enumerates
  exactly **39** IDs. Of those: **35 `PASS`**, **4 `BLOCKED`**, 0 `NOT_RUN`,
  0 `NOT_APPLICABLE`, 0 `FAIL`. A `PASS` here means CPU/synthetic mechanism
  evidence was actually run and observed, not that the underlying scientific
  method has been validated on real data. Full ID-by-ID breakdown, including
  exactly what each `BLOCKED` ID is waiting on, is in
  `docs/IMPLEMENTATION_CHECKLIST.md`.
- **Test suite**: 500/500 CPU tests pass
  (`python3 -m unittest discover -s pc_specific_psd/tests -v`), including 6
  tests added for the FLUX/unverified-adapter guard (acceptance ID `T03`,
  `PASS` — an unverified adapter is explicitly refused by the CLI with a
  `not_verified` message, at config-load time, for every command tier) and,
  from the later workflow-consolidation pass, 78 further tests across the 7
  new test files listed under "Test commands and results" above — all still
  CPU/synthetic, none adding a new §13 acceptance ID (see
  `IMPLEMENTATION_CHECKLIST.md`).

### 2. Real experiment execution (GPU/real data — not started, by design)

Nothing below has been run against real weights, real images, or real human
judgment; this sandbox (RTX 5060 8GB, no cached SDXL-Turbo/VAE weights, no
local image dataset) was never going to attempt it, and real generation is
deferred to a separate RTX Pro 4000/4090/5090 machine.

- No real PCA basis has been built — `basis.dataset_manifest_path` is
  `null` in the production config (`A01`, `BLOCKED`).
- No real SDXL-Turbo/VAE weights have been downloaded or run, and no
  genuine (non-`FakeAdapterPCA`) smoke test has been executed (`G01`, `G02`,
  `T02`, all `BLOCKED`).
- No probing images have been generated.
- No human annotation has been produced for any of the three review tracks
  (this is the plan's explicit hard human-annotation gate, not an
  oversight).
- No candidate selection, calibration, preview generation, full-pilot
  generation, real metric computation, or real analysis has been run.

None of these were approximated, faked, or silently skipped — the CPU test
suite exercises the same code paths with `FakeAdapterPCA`/`FakeMetricRunner`
standing in for the real model calls, so the mechanism is verified
independent of, not instead of, the real run.

**Next user-executable command and its budget**: supply a dataset manifest
path and set `basis.dataset_manifest_path` in
`configs/sdxl_turbo_pca_v1.yaml`, then on a machine with the generation venv
active run `python3 -m pc_specific_psd build-basis --config
pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml` (one-time basis fit, no
image budget) followed by `python3 -m pc_specific_psd probe --config
pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml` (84 images) — or, to run the
whole pipeline through both human-review stops in one resumable command,
`python3 -m pc_specific_psd workflow --config
pc_specific_psd/configs/sdxl_turbo_pca_v1.yaml` (see README's "The
`workflow` command" section).

### 3. Scientific effectiveness (no findings exist yet)

No PCA-edited images have been generated and no Q/D (quality/diversity)
claim can be made yet, in either direction — this pass delivers a tested
pipeline, not a result. There is no basis for saying PCA-specific PSD
editing has "already improved" anything; that question is not answerable
until real basis construction, probing, human annotation, calibration, and
metric computation (steps 1-7 above) have actually run on real data.
