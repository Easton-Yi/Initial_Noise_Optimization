# Implementation Checklist

Status values: `PASS`, `FAIL`, `NOT_RUN`, `BLOCKED`, `NOT_APPLICABLE`, per
`PC-Specific-PSD-Implementation-Prompt.md` §13. A `PASS` here means the cited
evidence was actually run and observed this pass, not that the underlying
scientific method has been validated on real data — most `PASS` rows below
are CPU/synthetic-data checks of the mechanism (algorithm, schema, gating,
budget) described by that ID, not a real-GPU/real-annotation result. Where an
ID has both a CPU-testable mechanism and a GPU/data/annotation-dependent
execution, both are stated explicitly rather than letting the mechanism PASS
imply the execution happened.

Full test suite evidence command, referenced by ID below as "full suite":

```bash
python3 -m unittest discover -s pc_specific_psd/tests -v
# Ran 500 tests ... OK
```

**ID accounting.** This table covers exactly the 39 IDs the Implementation
Prompt's §13 table actually enumerates (`S01`–`S03`, `C01`–`C03`, `A01`–`A04`,
`B01`–`B07`, `E01`–`E05`, `N01`–`N05`, `G01`–`G04`, `M01`–`M03`, `T01`–`T03`,
`D01`–`D02` = 3+3+4+7+5+5+4+3+3+2 = 39) — one row each, no additions, no
omissions. Beyond the plan's 11 required CLI commands, this pass also
implements 4 more (`export-preview-review`, `ingest-preview-review`,
`export-gallery-review`, `ingest-gallery-review`) to give the preview and
full-gallery review tracks their own export/ingest commands, exactly as the
plan's module layout calls for. These are **CLI-command extensions, not
additional §13 acceptance IDs** — they don't add rows to this table; their
correctness is covered by the existing `B05`–`B07`/`G03`/`M02` rows below
(review schema, budget branching, gallery handling), not by separate IDs.

A later, separate pass (the workflow-consolidation plan — see
`IMPLEMENTATION_DECISIONS.md`'s "Workflow consolidation" section) adds a
**16th** CLI command, `workflow`, plus two new review artifacts
(`ProbeGroupReviewRow`/`PreviewExclusionRow` in `review.py`) and a
calibration-independent smoke check (`adapters.smoke_check`). The same policy
applies: `workflow` is a **CLI-command extension**, not a new §13 ID — it
orchestrates the existing 15 commands' own functions (no new algorithm,
calibration, or generation logic), so its correctness is covered by the
existing rows it exercises end-to-end (`B06`/`B07` candidate nomination and
recheck, `N05` calibration gating, `G03`/`G04` budget branching and resume,
`M01`–`M03` metrics/analysis), not by separate IDs. It adds no new
acceptance criteria beyond what §13 already enumerates.

**Status accounting.** Of the 39 IDs: **35 `PASS`**, **4 `BLOCKED`**, 0
`NOT_RUN`, 0 `NOT_APPLICABLE`, 0 `FAIL` (35 + 4 = 39). Every remaining gap in
this pass turned out, on inspection, to be blocked on a specific missing
external input (a dataset manifest, or GPU hardware + downloaded weights) —
not a case of "could run now, just hasn't been" — so `NOT_RUN` is not used
here; see the per-row `BLOCKED` rationale for `A01`, `G01`, `G02`, `T02`
below for exactly what each one is waiting on.

## S — Scope / import / cwd-independence

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| S01 | PASS | `git status --short` (repo root), run after every change this pass | Only `?? pc_specific_psd/` ever appears; no tracked file outside it modified | — |
| S02 | PASS | `pc_specific_psd/_compat_common.py`, `compat_generation.py`, `compat_metrics.py`; `tests/test_compat_independence.py` | Two independent shims load only named `noise_init` modules via `sys.path` injection with an `__file__`-origin assertion; neither shim imports the other (checked via `ast`, not convention); no `noise_init` module is copied into this package | — |
| S03 | PASS | `tests/test_cli_dry_run.py::CwdIndependenceTests`; manual check: `find pc_specific_psd/configs pc_specific_psd/basis pc_specific_psd/outputs pc_specific_psd/calibration -type f` before/after every `--dry-run` invocation against `configs/sdxl_turbo_pca_v1.yaml` | Relative config paths resolve against the config file's own directory regardless of `cwd`; every `--dry-run` invocation (validate-config, build-basis, probe, generate-psd, inspect-basis, calibrate, export-review) wrote zero files | — |

## C — Prompt/model/schema fidelity

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| C01 | PASS | `manifests.py` (`PROMPTS`, `SeedBlock.sample_seed`); `tests/test_manifests.py::PromptTableTests` | 4 prompts × 3 seed blocks each, exact §14.2 `batch_seed` table; `sample_seed = batch_seed + base_index` verified | — |
| C02 | PASS (config-field parity only) | `configs/sdxl_turbo_pca_v1.yaml` header comment records source sha256 `99880bda712dc331e3dfe4ad117dfe6fcc00377dd54f4eb53636efb6704d6b50` for `noise_init/configs/sdxl_turbo_full.yaml`; fields copied by value (512×512, 1 step, guidance 0.0, `stabilityai/sdxl-turbo`, `madebyollin/sdxl-vae-fp16-fix`) | Config-level values match the original reference exactly | Runtime confirmation that the loaded pipeline actually resolves to these values requires real weights/GPU — see G01, T02 (`NOT_RUN`) |
| C03 | PASS | `runner.py` PC-condition schema (`pc_group_id`, `tau`, `gate_id`, `basis_hash`, `calibration_hash`, reference-condition id); `tests/test_runner.py` (`ref_record["pc_group_id"] is None`, `plus_record["pc_group_id"] == "B5"`) | PC-condition identity has its own fields; `alpha`/`gamma` never appear in this schema | `alpha`/`gamma` are not used by this package's runner at all (they belong to `noise_init.run_experiment`'s unrelated method), so there's nothing to repurpose in the first place |

## A — Basis construction correctness

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| A01 | `BLOCKED` | `basis.dataset_manifest_path: null` in `configs/sdxl_turbo_pca_v1.yaml`; `build-basis --dry-run` reports `ready: false` | Blocked on a missing required input: no real image dataset manifest exists in this sandbox | Needs a real dataset manifest (`source` + per-image `path`/`sha256`, ≥2000 images per plan §3) supplied by the user. Execution is not "possible but not yet performed" here — `build-basis` structurally cannot proceed without this path; it is not a scheduling gap |
| A02 | PASS (synthetic) | `tests/test_basis.py` | `d = C·p² = 4·5² = 100` enforced; full shared basis reconstructs input patches to numerical tolerance; patch flatten order (channel-major, row, col) matches `patch_codec.py` exactly | Exercised on synthetic float tensors, not real VAE latents |
| A03 | PASS (synthetic) | `tests/test_basis.py` (`StreamingMeanCovariance` vs `direct_mean_covariance` agreement; split-half stability; metadata-mismatch load rejection) | Float64 streaming covariance matches direct computation; `load_basis` rejects a patch-size/channel mismatch and an unrequested synthetic basis (`allow_synthetic=False`) | — |
| A04 | PASS | `basis.py` docstring + `BasisArtifact.mean` comment ("statistics/provenance only, never applied to noise"); `manifests.PC_GROUPS` uses plain `B1`–`B6` ids, not frequency-semantic labels | No mean/eigenvalue/whitening is ever applied to noise (`a = V^T q`, `q = V a` only); no PC group is auto-labeled by frequency content | — |

## B — Probing correctness / budget / review schema / nomination

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| B01 | PASS | `tests/test_patch_codec.py`, `tests/test_probing.py` (boundary-unchanged, `theta=0` identity checks) | Centered non-overlap grid, boundary passthrough, and full orthogonal block rotation all verified | — |
| B02 | PASS | `tests/test_probing.py::FairBudgetAngleTests` | `theta_B = arccos(1 - rho^2*D/(2*N_patch*|B|))` computed correctly; infeasible configuration raises rather than clamping | — |
| B03 | PASS | `probing.py` (`donor_seed` excludes `group_id`; rotation applied to raw PCA coefficients before any `normalize()` call) | Independent donor per `(block, base_index)` shared across all groups probed off that draw; no empirical rescale | — |
| B04 | PASS | `manifests.probing_image_count() == 84`; `manifests.rho020_supplement_image_count() == 72`; `tests/test_budget_accounting.py` | First round is exactly 84 images across 6 groups off one shared basis numbering; the ρ=0.20 follow-up budget is a separate, explicit 72 | — |
| B05 | PASS | `review.py::export_probe_review`; `tests/test_review.py` | Blind template covers all fields (layout/pose/shape/color/light/texture 0/1/2/NA, before/after plausibility, 3 artifact fields, valid-structural-change, evidence, confidence) | — |
| B06 | PASS | `review.py::select_candidates`; `tests/test_review.py::SelectCandidatesTests` (10 methods); `cli._load_and_ingest_probe_review` (routes `select-candidates`'s `--review-file`/`--mapping-file` through `review.ingest_probe_review()` before the nomination rule ever runs); `tests/test_cli.py::SelectCandidatesReviewFileIngestionRefusalTests` | ≥2/3 base draws in a prompt, ≥2/4 prompts rule; max 2 groups; priority by coverage then valid-change count; reserve list; correctly returns 0 or 1 candidates in under-evidence fixtures; there is no separate top-level `ingest-probe-review` command -- `select-candidates` (and `probe --rho020-followup`, and the candidate-recheck builder) are the sole ingestion entry points, verified via the real CLI to refuse (before nominating anything) on a missing or malformed review file | — |
| B07 | PASS | `probing.py::CandidateRecheckManifestTests` (in `tests/test_probing.py`) | Candidate/reserve recheck fires only off `select_candidates()`'s non-empty output, via its own gate, independent of the ρ=0.20 trigger; recorded status is a distinct field, never conflated with a safety or diversity claim | — |

## E — PSD editor correctness

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| E01 | PASS | `patch_codec.py::OverlapCodec` (`decode_center`, circular padding); `tests/test_patch_codec.py` | Genuine stride-1 circular-padded projection and `Center()` synthesis; not overlap-add, not swapped for the non-overlap codec | — |
| E02 | PASS | `psd_editor.py::apply_psd_edit`; `tests/test_psd_editor.py` | Per-PC coefficient-map spatial FFT; full basis vs. complement split; one shared `t_B(r; tau)` multiplier applied uniformly across the whole in-group block | — |
| E03 | PASS | `psd_editor.group_transfer_multiplier` (`t_B(r;tau) = exp(0.5*tau*w(r))`); `tests/test_psd_editor.py` | Positive/negative τ produce reciprocal-direction power changes at fixed gate; frozen `h_ref` reference matches locked `SAME_PHASE_ALPHA=0.9`/`SAME_PHASE_GAMMA=0.05` | — |
| E04 | PASS | `tests/test_tau_zero_correction.py::ForcedFullCodecPathAgreesWithShortcutTests` | Identity/all-PC-scalar/τ=0 forced-full-codec path agrees with both the τ=0 shortcut and the unmodified `same_phase_floor` reference via `torch.allclose` (not bitwise) | — |
| E05 | PASS | `psd_editor.apply_psd_edit` reconstructs via `codec.decode_center`/`decode_center_reference` (patch-space `Center(Σ_i ã_i(x,y) v_i)` synthesis) rather than summing per-PC power spectra; `spectral.py` measures the actual reconstructed output, never a sum of per-PC PSDs | Cross-PC interaction is handled correctly by construction (synthesis mixes PCs in patch space before any PSD is measured), verified indirectly via the identity/equivalence tests above rather than a dedicated cross-spectrum unit test | No test isolates a case with deliberately correlated PC coefficient maps to directly exercise cross-spectrum behavior; the identity and calibration tests only exercise independent per-PC filtering |

## N — Spectral / calibration correctness

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| N01 | PASS | `tests/test_spectral.py::RadialPSDParsevalTests` | Full-FFT and conjugate-weighted rFFT agree; Parseval holds for even/odd width/height; DC/Nyquist handled with no NaN | — |
| N02 | PASS (synthetic banks) | `calibration.py`; `tests/test_calibration_finite_set.py` | Fixed correction `c_{B,tau}(b)=sqrt(P_ref(b)/P~(b))` with epsilon/invalid-bin/gain-bound checks enforced; calibration and validation banks are seeded independently (`cfg.psd.calibration_bank_seed` vs `validation_bank_seed`) | Exercised against synthetic Gaussian banks (`torch.randn`), not real SDXL-Turbo noise statistics |
| N03 | PASS | `calibration.py` (`legacy-matched` vs `operator-clean` kept as separate code paths); `tests/test_calibration.py` | `legacy-matched` is only validated after a full `normalize()` pass; `operator-clean` never applies per-sample std and is reported independently | — |
| N04 | PASS | `spectral.py::TransferMatrixDiagnosticsTests` (in `tests/test_spectral.py`) | C×C transfer matrix, singular-value/condition-number diagnostics, linearity and shift-equivariance checks, and a constructed near-singular case are all flagged correctly | — |
| N05 | PASS (mechanism; real freeze `NOT_RUN`) | `config.py::_full_tier_missing`/`COMMAND_TIERS`; `calibrate --dry-run` against `configs/sdxl_turbo_pca_v1.yaml` (reports `gate_candidate_count: 3, group_ids: [B1..B6]`); `generate-psd --dry-run --stage preview` against the same config refuses citing the unresolved `psd.calibration_result_path` | Unlocked parameters are defined/frozen only via the noise-only calibration procedure (never against generated images); `generate-psd` structurally refuses when that freeze hasn't happened | `calibrate` has not actually been run to completion against a real or even a full synthetic bank in production form — only its selection *procedure* is unit-tested (see N02) |

## G — Real-generation provenance / independent seeds / resume

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| G01 | `BLOCKED` | — | Blocked on two missing upstream inputs: a real basis (blocked by `A01`) and real SDXL-Turbo weights + a GPU, neither present in this sandbox | Deferred to the RTX Pro 4000/4090/5090 machine; not something this sandbox could execute even if scheduled, so `BLOCKED` rather than `NOT_RUN` |
| G02 | `BLOCKED` | — | Blocked on the same missing inputs as `G01` (no real generation has happened yet to repeat) | Same GPU/weights/basis dependency as `G01` |
| G03 | PASS (mechanism; real preview/full execution `BLOCKED`, see `G01`) | `tests/test_runner_config_branch.py` -- **these are executed branches, not documentation-only constants**: `ZeroCandidateBranchTests` actually calls `runner.generate_preview`/`generate_full_pilot` with `[]` and asserts `None`; `OneCandidateBranchTests`/`TwoCandidateBranchTests` actually call `runner.generate_manifest` with a real `FakeAdapterPCA`, then count real `image.png` files written to disk (`_count_images`) and the adapter's real call count, asserting exactly 36/144 (one candidate) and 60/240 (two candidates); `tests/test_budget_accounting.py::test_generate_full_pilot_no_longer_takes_preview_run_dir` and its disjoint-sample-seed test | 0/1/2-candidate branches for both preview (36/60) and full-pilot (144/240 ceiling) budgets are real, executed conditionals on `select_candidates()`'s actual output length, confirmed by counting files actually written during the test run, not asserted against a fixed number. **Updated by the workflow-consolidation pass**: the earlier preview→full dedup-by-hash reuse path (`generate_full_pilot`'s `preview_run_dir` parameter) has been deleted from `runner.py` entirely — full-pilot draws now use an independent `base_index` range (`4`-`7`) that probing/preview never touch, and `manifests.assert_disjoint_sample_seeds` checks this at runtime rather than assuming it from the base_index ranges alone (see `IMPLEMENTATION_DECISIONS.md`'s "Workflow consolidation" section, decision #4) | No real preview or full-pilot run has actually been executed against real SDXL-Turbo images (blocked the same way as `G01`); nothing here is reported as a diversity result |
| G04 | PASS | `tests/test_runner.py` (`test_resume_skips_already_complete_samples`, `test_incomplete_existing_sample_dir_is_refused_not_overwritten`, `test_resuming_with_a_changed_config_is_refused`) | Resume skips already-complete samples, refuses to silently paper over a crashed mid-write sample dir, and refuses to resume under a changed `calibration_hash`/config rather than mixing results | Exercised with the `FakeAdapterPCA` test double, not a real model process |

## M — Metrics / analysis reporting

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| M01 | PASS (wrapper logic; real weights `NOT_RUN`) | `metrics.py` (bypasses `noise_init.metric_runner.evaluate_run`'s hardcoded alpha/gamma shape and `"image_"`-prefix filename filter directly); `tests/test_metrics.py` with `FakeMetricRunner` | HPSv3/CLIP/DreamSim/LPIPS/Vendi wrapping logic is exercised end-to-end with a deterministic fake standing in for the real model calls; every non-4-image gallery gets an explicit `not_applicable`/`incomplete` status instead of being silently skipped | No real metric model (HPSv3 weights, CLIP, DreamSim, LPIPS, Vendi) has actually been invoked — this requires the separate metrics venv and, for HPSv3, downloaded weights |
| M02 | PASS | `tests/test_metrics.py` (incomplete-gallery statuses); `tests/test_analysis.py` (block/prompt×seed-batch paired bootstrap) | 4-image/6-pair gallery handling and explicit incomplete-gallery statuses verified; block-paired bootstrap matches `noise_init.analysis`'s recipe (2000 replicates, seed 20260829, 95% CI) without duplicating draws across blocks | Bootstrap correctness checked on synthetic per-condition score arrays, not real metric outputs |
| M03 | PASS | `analysis.py` (separate reported axes for role/quality/diversity/artifact; gallery-review axis marked `incomplete` when not ingested); `tests/test_analysis_incomplete_axis.py`; `tests/test_analysis.py`'s no-interpolation guard | Axes are reported separately, never merged into one composite score; a comparison at an unmeasured operating point raises instead of interpolating; no "already improved Q-D" framing exists anywhere in the code or its output schema | — |

## T — Testing completeness / unverified-adapter (FLUX) guard

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| T01 | PASS | Full suite (500 tests, `python3 -m unittest discover -s pc_specific_psd/tests -v`) | Every algorithmic module has a dedicated CPU/synthetic test file; synthetic/mock tests (`FakeAdapterPCA`, `FakeMetricRunner`, `torch.randn` calibration banks) are structurally and textually distinct from any real-GPU/real-annotation path — none of the latter exist yet in this pass. The workflow-consolidation pass adds `tests/test_workflow.py`, `tests/test_workflow_cross_venv_exclusion.py`, `tests/test_review_group_consolidation.py`, `tests/test_preview_exclusion_review.py`, `tests/test_calibrate_candidates_filter.py`, `tests/test_full_pilot_condition_exclusion.py`, and `tests/test_smoke_check.py`, all exercised the same CPU/synthetic way | — |
| T02 | `BLOCKED` | `tests/test_cli.py::RealPreviewGenerationSmokeTest` | Blocked on missing real SDXL-Turbo weights and a GPU: this test exercises `generate-psd`'s real (non-dry-run) CLI code path end-to-end, but with `runner.SDXLTurboAdapterPCA` patched to `FakeAdapterPCA` — despite its name, it is a CLI-wiring smoke test, not a real-model smoke test | A genuine SDXL-Turbo smoke test (real latent shape/VAE coordinate/injection hash/repeat-generation/τ=0-vs-scalar-reference/metric I/O check) requires real weights and a GPU, neither present in this sandbox; not a scheduling gap, so `BLOCKED` rather than `NOT_RUN` |
| T03 | PASS | `config.py::_build_model_config` (`_SUPPORTED_ADAPTERS = ("sdxl_turbo",)`; any other `model.adapter` value, including `flux2klein`, raises `ConfigError` at config-load time with a message containing `not_verified`); `tests/test_config.py::test_model_adapter_flux2klein_is_refused_as_not_verified`, `test_model_adapter_unknown_value_is_refused_as_not_verified`; `tests/test_cli.py::UnverifiedAdapterRefusalTests` (4 methods: `validate-config`, `build-basis --dry-run`, `probe --dry-run`, and a real `generate-psd` call all refuse identically, since the guard runs in `config.resolve_config` before any per-command dispatch) | The CLI explicitly refuses, with an explicit `not_verified` message, any attempt to run a PCA experiment against an adapter other than the one this package's basis/PC groups/codec were built and tested against; `README.md`/`docs/IMPLEMENTATION_DECISIONS.md` state that the basis (`d=Cp²=100`), the six PC groups, and both patch codecs are SDXL-Turbo-VAE-specific and are not transferable to FLUX (or any other adapter) without a new basis, new PC groups, and full revalidation | The guard is a config-level allow-list check (`model.adapter` string), not a runtime introspection of a loaded FLUX pipeline, since FLUX is never loaded anywhere in this package -- `compat_generation.py` still re-exports `Flux2KleinAdapter` for a future pass, but nothing in `pc_specific_psd/` instantiates or dispatches to it |

## D — Docs / evidence completeness

| ID | Status | Evidence | Result | Limitations |
| --- | --- | --- | --- | --- |
| D01 | PASS | `pc_specific_psd/README.md`, `docs/IMPLEMENTATION_DECISIONS.md`, `docs/IMPLEMENTATION_CHECKLIST.md` (this file), `docs/IMPLEMENTATION_REPORT.md` | All four required deliverables exist and cover their required contents (commands, config/manifest format, decisions/deviations, acceptance table, report) | — |
| D02 | PASS | Every `PASS` row above cites a specific test file/class or CLI command actually run this pass; every `BLOCKED` row states exactly which missing external input it is waiting on (no `NOT_RUN`/`NOT_APPLICABLE` rows exist in this table -- see "Status accounting" above); `git status --short` confirms no out-of-scope writes | Evidence is traceable per-ID rather than asserted in aggregate | — |
