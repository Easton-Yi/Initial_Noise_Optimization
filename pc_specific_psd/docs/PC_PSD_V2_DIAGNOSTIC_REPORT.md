# PC-specific PSD v2 diagnostic handoff

## Outcome and evidence boundary

The v2 software path is implemented for the existing formal `OverlapCodec`
operator, fixed gate `r_s=4, beta=2`, existing B1 band, and 5%/10% final
paired-relative-L2 targets. CPU/synthetic regression evidence is recorded by
the test suite. The real basis file is absent in this worktree, so real-data
basis stability, v2 calibration, held-out noise validation, and image preview
are **NOT_RUN**. No numeric result is inferred for those stages.

The pre-existing v1 registry remains untouched. It records B1 `tau_plus=1`,
`tau_minus=-2` and gate `(4,2)`, but it has no v2 profile or candidate-level
effect diagnostics. It therefore proves neither that the targets are
reachable nor that covariance changed. Config validation will not accept that
legacy registry as a v2 result.

Repository audit at task start: root
`/home/zyw2004/comp3740_indiv_project/Initial_Noise_Optimization`, HEAD
`e85b62c7454a94f5f1ede21f481a366815a98558`. The user's modified v1 YAML,
review images/CSVs, candidate/approval/workflow JSON, and patch file were
preserved. The COCO manifest exists, but its `/workspace/datasets/...` source
images and the formal `pc_specific_psd/basis/sdxl_turbo_pca_v1_basis.pt` are
not available here; this is why no real stage was started.

## Call-chain audit

1. In v1, `select_gate_and_taus` evaluates a gate as a unit and ranks accepted
   tau values by distance of post-correction RMS from `target_rms=1`. One
   rejected tau rejects that gate. Since radial matching makes RMS nearly
   constant, this is not a meaningful dose ranking. V2 fixes the gate at
   `(4,2)`, evaluates every tau independently, and ranks each sign/target by
   `abs(paired_relative_l2_final-target)`. RMS is report-only. Exact numerical
   ties choose lower `abs(tau)`, then declaration order. Unreachable targets
   are explicit and do not enter preview.
2. The amplitude correction is fitted once from the fixed calibration noise
   bank. V2 persists that tensor in the registry; validation and generation
   load the frozen correction and never estimate it per generated sample.
   The registry is bound to the exact config file and formal basis hashes;
   changing either requires a new calibration.
3. Under `operator_clean`, white noise follows either
   `apply_psd_edit_tau_zero` (reference) or Overlap encode → fixed PC/frequency
   edit → Center synthesis → fixed radial correction (candidate). There is no
   per-sample normalization. Operator calculations use float32 inputs with
   FFT/correction arithmetic promoted internally; the adapter casts only at
   its model-injection boundary. A common downstream scheduler transformation
   is not included in the operator-layer metric and must be recorded by the
   adapter/model provenance.
4. `radial_psd` averages power over batch and channels. It uses `rfft2`,
   doubles non-DC/non-Nyquist columns for the omitted conjugate half, and uses
   integer-frequency radial annuli spanning the observed maximum radius. DC is
   included. Empty annuli divide by a clamped count and report zero; correction
   bins with either power below `1e-12` are recorded invalid and assigned a
   no-op multiplier rather than an unbounded ratio.
5. Tau zero uses the formal reference shortcut and an all-ones correction.
   Its identity/equivalence remains regression-tested and v2 calibration also
   performs a tau-zero consistency gate.

## Implemented diagnostics and safeguards

For each candidate, the registry records calibration bank identity, band,
gate, tau/sign, status/reason, pre- and post-correction aggregate paired L2,
per-sample paired L2, cosine, both RMS values, correction minimum/maximum and
`max(max(c),1/min(c))`, PSD error and valid-bin mask, transfer condition
number, covariance distance, and B1/complement Overlap-analysis coefficient
energies. Those coefficient energies are explicitly not claimed to be an
orthogonal decomposition of latent energy.

For final diagnostics, `valid_bin_mask` means reference-valid bins and is the
mask used for correction bounds and maximum PSD error. A separate
`final_power_valid_mask` exposes candidate-collapsed bins; such a collapse is
rejected without hiding its correction value from min/max statistics.

Covariance distance uses the complete 4×4 frequency response of the fixed,
linear, circular operator and compares `M M*` with proper real-FFT column
multiplicity. The reference numerical floor for this analytic calculation is
zero up to floating-point error; the configured nonzero resolution threshold
is `1e-6`. Tests demonstrate the key distinction: an orthogonal channel
rotation has nonzero paired L2 but zero covariance distance.

Correction safety is two-sided. Non-finite/non-positive valid-bin values fail,
and both excessive amplification and excessive attenuation fail the existing
bound. Split-half diagnostics now slice the leading/band columns before
principal angles, report radians and boundary `lambda_k-lambda_(k+1)` gaps,
and retain split seed plus image/sample counts. The two halves are disjoint at
the image-manifest level; sign flips and within-band rotations remain stable.

## Stage decision table

| Stage | Status | Evidence | Next step |
| --- | --- | --- | --- |
| A — effect-size selection | PASS (CPU/synthetic) | `tests/test_calibration_v2.py` | Run real calibration once the formal basis is available. |
| B — final/covariance diagnostics | PASS (CPU/synthetic) | `calibration_v2.py`, `spectral.py`, `tests/test_spectral.py` | Inspect real candidate registry; do not infer distribution change from L2 alone. |
| C — independent tau + symmetric bound | PASS (CPU/synthetic) | `tests/test_calibration_v2.py`, `tests/test_calibration.py` | Preserve the fixed bound and record independent failures. |
| D — split-half stability | PASS (code/tests), real result NOT_RUN | `basis.py`, `tests/test_basis.py` | Run `basis-stability`; interpret B1 only after its real angles/gap are available. |
| E — formal-operator preview | NOT_RUN | no real basis/model-backed v2 registry or validation artifact | Run only after v2 noise validation is PASS; maximum 60 images. |

A–C software acceptance does not imply Q–D improvement. If real validation
cannot resolve covariance/band-energy change, stop at noise-only diagnostics.
If validation passes but preview is weak, the supported conclusion is model
insensitivity to this intervention. A confirmatory full run is deliberately
blocked until the primary metric and success criterion are frozen before
viewing new final seeds.

Partial reachability is not promoted to complete success: the registry stores
`all_targets_reached=false`, the unreachable selections and their ranges, and
an explicit count. However, if tau-zero consistency passes and at least one
nonzero condition is safely selected, only those reachable conditions proceed
to held-out validation and the bounded preview. If no target is reachable, the
calibration status is `FAIL`.

The held-out validation artifact records the config, basis, and calibration
hashes, all of which are checked before preview. Its own hash is then pinned
in `run_manifest.json`, every `sample.jsonl`, and the generation-config hash.
Replacing validation under an existing run id therefore refuses resume rather
than silently mixing images validated under different evidence.

## Exact run order

Run from the repository root on the prepared GPU environment. These commands
do not rebuild the formal basis; the first creates two diagnostic half-bases
in memory and writes only its report.

```bash
.venv-generation/bin/python -m pc_specific_psd basis-stability --config pc_specific_psd/configs/sdxl_turbo_pca_v2.yaml --output pc_specific_psd/calibration/sdxl_turbo_pca_v2_basis_stability.json
.venv-generation/bin/python -m pc_specific_psd calibrate --config pc_specific_psd/configs/sdxl_turbo_pca_v2.yaml
.venv-generation/bin/python -m pc_specific_psd validate-noise --config pc_specific_psd/configs/sdxl_turbo_pca_v2.yaml
.venv-generation/bin/python -m pc_specific_psd generate-psd --config pc_specific_psd/configs/sdxl_turbo_pca_v2.yaml --stage preview --run-id sdxl_turbo_pca_v2_preview
.venv-generation/bin/python -m pc_specific_psd export-preview-review --config pc_specific_psd/configs/sdxl_turbo_pca_v2.yaml --review-output pc_specific_psd/configs/sdxl_turbo_pca_v2_preview_review.json --mapping-output pc_specific_psd/configs/sdxl_turbo_pca_v2_preview_mapping.json
.venv-metrics/bin/python -m pc_specific_psd metrics --config pc_specific_psd/configs/sdxl_turbo_pca_v2.yaml --run-id sdxl_turbo_pca_v2_preview --device cuda
```

If calibration status is `FAIL`, or validation reports `FAIL`, do not run the
remaining preview commands. Individual `TARGET_UNREACHABLE` entries are
excluded while independently selected entries may continue. Do not change
gate, band, bound, or target in response within this validation version. The
next experiment should change only the single bottleneck identified by the
frozen diagnostics. The legacy `workflow` command explicitly refuses v2; use
the ordered commands above.

## Local verification

The required command
`.venv-metrics/bin/python -m unittest discover -s pc_specific_psd/tests`
passes 542 tests. `git diff --check` also passes. Changes are intentionally
left as working-tree changes: **NOT_COMMITTED** and **NOT_PUSHED**.
