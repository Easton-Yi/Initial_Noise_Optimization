# Initial-noise quality–diversity experiment

This directory is a standalone implementation of the experiment in `docs/INITIAL_NOISE_QD_EXPERIMENT_SPEC.md`; it does not import code, models, or configuration from another checkout.

The implementation was informed by the 2D `rfft2` pink-noise convention and verified FLUX.2 latent-packing behavior in the prior DivGen work, but implements these locally. It adds immutable block/noise provenance, matched white-floor methods, standalone evaluation, and block-level Q–D analysis. Model weights are cached locally in `noise_init/cache/` by default.

`alpha` is the amplitude-spectrum exponent: pink expected PSD is proportional to `(1+r)^(-2 alpha)`. All final conditions use the configured common normalization profile by default.

The radial frequency grid uses DivGen-compatible integer FFT-bin coordinates (`fftfreq(H) * H`, `rfftfreq(W) * W`), not normalized cycles per pixel. Runs made with the earlier normalized-frequency implementation are invalid for this experiment; the grid version is recorded in every new run manifest and blocks metrics/analysis on mismatched prior runs.

The formal normalization profile uses population standard deviation (`unbiased=False`) uniformly for every condition. `divgen_compat` is available only as a separately named compatibility profile, using DivGen/Notebook sample standard deviation (`unbiased=True`); never mix it with the formal profile on one curve. White endpoints bypass FFT/IFFT so `alpha=0` and same-phase `gamma=1` are exact saved-white references before shared normalization.

## Setup

Run from `Initial_Noise_Optimization/noise_init` using `requirements.txt`. FLUX is gated in many environments, so export a Hugging Face token before the first model download. HPSv3 may need its own official metric-stage environment because its published Transformer pin can differ from the Diffusers generation environment; generation artefacts remain immutable and can be scored there later.

```bash
cd Initial_Noise_Optimization/noise_init
pip install -r requirements.txt
export HF_TOKEN=...  # required if the FLUX checkpoint is gated for this account
python3 run_experiment.py --config configs/flux2_klein.yaml --stage validate
```

## Optional preflight checks

These checks are recommended in a new environment but are not part of the formal experiment. They may be skipped after the implementation, environment, and model adapter have already been validated.

```bash
# CPU acceptance tests
python3 -m unittest discover -s tests -v

# One-prompt smoke test: eight PNGs and two 1×4 grids
python3 run_experiment.py --config configs/flux2_klein.yaml --stage generate --prompt "A photo of a red fox in snow" --batch-seed 10000 --conditions white,pink:0.5 --run-id flux_smoke
python3 run_experiment.py --config configs/flux2_klein.yaml --stage metrics --run-id flux_smoke
```

Do not run `analyze` for this deliberately incomplete two-condition smoke run.

## Optional FLUX.2 Klein baseline pilot: white through pink 0.7

`configs/flux2_klein.yaml` selects the primary model, **FLUX.2 Klein 4B, BF16, 1024×1024, four inference steps**, and enables all eight matched baseline conditions:

```text
white (= alpha 0.0), pink alpha = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
```

`configs/flux2_klein.yaml` explicitly uses the checked-in two-block `manifests/blocks_pilot.jsonl`. Use this step only with a deliberately small pilot manifest to validate the baseline procedure and inspect basic behaviour.

For an optional small baseline pilot, use the explicit pilot run ID:

```bash
python3 run_experiment.py --config configs/flux2_klein.yaml --stage generate --run-id flux2_baseline_pilot
python3 run_experiment.py --config configs/flux2_klein.yaml --stage metrics --run-id flux2_baseline_pilot
python3 run_experiment.py --config configs/flux2_klein.yaml --stage analyze --run-id flux2_baseline_pilot
```

These commands run the complete white-to-pink-0.7 alpha sweep because they omit `--conditions`. The proposed-method experiment is a full alpha sweep at every intermediate gamma, not an anchor-alpha sweep. Do not run this baseline-only configuration over the full formal manifest as a required pre-step: the full experiment below already generates its matched formal baseline once.

## Formal FLUX.2 comparison: baseline + both proposed initial-noise methods

The final comparison must use one frozen formal block manifest, matched base-noise batches, generation configuration, normalization profile, and metric versions. The simplest workflow is one fresh full run; do not append proposed methods to a completed baseline-only run because its run manifest is immutable.

`configs/flux2_klein_full.yaml` is the frozen full-grid configuration: it enables the complete baseline alpha sweep and both proposed methods at every intermediate gamma. It points to the checked-in six-block `manifests/blocks_formal.jsonl` (two prompts × three seed batches). Review and, if needed, extend that manifest before validation; then keep the approved version immutable. **Do not point the full configuration at `blocks_pilot.jsonl`, and do not run the formal grid using the checked-in two-block pilot manifest.** See [manifests/README.md](manifests/README.md) for the JSONL contract.

Then run the **entire** final grid—one shared eight-point baseline curve, nine same-phase alpha-sweep curves, and nine independent-white alpha-sweep curves—over the frozen formal block manifest:

```bash
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage validate
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage generate --run-id flux2_full
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage metrics --run-id flux2_full
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage analyze --run-id flux2_full
```

For **each quality–diversity metric pair**, this produces 19 primary Q–D curves: one baseline plus one fixed-gamma alpha-sweep curve for every `(method, gamma)` pair. All curve points, frontiers, matched-diversity values, and bootstrap results are retained in machine-readable tables. Default figures are deliberately compact: HPSv3 × DreamSim is the pre-registered primary pair, while the other pairs are robustness checks. `gamma=0` is represented by the shared baseline curve. At `gamma=1`, same-phase is exactly the baseline's cached base-white `epsilon` point, while independent-white is its cached independent `eta` point: it belongs to the same white-noise distribution but finite-sample metrics need not equal the baseline white point. Both are endpoint tensor tests only. The alpha-zero same-phase point is recorded as an exact alias of the cached baseline-white gallery, so each block needs 143 unique galleries rather than 152 duplicate galleries.

Final reported baseline and proposed-method results must use the same frozen formal block manifest, base-noise batches, generation configuration, normalization profile, metric versions, and complete-block intersection. The simplest workflow obtains them all from `outputs/flux2_full/`; previously generated baseline artefacts may be reused only when their provenance and hashes match exactly.

Do not use `--conditions` for this formal run: it is only for controlled smoke subsets.

`--force` can recreate derived metric outputs but cannot replace a cached source-noise batch, generated PNG, or immutable run configuration. Use a new `--run-id` for a deliberately new experiment.

## Compact analysis outputs

`analyze` writes all statistics under `analysis/tables/`, including every curve point, per-gamma matched-diversity values, per-gamma mean matched-diversity gain, Pareto frontiers, paired effects, and bootstrap intervals. It does not default to one figure per gamma.

With all six metric pairs enabled, it writes 13 PNG figures:

- `analysis/primary/baseline_qd.png`: HPSv3 × DreamSim baseline alpha curve;
- `analysis/primary/methods_qd_two_panel.png`: baseline plus nine same-phase curves on the left and baseline plus nine independent-white curves on the right; gamma colours run light-to-dark within each panel;
- `analysis/primary/matched_diversity_gain.png`: mean quality gain over the observed matched-diversity interval, by gamma, with paired-bootstrap CIs;
- for each of the five remaining metric pairs, one two-panel Q–D robustness figure and one matched-diversity gain summary.

The detailed 25-target matched-diversity values remain in `analysis/tables/matched_diversity.csv`; the gamma summary used in the figure is in `analysis/tables/matched_diversity_summary.csv`. To request an individual baseline-versus-one-gamma plot, add an explicit item to `analysis.optional_detail_curves`, for example:

```yaml
optional_detail_curves:
  - quality: hpsv3
    diversity: dreamsim_mean_pair_distance
    curve_id: same_phase_gamma_0.5
```

This produces only that requested detail under `analysis/optional_details/`; it does not alter the formal grid or omit any reported gamma from the tables.

## Minimum formal run order

```bash
# Validate first: blocks_formal.jsonl must exist and contain approved blocks.
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage validate
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage generate --run-id flux2_full
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage metrics --run-id flux2_full
python3 run_experiment.py --config configs/flux2_klein_full.yaml --stage analyze --run-id flux2_full
```
