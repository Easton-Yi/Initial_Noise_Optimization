# Initial-noise quality–diversity experiments

Compare text-to-image quality and diversity under three initial-noise families:

- **Baseline:** white noise and an alpha sweep of Fourier-filtered pink noise.
- **Same-phase (A):** a white-floor filter applied to the cached base white noise.
- **Independent-white (B):** a normalized pink component mixed with cached independent white noise.

The pipeline supports SDXL Turbo and FLUX.2 Klein, saves matched noise batches and images, evaluates quality/diversity metrics, and produces Q–D plots and spectral diagnostics. Each prompt/seed block contains four samples. Conditions within a block share the cached noise components.

All commands below run from the repository's `noise_init/` directory.

## Installation

Use separate environments for generation and HPSv3 evaluation. The dependency files specify the packages for each stage; choose a CUDA-enabled PyTorch installation appropriate for your GPU.

```bash
cd Initial_Noise_Optimization/noise_init

python3 -m venv ../.venv-generation
source ../.venv-generation/bin/activate
python3 -m pip install -r requirements_generation.txt
deactivate

python3 -m venv ../.venv-metrics
source ../.venv-metrics/bin/activate
python3 -m pip install -r requirements_metrics.txt
```

Generation and metric evaluation use the device configured in the YAML, normally `cuda`. Analysis, plotting and spectral validation can run on CPU and do not load generation or scoring models. Model downloads are cached in `noise_init/cache/` by default. If a checkpoint requires Hugging Face access, accept its access terms and set `HF_TOKEN` in the environment before downloading it.

## Choose a configuration

| Configuration | Model and experiment |
| --- | --- |
| `configs/sdxl_turbo_full_finer.yaml` | SDXL Turbo, baseline and both methods on family-specific grids |
| `configs/sdxl_turbo_geneval400_1group.yaml` | SDXL Turbo, 400 GenEval prompts with one seed group each |
| `configs/flux2_klein_full.yaml` | FLUX.2 Klein, baseline and both methods on the full grid |
| `configs/flux2_klein_full_finer.yaml` | FLUX.2 Klein with finer method grids |
| `configs/flux2_klein_geneval400_1group.yaml` | FLUX.2 Klein on the GenEval400 manifest |
| `configs/sdxl_turbo.yaml` | SDXL Turbo baseline pilot |

Before starting a run, review these settings:

| Section | What to set |
| --- | --- |
| `model`, `generation` | Checkpoint, device, dtype, image size, inference steps and generation batch size |
| `blocks.manifest` | Prompt/seed JSONL file; see [block manifests](manifests/README.md) |
| `baseline` | `enabled` and `alpha_values` |
| `same_phase_floor`, `independent_white` | Each family's `enabled`, `alpha_values` and `gamma_values` |
| `quality_metrics`, `diversity_metrics` | Metrics to evaluate |
| `analysis` | Primary metric pair, bootstrap repetitions, seed and confidence level |
| `paths.outputs_root` | Run output root, normally `outputs` |

Keep custom configurations under `configs/`. Relative manifest, cache and output paths are resolved from the directory containing `configs/`. Set `--run-id` explicitly to name the output directory; otherwise the runner uses `experiment.name`.

The checked-in formal manifest contains two prompts with three seed groups each. The pilot and GenEval configurations select their own manifests. Use `--stage validate` to inspect the block and condition counts before generation.

## Run SDXL Turbo finer

Generate the complete enabled grid:

```bash
source ../.venv-generation/bin/activate

python3 run_experiment.py --config configs/sdxl_turbo_full_finer.yaml --stage validate
python3 run_experiment.py --config configs/sdxl_turbo_full_finer.yaml --stage generate --run-id sdxl_turbo_full_finer
```

Evaluate the saved images:

```bash
source ../.venv-metrics/bin/activate

python3 run_experiment.py --config configs/sdxl_turbo_full_finer.yaml --stage metrics --run-id sdxl_turbo_full_finer
```

Calculate statistics and produce the standard Q–D figures, zoom figures and spectral diagnostics:

```bash
python3 run_experiment.py --stage analyze --run-id sdxl_turbo_full_finer
python3 plot_zoomed.py --run-id sdxl_turbo_full_finer
python3 noise_validation.py --run-id sdxl_turbo_full_finer
```

Run analysis before zoom: `plot_zoomed.py` reads `analysis/tables/all_curve_points.csv` and checks its analysis metadata. These three commands can also be used on an existing complete run without regenerating images or evaluating metrics again.

For log files, create `logs/` and append `2>&1 | tee logs/<name>.log` to a command.

## Run GenEval400 or FLUX.2 Klein

For GenEval400, use the same sequence with its configuration and a separate run ID:

```bash
source ../.venv-generation/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_1group.yaml --stage validate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_1group.yaml --stage generate --run-id sdxl_turbo_geneval400

source ../.venv-metrics/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_1group.yaml --stage metrics --run-id sdxl_turbo_geneval400
python3 run_experiment.py --stage analyze --run-id sdxl_turbo_geneval400
python3 plot_zoomed.py --run-id sdxl_turbo_geneval400
python3 analyze_prompt_subset.py --run-id sdxl_turbo_geneval400 --category-field geneval_metadata.tag
python3 noise_validation.py --run-id sdxl_turbo_geneval400
```

For FLUX.2 Klein, substitute `configs/flux2_klein_full.yaml` and `--run-id flux2_full` in the generation/evaluation commands, then run:

```bash
python3 run_experiment.py --stage analyze --run-id flux2_full
python3 plot_zoomed.py --run-id flux2_full
python3 noise_validation.py --run-id flux2_full
```

Generation and metrics require `--config`. Analysis and the standalone plotting/validation scripts read the saved run configuration.

## Analyze runs stored elsewhere

Use `--run-dir` for a run outside the default `outputs/` directory:

```bash
python3 run_experiment.py --stage analyze --run-dir /path/to/run
python3 plot_zoomed.py --run-dir /path/to/run
python3 analyze_prompt_subset.py --run-dir /path/to/run
python3 noise_validation.py --run-dir /path/to/run
```

Copy the complete run directory, including its `generations/`, `noise_cache/`, `metrics/`, `run_manifest.json` and `blocks.jsonl`. The analysis loader resolves recorded image paths within the selected run while checking image identities and hashes. Missing images or incomplete enabled-metric tables stop Q–D analysis.

Existing runs use the conditions, metrics and bootstrap settings saved in `run_manifest.json`, together with the saved `blocks.jsonl`. Editing a YAML file does not change an existing run's analysis grid.

## Select which curves to display

By default, figures display the complete grid of each enabled family. A and B may have different alpha and gamma grids.

| Option | Display selection |
| --- | --- |
| `--plot-alpha-range` | Alpha filter shared by A and B |
| `--plot-gamma-range` | Gamma filter shared by A and B |
| `--plot-same-alpha-range`, `--plot-same-gamma-range` | A-specific filters |
| `--plot-independent-alpha-range`, `--plot-independent-gamma-range` | B-specific filters |
| `--plot-baseline-alpha-range` | Baseline alpha filter |

Each option accepts an inclusive `MIN:MAX` range or comma-separated values from the run's grid. Family-specific options take priority over shared options. A filter selecting no points from an enabled family produces an error listing its available grid.

For example:

```bash
python3 plot_zoomed.py --run-id sdxl_turbo_full_finer \
  --plot-same-alpha-range 0.8:0.9 --plot-same-gamma-range 0.0125:0.05 \
  --plot-independent-alpha-range 0.8:0.9 --plot-independent-gamma-range 0.75:0.9 \
  --plot-baseline-alpha-range 0.2:0.5
```

The same display options are available for `run_experiment.py --stage analyze`, `analyze_prompt_subset.py` and `noise_validation.py`. They select plotted points without reducing the conditions used for statistics or spectral validation. Standard Q–D panels use independent axis limits; zoom panels share limits; category panels share limits across categories.

Use `--quality-metric` and `--diversity-metric` to choose an enabled Q–D metric pair, such as `clip_cosine` and `lpips_alex_mean_pair_distance`. Standard analysis produces the configured primary pair and the other enabled pairs; zoom can restrict output to a requested pair.

## Baseline-only and supplementary experiments

Set each family's `enabled` field before generation. Baseline-only, single-method and experiments without baseline use the same commands. For example, copy a configuration to `configs/supplement.yaml`, set `baseline.enabled: false`, choose the method grids, and run:

```bash
source ../.venv-generation/bin/activate
python3 run_experiment.py --config configs/supplement.yaml --stage validate
python3 run_experiment.py --config configs/supplement.yaml --stage generate --run-id supplement

source ../.venv-metrics/bin/activate
python3 run_experiment.py --config configs/supplement.yaml --stage metrics --run-id supplement
python3 run_experiment.py --stage analyze --run-id supplement
python3 plot_zoomed.py --run-id supplement
python3 noise_validation.py --run-id supplement
```

Without baseline, condition statistics and method curves remain available. Baseline-relative effects, matched-diversity comparisons and nearest-baseline-alpha diagnostics are marked unavailable. A disabled family contributes no conditions.

## Category plots

`analyze_prompt_subset.py` groups prompts by `geneval_metadata.tag`, falling back to `category`. Use `--category-field` for another metadata field, or `--category-manifest categories.jsonl` to supply labels with matching prompt IDs and text.

```bash
python3 analyze_prompt_subset.py --run-id sdxl_turbo_geneval400 --category-field geneval_metadata.tag
```

Each category receives full and zoom Q–D figures, metric summaries and its prompt/block lists under `analysis/prompt_subsets/`. Simple category names become directory names; other labels receive safe directory names recorded in `categories.csv`.

If every block lacks category metadata, the script records an unavailable report. If only some blocks lack labels, use `--exclude-missing` to explicitly analyze the labelled subset. Category statistics use the same prompt weighting as overall analysis.

## Spectral validation

`noise_validation.py` reconstructs every enabled condition from cached `base_white` and `independent_eta`, using the generation functions without resampling. It processes one block at a time and checks cache coverage against the saved `blocks.jsonl`.

```bash
python3 noise_validation.py --run-id sdxl_turbo_full_finer
```

The report separates the following evidence:

| Check | Meaning |
| --- | --- |
| Cache integrity | Required blocks, tensor shapes, finite values, metadata and raw hashes agree |
| Final-noise hashes | Reconstructed tensors agree bit-for-bit with recorded generation hashes |
| Saved statistics | Sample/channel summary statistics agree within the recorded tolerance |
| A construction | Complex Fourier coefficients follow the filter relation, with phase diagnostics |
| B construction | The mixing relation includes the actual finite-sample cross term |
| PSD diagnostics | Raw and final-input spectra, band energies, cosine/coherence and component predictions |

Close summary statistics alone do not establish tensor equality. Missing generation hashes leave that evidence unavailable. The prediction without B's cross term is a finite-sample diagnostic; its difference from empirical PSD is not itself a construction failure.

`nearest_baseline_alpha` is the closest point in the run's discrete baseline grid by normalized log-PSD distance. Its heatmap is saved as `equivalent_alpha_heatmap.png`. This diagnostic does not imply equivalent noise distributions or generation results. The JSON report records the search grid, distance, boundary status, FFT/bin settings and tolerances. Spectral summaries weight sample/channel units; they do not use prompt bootstrap.

Incomplete cache coverage stops validation by default. To explicitly inspect the available blocks:

```bash
python3 noise_validation.py --run-dir /path/to/run --allow-partial
```

The report records actual coverage and omissions, and incomplete-coverage figures are marked `PARTIAL`. Corrupt cache hashes still stop validation. Explicit inconsistencies return a nonzero exit code; inspect `validation_summary.json` for individual check statuses.

## Outputs and reruns

```text
outputs/<run-id>/
├── run_manifest.json
├── blocks.jsonl
├── noise_cache/<block-id>/
├── generations/<model>/<block-id>/<condition>/
├── metrics/
└── analysis/
    ├── analysis_manifest.json
    ├── tables/
    ├── primary/
    ├── robustness/<quality>__<diversity>/
    ├── optional_details/
    ├── prompt_subsets/
    └── frequency_validation/
```

Standard Q–D figures include `baseline_qd.png`, `methods_qd_two_panel.png`, `fixed_alpha_gamma_qd_two_panel.png` and `matched_diversity_gain.png`, as applicable to the enabled families. Zoom writes `methods_qd_two_panel_zoomed.png` and `fixed_alpha_gamma_qd_two_panel_zoomed.png` alongside the standard figures. Optional detail plots are controlled by `analysis.optional_detail_curves`.

Spectral outputs include `validation_summary.json`, `noise_frequency_summary.csv`, `radial_psd.csv`, `radial_psd_selected.png`, `theoretical_vs_empirical_psd.png`, `equivalent_alpha_heatmap.png` and `coherence_heatmap.png`.

**Rerunning analysis or plotting overwrites the corresponding tables, metadata and same-named figures.** These commands leave source images, metrics, noise caches and the saved run configuration unchanged. Save a separate copy of any derived output you want to keep before rerunning.

Generation can resume with the same configuration and run ID. Use a new run ID when changing the experiment configuration or block manifest. `--force` can recreate derived metric outputs; it does not replace cached source noise, generated PNGs or the saved run configuration. Use `--conditions` only for deliberately restricted smoke runs; complete Q–D analysis requires the full enabled grid.

## Statistical and noise definitions

For each condition and metric, analysis averages a prompt's seed groups and then gives every prompt equal weight. Paired bootstrap draws resample prompts together across conditions. Matched-diversity comparisons recompute curves, Pareto frontiers and interpolation within each draw, without extrapolation. The saved analysis metadata records the statistical unit, sample counts, bootstrap settings and display selection.

`alpha` is the amplitude-spectrum exponent: expected pink PSD is proportional to `(1+r)^(-2 alpha)`, with `r` measured in integer FFT-bin coordinates. Final noise uses the configured normalization profile. At intermediate gamma, B normalizes its pink component before mixing it with independent white noise and then applies final normalization. The standard profile uses per-sample, per-channel zero mean and population unit standard deviation.

## Tests and sample figures

Run the CPU tests or render small synthetic examples without downloading models:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -v
MPLBACKEND=Agg OMP_NUM_THREADS=1 python3 tests/render_formal_examples.py --output /tmp/noise-figure-examples
```

The examples directory must not already exist. It contains synthetic finer, GenEval, single-method, no-baseline and baseline-only runs; `examples.json` lists their figure directories. The synthetic scores exercise the analysis and plotting pipeline and are not experimental model results.
