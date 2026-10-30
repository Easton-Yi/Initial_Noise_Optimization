# Stage 1: White-Floor Pink Noise Initialisation Experiments

This directory implements and evaluates the first stage of FloorRefine on its own: the construction of white, pink and **white-floor pink** initial noise, image generation with SDXL-Turbo and FLUX.2 [klein] 4B, quality and diversity metrics, prompt-level statistics, quality–diversity (Q–D) plots, generation grids, and spectral validation of the noise tensors themselves.

← [Back to the project overview](../README.md) · Next stage: [FloorRefine refinement and DivGen comparisons](../denoising_optimization/README.md)

## Contents

- [What this directory does](#what-this-directory-does)
- [Implementation map](#implementation-map)
- [Environments](#environments)
- [Configuration](#configuration)
- [Experiment configurations and sizes](#experiment-configurations-and-sizes)
- [Full pipeline: SDXL-Turbo, GenEval400](#full-pipeline-sdxl-turbo-geneval400)
- [Full pipeline: FLUX.2 [klein] 4B, GenEval400](#full-pipeline-flux2-klein-4b-geneval400)
- [Smoke runs, pilots and supplementary grids](#smoke-runs-pilots-and-supplementary-grids)
- [Resume, skip, overwrite and partial results](#resume-skip-overwrite-and-partial-results)
- [Metrics](#metrics)
- [Statistical analysis](#statistical-analysis)
- [Standard and zoomed Q–D plots](#standard-and-zoomed-qd-plots)
- [Category analysis](#category-analysis)
- [Generation grids](#generation-grids)
- [Noise validation](#noise-validation)
- [Coupled-sampling comparison baselines](#coupled-sampling-comparison-baselines)
- [Output layout](#output-layout)
- [Troubleshooting](#troubleshooting)
- [Tests and synthetic examples](#tests-and-synthetic-examples)

## What this directory does

| Capability | Description |
| --- | --- |
| Noise families | White and pink baselines (an α sweep), same-phase white-floor pink noise and independent-white white-floor pink noise (α × γ grids). The constructions are defined in the [project overview](../README.md#stage-1-white-floor-pink-noise-initialisation). |
| Models | SDXL-Turbo (512 × 512, 1 denoising step) and FLUX.2 [klein] 4B (1024 × 1024, 4 denoising steps). |
| Experiments | Small pilot and parameter-exploration runs on two development prompts, and fixed-parameter runs on 400 GenEval prompts (GenEval400). |
| Pipeline | `validate` → `generate` → `metrics` → `analyze`, followed by zoomed plots, category plots, generation grids and noise validation. |
| Provenance | Every run freezes its resolved configuration and block manifest, caches the base noise tensors, records tensor and image hashes, and refuses to mix results from different configurations. |

## Implementation map

All entry points are run from the `noise_init/` directory.

| Function | Entry point or module | Input | Output |
| --- | --- | --- | --- |
| Experiment runner (validate, generate, metrics, analyze, all) | [`run_experiment.py`](run_experiment.py) | YAML config, block manifest | `outputs/<run-id>/` |
| Noise construction | [`noise_methods.py`](noise_methods.py) (`pink`, `same_phase_floor`, `independent_white`, `normalize`, `construct_noise`) | Cached base and η tensors | Initial-noise tensors |
| Model adapters | [`model_adapters.py`](model_adapters.py) (`SDXLTurboAdapter`, `Flux2KleinAdapter`) | Initial noise, prompt | PIL images; verifies that the supplied noise reached the pipeline's `prepare_latents` |
| Metric runner | [`metric_runner.py`](metric_runner.py) (`evaluate_run`) | Generated images | `metrics/per_image.csv`, `per_pair.csv`, `per_group.csv`, `metric_manifest.json` |
| Statistical analysis and standard plots | [`analysis.py`](analysis.py) (called by `run_experiment.py --stage analyze`) | Complete metrics | `analysis/tables/`, `analysis/primary/`, `analysis/robustness/` |
| Zoomed plots | [`plot_zoomed.py`](plot_zoomed.py) | `analysis/tables/all_curve_points.csv` | `*_zoomed.png`, `analysis/zoom_manifest.json` |
| Category analysis | [`analyze_prompt_subset.py`](analyze_prompt_subset.py) | Complete metrics, prompt categories | `analysis/prompt_subsets/` |
| Generation comparison grid | [`make_comparison_grid.py`](make_comparison_grid.py) | Saved `grid_1x4.png` files | `analysis/generation_grids/*.png` |
| Noise validation | [`noise_validation.py`](noise_validation.py) | `noise_cache/`, generation records | `analysis/frequency_validation/` |
| Configurations | [`configs/*.yaml`](configs/) | — | — |
| Block manifests | [`manifests/*.jsonl`](manifests/) ([description](manifests/README.md)) | — | — |
| Shared run support | [`run_support.py`](run_support.py) (condition grid, run resolution, display selection), [`io_utils.py`](io_utils.py) (hashing, immutable records) | — | — |
| Result integrity | [`result_integrity.py`](result_integrity.py) | Generation records, metric tables | Raises on missing, duplicated or changed results |
| Coupled-sampling baselines | [`coupled_sampling.py`](coupled_sampling.py), [`coupled_run.py`](coupled_run.py), [`compare_initializations.py`](compare_initializations.py) | `*_coupled.yaml` configs | See [Coupled-sampling comparison baselines](#coupled-sampling-comparison-baselines) |

[`analyze_prompt_default-subset.py`](analyze_prompt_default-subset.py) is an alias that calls `analyze_prompt_subset.py`.

## Environments

Generation and metric evaluation use **two separate Python environments**. The official HPSv3 package (`hpsv3==1.0.0`) requires `transformers==4.45.2`, while the FLUX.2 [klein] pipeline in `diffusers==0.37.1` needs a newer `transformers`. Keeping the stages separate lets both pins hold.

| Environment | Requirements file | Used for |
| --- | --- | --- |
| `.venv-generation` (project root) | [`requirements_generation.txt`](requirements_generation.txt) | `validate`, `generate` |
| `.venv-metrics` (project root) | [`requirements_metrics.txt`](requirements_metrics.txt) | `metrics`, `analyze`, plotting, grids, noise validation, tests |

Both files pin Python 3.12-compatible packages including `torch==2.11.0`, `torchvision==0.26.0`, `diffusers==0.37.1`, `lpips==0.1.4`, `dreamsim==0.2.1` and `vendi-score`. The generation file pins `transformers==4.56.2`; the metrics file pins `transformers==4.45.2` and `hpsv3==1.0.0`. Choose the CUDA build of PyTorch that matches your GPU and driver.

```bash
# From the project root (the directory containing noise_init/)
cd /path/to/Initial_Noise_Optimization

python3.12 -m venv .venv-generation
.venv-generation/bin/python -m pip install --upgrade pip
.venv-generation/bin/python -m pip install -r noise_init/requirements_generation.txt

python3.12 -m venv .venv-metrics
.venv-metrics/bin/python -m pip install --upgrade pip
.venv-metrics/bin/python -m pip install -r noise_init/requirements_metrics.txt
```

To install a specific CUDA wheel of PyTorch first, run for example `.venv-metrics/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128` before installing the requirements file.

**Hardware.** Generation and metrics run on the device set in the YAML (`model.device`, normally `cuda`). The metric stage loads one evaluation model at a time and releases it before loading the next. Analysis, zoomed plots, category analysis, comparison grids and noise validation run on CPU and load no neural networks.

**Model downloads and cache.** Generator checkpoints are downloaded from Hugging Face into `model.cache_dir`, which the checked-in configs set to `cache` (resolved to `noise_init/cache/`, relative to the directory that contains `configs/`). Metric models (CLIP, HPSv3, DreamSim, LPIPS) use their libraries' default caches. If a checkpoint requires accepting access terms on Hugging Face, accept them and export `HF_TOKEN` before the first generation run.

## Configuration

Each YAML file in `configs/` fully specifies one experiment. Relative paths inside a config (manifest, cache, outputs) are resolved against the parent of `configs/`, i.e. `noise_init/`, regardless of the shell's working directory.

| Section | Fields | Effect |
| --- | --- | --- |
| `experiment` | `name` | Default run ID when `--run-id` is omitted |
| | `master_seed` | Seed used to derive the independent η draws |
| | `gallery_size` | Must be 4 (images per group) |
| | `normalization_profile` | `per_sample_per_channel_zero_mean_unit_std` for all white-floor experiments |
| `model` | `adapter` | `sdxl_turbo` or `flux2_klein` |
| | `checkpoint`, `vae_checkpoint`, `revision` | Hugging Face model IDs (SDXL-Turbo uses `madebyollin/sdxl-vae-fp16-fix` as its VAE) |
| | `dtype`, `device`, `cpu_offload`, `cache_dir` | Precision (`float16` for SDXL-Turbo, `bfloat16` for FLUX), device, optional CPU offload, model cache |
| `generation` | `height`, `width`, `num_inference_steps`, `guidance_scale` | Resolution, denoising steps and guidance (SDXL-Turbo: 512, 1, 0.0; FLUX: 1024, 4, 1.0) |
| | `generation_batch_size`, `release_model_after_generation`, `output_format` | Images are generated one at a time; the model is released after generation; PNG output |
| `blocks` | `manifest` | Prompt and seed-group manifest (JSONL) |
| `baseline` | `enabled`, `alpha_values` | White (α = 0) and pink conditions |
| `same_phase_floor` | `enabled`, `alpha_values`, `gamma_values` | Same-phase white-floor pink grid |
| `independent_white` | `enabled`, `alpha_values`, `gamma_values` | Independent-white white-floor pink grid |
| `quality_metrics` | `clip`, `hpsv3` (`maniqa` is present but disabled) | Per-image quality and alignment metrics |
| `diversity_metrics` | `dreamsim`, `lpips` (`backbone: alex`), `vendi_clip` | Group diversity metrics; `vendi_clip` must use the same CLIP checkpoint as `clip` |
| `analysis` | `primary_metric_pair`, `bootstrap_replicates`, `bootstrap_seed`, `confidence_level`, `optional_detail_curves` | Primary Q–D pair (`hpsv3` × `dreamsim_mean_pair_distance` in the formal configs) and bootstrap settings (2000 replicates, 95 %) |
| `paths` | `outputs_root` | Root for run directories, normally `outputs` |

A condition is one (family, α, γ) point. Each enabled family contributes the Cartesian product of its α and γ lists; the baseline contributes one condition per α. Formal grids exclude the γ = 0 and γ = 1 endpoints (they are covered by unit tests). The same-phase condition with α = 0 is identical to the white baseline, so it is recorded as an alias of the white images instead of being generated again.

**Block manifests.** Each line of a manifest is one *block* (one prompt × one seed group of four images) with `block_id`, `prompt_id`, `prompt`, `seed_batch_id` and `batch_seed`. Image $i$ of the group uses base white noise seeded with `batch_seed + i`; the independent η draws are derived from `master_seed`, `block_id` and $i$. The GenEval400 manifest additionally records the GenEval category, source row and commit, and its selection rule is described in the [project overview](../README.md#prompts-and-comparison-units).

## Experiment configurations and sizes

Counts below are the expanded grids reported by `--stage validate`. Images = blocks × conditions × 4; the same-phase α = 0 alias (where present) is listed as a condition but reuses the white images.

| Config | Model, steps | Blocks | Baseline α | Same-phase α × γ | Independent-white α × γ | Conditions | Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`sdxl_turbo_geneval400_1group.yaml`](configs/sdxl_turbo_geneval400_1group.yaml) | SDXL-Turbo, 1 | 400 | 0–0.5 (6) | {0.8, 0.9} × {0.025, 0.038, 0.05} | {0.8, 0.9} × {0.75, 0.825, 0.9} | 18 | 28,800 |
| [`flux2_klein_geneval400_1group.yaml`](configs/flux2_klein_geneval400_1group.yaml) | FLUX.2 [klein], 4 | 400 | 0–0.5 (6) | {0.8, 0.9} × {0.025, 0.038, 0.05} | {0.8, 0.9} × {0.75, 0.825, 0.9} | 18 | 28,800 |
| [`sdxl_turbo_full_finer.yaml`](configs/sdxl_turbo_full_finer.yaml) | SDXL-Turbo, 1 | 6 | 0–0.7 (8) | {0.6, …, 0.9} × {0.0125, 0.025, 0.05, 0.075, 0.1} | {0.6, …, 0.9} × {0.25, 0.5, 0.75, 0.9, 0.925} | 48 | 1,152 |
| [`flux2_klein_full_finer.yaml`](configs/flux2_klein_full_finer.yaml) | FLUX.2 [klein], 4 | 6 | 0–0.7 (8) | {0.8, 0.9} × {0.0125, 0.025, 0.05} | {0.8, 0.9} × {0.5, 0.75, 0.9} | 20 | 480 |
| [`sdxl_turbo_full.yaml`](configs/sdxl_turbo_full.yaml), [`flux2_klein_full.yaml`](configs/flux2_klein_full.yaml) | coarse α 0–0.7 × γ 0.1–0.9 exploration | 6 | 0–0.7 (8) | 8 × 9 | 8 × 9 | 152 | 3,648 |
| [`sdxl_turbo_full_finerplus.yaml`](configs/sdxl_turbo_full_finerplus.yaml) | SDXL-Turbo, 1; high-α exploration | 6 | 0–0.7 (8) | {0.9, …, 1.5} × {0.025, …, 0.125} | same | 58 | 1,392 |
| [`sdxl_turbo_pilot_finerplus.yaml`](configs/sdxl_turbo_pilot_finerplus.yaml) | as above, pilot blocks | 2 | 0–0.7 (8) | 5 × 5 | 5 × 5 | 58 | 464 |
| [`sdxl_turbo.yaml`](configs/sdxl_turbo.yaml) | SDXL-Turbo baseline-only pilot | 2 | 0–0.7 (8) | disabled | disabled | 8 | 64 |
| [`flux2_klein.yaml`](configs/flux2_klein.yaml) | FLUX.2 [klein] pilot | 2 | 0–0.7 (8) | {0.6, …, 0.9} × 5 γ | {0.6, …, 0.9} × 5 γ | 48 | 384 |
| `*_geneval400_coupled.yaml` | Coupled-sampling baselines | 400 | — | — | — | 2 | 3,200 |

The `*_full*` configs use [`manifests/blocks_formal.jsonl`](manifests/blocks_formal.jsonl) (two development prompts × three seed groups); the pilots use [`manifests/blocks_pilot.jsonl`](manifests/blocks_pilot.jsonl) (two prompts × one seed group); the GenEval configs use [`manifests/blocks_geneval400_1group.jsonl`](manifests/blocks_geneval400_1group.jsonl).

The fixed-parameter GenEval400 evaluation is the primary Stage 1 result; the six-block and pilot configs were used to choose its α and γ ranges. A run's actual grid is the one frozen in its own `run_manifest.json`; read that file rather than inferring settings from a config or directory name.

## Full pipeline: SDXL-Turbo, GenEval400

Run every command from `noise_init/`. The run ID `sdxl_turbo_geneval400` is an example; any new name works.

**1. Validate the configuration** (generation environment; reads the YAML and manifest only):

```bash
cd /path/to/Initial_Noise_Optimization/noise_init
source ../.venv-generation/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_1group.yaml --stage validate
# Expected: valid: 400 blocks, 18 conditions
```

**2. Generate** (generation environment, GPU). Creates `outputs/sdxl_turbo_geneval400/` with the frozen config, the block manifest, the noise cache and the images:

```bash
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_1group.yaml \
  --stage generate --run-id sdxl_turbo_geneval400
deactivate
```

**3. Compute metrics** (metrics environment, GPU). Requires the generated images; writes `outputs/sdxl_turbo_geneval400/metrics/`:

```bash
source ../.venv-metrics/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_1group.yaml \
  --stage metrics --run-id sdxl_turbo_geneval400
```

**4. Analyse** (metrics environment, CPU). Requires complete images and metrics; writes `analysis/tables/`, `analysis/primary/` and `analysis/robustness/`. `--config` is optional here: the analysis reads the run's frozen configuration.

```bash
python3 run_experiment.py --stage analyze --run-id sdxl_turbo_geneval400
```

**5. Zoomed Q–D plots** (CPU). Requires step 4:

```bash
python3 plot_zoomed.py --run-id sdxl_turbo_geneval400
```

**6. Category analysis** (CPU). Requires complete metrics:

```bash
python3 analyze_prompt_subset.py --run-id sdxl_turbo_geneval400 --category-field geneval_metadata.tag
```

**7. Generation comparison grid** for one prompt (CPU). Requires the generated `grid_1x4.png` files:

```bash
python3 make_comparison_grid.py --config configs/sdxl_turbo_geneval400_1group.yaml \
  --run-id sdxl_turbo_geneval400 --prompt-id geneval_0000 --alpha 0.9
```

**8. Noise validation** (CPU). Requires the noise cache and generation records; independent of metrics:

```bash
python3 noise_validation.py --run-id sdxl_turbo_geneval400
deactivate
```

## Full pipeline: FLUX.2 [klein] 4B, GenEval400

The FLUX pipeline uses the same stages with its own config and run ID. It generates 1024 × 1024 images with 4 denoising steps in `bfloat16`; check that `black-forest-labs/FLUX.2-klein-4B` is accessible with your Hugging Face account before starting.

**1. Validate:**

```bash
cd /path/to/Initial_Noise_Optimization/noise_init
source ../.venv-generation/bin/activate
python3 run_experiment.py --config configs/flux2_klein_geneval400_1group.yaml --stage validate
# Expected: valid: 400 blocks, 18 conditions
```

**2. Generate:**

```bash
python3 run_experiment.py --config configs/flux2_klein_geneval400_1group.yaml \
  --stage generate --run-id flux2_klein_geneval400
deactivate
```

**3. Compute metrics:**

```bash
source ../.venv-metrics/bin/activate
python3 run_experiment.py --config configs/flux2_klein_geneval400_1group.yaml \
  --stage metrics --run-id flux2_klein_geneval400
```

**4. Analyse:**

```bash
python3 run_experiment.py --stage analyze --run-id flux2_klein_geneval400
```

**5. Zoomed Q–D plots:**

```bash
python3 plot_zoomed.py --run-id flux2_klein_geneval400
```

**6. Category analysis:**

```bash
python3 analyze_prompt_subset.py --run-id flux2_klein_geneval400 --category-field geneval_metadata.tag
```

**7. Generation comparison grid:**

```bash
python3 make_comparison_grid.py --config configs/flux2_klein_geneval400_1group.yaml \
  --run-id flux2_klein_geneval400 --prompt-id geneval_0000 --alpha 0.9
```

**8. Noise validation:**

```bash
python3 noise_validation.py --run-id flux2_klein_geneval400
deactivate
```

To keep logs, create `logs/` and append `2>&1 | tee logs/<name>.log` to any command.

## Smoke runs, pilots and supplementary grids

**Single-prompt smoke run.** `--prompt` with `--batch-seed` replaces the manifest with one block (`block_id = smoke_<batch-seed>`). `--conditions` restricts generation to selected enabled conditions, written as `white`, `pink:<alpha>`, `same:<alpha>:<gamma>` or `independent:<alpha>:<gamma>`:

```bash
source ../.venv-generation/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_1group.yaml \
  --stage generate --run-id smoke_sdxl \
  --prompt "a photo of a bench" --batch-seed 4100000 \
  --conditions white,pink:0.3,same:0.9:0.05,independent:0.9:0.9
```

A restricted smoke run can be scored with `--stage metrics` to check the evaluation environment. Q–D analysis requires the complete frozen grid, so it rejects runs generated with `--conditions`.

**Development pilots and six-block grids.** Use the same command sequence as above with a pilot or six-block config, for example `configs/sdxl_turbo_full_finer.yaml` with `--run-id sdxl_turbo_full_finer`, or `configs/flux2_klein_full_finer.yaml` with `--run-id flux2_klein_full_finer`. These manifests have several seed groups per prompt, so `make_comparison_grid.py` needs `--gamma` (see [Generation grids](#generation-grids)).

**Supplementary grids and single-family runs.** Copy a config, set each family's `enabled` flag and grids, and run it under a new run ID. For example, to add same-phase points at γ ∈ {0.075, 0.1} without regenerating the baseline:

```bash
cp configs/sdxl_turbo_geneval400_1group.yaml configs/sdxl_turbo_geneval400_supp.yaml
# Edit the copy: experiment.name: sdxl_turbo_geneval400_supp
#                baseline.enabled: false
#                same_phase_floor.gamma_values: [0.075, 0.1]
#                independent_white.gamma_values: [0.92]
source ../.venv-generation/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_supp.yaml --stage validate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_supp.yaml --stage generate --run-id sdxl_turbo_geneval400_supp
deactivate
source ../.venv-metrics/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_supp.yaml --stage metrics --run-id sdxl_turbo_geneval400_supp
python3 run_experiment.py --stage analyze --run-id sdxl_turbo_geneval400_supp
```

Because the block manifest and `master_seed` are unchanged, the supplementary run uses the same base noise as the main run. Without the baseline, condition-level statistics and method curves are still produced, while baseline-relative effects, matched-diversity comparisons and nearest-baseline-α diagnostics are marked unavailable.

**The `all` stage.** `--stage all` runs generate, metrics and analyze in one process. It is suitable only when one environment can both generate and score, for example a smoke test with HPSv3 disabled. For the formal two-environment workflow, run the stages separately as shown above.

## Resume, skip, overwrite and partial results

| Situation | Behaviour |
| --- | --- |
| Rerunning `generate` with the same config and run ID | Completed groups (all four PNGs present and the recorded noise hashes match) are skipped; missing groups are generated. |
| A group directory exists but is incomplete | Generation stops instead of overwriting source images. Move the incomplete directory aside or use a new run ID. |
| Changing the config or manifest for an existing run ID | Refused (`Run manifest differs … choose a different --run-id`). Every scientific setting is frozen in `run_manifest.json`; use a new run ID. |
| `--force` | Recomputes derived metric outputs from scratch. It never replaces cached noise, generated PNGs or the frozen run configuration. |
| Rerunning `metrics` | Scores already recorded for the same image hash, prompt and metric configuration are reused; only missing scores are computed. `per_image.csv` is checkpointed every 100 images. |
| Incomplete runs and `analyze` | Analysis requires every frozen block × condition group and every enabled metric. Missing images or metric rows stop analysis with a list of examples. |
| Rerunning analysis, zoom, category analysis or validation | Overwrites the corresponding tables, manifests and same-named figures. Source images, metrics, noise caches and the frozen configuration are not touched. Copy any derived output you want to keep before rerunning. |
| Editing a YAML after a run | Has no effect on that run's analysis, which always uses the frozen `run_manifest.json` and `blocks.jsonl`. |

## Metrics

`--stage metrics` evaluates every generated image and group with the metrics enabled in the frozen config. It runs one model at a time over the complete run and releases it before loading the next.

| Metric (column value of `metric`) | Unit of computation | Group value | Direction |
| --- | --- | --- | --- |
| `clip_cosine` | One image and its prompt: cosine similarity of L2-normalised `openai/clip-vit-large-patch14` image and text embeddings | Mean of the four images (`n = 4`, with `std`, `minimum`, `maximum`) | Higher means closer text–image agreement |
| `hpsv3` | One image and its prompt: official HPSv3 reward mean μ | Mean of the four images (`n = 4`) | Higher means higher predicted human preference |
| `dreamsim_mean_pair_distance` | One image pair: DreamSim distance | Mean over the six unordered pairs (`n = 6`) | Higher means more perceptual diversity |
| `lpips_alex_mean_pair_distance` | One image pair: LPIPS with AlexNet, images resized to 256 × 256 and scaled to [−1, 1] | Mean over the six pairs (`n = 6`) | Higher means more perceptual diversity |
| `vendi_clip` | Whole group: Vendi score of the 4 × 4 cosine kernel of normalised CLIP image embeddings | One value per group (`n = 4`) | Higher means more feature-space diversity (range 1–4) |

Output tables in `metrics/`:

- `per_image.csv` — one row per image and quality metric (`block_id`, `condition_id`, `base_index`, `metric`, `score`, plus image and prompt hashes).
- `per_pair.csv` — one row per image pair and pairwise diversity metric (`left_base_index`, `right_base_index`).
- `per_group.csv` — one row per group and metric (`score`, `n`).
- `metric_manifest.json` — package versions, the metric configuration hash and `complete: true`.

All diversity metrics here measure **within-group pairwise diversity**: how different the four images of one prompt and condition are from each other. Stage 1 does not compute changes between matched images of two conditions; same-seed paired changes are used in the refinement experiments.

## Statistical analysis

`run_experiment.py --stage analyze` (implemented in [`analysis.py`](analysis.py)) first checks that images and metrics are complete and unchanged, then computes:

| Output table (`analysis/tables/`) | Content |
| --- | --- |
| `all_curve_points.csv` | For every condition and every quality × diversity metric pair: prompt-level mean quality and diversity, bootstrap standard errors and 95 % percentile intervals, prompt and block counts |
| `paired_effects.csv` | Per-prompt differences between each condition and its reference: white for baseline conditions, and the baseline pink condition with the same α for white-floor conditions |
| `bootstrap_results.csv` | Bootstrap mean, standard error and interval of the quality difference for each paired effect; `unavailable` when the reference condition is not in the run |
| `qd_frontiers.csv` | Pareto-optimal points (both quality and diversity maximised) of every curve |
| `matched_diversity.csv`, `matched_diversity_summary.csv` | Quality difference between each white-floor fixed-γ curve and the baseline α curve at 25 matched diversity values in their shared diversity range, with paired bootstrap intervals |

Statistical rules:

- **Prompt-equal aggregation.** Group scores are first averaged over the seed groups of a prompt, then every prompt receives equal weight.
- **Paired bootstrap.** Prompts are resampled with replacement (2000 replicates, seed from the config), and the same resampled prompts are used for all conditions in a draw.
- **Curves.** The baseline forms one curve over α. Each white-floor family forms one curve per fixed γ, ordered by α.
- **Matched diversity.** Within each bootstrap draw, the baseline and method curves are recomputed, reduced to their Pareto frontiers, and linearly interpolated at the target diversity values. No extrapolation is performed, and comparisons outside the shared diversity range are reported as unavailable.
- **Completeness.** Analysis refuses runs with missing groups or metric rows; there is no partial-coverage mode for Q–D statistics. Noise validation has an explicit partial mode (see [Noise validation](#noise-validation)).

`analysis/analysis_manifest.json` records the aggregation and bootstrap settings, the input hashes and the display selection.

## Standard and zoomed Q–D plots

### Standard figures

Produced by `--stage analyze` for the primary metric pair in `analysis/primary/`, and for every other enabled pair in `analysis/robustness/<quality>__<diversity>/`:

| Figure | Content |
| --- | --- |
| `baseline_qd.png` | Baseline curve over α (diversity on the x-axis, quality on the y-axis), α-labelled |
| `methods_qd_two_panel.png` | One panel per white-floor family: the baseline curve and one curve per fixed γ (points ordered by α), with an inset zoomed on the white-floor points |
| `fixed_alpha_gamma_qd_two_panel.png` | Diagnostic view: one curve per fixed α (points ordered by γ), with baseline α points as references |
| `matched_diversity_gain.png` | Mean quality difference at matched diversity for each γ, with bootstrap intervals |

**Fixed-γ versus fixed-α curves.** A fixed-γ curve shows how a white-floor family moves through the Q–D plane as α increases at constant floor level; it is directly comparable to the baseline α curve. A fixed-α curve shows the effect of raising γ at constant pink exponent, and is a diagnostic of the floor's effect.

Use `--quality-metric` and `--diversity-metric` (values: `hpsv3`, `clip_cosine`; `dreamsim_mean_pair_distance`, `lpips_alex_mean_pair_distance`, `vendi_clip`) to change which enabled pair is treated as primary in the figures. Optional single-curve figures are controlled by `analysis.optional_detail_curves` in the config.

### Zoomed figures

[`plot_zoomed.py`](plot_zoomed.py) redraws the white-floor Q–D figures with shared axes fitted to the plotted points. It reads the existing `analysis/tables/all_curve_points.csv`, checks that the table matches the current inputs and analysis settings, and **does not recompute any statistics**. It writes, alongside the standard figures:

- `methods_qd_two_panel_zoomed.png` (fixed-γ curves with baseline α points)
- `fixed_alpha_gamma_qd_two_panel_zoomed.png` (fixed-α curves with γ labels)
- `analysis/zoom_manifest.json`

By default it writes figures for the primary pair and every other enabled pair; passing `--quality-metric` or `--diversity-metric` restricts it to that pair.

### Display selection

All plotting commands (`run_experiment.py --stage analyze`, `plot_zoomed.py`, `analyze_prompt_subset.py`, `noise_validation.py`) accept display filters. They select which existing grid points are drawn; they never change the conditions used for statistics or validation.

| Option | Applies to |
| --- | --- |
| `--plot-alpha-range`, `--plot-gamma-range` | Both white-floor families |
| `--plot-same-alpha-range`, `--plot-same-gamma-range` | Same-phase only (overrides the shared option) |
| `--plot-independent-alpha-range`, `--plot-independent-gamma-range` | Independent-white only (overrides the shared option) |
| `--plot-baseline-alpha-range` | Baseline α points |

Each option takes an inclusive `MIN:MAX` range or a comma-separated list of values from the run's grid. A filter that selects no points from an enabled family stops with the available grid listed.

Example: zoom on the higher half of each family's γ range and on mid-range baseline points, for the CLIP × LPIPS pair only:

```bash
python3 plot_zoomed.py --run-id sdxl_turbo_geneval400 \
  --plot-same-alpha-range 0.8:0.9 --plot-same-gamma-range 0.038,0.05 \
  --plot-independent-alpha-range 0.9 --plot-independent-gamma-range 0.825:0.9 \
  --plot-baseline-alpha-range 0.2:0.5 \
  --quality-metric clip_cosine --diversity-metric lpips_alex_mean_pair_distance
```

## Category analysis

[`analyze_prompt_subset.py`](analyze_prompt_subset.py) repeats the prompt-level analysis within each prompt category. By default it reads `geneval_metadata.tag`, falling back to `category`; `--category-field` selects another (dotted) metadata field, and `--category-manifest file.jsonl` supplies labels by `block_id` or `prompt_id` (with matching prompt text).

```bash
python3 analyze_prompt_subset.py --run-id sdxl_turbo_geneval400 --category-field geneval_metadata.tag
```

For each category it writes, under `analysis/prompt_subsets/<category>/`: the analysis tables, `metric_summary.csv`, `curve_points.csv`, the prompt and block lists, and full and zoomed fixed-α and fixed-γ figures that share axes across all categories. `categories.csv` maps categories to directory names and `analysis_info.json` records the settings. If no block has category metadata the report is marked unavailable; if only some blocks lack labels, `--exclude-missing` analyses the labelled subset explicitly.

## Generation grids

**Per-group grid.** During generation, every group directory receives `grid_1x4.png`: the four images of that block and condition in base-index order (left to right, seeds `batch_seed + 0 … 3`).

**Cross-condition comparison grid.** [`make_comparison_grid.py`](make_comparison_grid.py) assembles saved `grid_1x4.png` files into one labelled figure for one prompt. It loads no model and samples no noise.

- **Left panel:** the baseline α sweep for one seed group, one row per α (default `--baseline-alphas 0.0,0.1,0.2,0.3,0.4,0.5`, white first), from the first seed group of the prompt unless `--baseline-seed-batch` is given.
- **Right panel, one seed group per prompt (GenEval runs):** for the chosen `--alpha`, one row per γ of the same-phase grid followed by one row per γ of the independent-white grid. The γ lists default to the run's frozen grids and can be overridden with `--same-phase-gammas` and `--independent-gammas`.
- **Right panel, several seed groups per prompt (development runs):** one row per seed group for each family at one γ. Pass `--gamma`, or `--same-phase-gamma` and `--independent-gamma` explicitly. When the two families have disjoint γ grids of equal length, `--gamma` is matched to the other family's value at the same position in its grid.
- Every row keeps all four images of the group. Row labels give the family and α or γ (and the seed for multi-seed runs); the title is the prompt text and the footer is the model ID.

```bash
# One seed group per prompt (GenEval)
python3 make_comparison_grid.py --config configs/sdxl_turbo_geneval400_1group.yaml \
  --run-id sdxl_turbo_geneval400 --prompt-id geneval_0000 --alpha 0.9

# Several seed groups per prompt (development grid), explicit per-family gamma
python3 make_comparison_grid.py --config configs/sdxl_turbo_full_finer.yaml \
  --run-id sdxl_turbo_full_finer --prompt-id p001 --alpha 0.9 \
  --same-phase-gamma 0.05 --independent-gamma 0.9 --output /tmp/p001_alpha0.9.png
```

The default output is `outputs/<run-id>/analysis/generation_grids/<prompt-id>_alpha<α>.png` for GenEval runs and `<prompt-id>_alpha<α>_gamma<γ>.png` for multi-seed runs (written over any existing file of the same name); `--output` selects another path. If a required `grid_1x4.png` does not exist, the script stops and names the missing file; generate that condition first.

## Noise validation

[`noise_validation.py`](noise_validation.py) checks the **initial-noise tensors**, not the generated images. It reconstructs every enabled condition from the cached `base_white.pt` and `independent_eta.pt` of each block, using the same construction functions as generation, without resampling. It runs on CPU, processes one block at a time, needs the noise cache and the generation records, and does not need metrics.

```bash
python3 noise_validation.py --run-id sdxl_turbo_geneval400
```

The report separates three kinds of evidence:

**1. Numerical correctness and provenance**

| Check (`validation_summary.json` → `checks`) | What is verified |
| --- | --- |
| `construction_version` | The run uses the supported construction and integer FFT-bin frequency grid |
| `cache_coverage` | Every block in `blocks.jsonl` has a complete cache directory (no missing, extra or incomplete blocks) |
| `raw_cache_integrity` | Cached tensors are finite FP32 with the expected shape, seeds match `batch_seed + i` and the derived η seeds, and tensor hashes match the cache metadata; a mismatch always stops validation |
| `final_noise_bit_exact` | Each reconstructed final tensor matches the `final_noise_hash` recorded at generation, bit for bit |
| `saved_summary_statistics` | Per-sample, per-channel mean, population standard deviation, L2 norm and PSD summaries match the statistics saved at generation (relative tolerance 1e-4) |
| `sample_index_coverage`, `recorded_generation_tensor` | Each group has the four expected sample indices with consistent seeds and hashes, so the validated tensor is the one that was supplied to the generator |
| `exact_construction` | **Same-phase and pink:** the complex Fourier coefficients equal $\mathcal{F}[z]$ times the filter multiplier, within float32 tolerance, and the maximum phase error over non-negligible bins is reported. **Independent-white:** the coefficients equal $\sqrt{1-\gamma}\,\mathcal{F}[\text{normalised pink}] + \sqrt{\gamma}\,\mathcal{F}[\eta]$, and the power identity including the actual finite-sample cross term $2\sqrt{\gamma(1-\gamma)}\,\mathrm{Re}(P\bar{E})$ holds |
| `unexpected_generation_records` | No generation records outside the frozen grid |

Close summary statistics alone do not establish tensor equality; only the recorded hashes do.

**2. Descriptive spectral diagnostics** (no pass/fail threshold)

- **Radial PSD.** Power $|\mathcal{F}|^2$ from `rfft2` with the real-FFT column multiplicity, averaged in 24 radial shells of the integer FFT-bin radius (`--radial-bins` changes this). The DC bin is excluded from shells and reported separately.
- **Normalisation effect.** Both the pre-normalisation (`raw`) and the final (`final`) spectra are recorded.
- **Theoretical versus empirical.** For pink and same-phase conditions, the prediction is $|\mathcal{F}[z]|^2$ times the squared multiplier. For independent-white conditions, the prediction is the weighted sum of the component spectra without the cross term; the measured cross-term magnitude is reported, and its difference from the empirical spectrum reflects finite-sample variation, not a construction error.
- **Band energy.** Mean fraction of non-DC energy in the low, middle and high thirds of the radius range.
- **Cosine and coherence to the base draw.** `cosine_to_base` is the per-sample spatial cosine with the normalised base white draw. `coherence_to_base` is a pooled, power-weighted magnitude-squared Fourier coherence with the base draw; it summarises how much of the spectrum stays aligned with the base sample and is not a per-frequency proof of phase preservation.
- **Nearest baseline α.** `nearest_baseline_alpha` is the baseline grid point whose normalised log-PSD is closest (mean squared difference over non-empty shells) to each white-floor condition. It describes spectral shape on the run's discrete grid; it does not imply equivalent noise distributions or equivalent generated images. Boundary hits are flagged.

**3. Questions that need image experiments.** Whether a spectral profile improves quality or diversity is answered only by the generation and metric stages. A successful validation run shows that the noise was constructed as specified; it does not test the scientific hypothesis.

**Outputs** (`outputs/<run-id>/analysis/frequency_validation/`):

| File | Content |
| --- | --- |
| `validation_summary.json` | Overall `status`, every check with its status, coverage, counts, tolerances, spectral settings and per-block-condition evidence |
| `noise_frequency_summary.csv` | One row per condition: mean, std, L2 norm, cosine and coherence to base, band fractions, DC power, nearest baseline α and its distance |
| `radial_psd.csv` | One row per condition and radial shell: shell-mean PSD and shell energy for raw, final, prediction and cross-term curves |
| `radial_psd_selected.png` | Normalised log-PSD of the selected white-floor conditions over the grey baseline α sweep |
| `theoretical_vs_empirical_psd.png` | Empirical (pre-final-normalisation) versus predicted PSD shape |
| `equivalent_alpha_heatmap.png` | Nearest baseline α over the α × γ grid of each family |
| `coherence_heatmap.png` | Coherence with the base draw over the α × γ grid |

The command prints the overall status and each check, and exits with a non-zero code if a check fails or validation cannot run. Read `validation_summary.json` first: `checks.final_noise_bit_exact` and `checks.exact_construction` should be `passed`, then inspect the figures for spectral behaviour.

**Partial runs.** Incomplete cache coverage stops validation by default. `--allow-partial` validates the available blocks, records the omissions and marks figures `PARTIAL`; corrupt caches still stop validation.

```bash
python3 noise_validation.py --run-id sdxl_turbo_geneval400 --allow-partial \
  --plot-same-gamma-range 0.05 --plot-independent-gamma-range 0.9
```

## Coupled-sampling comparison baselines

The two `*_geneval400_coupled.yaml` configs generate two published batch-coupling baselines on the same 400 prompts and base draws, for comparison with the white-floor conditions:

- **Antithetic pairs** (`antithetic/pairs_01_23`): images 1 and 3 use the negated noise of images 0 and 2.
- **Repulsive Gaussian coupling** (`repulsive/strength_1`): the first image keeps its noise; the other three are transformed to have pairwise correlation −1/3 with standard Gaussian marginals. The transform is vendored from the authors' MIT-licensed implementation (upstream revision recorded in `coupled_sampling.py`).

These configs use `normalization_profile: none` and disable the white-floor families. They run through the same stages:

```bash
source ../.venv-generation/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_coupled.yaml --stage validate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_coupled.yaml --stage generate --run-id sdxl_turbo_geneval400_coupled
deactivate
source ../.venv-metrics/bin/activate
python3 run_experiment.py --config configs/sdxl_turbo_geneval400_coupled.yaml --stage metrics --run-id sdxl_turbo_geneval400_coupled
python3 run_experiment.py --stage analyze --run-id sdxl_turbo_geneval400_coupled
```

For coupled runs, `analyze` writes a categorical summary to `outputs/comparisons/<run-id>_categorical/` instead of Q–D curves. [`compare_initializations.py`](compare_initializations.py) `--spec spec.yaml` builds paired prompt-level comparisons across runs from a YAML file that names each run (`run`), condition ID (`condition`) and label (`name`) under `selections`, the reference labels under `references`, and an `output` directory outside every source run. It checks that the selected runs share prompts, seeds, generation settings and metric versions before comparing them.

## Output layout

```text
outputs/<run-id>/
├── run_manifest.json            # frozen resolved config and its hash
├── environment.json             # Python, platform, torch and CUDA versions
├── blocks.jsonl                 # frozen prompt/seed blocks
├── noise_cache/<block-id>/
│   ├── base_white.pt            # four base white draws (FP32)
│   ├── independent_eta.pt       # four independent eta draws (FP32)
│   └── noise_metadata.json      # seeds, tensor hashes, statistics
├── generations/<model>/<block-id>/<family>/<condition>/
│   ├── image_00.png … image_03.png
│   ├── grid_1x4.png
│   ├── samples.jsonl            # per-image provenance and hashes
│   └── noise_statistics.json    # pre- and post-normalisation statistics
├── metrics/
│   ├── per_image.csv
│   ├── per_pair.csv
│   ├── per_group.csv
│   └── metric_manifest.json
└── analysis/
    ├── analysis_manifest.json
    ├── zoom_manifest.json
    ├── tables/
    ├── primary/
    ├── robustness/<quality>__<diversity>/
    ├── optional_details/
    ├── prompt_subsets/
    ├── generation_grids/
    └── frequency_validation/
```

Condition directories are named `baseline/alpha_0p3`, `same_phase/alpha_0p9_gamma_0p05` and `independent_white/alpha_0p9_gamma_0p9` (decimal points written as `p`). Same-phase α = 0 alias directories contain `samples.jsonl` pointing to the white images instead of their own PNGs.

| Question | File |
| --- | --- |
| Was generation complete and verified? | Every group has four PNGs and a `samples.jsonl`; `analyze` and `noise_validation.py` check coverage against `blocks.jsonl` |
| Were metrics complete? | `metrics/metric_manifest.json` has `complete: true`, and `analyze` validates every expected row |
| Which settings produced this run? | `run_manifest.json` (`resolved_config`) and `blocks.jsonl` |

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'io_utils'` (or another module) | The scripts use flat imports; run them from `noise_init/`. |
| `Missing frozen run manifest` | `--run-id` names a directory that does not exist under `outputs/`; check the ID, or pass `--run-dir /path/to/run`. |
| `Run manifest differs … choose a different --run-id` | The config or manifest changed for an existing run; use a new run ID. |
| `Incomplete existing generation … refusing to overwrite source images` | An interrupted group left a partial directory; move it aside and rerun `generate`. |
| `HPSv3 is enabled but unavailable` | `metrics` was run outside `.venv-metrics`; activate the metrics environment. |
| Errors importing `Flux2KleinPipeline` | The generation environment needs `diffusers==0.37.1`; use `.venv-generation`. |
| `Metric integrity check failed … rerun metrics` | Metrics are missing for some groups, or images changed after scoring; rerun `--stage metrics`. |
| `Missing …all_curve_points.csv; run run_experiment.py --stage analyze first` or `Aggregate table lacks current … metadata` | Run `--stage analyze` before `plot_zoomed.py`, and rerun it after metrics change. |
| `Cache coverage failed` in noise validation | Some blocks have no cache yet; finish generation or pass `--allow-partial`. |
| Moved run directory | Copy the whole run directory (including `generations/`, `noise_cache/`, `metrics/`, `run_manifest.json` and `blocks.jsonl`) and pass `--run-dir /new/path/<run-id>` to `analyze`, `plot_zoomed.py`, `analyze_prompt_subset.py` and `noise_validation.py`. Recorded absolute image paths are re-based inside the selected run and verified by hash. `make_comparison_grid.py` locates runs through `--config` and `--run-id` under `outputs_root`, so keep moved runs under `noise_init/outputs/` for grids. |

## Tests and synthetic examples

The unit tests run on CPU and need no model downloads. They cover noise construction identities, spectral validation, the analysis and plotting pipeline on synthetic runs, result-integrity checks, run relocation and the comparison grid.

```bash
cd /path/to/Initial_Noise_Optimization/noise_init
source ../.venv-metrics/bin/activate
MPLBACKEND=Agg OMP_NUM_THREADS=1 python3 -m unittest discover -s tests -v
```

To render example figures from synthetic runs (finer grid, GenEval-style, single-family, no-baseline and baseline-only), give an output directory that does not yet exist:

```bash
MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONPATH=. python3 tests/render_formal_examples.py --output /tmp/noise-figure-examples
```

`examples.json` in the output directory lists the figure folders. The synthetic scores only exercise the pipeline and are not model results.
