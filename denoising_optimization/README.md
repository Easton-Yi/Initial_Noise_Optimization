# Stage 2: FloorRefine Refinement, Initialisation Ablation and DivGen Comparison

This directory runs the complete FloorRefine method — white-floor pink noise initialisation followed by quality-guided refinement of the initial noise — on 50 GenEval prompts with SDXL-Turbo and FLUX.2 [klein] 4B. It also contains an ablation over the starting noise under the same refinement, and a comparison with DivGen run under the same generation protocol. Each notebook covers generation, evaluation and summary.

← [Back to the project overview](../README.md) · Previous stage: [Initialisation experiments and evaluation](../noise_init/README.md)

## Contents

- [Experiment design](#experiment-design)
- [Notebook overview](#notebook-overview)
- [Shared protocol](#shared-protocol)
- [FloorRefine implementation](#floorrefine-implementation)
- [DivGen implementation](#divgen-implementation)
- [Environments, kernels and the MANIQA subprocess](#environments-kernels-and-the-maniqa-subprocess)
- [Running the FloorRefine notebooks](#running-the-floorrefine-notebooks)
- [Running the DivGen notebooks](#running-the-divgen-notebooks)
- [Evaluation workflow](#evaluation-workflow)
- [Metrics and outputs of the evaluation](#metrics-and-outputs-of-the-evaluation)
- [Statistics and interpretation](#statistics-and-interpretation)
- [Output map](#output-map)
- [Completion and failure records](#completion-and-failure-records)
- [Troubleshooting](#troubleshooting)

## Experiment design

The notebooks answer three questions:

1. **Complete FloorRefine versus DivGen.** How does FloorRefine (same-phase white-floor pink start + quality-only refinement) compare with the project's DivGen implementation on the same prompts, base draws and generation settings?
2. **Initialisation under the same refinement.** With the FloorRefine refinement fixed, how do white, pink (α = 0.2) and same-phase white-floor pink (α = 0.9, γ = 0.05) starting points differ?
3. **Initialisation within DivGen.** With the DivGen optimisation fixed, how do pink (α = 0.2) and same-phase white-floor pink starting points differ?

Questions 2 and 3 compare **initialisations under one fixed optimisation procedure**, so differences can be attributed to the starting noise. Question 1 compares **two different optimisation procedures**. FloorRefine and DivGen differ in their reward model (HPSv2.0 versus HPSv2.1), diversity objective (none versus DINO-DPP), optimiser, update budget, constraint, selection rule and numerical precision, so a difference between them cannot be attributed to the presence or absence of a diversity reward alone.

| Experiment | Notebook pair | Initialisations | Optimisation |
| --- | --- | --- | --- |
| FloorRefine + initialisation ablation | `FloorRefine-*.ipynb` | white; pink α = 0.2 (`pink02`); same-phase α = 0.9, γ = 0.05 (`whitefloor`) | Quality-only (HPSv2.0), Adam, 20 updates, 10 % edit bound |
| DivGen + initialisation comparison | `DivGen-*.ipynb` | pink α = 0.2 (`pink`); same-phase α = 0.9, γ = 0.05 (`whitefloor`) | DivGen: HPSv2.1 + DINO-DPP + norm regulariser, SGD, 149 updates |
| FloorRefine versus DivGen | Section 14 of each `DivGen-*.ipynb` | FloorRefine `whitefloor` versus both DivGen starts | Reads both completed evaluations; no generation |

In the notebooks and output folders, `whitefloor` denotes the same-phase white-floor pink noise construction; the independent-white construction is evaluated in [`noise_init/`](../noise_init/README.md) only.

## Notebook overview

| Notebook | Model | Purpose | Initialisations | Objective | Denoising steps | Optimisation updates | Main outputs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`FloorRefine-SDXL-GenEval50.ipynb`](FloorRefine-SDXL-GenEval50.ipynb) | SDXL-Turbo, 512 × 512 | FloorRefine and initialisation ablation | white, pink02, whitefloor | Mean HPSv2.0, quality only | 1 (`NUM_STEPS=1`; 4 is available as an additional setting) | 20 | 1,200 images (50 × 3 × {baseline, refined} × 4), per-group optimisation records, evaluation tables |
| [`FloorRefine-FLUX2-Klein-4B-4step-GenEval50-GPU-Resident.ipynb`](FloorRefine-FLUX2-Klein-4B-4step-GenEval50-GPU-Resident.ipynb) | FLUX.2 [klein] 4B, 1024 × 1024 | Same experiment on FLUX | white, pink02, whitefloor | Mean HPSv2.0, quality only | 4 | 20 | As above |
| [`DivGen-SDXL-GenEval50.ipynb`](DivGen-SDXL-GenEval50.ipynb) | SDXL-Turbo, 512 × 512 | DivGen under the shared protocol; comparison with FloorRefine | pink, whitefloor | HPSv2.1 + DINO-DPP + regulariser | 1 (`NUM_STEPS=1`; 4 is available) | 149 (150 evaluated candidates); selected candidates saved after 21 and 150 evaluations | 1,200 images (50 × 2 × {baseline, 21, 150} × 4), histories, timings, evaluation tables, FloorRefine comparison |
| [`DivGen-FLUX2-Klein-4B-4step-GenEval50-GPU-Resident.ipynb`](DivGen-FLUX2-Klein-4B-4step-GenEval50-GPU-Resident.ipynb) | FLUX.2 [klein] 4B, 1024 × 1024 | As above, on FLUX | pink, whitefloor | HPSv2.1 + DINO-DPP + regulariser | 4 | 149 (150 evaluated candidates); saved at 21 and 150 | As above |

Every notebook has the same structure. Locate cells by their section headings:

| Sections | Kernel | Content |
| --- | --- | --- |
| 1. Prepare the generation environment | Generation | HPSv2 preparation and imports |
| 2. Settings | Generation | Configuration cell and the fixed GenEval50 prompt list |
| 3–5. Optimisation, CPU checks, sampling and rewards | Generation | Function definitions; Section 4 runs CPU checks before any model is loaded (FloorRefine: objective, gradients and edit bound; DivGen: shared base draws) |
| 6. Generate the images | Generation | `RUN_DIR = generate_batch()` |
| 7. Select the images for evaluation | Metrics | Verifies a completed run; no generation |
| 8. Prepare the evaluation worker | Metrics | Writes the embedded worker script |
| 9. Compute core metrics | Metrics | Runs core metrics through `.venv-metrics` |
| 10. Compute MANIQA | Metrics | Runs MANIQA through `.venv-maniqa` |
| 11. Inspect metric versions | Metrics | Version table, optional comparison with another evaluation |
| 12. Combine the results | Metrics | Validated merge of core and MANIQA tables |
| 13. Summarise | Metrics | Prompt-level statistics, plots, cost summaries |
| 14. (DivGen only) Compare with quality-only FloorRefine | Metrics | Paired FloorRefine-versus-DivGen comparison |

## Shared protocol

All four notebooks use the same prompts, base noise and sampling protocol for a given model:

- **Prompts.** The fixed GenEval50 subset of the 400-prompt GenEval pool used in Stage 1, embedded in each notebook as `GENEVAL50` (Section 2). Categories were allocated proportionally with the largest-remainder method, and prompts were chosen within each category by a seeded SHA-256 ranking (`selection_seed = 20261023`): 7 single-object, 9 two-object, 7 counting, 9 colour, 9 position and 9 colour-attribute prompts. Each prompt keeps its GenEval400 prompt ID and batch seed.
- **Base draws.** For batch seed $s_b$, the four base white draws use CPU generators seeded with $s_b + i$, $i = 0, \dots, 3$, and are saved per prompt as `shared_base_white.pt`. All initialisations of a prompt are built from these same draws.
- **Noise construction.** The project's own functions from `noise_init/noise_methods.py` (`pink`, `same_phase_floor`, `normalize`), with the `per_sample_per_channel_zero_mean_unit_std` normalisation. For FLUX, noise is constructed on the 128 × 64 × 64 patchified latent grid before sequence packing.
- **Sampling.** SDXL-Turbo uses `EulerAncestralDiscreteScheduler` with trailing timestep spacing, guidance 0 and the `madebyollin/sdxl-vae-fp16-fix` VAE; the initial noise is multiplied by the scheduler's `init_noise_sigma`, and the ancestral noise of image $i$ comes from a CPU generator seeded with $s_b + i + 200000011$, restored identically for every candidate. FLUX.2 [klein] uses the deterministic FlowMatch Euler schedule of the official `Flux2KleinPipeline` with no ancestral noise.
- **Pairing checks.** FloorRefine verifies that the saved initial noise of each initialisation equals the noise rebuilt from the shared base draws, and that all three initialisations use identical sampler random states. DivGen verifies that its two initialisations share the same base-draw hash and rollout seeds.

## FloorRefine implementation

FloorRefine optimises an offset to the initial noise of a four-image group using a differentiable quality reward on the decoded images.

| Element | Implementation |
| --- | --- |
| Reference noise | The constructed, normalised initial noise of each initialisation, saved in `prefix.pt`. Edits are applied before the first denoising step (`EDIT_AFTER_STEP = 0`). |
| Offset | $\delta$, parameterised in units of each image's RMS noise amplitude and initialised to zero (`OPT_RESTARTS = 1`). |
| Rollout | Each candidate $z_{\mathrm{ref}} + \delta$ runs the complete denoising rollout and VAE decode for all four images. |
| Reward | Mean HPSv2.0 score of the four images (official `xswu/HPSv2` v2.0 weights, ViT-H-14), computed from continuous decoded pixels with the official HPSv2 tensor preprocessing. The notebook checks that this tensor path matches the official preprocessing of the saved PNG within a fixed tolerance. |
| Objective | `DIVERSITY_ENABLED = False`, `QUALITY_WEIGHT = 1.0`: maximise mean HPSv2. |
| Constraint | After each update, every image's offset is projected onto $\lVert\delta_i\rVert_2 \le$ `MAX_REL_EDIT` $\cdot \lVert z_{\mathrm{ref},i}\rVert_2$ with `MAX_REL_EDIT = 0.10`. |
| Optimiser | Adam, `OPT_LR = 0.01`, `OPT_STEPS = 20` updates, optimiser seed `OPT_SEED = 731`. Candidates are evaluated at iterations 0–20 (21 evaluations, the first being the zero offset). |
| Candidate selection | The unedited baseline is the initial best candidate. A candidate replaces the current best only if its mean HPSv2 is strictly higher, so the selected group can be the baseline, and it is the best evaluated candidate rather than the last iterate. The whole group is selected together. |
| Saved conditions | `baseline` (unedited) and `edit_1` (the selected candidate); `STRENGTHS = [0.0, 1.0]`. |
| Gradient computation | Two passes per update: (1) evaluate all four rollouts without gradients and compute the reward gradient with respect to each final latent; (2) recompute each image's rollout with gradients and back-propagate that upstream gradient (a vector–Jacobian product), one image at a time, with activation checkpointing. The recomputed forward is checked against pass (1). This keeps only one image graph in memory. |
| Precision (SDXL) | FP16 UNet with a fixed VJP scale of 128 to avoid FP16 gradient underflow (divided out afterwards); FP32 VAE and HPSv2; FP32 offset. |
| Precision (FLUX) | BF16 transformer and VAE, FP32 offset (each forward casts to BF16), VJP scale 1. |
| Frozen networks | UNet or transformer, VAE, text encoders and HPSv2 are frozen (`requires_grad_(False)`). |

Built-in checks run before the optimisation of every group: segmented and uninterrupted sampling must agree, the HPSv2 gradient must reach the initial noise, and after optimisation the selected offset is replayed to confirm that it reproduces the saved candidate. The FLUX notebook additionally runs a **preflight** once per loaded pipeline: it compares the custom four-step rollout with the official `Flux2KleinPipeline` latent output and checks that the gradient to the initial noise is finite and non-zero; any mismatch stops the run.

The notebooks keep a joint quality–diversity objective (`DIVERSITY_ENABLED = True`) for separate experiments. The formal FloorRefine runs use the quality-only setting above, and HPSv3 is never used during optimisation or selection.

## DivGen implementation

The DivGen notebooks run the upstream DivGen optimisation under the shared generation protocol above.

**Upstream source.** [DivGen](https://github.com/anneharrington/divgen) at revision `6166a2d4c1fb06dd6984adcd18a3261df7030e74`, MIT License, Copyright (c) 2026 Anne Harrington, A. Sophia Koepke, Shyamgopal Karthik. The selected upstream files (model utilities, HPS reward, DINO backend, DPP objective, trainer, optimiser helpers) are embedded in the notebook (Section 3) with their SHA-256 hashes. At run time they are verified, written to `denoising_optimization/_vendor/divgen/<revision>/`, and executed with imports of unused branches removed; function bodies are unchanged. No separate clone is needed. The licence text is reproduced at the end of each DivGen notebook.

| Element | Implementation |
| --- | --- |
| Trainer | Upstream `LatentNoiseTrainer.train()`; a subclass only records history, timing and selected candidates. |
| Optimised variable | The initial noise itself (no separate offset). SDXL: an FP16 parameter, as upstream. FLUX: an FP32 master parameter with BF16 forward passes. |
| Quality term | Upstream HPS loss with HPSv2.1 weights (`HPS_v2.1_compressed.pt`), weight 3. For a four-image group with one text embedding, the upstream HPS reduction uses the first image's score; this behaviour is preserved and recorded as `hps_reduction = official_diagonal`. |
| Diversity term | Upstream DPP objective on normalised DINOv2 (`facebook/dinov2-base`) CLS embeddings of the four images, weight 150. |
| Regulariser | Upstream norm regulariser, weight 0.01: $0.5\lVert z\rVert^2 - (d-1)\log\lVert z\rVert$, averaged over the images. |
| Optimiser | SGD, learning rate 6, momentum 0.9, gradient-norm clipping 0.1. |
| Budget | `N_EVALUATIONS = 150` candidates (149 updates). `SAVE_BUDGETS = [21, 150]` saves the best candidate found within the first 21 and within all 150 evaluations of the same trajectory. |
| Selection | Upstream rule: the candidate with the lowest weighted reward loss (HPS + DPP, before the regulariser) among the evaluated candidates, including the initial one. |
| Gradient | One backward pass through the connected graph of all four images, as upstream; activation checkpointing is used for UNet or transformer and VAE calls. No VJP scaling, retry or gradient replacement is applied; non-finite gradients stop the run. |
| Edit size | Not constrained; the relative change of the selected noise is recorded as `max_relative_edit`. |

**Adaptations for the shared protocol.**

- Images are generated by a `SharedGenerationAdapter` that uses the FloorRefine sampler, decode path and per-image random streams instead of the upstream pipeline call. Images are generated one at a time while keeping their graphs connected, so the DPP term still sees the four images together.
- Initial noise is built in FP32 with the project's noise functions and the same normalisation as FloorRefine, then converted to the optimisation dtype; the conversion error is checked and recorded in `initialization.json`.
- Decoded pixels are cast to the HPS encoder's input dtype; the HPS score and loss are unchanged.
- The pink exponent is set to α = 0.2 for both models, and FLUX runs at 4 steps and 1024 × 1024. These settings evaluate DivGen under this project's protocol; they are not a reproduction of the upstream default configuration, and the hyperparameters were not retuned for FLUX.

**Aligned with FloorRefine:** prompts, base draws, noise construction and normalisation, scheduler, scheduler randomness, VAE, resolution, denoising steps, four images per prompt, and the evaluation worker and metric definitions.

**Different from FloorRefine:** reward model (HPSv2.1 with the upstream reduction versus mean HPSv2.0), the DPP diversity term and regulariser, optimiser and learning rate, number of updates, the absence of an edit bound, the selection rule, the gradient computation, and (for SDXL) the precision of the optimised noise.

## Environments, kernels and the MANIQA subprocess

Three kinds of Python environment are used:

| Role | Environment | Used by | Key packages |
| --- | --- | --- | --- |
| Generation and optimisation, SDXL | e.g. `.venv-generation` (project root), registered as a Jupyter kernel | Sections 1–6 of the SDXL notebooks | `torch` (CUDA), `diffusers==0.37.1`, `transformers==4.56.2`, `accelerate`, `hpsv2`, `timm`, `pandas`, `matplotlib`, `ipykernel` |
| Generation and optimisation, FLUX | a separate environment, e.g. `.venv-flux`, registered as a Jupyter kernel | Sections 1–6 of the FLUX notebooks | as above but `transformers==4.57.6`; the GPU must support BF16 |
| Core metrics | `.venv-metrics` (project root) | Sections 7–14 (the notebook kernel) and the core metric worker (`CORE_PYTHON`) | `hpsv3==1.0.0`, `transformers==4.45.2`, `lpips`, `dreamsim`, `vendi-score`, `pandas`, `ipykernel` |
| MANIQA | `.venv-maniqa` (project root) | MANIQA worker only (`MANIQA_PYTHON`); never selected as a kernel | `torch==2.11.0`, `torchvision==0.26.0`, `pyiqa==0.1.16` |

The separation resolves real pin conflicts: HPSv3 requires an older `transformers` than the FLUX.2 [klein] pipeline, HPSv2 belongs to the generation stage only, and `pyiqa` pulls its own dependency versions.

**Creating the environments** (adjust the CUDA wheel index to your machine):

```bash
cd /path/to/Initial_Noise_Optimization

# SDXL generation (extends the Stage 1 generation environment)
python3.12 -m venv .venv-generation
.venv-generation/bin/python -m pip install -r noise_init/requirements_generation.txt
.venv-generation/bin/python -m pip install hpsv2 timm pandas safetensors ipykernel
.venv-generation/bin/python -m ipykernel install --user --name venv-generation --display-name ".venv-generation"

# FLUX generation
python3.12 -m venv .venv-flux
.venv-flux/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv-flux/bin/python -m pip install "diffusers==0.37.1" "transformers==4.57.6" accelerate safetensors pillow numpy pandas matplotlib hpsv2 timm ipykernel
.venv-flux/bin/python -m ipykernel install --user --name venv-flux --display-name ".venv-flux"

# Core metrics (see noise_init/README.md); register it as the metrics kernel
python3.12 -m venv .venv-metrics
.venv-metrics/bin/python -m pip install -r noise_init/requirements_metrics.txt
.venv-metrics/bin/python -m pip install pandas ipykernel
.venv-metrics/bin/python -m ipykernel install --user --name venv-metrics --display-name ".venv-metrics"

# MANIQA (no kernel registration)
python3.12 -m venv .venv-maniqa
.venv-maniqa/bin/python -m pip install --upgrade pip
.venv-maniqa/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv-maniqa/bin/python -m pip install "pyiqa==0.1.16"
.venv-maniqa/bin/python -m pip check
```

**HPSv2 preparation.** Section 1 of each notebook patches the installed `hpsv2` package once: it removes an unused `from turtle import forward` line from its bundled OpenCLIP code (keeping a `.py.bak` backup) and downloads the CLIP BPE vocabulary file if it is missing, verifying its SHA-256. Run these cells in the generation kernel. The HPSv2, HPSv2.1, DINOv2 and generator checkpoints are downloaded from Hugging Face on first use.

**Required project files.** The notebooks locate the project root by searching upward from the working directory (or from `PROJECT_ROOT` if it is set) for `noise_init/noise_methods.py`, `noise_init/metric_runner.py` and the two loader modules `pc_specific_psd/_compat_common.py` and `pc_specific_psd/compat_metrics.py`. These loaders import `noise_methods` and `MetricRunner` from `noise_init/`; the evaluation worker also imports `MetricRunner` through `pc_specific_psd.compat_metrics`. Both modules, together with `pc_specific_psd/__init__.py`, must be present for generation and evaluation to run.

**Paths and switches** (set in the configuration cells):

| Variable | Where | Meaning |
| --- | --- | --- |
| `PROJECT_ROOT` | Sections 2 and 7 | `None` searches upward from the working directory; otherwise an absolute project path |
| `CACHE_DIR` | Section 2 | Model cache directory. The FloorRefine notebooks set an absolute example path; change it to `/path/to/Initial_Noise_Optimization/noise_init/cache` or `None` (default Hugging Face cache). The DivGen notebooks derive it from `PROJECT_ROOT`. |
| `OUTPUT_ROOT` | Section 2 | `None` (FloorRefine) or the default under `denoising_optimization/outputs/`; change only together with `RESULTS_ROOT` |
| `RUN_NAME` | Section 2 | `None` builds a name from the model, steps, objective and budget; a run is resumed only with the same name and settings |
| `RESULTS_ROOT`, `RUN_TO_SCORE` | Section 7 | Where to find runs, and which run to evaluate (`None` = the run recorded in `latest_generation.json`; or a folder name or absolute path) |
| `CORE_PYTHON`, `MANIQA_PYTHON` | Section 7 | Interpreters for the metric workers: `PROJECT_ROOT/.venv-metrics/bin/python` and `PROJECT_ROOT/.venv-maniqa/bin/python` |
| `RUN_MANIQA` | Section 7 | `True` runs MANIQA after the core metrics; `False` evaluates core metrics only |
| `EVALUATION_TAG` | Section 7 | Name of the evaluation folder inside the run (`metrics_revalidated_v1` for FloorRefine, `evaluation_v1` for DivGen) |
| `CORE_METRICS` | Section 7 | `['clip', 'hpsv3', 'dreamsim', 'lpips', 'vendi_clip']` |
| `REFERENCE_METRIC_MANIFEST` | Section 7 | Optional `metric_manifest.json` from another evaluation, compared in Section 11 |

**How the metric workers run.** Section 8 writes the embedded worker source (`METRICS_WORKER_SOURCE`) to `<run>/<EVALUATION_TAG>/isolated_metrics_worker.py`. Sections 9 and 10 call `execute_metric(metric, python, output)`, which starts the worker as a **separate process** with `subprocess.Popen`, using the selected interpreter:

```text
<python> <EVAL_ROOT>/isolated_metrics_worker.py --run-dir <RUN_DIR> --project-root <PROJECT_ROOT> \
         --output-dir <EVAL_ROOT>/core|maniqa --device cuda --metrics <metric>
```

- One process is started per metric, and the notebook waits for each to exit before starting the next. Core metrics and MANIQA therefore run **sequentially**, not in parallel.
- The worker reads the run's `manifest.json`, re-checks every image hash and writes its tables to the output directory. `PYTHONPATH` and `PYTHONHOME` are removed from the child environment so the notebook kernel's packages cannot leak into it.
- The child's combined stdout and stderr are streamed into the notebook and appended to `<output>/<metric>.log`. A non-zero return code raises an error that names the log. On success, the notebook also requires `metric_status.csv` to show `ok` for that metric.
- If a metric fails inside the worker, its status is `error`, its traceback is written to `<output>/<metric>_error.txt`, and its partial rows are discarded.
- MANIQA runs entirely inside `.venv-maniqa` while the notebook stays on the metrics kernel. No kernel switch and no MANIQA kernel registration are needed.

**GPU memory.** Generation releases its models at the end of `generate_batch()`, but the kernel process keeps its CUDA context. Shut down the generation kernel before starting evaluation so that only one set of models occupies the GPU.

## Running the FloorRefine notebooks

1. **Open** the notebook from `denoising_optimization/` and select the generation kernel (`.venv-generation` for SDXL, `.venv-flux` for FLUX).
2. **Section 1:** run the HPSv2 preparation and import cells.
3. **Section 2:** review the configuration cell. Settings that may be changed:
   - `CACHE_DIR` — set to your model cache.
   - `NUM_STEPS` (SDXL only) — `1` for the main experiment, `4` for the additional four-step comparison; the run name changes automatically.
   - `PROMPT_LIMIT` (FLUX only) — `1` for a one-prompt trial saved as a separate `geneval1_…` run, `None` for all 50 prompts.
   - `SHOW_INLINE` — display figures inline; they are saved either way.

   Keep the formal settings unchanged for reproduction: `INIT_CONFIGS`, `MAX_REL_EDIT = 0.10`, `OPT_STEPS = 20`, `OPT_LR = 0.01`, `OPT_SEED = 731`, `DIVERSITY_ENABLED = False`, `QUALITY_WEIGHT = 1.0`, `NORMALIZATION_PROFILE`. Then run the next cell, which defines the fixed `GENEVAL50` list and `PROMPT_SOURCE`.
4. **Sections 3–5:** run all cells. Section 4 runs CPU checks of the objective, chain-rule gradients, edit bound and best-candidate replay; it needs no GPU.
5. **Section 6:** run `RUN_DIR = generate_batch()`. For each prompt and initialisation, progress lines are printed to the notebook, while the detailed log of each group is written to `prompts/<prompt-id>/<init>_generation.log`.
6. **Resume.** Rerun Sections 1–6 with the same `RUN_NAME` and settings (`RESUME = True`). Completed groups are verified by status, prompt, seed and image hashes and skipped; an incomplete group directory is renamed to `<init>_incomplete_<timestamp>` and regenerated. Changing any setting under an existing run name is refused; choose a new `RUN_NAME`.
7. **Check completion.** The run is complete when `<run>/manifest.json` has `"status": "complete"`, and the notebook prints the path to evaluate.

Default run directories: `outputs/initial_noise_optimize_geneval50/geneval50_quality20_sdxl1step_v1/` (SDXL, one step) and `outputs/initial_noise_optimize_geneval50_flux2/geneval50_quality20_flux2_klein4b_4step_resident_bf16_v1/` (FLUX).

**FLUX-specific behaviour.** The FLUX notebook requires `diffusers==0.37.1`, a CUDA GPU with BF16 support, and the distilled `black-forest-labs/FLUX.2-klein-4B` checkpoint (it checks all three). All selected prompts are encoded once with the Qwen3 text encoder, which is then released; the transformer, VAE and HPSv2 stay resident on the GPU during optimisation, with block-level gradient checkpointing and no CPU offload. Run a one-prompt trial (`PROMPT_LIMIT = 1`) before the full batch to confirm that GPU memory is sufficient.

## Running the DivGen notebooks

1. **Open** the notebook from `denoising_optimization/` with the generation kernel (`.venv-generation` for SDXL, `.venv-flux` for FLUX).
2. **Section 1:** run the HPSv2 preparation cell.
3. **Section 2:** run the configuration cell and the `GENEVAL50` cell. Settings that may be changed: `PROMPT_LIMIT` (`1` for a separate trial, `None` for all 50 prompts), `NUM_STEPS` (SDXL only: `1` or `4`), `RUN_NAME`, and `LOG_EVERY`. Keep `INIT_CONFIGS`, `N_EVALUATIONS`, `SAVE_BUDGETS`, the loss weights, `LR`, `MOMENTUM` and `GRAD_CLIP` unchanged for reproduction.
4. **Section 3:** load the embedded upstream source and the shared generation protocol.
5. **Section 4:** run the CPU check that both initialisations use the same base draws.
6. **Section 5:** define the recording trainer.
7. **Section 6:** run `RUN_DIR = generate_batch()`. Pink and whitefloor groups alternate in order across prompts. Each group writes `history.csv` (HPS, DPP, best iteration, gradient norms, noise change, cumulative time) as it runs.
8. **Resume.** Completed groups (`group.json` with `"status": "complete"`, matching configuration and environment hashes, and verified image and artifact hashes) are skipped. An interrupted group restarts from its initial noise. A change of settings or package versions under the same run name is refused.
9. **Check completion.** `<run>/manifest.json` has `"status": "complete"` and `paired_initialization_verified: true`.

Default run directories: `outputs/divgen_geneval50/geneval50_divgen_shared_sdxl1step_pink0.2_whitefloor_eval150_v1/` and `outputs/divgen_geneval50_flux2/geneval50_divgen_shared_flux2_klein4b_4step_pink0.2_whitefloor_eval150_resident_bf16_v1/`.

**Comparison with FloorRefine (Section 14).** Set `FLOORREFINE_RUN` to the completed FloorRefine run directory with the same model and number of denoising steps, and `FLOORREFINE_METRICS` to its `metrics_revalidated_v1/combined_with_maniqa` folder. For SDXL with one step, the FloorRefine run is `outputs/initial_noise_optimize_geneval50/geneval50_quality20_sdxl1step_v1`. The FLUX notebook's default already points to `geneval50_quality20_flux2_klein4b_4step_resident_bf16_v1`. The cell checks that both runs use the same ordered prompts, batch seeds and generation settings, and that their evaluations used the same worker, metric code and package versions; it refuses to compare otherwise. If the FloorRefine paths do not exist, the section prints a message and skips the comparison.

## Evaluation workflow

Evaluation never generates images and can be run on any completed run, including runs copied from another machine (keep the run folder intact; image paths are resolved relative to the run directory).

1. Shut down the generation kernel and open the same notebook with the **metrics kernel** (`.venv-metrics`).
2. **Section 7:** set `RUN_TO_SCORE` (or leave `None` for the last generated run), `RUN_MANIQA` and `EVALUATION_TAG`, then run the cell. It checks that the run is complete, that every prompt × initialisation × condition × image is present exactly once, that each refined condition is paired with its own baseline, and that every image matches its recorded hash. It prints the run path, the number of denoising steps and the interpreters to be used.
3. **Section 8:** write the worker script.
4. **Section 9:** compute the five core metrics (each in its own `.venv-metrics` process); results go to `<EVAL_ROOT>/core/`.
5. **Section 10:** compute MANIQA in `.venv-maniqa` if `RUN_MANIQA = True`; results go to `<EVAL_ROOT>/maniqa/`.
6. **Section 11:** inspect evaluator versions.
7. **Section 12:** merge the validated tables into `<EVAL_ROOT>/combined_with_maniqa/` (or `combined_core/` without MANIQA). The merge requires every requested task to have status `ok`, checks that the generation manifest and worker are unchanged since scoring, rejects duplicate rows, and requires complete group coverage for every metric.
8. **Section 13:** compute summaries and plots.
9. **Section 14 (DivGen):** optional comparison with FloorRefine.

**Resuming and re-running metrics.** Each completed task is recorded in `completed_tasks.json` with a signature of the worker, environment and input images; rerunning a section skips completed tasks and computes only missing or failed ones. The worker refuses to write into an evaluation folder whose saved `metric_manifest.json` describes a different environment, worker or input; use a new `EVALUATION_TAG` after changing evaluator code or packages.

**Adding MANIQA later.** Evaluate with `RUN_MANIQA = False` first (merged into `combined_core/`), then set `RUN_MANIQA = True` and rerun Sections 7–13: the core tasks are skipped, MANIQA is computed, and the merge writes `combined_with_maniqa/`.

## Metrics and outputs of the evaluation

Only the following metrics are computed by the evaluation worker in these notebooks.

| Task | Output metric (`metric` column) | Unit | Description |
| --- | --- | --- | --- |
| `hpsv3` | `hpsv3` | Image → group mean | Official HPSv3 reward mean, via `noise_init/metric_runner.py`. Independent of the HPSv2 optimisation reward. |
| `clip` | `clip_cosine` | Image → group mean | Cosine similarity between `openai/clip-vit-large-patch14` image and prompt embeddings (text alignment). |
| `clip` | `clip_mean_cosine_distance`, `clip_regularized_logdet` | Group | Additional CLIP-embedding diversity summaries: mean pairwise cosine distance, and $\log\det(K + 10^{-6}I)$ of the cosine kernel. |
| `dreamsim` | `dreamsim_mean_pair_distance` | Six pairs → group mean | Within-group perceptual diversity. |
| `lpips` | `lpips_alex_mean_pair_distance` | Six pairs → group mean | Within-group perceptual diversity (AlexNet, 256 × 256). |
| `vendi_clip` | `vendi_clip` | Group | Vendi score of the CLIP cosine kernel (1–4). |
| `maniqa` | `maniqa` | Image → group mean | No-reference perceptual image-quality prediction from `pyiqa`'s MANIQA with its default checkpoint and preprocessing; each image uses a seed derived from its hash, so scores do not depend on evaluation order. |
| `basic` (run with the core metrics) | `edge_gradient_mean` | Image → group mean | Mean gradient magnitude of the grayscale image in [0, 1] (`numpy.gradient`). |
| `basic` | `laplacian_variance` | Image → group mean | Variance of the interior five-point Laplacian of the grayscale image. |
| `dreamsim`, `lpips` | `dreamsim_distance_to_same_seed_baseline`, `lpips_distance_to_same_seed_baseline` (in `paired_change.csv`) | Image | Distance between each optimised image and the unoptimised image with the same prompt, initialisation and seed. |

**Within-group diversity versus same-seed change.** DreamSim, LPIPS and CLIP-Vendi in `per_group.csv` measure how different the four images of one group are from each other. The same-seed paired change in `paired_change.csv` measures how much optimisation moved each image away from its own baseline; it is not a diversity measure, and a larger change is not by itself better.

**Optimisation and cost records** (from generation, not from the worker):

| Record | FloorRefine | DivGen |
| --- | --- | --- |
| HPSv2 reward | `baseline_hpsv2`, `selected_hpsv2`, `delta_hpsv2` in `optimization_summary.csv` (HPSv2.0, mean of four images) | `hpsv2_1_objective` per iteration in `history.csv` (upstream reduction) |
| Selected iteration | `best_iteration` (0 means the baseline was kept) | `selected_iteration` in `timing.csv` / `timing_per_prompt.csv` |
| Edit size | `selected_max_relative_edit` (≤ 0.10); per image in `edit_stats.csv` | `max_relative_edit` in `timing.csv` |
| Optimisation time | `optimization_seconds`: the optimisation loop only (all candidate rollouts, reward evaluations and backward passes); excludes model setup, baseline generation, checks and final decoding | `optimization_seconds`: candidate generation, reward evaluation and updates, including the initial candidate; excludes model loading and file export |
| Other timings | Per group `timings.json`: setup, baseline, checks, quality setup, optimisation, replay, decoding and total wall time | `prompt_setup_seconds`, `warm_start_compute_seconds`, `warm_start_method_seconds`, `warm_start_end_to_end_seconds` (full trajectory only); model loading in `setup_<timestamp>.json` |
| Peak GPU memory | `peak_cuda_allocated_gb` during optimisation | `peak_cuda_allocated_gib`, `peak_cuda_reserved_gib` |

The two methods time different spans, so their recorded times are not directly comparable as end-to-end runtimes. The 21-evaluation DivGen timings are prefixes of the 150-evaluation trajectory, not separately timed runs.

## Statistics and interpretation

Section 13 builds prompt-level tables from `per_group.csv` (one score per prompt, initialisation, condition and metric):

- **Prompt-equal weighting.** Each prompt contributes one group per condition, and every prompt has equal weight.
- **Paired bootstrap.** 2,000 resamples of the 50 prompts with seed 20261023; the same resampled prompt indices are used for every condition and contrast, and intervals are 95 % percentile intervals.
- **Baseline-relative effects.** For each initialisation, the optimised score minus its own baseline score per prompt, with a bootstrap interval. FloorRefine also reports `relative_pct` = 100 × (mean change) / (mean baseline) and `fraction_increased` (share of prompts whose score rose).
- **Initialisation contrasts.** FloorRefine: `whitefloor − white` and `whitefloor − pink02`, both for the final optimised score (`optimized`) and for the improvement over baseline (`gain`). DivGen: `whitefloor − pink` for each saved condition.
- **Cross-method contrast (DivGen Section 14).** FloorRefine `whitefloor` optimised score minus each DivGen initialisation at each saved budget, per prompt, with bootstrap intervals.
- **Category summaries.** Mean scores per GenEval category, initialisation, condition and metric.
- **Q–D figures.** Arrows from each method's own unoptimised mean to its selected-candidate mean, for every quality × diversity metric pair.

**What the intervals support.** Each prompt has a single four-image group, so the intervals describe variation across the 50 selected prompts, not uncertainty from repeated independent seed groups within a prompt. GenEval supplies the prompts only; detector-based GenEval accuracy is not computed. Quality, text alignment and diversity should be read together, and differences between FloorRefine and DivGen reflect the whole optimisation procedure, not a single component.

## Output map

### FloorRefine run directory

```text
outputs/initial_noise_optimize_geneval50[_flux2]/
├── latest_generation.json                 # last generated run, used when RUN_TO_SCORE=None
└── <run-name>/
    ├── batch_config.json                  # all settings, prompt list, versions and fingerprint
    ├── selected_prompts.json
    ├── memory_policy.json
    ├── progress.json                      # completed (prompt, initialisation) pairs
    ├── optimization_summary.csv           # HPSv2 change, best iteration, edit size, time, memory
    ├── manifest.json                      # status=complete, all image records
    ├── images.csv
    ├── optimizer_history.csv              # all optimisation histories
    ├── prompts/<prompt-id>/
    │   ├── prompt.json
    │   ├── shared_base_white.pt           # four base draws and seeds
    │   ├── comparison_all.png             # all initialisations, baseline and optimised
    │   ├── <init>_generation.log
    │   └── <init>/                        # white | pink02 | whitefloor
    │       ├── run_config.json, manifest.json, timings.json
    │       ├── prefix.pt                  # initial noise and sampler random states
    │       ├── quality_diagnostics.json   # HPSv2 preprocessing and gradient checks
    │       ├── baseline_final_latents.pt, final_latents.pt
    │       ├── optimizer_history.csv/.jsonl, optimization_curve.png
    │       ├── optimization.pt, best_so_far.pt
    │       ├── edit_stats.csv, final_latent_distances.csv
    │       ├── images/baseline/image_00..03.png
    │       ├── images/edit_1/image_00..03.png
    │       ├── images.csv
    │       └── comparison.png
    └── metrics_revalidated_v1/            # EVALUATION_TAG
        ├── isolated_metrics_worker.py
        ├── core/                          # per_image, per_pair, per_group, paired_change,
        │                                  # metric_status.csv, completed_tasks.json,
        │                                  # metric_manifest.json, <metric>.log
        ├── maniqa/                        # same layout, MANIQA only
        └── combined_with_maniqa/          # or combined_core/
            ├── per_image.csv, per_pair.csv, per_group.csv, paired_change.csv, metric_status.csv
            ├── evaluation_sources.json
            ├── per_prompt_scores.csv, per_prompt_delta.csv
            ├── summary_prompt_equal_weight.csv
            ├── paired_initialization_comparisons.csv
            ├── summary_by_category.csv
            ├── optimization_summary_means.csv
            ├── excluded_incomplete_metrics.csv
            └── mean_<quality>_vs_<diversity>.png
```

### DivGen run directory

```text
outputs/divgen_geneval50[_flux2]/
├── latest_generation.json
└── <run-name>/
    ├── manifest.json                      # config, environment, hashes, status, all image records
    ├── images.csv
    ├── timing_per_prompt.csv              # one row per prompt, initialisation and saved budget
    ├── setup_<timestamp>.json             # model loading time, memory, parameter dtypes
    ├── images/<prompt-id>__<init>__{baseline,budget_0021,budget_0150}/00..03.png
    ├── prompts/<prompt-id>/
    │   ├── shared_base_white.pt
    │   └── <init>/                        # pink | whitefloor
    │       ├── group.json                 # status=complete, records, budgets, artifact hashes
    │       ├── initialization.pt/.json    # initial noise and conversion report
    │       ├── history.csv                # per-iteration HPS, DPP, gradients, time
    │       ├── official_history.csv       # upstream trainer history
    │       ├── timing.csv
    │       └── noise_0021.pt, noise_0150.pt   # selected noise at each budget
    └── evaluation_v1/                     # EVALUATION_TAG
        ├── isolated_metrics_worker.py
        ├── core/, maniqa/                 # as for FloorRefine
        └── combined_with_maniqa/
            ├── per_*.csv, paired_change.csv, metric_status.csv, evaluation_sources.json
            ├── summary_prompt_equal_weight.csv, paired_initialization_comparisons.csv
            ├── per_prompt_scores.csv, per_prompt_delta.csv, summary_by_category.csv
            ├── timing_summary.csv, paired_timing_comparisons.csv, warm_end_to_end_timing_summary.csv
            ├── divgen_initialization_qd_arrows_eval0021/0150.{png,pdf}
            └── floorrefine_vs_divgen_paired.csv, floorrefine_vs_divgen_recorded_cost.csv,
                floorrefine_vs_divgen_qd_arrows_eval0021/0150.{png,pdf}   # Section 14
```

Running a DivGen notebook also writes the verified upstream source files to `denoising_optimization/_vendor/divgen/<revision>/`.

## Completion and failure records

| Check | Evidence |
| --- | --- |
| All prompts and initialisations generated | Run `manifest.json` has `"status": "complete"`; Section 7 additionally verifies full prompt × initialisation × condition × image coverage and every image hash |
| A single group completed | FloorRefine: `<init>/manifest.json` with `"status": "complete"`. DivGen: `<init>/group.json` with `"status": "complete"` |
| A group failed | `failure.json` in the group directory (error message, and for DivGen the iteration and memory state); FloorRefine also keeps the group log `<init>_generation.log` and archives incomplete attempts as `<init>_incomplete_<timestamp>` |
| A metric failed | `metric_status.csv` shows `error` for that task; see `<task>_error.txt` and `<task>.log` in `core/` or `maniqa/` |
| Evaluation complete | Every requested task is `ok` in `combined_*/metric_status.csv`, and Section 12 completed without error (it rejects missing tasks and incomplete group coverage); `evaluation_sources.json` lists the merged sources and the generation manifest hash |
| Provenance | `batch_config.json` / `manifest.json` (settings, package versions, `noise_methods.py` hash), `metric_manifest.json` (evaluator versions, worker hash, interpreter, installed packages), `setup_*.json` and per-group `timings.json` |

Reaching the end of a notebook does not by itself show that every metric succeeded; check `metric_status.csv` and the merge output.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Set PROJECT_ROOT to the project root containing noise_init and pc_specific_psd` | The kernel started outside the project, or a required project file is missing (see [Required project files](#environments-kernels-and-the-maniqa-subprocess)); set `PROJECT_ROOT` to the absolute project path. |
| `The saved configuration does not match` / `Run settings or environment differ` | Settings or package versions changed under an existing run name; restore them or set a new `RUN_NAME`. |
| `Generation is incomplete` in Section 7 | Resume Sections 1–6 in the generation kernel with the same settings. |
| `Interpreter missing: …/.venv-maniqa/bin/python` | Create `.venv-maniqa`, or set `RUN_MANIQA = False`. |
| `Environment, worker or inputs changed. Choose a new EVALUATION_TAG` | The evaluator environment or worker differs from the one recorded in this evaluation folder; use a new `EVALUATION_TAG`. |
| A metric exits with `-9` | The worker process was killed, typically for lack of memory; compare `oom_kill` in the cgroup `memory.events` printed in the log, and make sure the generation kernel is shut down. Images are not affected. |
| `Evaluation settings differ` in DivGen Section 14 | The FloorRefine and DivGen evaluations used different evaluator versions; re-evaluate one run with the same environment under a new `EVALUATION_TAG`. |
| FLUX: `requires a CUDA GPU with BF16 support` or `Install diffusers==0.37.1` | Use a BF16-capable GPU and the FLUX generation environment. |
| FLUX preflight mismatch | The custom rollout disagrees with `Flux2KleinPipeline`; check the `diffusers` version and the checkpoint before running the batch. |
