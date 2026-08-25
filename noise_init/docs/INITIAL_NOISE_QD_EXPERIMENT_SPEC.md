# Initial-Noise Quality–Diversity Experiment Specification

Status: implementation specification  
Primary objective: build a reproducible baseline and two white-floor initial-noise methods for text-to-image quality–diversity evaluation  
Primary model: FLUX.2 Klein 4B, 4 inference steps  
Secondary model: SDXL-Turbo  
Initial gallery size: 4 images per `prompt × seed-batch × noise-condition`

Repository scope: all implementation, configuration, tests, generated manifests, and documentation for this task must live under `noise_init/`. Do not modify files outside `noise_init/` unless the user explicitly approves a narrowly identified integration change.

## 1. Scope and scientific question

This project studies whether changing only the initial latent-noise spectrum can improve the quality–diversity trade-off of a fixed pretrained text-to-image generator.

The implementation must answer three questions in order:

1. Baseline: how do white noise and simple pink noise with different exponents affect image quality and within-prompt diversity?
2. Same-phase PSD floor: at a fixed pink exponent, can restoring spectral power—especially suppressed high-frequency power—recover quality while retaining pink-noise diversity?
3. Independent-white replenishment: for the same expected PSD, does injecting independent Gaussian randomness behave differently from restoring amplitudes while retaining the base noise's Fourier phase?

This specification covers direct generation from designed initial noise. It does **not** include DivGen/noise optimization in the first implementation. A fixed optimizer can be added later as a separate experimental factor without changing the data contract defined here.

## 2. Corrections and fixed decisions

### 2.1 Model interpretation

FLUX.2 Klein 4B is itself a distilled four-step model according to its official model card. SDXL-Turbo is also distilled. Therefore, these two models provide a useful cross-architecture/model comparison, but they do **not** by themselves support a distilled-versus-non-distilled conclusion.

### 2.2 Experimental unit and control

The primary experimental block is:

$$
b=(\text{prompt},\text{seed batch}).
$$

Each block contains four independent white Gaussian base tensors:

$$
E_b=\{\epsilon_{b,0},\epsilon_{b,1},\epsilon_{b,2},\epsilon_{b,3}\},
\qquad \epsilon_{b,i}\sim\mathcal N(0,I).
$$

Every baseline and proposed noise condition within the block must be derived from this same saved base batch. Prompt and seed may vary between blocks, but both are fixed within every matched comparison across noise conditions.

Never generate new base white noise separately for different alpha or gamma values.

### 2.3 Four images are a gallery, not four statistical replicates

For each block and condition:

- quality is calculated for each of the four images and summarized within the block;
- diversity is calculated once from the four-image set;
- the four images or six image pairs must not be treated as four or six independent experimental units.

### 2.4 Model and sampler invariants

Within a model-specific experiment, the following must remain identical across all noise methods and parameter values:

- model checkpoint and revision;
- VAE and text encoder revisions;
- prompt text;
- height and width;
- inference steps;
- guidance scale;
- scheduler/sampler and all scheduler parameters;
- dtype and device policy;
- latent shape and latent scaling convention;
- image decoding and file format;
- metric versions and checkpoints.

Only the initial-noise construction may change.

## 3. Prompt–seed block manifest

Use a manifest as the source of truth rather than deriving experimental blocks implicitly during generation.

Example `blocks.jsonl`:

```json
{"block_id":"p000_s000","prompt_id":"p000","prompt":"A photo of a red fox in a snowy forest","seed_batch_id":"s000","batch_seed":10000}
{"block_id":"p001_s000","prompt_id":"p001","prompt":"A ceramic teapot on a wooden table","seed_batch_id":"s000","batch_seed":11000}
```

For a pilot, one seed batch per prompt is acceptable. For the final experiment, support multiple seed batches per prompt:

```text
prompt p000 × seed batches s000, s001, s002
prompt p001 × seed batches s000, s001, s002
...
```

The manifest builder must use a stable deterministic rule. Do not use Python's built-in `hash()`. Either store explicit seeds or derive them using SHA-256 from `master_seed`, `prompt_id`, and `seed_batch_id`.

Within one batch, create the four base samples with explicit sample seeds:

$$
s_{b,i}=s_b+i,\qquad i\in\{0,1,2,3\}.
$$

Save these seeds and the resulting tensor hashes.

## 4. Noise definitions

### 4.1 Frequency grid and pink filter

For spatial frequency coordinates $(u,v)$, define radial frequency:

$$
r(u,v)=\sqrt{u^2+v^2}.
$$

Use the amplitude filter:

$$
H_\alpha(r)=\frac{1}{(1+r)^\alpha}.
$$

This convention must be documented clearly: $\alpha$ is the **amplitude-spectrum exponent**, so the expected PSD is proportional to $(1+r)^{-2\alpha}$.

Use `torch.fft.rfft2`/`irfft2` over the two spatial dimensions. Build the radial grid from `torch.fft.fftfreq` and `torch.fft.rfftfreq` in the same frequency units for every method. The `1+r` term keeps the DC gain finite.

### 4.2 Baseline conditions

Baseline parameter grid:

```text
alpha = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
```

For base sample $\epsilon_{b,i}$:

$$
\widehat z^{\mathrm{pink}}_{b,i,\alpha}(r)
=H_\alpha(r)\widehat\epsilon_{b,i}(r).
$$

The `alpha=0.0` condition is the white endpoint because $H_0(r)=1$. The implementation should label it `white` in figures and data while retaining numeric `alpha=0.0` for analysis.

### 4.3 Same-phase PSD floor

Define:

$$
G_{\alpha,\gamma}(r)
=\sqrt{(1-\gamma)H_\alpha(r)^2+\gamma},
$$

$$
\widehat z^{\mathrm{same}}_{b,i,\alpha,\gamma}(r)
=\widehat\epsilon_{b,i}(r)G_{\alpha,\gamma}(r).
$$

Properties that must be tested:

- `gamma=0`: identical to the corresponding simple pink condition;
- `gamma=1`: identical to the block's base white tensor, subject to the shared postprocessing policy;
- the multiplier is positive and real, so Fourier phase is unchanged;
- expected PSD multiplier is $(1-\gamma)H_\alpha^2+\gamma$.

This method is not equivalent to merely lowering alpha: lowering alpha produces another power law, while the floor approaches a non-zero high-frequency constant.

### 4.4 Independent-white replenishment

Create and cache an independent white batch:

$$
\eta_{b,i}\sim\mathcal N(0,I),\qquad \eta_{b,i}\perp\epsilon_{b,i}.
$$

Derive its seed from a separate deterministic namespace such as `sha256(master_seed, block_id, "independent_eta", i)`. Use the same saved $\eta$ samples for all relevant alpha and gamma conditions in that block.

Construct:

$$
\widehat z^{\mathrm{ind}}_{b,i,\alpha,\gamma}(r)
=\sqrt{1-\gamma}\,H_\alpha(r)\widehat\epsilon_{b,i}(r)
+\sqrt\gamma\,\widehat\eta_{b,i}(r).
$$

The equivalent spatial implementation is:

$$
z^{\mathrm{ind}}
=\sqrt{1-\gamma}\,z^{\mathrm{pink}}
+\sqrt\gamma\,\eta.
$$

Properties that must be tested:

- `gamma=0`: identical to the corresponding simple pink condition;
- `gamma=1`: identical to the cached independent $\eta$ sample, not to $\epsilon$;
- before optional normalization, the expected PSD matches the same-phase method when $\epsilon$ and $\eta$ are independent unit-variance Gaussian fields;
- finite-sample spectra and phases need not match the same-phase method.

### 4.5 Normalization policy

Normalization is a scientifically important configuration, not an implementation detail.

The primary matched-comparison profile should apply the same postprocessing function to every final latent, including white, pink, same-phase, and independent-white conditions:

```text
normalization = per_sample_per_channel_zero_mean_unit_std
```

For each sample and channel:

$$
z\leftarrow\frac{z-\mu(z)}{\sigma(z)+\varepsilon}.
$$

This removes mean/variance as uncontrolled differences and makes same-phase endpoint tests exact after the shared transformation.

If strict compatibility with an existing DivGen implementation is required, support an explicitly named `divgen_compat` profile rather than silently changing behavior. Never combine results from different normalization profiles on one curve.

Record pre-normalization and post-normalization mean, standard deviation, L2 norm, and empirical radial PSD summary for every noise tensor.

## 5. Method parameter grids

The configuration must support arbitrary lists rather than hard-coded values.

Recommended pilot:

```yaml
baseline:
  alpha_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

same_phase_floor:
  enabled: false        # enable after baseline validation
  alpha_values: [0.5]  # replace/add anchors based on baseline failure region
  gamma_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

independent_white:
  enabled: false
  alpha_values: [0.5]
  gamma_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

Do not choose the final anchor alpha after looking only at one cherry-picked prompt. Select it from the aggregate baseline failure/trade-off region using a documented rule, then freeze the configuration before running the proposed methods.

## 6. Model adapter contract

Implement one adapter per model behind a common interface:

```python
class T2IModelAdapter(Protocol):
    def latent_spec(self, height: int, width: int, batch_size: int) -> LatentSpec: ...
    def generate(
        self,
        prompt: str,
        latents: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> list[PIL.Image.Image]: ...
```

Requirements:

- infer or validate model-specific latent shape; do not reuse SDXL latent assumptions for FLUX;
- inject the supplied latents at the actual initial-noise entry point;
- ensure the pipeline does not silently resample or overwrite them;
- for packed-token models, perform packing exactly once in the adapter;
- generation with the same prompt, latent, and config must be reproducible within the limits of the CUDA stack;
- permit `generation_batch_size=1` so a four-image gallery can be generated sequentially on limited VRAM;
- return images in base-index order `0,1,2,3`.

### 6.1 FLUX.2 Klein 4B preflight

Use the official `black-forest-labs/FLUX.2-klein-4B` checkpoint and the Diffusers `Flux2KleinPipeline` unless the existing repository already has a validated adapter.

Before implementing the full experiment:

1. load the model with `torch.bfloat16` where supported;
2. use the configured server memory profile;
3. generate one image from one explicitly supplied latent at four steps;
4. rerun and confirm the image hash or pixel difference is stable;
5. change only the latent and confirm the output changes;
6. verify no extra random latent is sampled inside the pipeline.

The official model card reports approximately 13 GB VRAM for FLUX.2 Klein 4B. Use a **24 GB GPU as the standard target**:

- load FLUX in BF16;
- start without CPU offload;
- default to `generation_batch_size=1` or `2` because simultaneous batch-of-four generation is not scientifically required;
- generate the four saved latents sequentially while preserving their order;
- unload FLUX and clear GPU memory before loading HPSv3 or other large metric models;
- score HPSv3 in a separate stage with a small metric batch size.

A 32 GB GPU is preferable if the implementation must keep more models resident or generate all four 1024×1024 images simultaneously, but it is not required for the specified staged pipeline. CPU offload is a fallback after a measured OOM, not the default server path. Do not begin the full grid until the smoke test passes.

### 6.2 SDXL-Turbo adapter

Implement as the secondary model using its documented one/few-step settings. Keep its result tree and aggregate tables separate from FLUX. Never pool raw metric values across models unless the analysis explicitly models the model factor.

## 7. Repository architecture

Keep the implementation deliberately small. Adapt names to the existing repository and reuse working code. Do not create one class/file per metric or method unless the current repository already follows that pattern.

Preferred maximum structure:

```text
noise_init/
  configs/
    flux2_klein.yaml
    sdxl_turbo.yaml
  docs/
    INITIAL_NOISE_QD_EXPERIMENT_SPEC.md
  prompts.jsonl
  run_experiment.py       # one entry point with validate/generate/metrics/analyze/all stages
  noise_methods.py        # base sampling, pink, same-phase, independent-white, normalization
  model_adapters.py       # FLUX and SDXL adapters only
  metric_runner.py        # all quality/diversity metrics and embedding cache
  analysis.py             # aggregation, bootstrap, Pareto/interpolation, plots
  io_utils.py             # manifests, hashes, paths, resume checks
  tests/
    test_noise_methods.py
    test_metrics.py
    test_pipeline_small.py
  outputs/
```

Complexity rules:

- prefer plain functions and small dataclasses over factories, registries, dependency injection, or deep class inheritance;
- use at most one thin model-adapter abstraction;
- keep all three noise formulas in one clearly documented module;
- keep metric implementations in one module unless it becomes unmanageably large;
- use one YAML configuration and command-line overrides rather than many near-duplicate scripts;
- do not introduce Hydra, a database, a workflow engine, distributed execution, or a plugin system solely for this experiment;
- separate generation, metrics, and analysis by stage so expensive work can resume, but expose them through one command;
- add a new abstraction only when two real implementations already require it.

Do not duplicate existing repository functionality. The first implementation step must audit the current code and map existing modules to this compact architecture before editing.

## 8. Configuration contract

Example:

```yaml
experiment:
  name: flux2_klein_initial_noise_qd
  master_seed: 20260825
  gallery_size: 4
  normalization_profile: per_sample_per_channel_zero_mean_unit_std

model:
  adapter: flux2_klein
  checkpoint: black-forest-labs/FLUX.2-klein-4B
  revision: null
  dtype: bfloat16
  device: cuda
  cpu_offload: false

generation:
  height: 1024
  width: 1024
  num_inference_steps: 4
  guidance_scale: 1.0
  generation_batch_size: 1
  release_model_after_generation: true
  output_format: png

blocks:
  manifest: manifests/blocks.jsonl

baseline:
  enabled: true
  alpha_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

same_phase_floor:
  enabled: false
  alpha_values: [0.5]
  gamma_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

independent_white:
  enabled: false
  alpha_values: [0.5]
  gamma_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

quality_metrics:
  clip:
    enabled: true
    checkpoint: openai/clip-vit-large-patch14
    output_name: clip_cosine
  hpsv3:
    enabled: true
    checkpoint: official_default
  maniqa:
    enabled: false

diversity_metrics:
  dreamsim:
    enabled: true
    checkpoint: official_default
    aggregation: mean_unordered_pair_distance
  lpips:
    enabled: true
    backbone: alex
    aggregation: mean_unordered_pair_distance
  vendi_clip:
    enabled: true
    embedding_checkpoint: openai/clip-vit-large-patch14

analysis:
  bootstrap_replicates: 2000
  bootstrap_seed: 20260825
  confidence_level: 0.95
  compare_on_pareto_frontier: true
  prohibit_extrapolation: true
```

All configuration values must be copied into an immutable run manifest before generation starts.

## 9. CLI and resumability

Use one entry point with explicit stages; exact argument style may follow the existing repository:

```bash
python noise_init/run_experiment.py --config noise_init/configs/flux2_klein.yaml --stage validate
python noise_init/run_experiment.py --config noise_init/configs/flux2_klein.yaml --stage generate
python noise_init/run_experiment.py --config noise_init/configs/flux2_klein.yaml --stage metrics
python noise_init/run_experiment.py --config noise_init/configs/flux2_klein.yaml --stage analyze
python noise_init/run_experiment.py --config noise_init/configs/flux2_klein.yaml --stage all
```

Also support a one-prompt smoke test:

```bash
python noise_init/run_experiment.py \
  --config noise_init/configs/flux2_klein.yaml \
  --stage generate \
  --prompt "A photo of a red fox in snow" \
  --batch-seed 10000 \
  --conditions white,pink:0.5
```

Every stage must be resumable:

- generation skips a sample only if the image, sample metadata, and expected latent hash all exist;
- evaluation skips a metric only if its version/checkpoint and input image hash match;
- aggregation and plotting are deterministic rebuilds from immutable records;
- `--force` may replace derived outputs but must not silently replace source images or cached base noises.

## 10. Output and provenance contract

Recommended structure:

```text
outputs/<run_id>/
  run_manifest.json
  environment.json
  blocks.jsonl
  noise_cache/
    <block_id>/
      base_white.pt
      independent_eta.pt
      noise_metadata.json
  generations/
    <model_id>/<block_id>/
      baseline/alpha_0p0/
        image_00.png
        image_01.png
        image_02.png
        image_03.png
        grid_1x4.png
        samples.jsonl
      baseline/alpha_0p1/
      ...
      same_phase/alpha_0p5_gamma_0p1/
      independent_white/alpha_0p5_gamma_0p1/
  metrics/
    per_image.csv
    per_pair.csv
    per_group.csv
    metric_manifest.json
  analysis/
    aggregate_conditions.csv
    paired_effects.csv
    qd_frontiers.csv
    qd_improvement_at_matched_diversity.csv
    bootstrap_intervals.csv
  plots/
    baseline/
    methods/
    qd/
  logs/
```

Use lossless PNG for metric input. The 1×4 grid is for visual inspection only and must never be passed to image metrics.

Every generated sample record must contain at least:

```text
run_id, model_id, block_id, prompt_id, prompt, seed_batch_id,
batch_seed, base_index, sample_seed, method, alpha, gamma,
normalization_profile, base_noise_hash, eta_noise_hash,
final_noise_hash, image_path, image_hash, generation_config_hash
```

## 11. Metric definitions

### 11.1 Quality

For image $x_{b,c,i}$, prompt $p_b$, and quality metric $q_m$:

$$
Q^{(m)}_{b,c}
=\frac{1}{4}\sum_{i=0}^{3}q_m(x_{b,c,i},p_b).
$$

Store all four raw scores plus block mean, standard deviation, minimum, and maximum.

Required metrics:

- `clip_cosine`: raw cosine similarity between the normalized prompt and image embeddings from one frozen, explicitly recorded CLIP checkpoint. Do not call it generic `CLIPScore` unless the exact CLIPScore scaling convention is implemented.
- `hpsv3`: use the official HPSv3 inferencer and store the scalar preference mean returned by the official API.
- `maniqa`: optional and disabled until the required metrics are stable.

If HPSv3 cannot run within available memory, do not silently substitute HPSv2. Add HPSv2 under a different metric name and keep HPSv3 pending or run it in a separate environment.

### 11.2 Pairwise diversity

Four images produce six unordered pairs. For DreamSim or LPIPS distance $d_m$:

$$
D^{(m)}_{b,c}
=\frac{1}{6}\sum_{0\le i<j<4}d_m(x_{b,c,i},x_{b,c,j}).
$$

Store each pair score in `per_pair.csv` and the six-pair mean in `per_group.csv`.

Required:

- `dreamsim_mean_pair_distance`;
- `lpips_alex_mean_pair_distance`.

For both, larger values must consistently mean greater diversity.

### 11.3 Vendi diversity

`Vendi` is incomplete unless its similarity kernel is specified. Define the primary metric as `vendi_clip`:

1. extract normalized CLIP image embeddings $X\in\mathbb R^{4\times d}$;
2. calculate $K=XX^\top$;
3. symmetrize numerically and validate diagonal values near one;
4. calculate `vendi.score_K(K)`.

For four samples the effective-number interpretation should usually lie in $[1,4]$, up to numerical tolerance. Never mix Vendi scores obtained from different embedding kernels.

### 11.4 Metric execution

Metric models must be loaded once per stage, not once per image. Cache reusable image embeddings keyed by image hash and metric checkpoint. Run large metric models sequentially if required by memory.

## 12. Aggregation and statistical analysis

For condition $c$, where a baseline condition is a fixed alpha and a proposed condition is a fixed `(method, alpha, gamma)`:

$$
\bar Q_c=\frac{1}{B}\sum_{b=1}^{B}Q_{b,c},
\qquad
\bar D_c=\frac{1}{B}\sum_{b=1}^{B}D_{b,c}.
$$

Thus each aggregate Q–D point is:

$$
(\bar D_c,\bar Q_c).
$$

Rules:

- give every complete block equal weight;
- use the same set of complete blocks for every condition compared on a curve;
- if a block is missing one condition, either regenerate it or use the common complete-block intersection and report the reduced count;
- never average already-aggregated prompt means with unequal hidden weights;
- produce separate Q–D plots for every quality/diversity metric pair;
- never combine DreamSim, LPIPS, and Vendi into an unnamed single diversity number.

### 12.1 Confidence intervals

Use paired cluster bootstrap:

- with one seed batch per prompt, resample prompt/block units;
- with multiple seed batches per prompt, use hierarchical bootstrap by resampling prompts and then seed batches within prompt, or use prompt-level clustering;
- resample complete blocks, not images or image pairs;
- recompute aggregate points, Pareto frontiers, interpolation, and improvements inside each bootstrap replicate.

Report point estimate, standard error, and 95% confidence interval.

### 12.2 Paired effects

For baseline alpha relative to white:

$$
\Delta Q_{b,\alpha}=Q_{b,\alpha}-Q_{b,\mathrm{white}},
\qquad
\Delta D_{b,\alpha}=D_{b,\alpha}-D_{b,\mathrm{white}}.
$$

For each proposed condition relative to its matched simple-pink anchor, calculate the analogous paired differences. Store them in `paired_effects.csv`.

## 13. Required plots

### 13.1 Baseline

For each quality metric:

- quality versus alpha, with mean and 95% CI;

For each diversity metric:

- diversity versus alpha, with mean and 95% CI;

For every quality/diversity pair:

- Q–D scatter/curve with one labeled point per alpha;
- both PNG and vector PDF/SVG output;
- display block count and metric/checkpoint names in metadata or caption.

### 13.2 Proposed methods

For each fixed anchor alpha:

- quality versus gamma;
- diversity versus gamma;
- Q–D curves for baseline, same-phase floor, and independent-white replenishment;
- visually distinguish raw condition paths from Pareto frontiers;
- label endpoints and intermediate gamma values.

### 13.3 Same-diversity comparison

Raw parameter sweeps may be non-monotonic. Therefore:

1. retain and plot every raw condition point;
2. determine the empirical non-dominated/Pareto frontier separately;
3. sort frontier points by diversity;
4. use piecewise-linear interpolation only within the diversity range shared by baseline and the proposed method;
5. prohibit extrapolation;
6. calculate:

$$
\Delta Q(D^*)=Q_{\mathrm{ours}}(D^*)-Q_{\mathrm{baseline}}(D^*).
$$

If the curves have no adequate overlapping diversity interval, report that the matched-diversity comparison is unavailable rather than forcing a claim.

## 14. Validation and acceptance tests

### 14.1 Noise tests

- same seed produces bitwise-identical cached FP32 CPU base noise;
- four base samples in a block are distinct;
- all alpha/gamma conditions reference the same `base_noise_hash` for a given block and base index;
- `same(alpha, gamma=0)` equals `pink(alpha)` within tolerance;
- `same(alpha, gamma=1)` equals `white(base epsilon)` under the shared normalization policy;
- `ind(alpha, gamma=0)` equals `pink(alpha)`;
- `ind(alpha, gamma=1)` equals cached independent eta;
- IFFT outputs are real and have the expected shape;
- pre/post normalization statistics are finite;
- an empirical PSD test over many synthetic samples confirms the target expected spectra within a stated tolerance.

### 14.2 Generation tests

- a one-prompt, two-condition smoke run produces exactly eight individual images and two 1×4 grids;
- every output image maps to one recorded latent hash;
- rerunning without `--force` creates no duplicate samples;
- supplied latents are not silently overwritten by the model adapter;
- base-index order is preserved in individual files and grids;
- model settings are identical across compared conditions.

### 14.3 Metric tests

- four identical images give DreamSim and LPIPS distances near zero;
- four identical image embeddings give Vendi near one;
- pairwise metrics produce exactly six records for a four-image gallery;
- all quality metrics use the correct prompt;
- grid images are excluded from metric discovery;
- metric records are invalidated if image hash or metric checkpoint changes.

### 14.4 Aggregation tests

- one aggregate row exists per complete condition;
- aggregate values equal manual calculations on a synthetic fixture;
- conditions compared on one curve use the same block set;
- bootstrap resamples blocks, not individual images;
- interpolation never produces values outside the observed overlap;
- dominated points are not included in the reported Pareto frontier.

## 15. Implementation phases and gates

The coding agent may implement the complete pipeline in one task. These phases are internal implementation order and validation gates, not a requirement for eight separate conversations or an elaborate workflow system. The agent should proceed through them sequentially, run the small tests at each gate, and continue automatically unless a real blocker or scientific ambiguity is found.

### Phase 0 — Repository audit and experiment contract

Tasks:

- inspect existing white/pink code, model loading, generation CLI, output layout, dependencies, and tests;
- identify reusable modules and conflicts with this specification;
- document the actual latent injection point and model-specific latent shapes;
- propose the smallest change set;
- create validated configuration schemas and the block manifest format.

Gate:

- no full generation;
- user can inspect the proposed file/module map and one resolved config;
- ambiguities are recorded rather than silently guessed.

### Phase 1 — Deterministic noise library

Tasks:

- implement/correct base sampling, spectral filtering, normalization, hashing, and cache;
- implement all baseline alpha conditions only;
- add endpoint, pairing, shape, determinism, and PSD tests;
- save noise metadata.

Gate:

- all unit tests pass on CPU;
- one block yields four distinct bases and eight correctly paired baseline conditions;
- alpha zero is correctly labeled white.

### Phase 2 — Baseline generation

Tasks:

- complete FLUX.2 Klein adapter smoke test;
- run one prompt, one seed batch, white and alpha 0.5;
- verify eight individual images, two grids, metadata, hashes, and resume behavior;
- after approval, run all eight alpha conditions for a small prompt pilot;
- only then scale to the baseline manifest.

Gate:

- no OOM in the documented device policy;
- no hidden latent resampling;
- expected output counts and provenance pass validation.

### Phase 3 — Metrics and baseline plots

Tasks:

- implement CLIP, HPSv3, DreamSim, LPIPS, and Vendi-CLIP adapters;
- add embedding cache and metric sanity tests;
- create per-image, per-pair, and per-group tables;
- aggregate by fixed alpha across blocks;
- produce quality–alpha, diversity–alpha, and all Q–D baseline plots with confidence intervals.

Gate:

- tables can be regenerated without image generation;
- plots can be regenerated without metric inference;
- all baseline conditions use the same complete-block set;
- a manual small-case metric calculation agrees with stored results.

### Phase 4 — Same-phase PSD floor

Tasks:

- freeze selected anchor alpha values from the baseline rule;
- implement same-phase floor and endpoint/PSD tests;
- run a one-block gamma smoke sweep;
- run all approved blocks with generation settings unchanged;
- evaluate metrics and add same-phase curves.

Gate:

- gamma zero matches the cached pink baseline rather than regenerating it;
- gamma one matches the cached base-white endpoint;
- only the noise method and gamma differ from baseline.

### Phase 5 — Independent-white replenishment

Tasks:

- implement deterministic independent eta cache;
- implement spatial combination and optional frequency-domain equivalence test;
- run endpoint and expected-PSD tests;
- run one-block smoke sweep, then the approved full manifest;
- evaluate metrics and add independent-white curves.

Gate:

- eta is independent of epsilon and fixed across gamma within a block;
- gamma zero reuses/matches the pink anchor;
- finite-sample difference from same-phase is preserved and documented.

### Phase 6 — Final Q–D comparison

Tasks:

- compute raw curves and Pareto frontiers;
- determine shared diversity ranges;
- calculate quality improvement at matched diversity with paired bootstrap confidence intervals;
- produce final comparison figures and machine-readable tables;
- document limitations, including model scope, prompt coverage, seed coverage, and normalization profile.

Gate:

- no extrapolated matched-diversity claims;
- every plotted point traces back to complete block records and immutable manifests;
- baseline and proposed methods use identical prompts, seed batches, model settings, and metric versions.

### Phase 7 — Secondary-model replication

Only after the FLUX pipeline passes all gates:

- run the same approved experiment contract with SDXL-Turbo;
- keep model-specific aggregate curves separate;
- compare whether the direction and shape of the trade-off improvement generalize;
- do not interpret this as a distilled-versus-non-distilled comparison.

## 16. Definition of done

The implementation is complete when:

1. a fresh environment can reproduce a run from a committed config and block manifest;
2. every block uses four saved base noises shared across all matched conditions;
3. baseline white through pink 0.7 generates four individual PNGs and one 1×4 grid per condition;
4. all required raw and grouped metrics are stored with versions and hashes;
5. baseline quality–alpha, diversity–alpha, and Q–D curves are reproducible;
6. both proposed methods pass endpoint and PSD tests;
7. proposed-method curves are produced using otherwise identical generation/evaluation settings;
8. quality improvement at matched diversity is reported only over observed overlap with uncertainty;
9. all data, plots, and claims are traceable to immutable manifests and complete prompt–seed blocks.

## 17. Step-by-step prompts for a coding agent

### Recommended master prompt — direct complete implementation

Use this when the full `noise_init/` repository is available and the agent is authorized to implement the complete pipeline directly:

```text
Read noise_init/docs/INITIAL_NOISE_QD_EXPERIMENT_SPEC.md completely and implement the specified experiment end to end, working only under noise_init/. Start by auditing and reusing the current white/pink generation code; do not create a parallel framework. Keep the implementation compact: one experiment entry point, one noise-method module, one model-adapter module, one metric module, one analysis module, and only the utilities/tests genuinely needed.

Internally follow Phases 0–6 in order. At each phase, run the small acceptance tests before continuing, but continue automatically unless there is a real blocker or a choice that changes the scientific experiment. Implement baseline first, then all required metrics and baseline curves, then same-phase PSD floor, then independent-white replenishment, and finally the matched-diversity Q–D comparison. Apart from initial-noise construction, keep model, prompt, sampler, steps, guidance, resolution, normalization, and metric versions identical across methods.

Target a 24 GB CUDA server: use BF16, generate the four gallery samples sequentially or in a small batch, release the generation model before loading HPSv3, and use CPU offload only if the measured smoke test requires it. Make every expensive stage resumable. Before launching a large prompt grid, complete a one-prompt smoke run and report the command, output count, test results, and estimated storage/runtime. Do not silently substitute metrics, change normalization, extrapolate Q–D curves, or modify files outside noise_init/.
```

The following narrower prompts are optional troubleshooting or staged-review prompts. They are not required if the master prompt is used successfully.

### Prompt A — audit only

```text
Read initial-noise-quality-diversity-specification.md completely. Audit the current repository against Phase 0 only. Do not implement the full experiment yet. Inspect existing white/pink noise construction, model adapters, generation entry points, CLIs, output formats, dependencies, and tests. Identify what can be reused, where supplied latents enter each model, any normalization mismatch, and any gap that would violate the specification. Return: (1) current architecture, (2) spec-to-code mapping, (3) smallest proposed file changes, (4) unresolved risks, and (5) exact Phase 1 acceptance commands. Do not modify files unless needed to record the audit in a clearly named planning document.
```

### Prompt B — deterministic baseline noise

```text
Implement Phase 1 of initial-noise-quality-diversity-specification.md only, using the approved Phase 0 mapping. Reuse existing code where correct. Implement deterministic four-sample base batches, the alpha 0.0–0.7 baseline spectral transform, one explicit normalization policy, hashes/cache/metadata, and CPU tests for determinism, pairing, endpoints, shapes, and expected PSD. Do not load a text-to-image model. Run the tests and report changed files, commands, results, and remaining risks.
```

### Prompt C — baseline generation smoke test

```text
Implement Phase 2 only. First validate the FLUX.2 Klein latent injection point and run one prompt × one seed batch for only white and pink alpha=0.5, generating samples sequentially if required for memory. Save four individual PNGs and one 1×4 grid per condition plus complete provenance. Verify the model does not overwrite supplied latents and reruns resume safely. Do not launch the full alpha/prompt grid until the smoke-test artifacts and validation results are reported.
```

### Prompt D — full baseline

```text
Using the approved Phase 2 pipeline, run/prepare the full baseline implementation for alpha=[0.0,0.1,...,0.7] over the configured prompt–seed block manifest. Keep every model/sampler setting fixed and reuse each block's cached four base noises across all alpha conditions. Make execution resumable. Validate output counts, hashes, complete-block coverage, and one 1×4 grid per condition. Do not implement ours methods yet.
```

### Prompt E — metrics and baseline curves

```text
Implement Phase 3 only. Add versioned adapters for CLIP cosine and HPSv3 quality, DreamSim and LPIPS pairwise diversity, and CLIP-kernel Vendi diversity. Store per-image, per-pair, and per-group records; never score grid images. Aggregate fixed-alpha points over complete prompt–seed blocks, add paired cluster-bootstrap intervals, and generate quality–alpha, diversity–alpha, and every requested Q–D baseline curve. Run metric sanity and aggregation fixture tests. Do not implement the two proposed noise methods.
```

### Prompt F — same-phase method

```text
Implement Phase 4 only for the approved anchor alpha and gamma grid. Use the exact formula and shared normalization in the specification. Add gamma endpoint, phase-preservation, pairing, and expected-PSD tests. Reuse cached pink/white endpoints where applicable. Run a one-block smoke sweep before scaling. Then evaluate with the already frozen metrics and add same-phase plots without changing baseline records.
```

### Prompt G — independent-white method

```text
Implement Phase 5 only. Add deterministic independent eta batches, keep eta fixed across gamma within each prompt–seed block, implement the spatial combination equivalent to the specified Fourier formula, and add endpoint/independence/expected-PSD tests. Run a one-block smoke sweep before scaling, then evaluate with unchanged generation and metric settings and add independent-white plots.
```

### Prompt H — final comparison

```text
Implement Phase 6 only. From immutable per-group records, compute raw Q–D paths, empirical Pareto frontiers, common diversity ranges, and the quality improvement of each proposed method over baseline at matched diversity. Recompute the full comparison within paired cluster-bootstrap replicates, prohibit extrapolation, and output both figures and machine-readable tables. Audit that all compared methods use identical complete prompt–seed blocks, model settings, normalization, and metric versions. Summarize supported conclusions separately from limitations.
```

## 18. Primary references

- DivGen official repository and batch/noise initialization: <https://github.com/anneharrington/divgen>
- FLUX.2 Klein 4B official model card: <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- HPSv3 official implementation: <https://github.com/MizzenAI/HPSv3>
- DreamSim official implementation: <https://github.com/carpedm20/dreamsim>
- LPIPS official implementation: <https://github.com/richzhang/PerceptualSimilarity>
- Vendi Score official implementation: <https://github.com/vertaix/Vendi-Score>
