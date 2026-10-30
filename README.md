# FloorRefine: Initial-Noise Design for the Quality–Diversity Trade-off in Text-to-Image Diffusion

This repository contains the research code for **FloorRefine**, an inference-time method for improving the quality–diversity trade-off of text-to-image diffusion models without retraining any network. FloorRefine works entirely on the *initial noise* of the sampler:

1. **White-floor pink noise initialisation** reshapes the spectrum of the starting Gaussian noise so that a group of images generated for the same prompt can differ more from one another.
2. **Quality-guided refinement** then makes a small, bounded edit to that initial noise, using a differentiable image-quality reward evaluated on the fully generated image.

The code supports two distilled few-step generators, **SDXL-Turbo** and **FLUX.2 [klein] 4B**, and evaluates every result with independent quality, text-alignment and diversity metrics.

## Contents

- [Research question and scope](#research-question-and-scope)
- [Method overview](#method-overview)
- [Stage 1: White-floor pink noise initialisation](#stage-1-white-floor-pink-noise-initialisation)
- [Stage 2: Quality-guided refinement](#stage-2-quality-guided-refinement)
- [Data and evaluation](#data-and-evaluation)
- [Repository layout and where to start](#repository-layout-and-where-to-start)
- [Supplementary demonstration notebooks](#supplementary-demonstration-notebooks)

## Research question and scope

Few-step text-to-image generators often produce visually similar images for the same prompt when only the random seed changes. Increasing that variation tends to cost image quality, and improving quality tends to reduce variation. This project asks:

> Can the quality–diversity trade-off of a pretrained text-to-image diffusion model be improved by designing and refining only its initial noise, while the generator, text encoders, decoder and reward models stay frozen?

Scope and assumptions:

- **Inference-time only.** No network weight is trained or fine-tuned. The only optimised quantity is an additive offset to the initial noise.
- **Group-level evaluation.** Diversity is a property of a *group* of images generated for one prompt. Every experiment generates groups of four images per prompt and condition.
- **Measured, not assumed.** The method is motivated by a hypothesis about spectral content and diversity. Whether it improves quality, diversity or both in a given setting is determined by the evaluation described below, not guaranteed by construction.

## Method overview

```text
 base Gaussian draw z  (four images per prompt, fixed seeds)
        │
        ▼
 Stage 1: white-floor pink noise construction        noise_init/noise_methods.py
        │   same-phase (main FloorRefine start) or independent-white
        ▼
 reference initial noise z_ref   (normalised)
        │
        ▼
 Stage 2: quality-guided refinement                  denoising_optimization/FloorRefine-*.ipynb
        │   z = z_ref + δ,  ||δ_i|| ≤ 0.10 · ||z_ref,i||  for each image i
        │   repeat: full denoising rollout → VAE decode → HPSv2 reward → gradient w.r.t. δ
        ▼
 selected four-image group (the best evaluated candidate, which may be the unedited baseline)
        │
        ▼
 independent evaluation: HPSv3, CLIP alignment, MANIQA, DreamSim, LPIPS-Alex, CLIP-Vendi
```

The two stages play complementary roles. The initialisation supplies a starting point whose spectral profile is intended to preserve useful within-prompt variation. The refinement then uses quality feedback to improve each group while the edit bound keeps the result close to that starting point. The edit bound limits how far the noise moves; it does not guarantee that the spectrum or the diversity of the starting point is unchanged, so both are measured after refinement.

The complete FloorRefine experiments use the **same-phase** construction with α = 0.9 and γ = 0.05 as the starting point. Stage 1 evaluates both white-floor constructions on their own; the refinement experiments compare the same-phase start against white-noise and pink-noise starts (see [Data and evaluation](#data-and-evaluation)).

## Stage 1: White-floor pink noise initialisation

### Motivation

Pink noise is obtained by attenuating the high-frequency Fourier components of white Gaussian noise. Its stronger low-frequency structure can increase the variation between images generated for the same prompt, but strong filtering can also reduce image quality. A white-floor construction adds back a controlled white contribution, adjusting the spectral balance with the aim of keeping useful diversity while reducing the quality loss. Two constructions are studied:

- **Same-phase white-floor pink noise**
- **Independent-white white-floor pink noise**

### Common definitions

All constructions act on each latent channel of each sample separately, on the 2-D spatial grid of the latent (SDXL-Turbo: 4 × 64 × 64 at 512 × 512; FLUX.2 [klein]: the 128 × 64 × 64 patchified latent at 1024 × 1024, before sequence packing).

- **Frequency radius.** With the real 2-D FFT over the two spatial axes, frequencies are measured in integer FFT-bin units, $f_y = \mathrm{fftfreq}(H)\cdot H$ and $f_x = \mathrm{rfftfreq}(W)\cdot W$, and $r = \sqrt{f_x^2 + f_y^2}$.
- **Pink amplitude response.**

$$H_\alpha(r) = (1 + r)^{-\alpha}, \qquad \alpha \ge 0 .$$

  The expected power spectrum of pink noise is therefore proportional to $(1+r)^{-2\alpha}$. $\alpha = 0$ is white noise.
- **Pink noise.** $\mathrm{pink}_\alpha(z) = \mathcal{F}^{-1}\big[\mathcal{F}[z]\cdot H_\alpha\big]$, where $\mathcal{F}$ is `torch.fft.rfft2` and $\mathcal{F}^{-1}$ is `torch.fft.irfft2`.
- **Normalisation.** $\mathrm{normalise}(x)$ subtracts the mean and divides by the population standard deviation (`unbiased=False`) plus $\epsilon = 10^{-8}$, computed separately for every sample and channel over the spatial axes. This is the `per_sample_per_channel_zero_mean_unit_std` profile used by all formal experiments. Subtracting the mean removes the zero-frequency (DC) component.

### Same-phase white-floor pink noise

$$z_{\mathrm{same}} = \mathrm{normalise}\Big(\mathcal{F}^{-1}\big[\mathcal{F}[z]\cdot\sqrt{(1-\gamma)\,H_\alpha(r)^2 + \gamma}\,\big]\Big), \qquad \gamma \in [0, 1].$$

- The construction mixes **filter power responses**: the squared multiplier is a convex combination of the pink power response $H_\alpha^2$ and a flat response.
- The multiplier is real and positive, so the Fourier phase of the base draw $z$ is preserved at every non-zero frequency (normalisation only alters the DC bin and applies a positive scale).
- $\gamma = 0$ gives the normalised pink condition, and $\gamma = 1$ or $\alpha = 0$ gives the normalised base white draw exactly.

### Independent-white white-floor pink noise

$$z_{\mathrm{ind}} = \mathrm{normalise}\Big(\sqrt{1-\gamma}\;\mathrm{normalise}\big(\mathrm{pink}_\alpha(z)\big) + \sqrt{\gamma}\;\eta\Big), \qquad \eta \sim \mathcal{N}(0, I),\ \eta \perp z .$$

- The construction mixes **noise samples**: the pink component is first normalised to a consistent scale, then mixed in the spatial domain with an independently sampled white draw $\eta$, and the mixture is normalised again.
- Because $\eta$ is independent of $z$, the result generally does not keep the Fourier phases of the original draw.
- $\gamma = 0$ gives the normalised pink condition; $\gamma = 1$ gives $\mathrm{normalise}(\eta)$, which has the same distribution as normalised white noise but is a different sample.

### Interpreting α and γ

- **γ does not mean the same thing in the two constructions.** In the same-phase construction, γ is a floor on the power response that is compared against $(1-\gamma)H_\alpha(r)^2$; for α near 0.9 that pink term is already small at high frequencies, so even a small γ strongly lifts the high-frequency band. In the independent-white construction, γ is the variance share of the white sample relative to a unit-variance pink sample. Equal γ values are therefore not equivalent interventions, and the experiment grids use small γ for same-phase (for example 0.025–0.05) and large γ for independent-white (for example 0.75–0.9).
- **A floor on the filter is not a floor on the sample spectrum.** The floor applies to the expected power response. The periodogram of any single finite sample fluctuates around its expectation at each frequency bin, the independent-white mixture contains a finite-sample cross term between the pink and white components, and the final normalisation rescales the whole spectrum. The spectral diagnostics in `noise_init/` measure these effects directly.
- The endpoints γ = 0 and γ = 1 are covered by unit tests; formal generation grids use interior γ values only.

## Stage 2: Quality-guided refinement

The formal FloorRefine refinement is **quality-only initial-noise refinement**:

| Component | Implementation |
| --- | --- |
| Reference | The constructed, normalised initial noise $z_{\mathrm{ref}}$ of each image is kept fixed. |
| Optimised variable | An additive offset $\delta$ applied before the first denoising step, $z = z_{\mathrm{ref}} + \delta$. The offset is parameterised in units of each image's RMS noise amplitude and starts at zero. |
| Candidate evaluation | Every candidate runs the **complete** denoising rollout (1 step for SDXL-Turbo, 4 steps for FLUX.2 [klein]) with fixed per-image scheduler randomness, then decodes the final latent with the VAE. |
| Reward | Mean **HPSv2** (v2.0) score of the four decoded images, computed differentiably on continuous pixels. HPSv2 is used only here, as optimisation feedback. |
| Gradient | Back-propagated through the reward model, VAE and every denoising step to $\delta$. Gradients are recomputed one image at a time with activation checkpointing to bound memory. |
| Frozen networks | Generator (UNet or transformer), VAE, text encoders and reward model. |
| Edit constraint | After every update, each image's offset is projected so that $\lVert\delta_i\rVert_2 \le 0.10\,\lVert z_{\mathrm{ref},i}\rVert_2$. |
| Optimiser | Adam, learning rate 0.01, 20 updates (21 evaluated candidates, the first being the zero offset). |
| Selection | The whole four-image group is selected by mean HPSv2. The unedited baseline is the initial best candidate, and a later candidate replaces it only if its mean reward is strictly higher, so the selected group may be the baseline itself. The selected candidate is the best evaluated one, not necessarily the last iterate. |

**HPSv3 is used only for independent evaluation.** It never guides the optimisation or the candidate selection. The formal method has no explicit diversity reward; the notebooks retain a switch for a joint quality–diversity objective, but the FloorRefine experiments run with it disabled.

**Denoising steps versus optimisation updates.** A *denoising step* is one evaluation of the generator inside a single image rollout; the number is fixed by the model (1 for SDXL-Turbo, 4 for FLUX.2 [klein]). An *optimisation update* is one Adam step on the noise offset; each update requires a complete rollout, decode and reward evaluation of all four images plus the corresponding backward passes. FloorRefine uses 20 updates for both models.

## Data and evaluation

### Prompts and comparison units

| Experiment | Prompts | Images per prompt and condition | Pairing |
| --- | --- | --- | --- |
| Stage 1 pilot and parameter exploration (`noise_init/`) | Two development prompts (a red fox in a snowy forest; a ceramic teapot on a wooden table), with one or three seed groups each | 4 | All conditions in a block share the same base draws |
| Stage 1 fixed-parameter evaluation (`noise_init/`) | **GenEval400**: 400 prompts from the GenEval prompt set | 4 (one seed group per prompt) | All conditions share the same base draws |
| Refinement and DivGen comparisons (`denoising_optimization/`) | **GenEval50**: 50 prompts selected from GenEval400 | 4 (one seed group per prompt) | All initialisations and methods share the same base draws and scheduler seeds |

- **GenEval400** is drawn from the official GenEval `evaluation_metadata.jsonl` at commit `af4902f24d3ca90ebbb446dd9891a59e0f82725f`, proportionally across the six GenEval categories (largest-remainder allocation, then a seeded SHA-256 ranking within each category, seed 20261009). It contains 58 single-object, 72 two-object, 58 counting, 68 colour, 72 position and 72 colour-attribute prompts, with the prompt text unchanged. The full manifest, including source rows, hashes and category tags, is [`noise_init/manifests/blocks_geneval400_1group.jsonl`](noise_init/manifests/blocks_geneval400_1group.jsonl).
- **GenEval50** is a fixed subset of GenEval400, chosen with the same proportional rule (selection seed 20261023): 7 single-object, 9 two-object, 7 counting, 9 colour, 9 position and 9 colour-attribute prompts. Each GenEval50 prompt keeps its GenEval400 prompt ID, text and batch seed. The list is embedded in each refinement notebook.
- **Seeds.** Each prompt has a batch seed $s_b$; the four images of a group use base draws seeded with $s_b + 0, \dots, s_b + 3$. Every condition and method for that prompt reuses these draws, so comparisons are paired at the level of the prompt and the image index.
- **Statistical unit.** Results are averaged per prompt with equal prompt weight. Confidence intervals come from a paired bootstrap over prompts. With one seed group per prompt, these intervals describe variation across the selected prompts, not repeated-seed variation within a prompt.
- These prompt sets are fixed evaluation subsets drawn from GenEval. They are not an official train/test split, and the full GenEval benchmark is not reproduced.
- **The official GenEval detector-based evaluator is not run.** GenEval supplies the prompts only; GenEval compositional accuracy is not computed.

### Metrics

| Category | Metric | What it measures | Where it is used |
| --- | --- | --- | --- |
| Optimisation feedback | HPSv2 (v2.0) | Differentiable predicted human preference used as the refinement reward; the baseline and selected group scores are recorded | FloorRefine refinement only |
| Predicted human preference | HPSv3 | Predicted human preference score of each image (the mean μ of its predicted reward); averaged over the group | Stage 1 and refinement evaluation |
| Text alignment | CLIP cosine (`openai/clip-vit-large-patch14`) | Cosine similarity between the image and prompt embeddings; a measure of text–image agreement, not of image quality | Stage 1 and refinement evaluation |
| Perceptual diversity | DreamSim distance | Mean DreamSim distance over the six unordered image pairs of a group | Stage 1 and refinement evaluation |
| Perceptual diversity | LPIPS-Alex distance | Mean LPIPS (AlexNet backbone, images resized to 256 × 256) over the six pairs of a group | Stage 1 and refinement evaluation |
| Feature-space diversity | CLIP-Vendi | Vendi score of the four-image cosine kernel built from normalised CLIP image embeddings (effective number of distinct images, between 1 and 4) | Stage 1 and refinement evaluation |
| Additional quality assessment | MANIQA (via `pyiqa`) | No-reference perceptual image-quality prediction; a learned quality assessment, not a sharpness measure | Refinement evaluation, run in its own environment |
| Additional diagnostics | Edge-gradient mean and Laplacian variance | Simple grayscale image statistics recorded alongside the main metrics | Refinement evaluation |
| Additional diagnostics | Same-seed paired change | DreamSim and LPIPS distance between each refined image and its own unrefined baseline (same prompt, same seed); measures how much refinement changed an image, not group diversity | Refinement evaluation |
| Additional diagnostics | Optimisation records | HPSv2 before and after refinement, selected iteration, relative edit size, optimisation time and peak GPU memory | Refinement experiments |
| Noise diagnostics | Spectral validation | Construction checks and radial power spectra of the initial noise tensors themselves | Stage 1 (`noise_init/noise_validation.py`) |

Higher values are better for HPSv3, CLIP cosine and MANIQA, and indicate more diversity for DreamSim, LPIPS and CLIP-Vendi. Stage 1 additionally reports quality–diversity curves, Pareto frontiers and matched-diversity quality differences. Paired same-seed changes are diagnostics: a larger change is not by itself better.

## Repository layout and where to start

| Path | Role |
| --- | --- |
| [`noise_init/`](noise_init/README.md) | Stage 1. Implements white, pink and both white-floor pink noise constructions and evaluates them on their own: generation with SDXL-Turbo and FLUX.2 [klein], metrics, statistics, quality–diversity plots, generation grids and spectral validation of the noise tensors. |
| [`denoising_optimization/`](denoising_optimization/README.md) | Stage 2 and the complete method. Four notebooks run FloorRefine, an ablation over initialisations under the same refinement, and the DivGen comparison, each including generation, evaluation and summary. |
| [`white_vs_pink_noise/`](white_vs_pink_noise/) | Demonstration notebooks for the pink-noise construction (not required for reproduction). |
| [`samePhase_vs_indiWhite_noise/`](samePhase_vs_indiWhite_noise/) | Demonstration notebook comparing the two white-floor constructions (not required for reproduction). |

Detailed instructions:

- [Initialisation experiments and evaluation](noise_init/README.md)
- [FloorRefine refinement and DivGen comparisons](denoising_optimization/README.md)

A suggested reading order for a new reader is: this page → [`noise_init/README.md`](noise_init/README.md) for the noise constructions and the Stage 1 evaluation → [`denoising_optimization/README.md`](denoising_optimization/README.md) for the full method and the comparisons.

Generated images, noise caches, metrics and model caches are written under `outputs/` and `cache/` directories, which are excluded from version control by `.gitignore`.

## Supplementary demonstration notebooks

These notebooks build intuition for the noise constructions on small synthetic tensors. They do not load a diffusion model and are not part of the formal experiment pipeline. Run them from their own directory with any Python environment that has PyTorch, NumPy and Matplotlib (for example the metrics environment described in [`noise_init/README.md`](noise_init/README.md#environments)).

### `white_vs_pink_noise/`

| File | What it shows |
| --- | --- |
| [`noise_init.ipynb`](white_vs_pink_noise/noise_init.ipynb) | **Start here.** Builds pink noise from a single 128 × 128 white draw for α ∈ {0, 0.2, 0.5, 1.0} and shows that pink noise is the same draw with a reweighted spectrum, not a new sample. Plots the spatial appearance (fine static becoming smooth blobs), the 2-D log-magnitude spectrum, and the radially averaged power spectrum against the $(1+r)^{-2\alpha}$ shape. Saves `white_vs_pink_psd_curve.png` and `white_vs_pink_spatial_spectrum.png`. |
| [`fft_pipeline.ipynb`](white_vs_pink_noise/fft_pipeline.ipynb) | A step-by-step walkthrough of the FFT pipeline behind pink noise: the `rfft2` frequency grid, the radius $r$ in FFT-bin units, the filter $H_\alpha(r)$, spectra before and after filtering, and radial power spectra before and after normalisation. Its first code cell checks the inline implementation against `noise_init/noise_methods.py`. |
| `white_vs_pink_*.png` | Figures saved by the notebooks for presentations. |

### `samePhase_vs_indiWhite_noise/`

| File | What it shows |
| --- | --- |
| [`noise_init.ipynb`](samePhase_vs_indiWhite_noise/noise_init.ipynb) | Compares the same-phase and independent-white constructions at α = 0.9 with method-specific γ sweeps. It checks the γ = 0 and γ = 1 identities numerically, shows spatial appearance and 2-D spectra, compares radially averaged spectra with each construction's component prediction (including the finite-sample cross term of the independent-white mixture), and plots the correlation of each construction with the pink component of the shared base draw as γ grows. |

The final section of `samePhase_vs_indiWhite_noise/noise_init.ipynb` reconstructs both constructions from a real cached noise block (`noise_cache/<block-id>/` inside a Stage 1 run). It needs an existing Stage 1 run; set `NOISE_CACHE_DIR` to a block directory of that run before executing it. All other sections are self-contained.
