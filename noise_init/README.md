# Initial-noise quality–diversity experiment

This directory is a standalone implementation of the experiment in `docs/INITIAL_NOISE_QD_EXPERIMENT_SPEC.md`; it does not import code, models, or configuration from another checkout.

The implementation was informed by the 2D `rfft2` pink-noise convention and verified FLUX.2 latent-packing behavior in the prior DivGen work, but implements these locally. It adds immutable block/noise provenance, matched white-floor methods, standalone evaluation, and block-level Q–D analysis. Model weights are cached locally in `noise_init/cache/` by default.

`alpha` is the amplitude-spectrum exponent: pink expected PSD is proportional to `(1+r)^(-2 alpha)`. All final conditions use the configured common normalization profile by default.

## Setup

Run from `Initial_Noise_Optimization/noise_init` using `requirements.txt`. FLUX is gated in many environments, so export a Hugging Face token before the first model download. HPSv3 may need its own official metric-stage environment because its published Transformer pin can differ from the Diffusers generation environment; generation artefacts remain immutable and can be scored there later.

```bash
cd Initial_Noise_Optimization/noise_init
pip install -r requirements.txt
export HF_TOKEN=...  # required if the FLUX checkpoint is gated for this account
python3 run_experiment.py --config configs/flux2_klein.yaml --stage validate
```

## 1. One-prompt smoke test

This is only a preflight check: it generates exactly eight PNG images (four white and four pink `alpha=0.5`) plus two 1×4 grids. It is not a baseline result and must not be used for the final Q–D curve.

```bash
python3 run_experiment.py --config configs/flux2_klein.yaml --stage generate --prompt "A photo of a red fox in snow" --batch-seed 10000 --conditions white,pink:0.5 --run-id flux_smoke
python3 run_experiment.py --config configs/flux2_klein.yaml --stage metrics --run-id flux_smoke
```

Confirm that the smoke output contains the expected image/hash/provenance records before continuing. Do not run `analyze` for this deliberately incomplete two-condition smoke run.

## 2. FLUX.2 Klein formal baseline: white through pink 0.7

`configs/flux2_klein.yaml` selects the primary model, **FLUX.2 Klein 4B, BF16, 1024×1024, four inference steps**, and enables all eight matched baseline conditions:

```text
white (= alpha 0.0), pink alpha = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7
```

Before the formal run, replace or extend `manifests/blocks.jsonl` with the approved prompt × seed-batch blocks. The checked-in two-block manifest is a pilot example. Every final block must be present before launch; use multiple `seed_batch_id` values per prompt for the final experiment.

```bash
# No --conditions: this intentionally runs every white/pink baseline condition.
python3 run_experiment.py --config configs/flux2_klein.yaml --stage generate --run-id flux2_baseline_pilot
python3 run_experiment.py --config configs/flux2_klein.yaml --stage metrics --run-id flux2_baseline_pilot
python3 run_experiment.py --config configs/flux2_klein.yaml --stage analyze --run-id flux2_baseline_pilot
```

The baseline outputs are under `outputs/flux2_baseline_pilot/`. Inspect the aggregate curves to select an anchor alpha using the aggregate failure/trade-off region—not one prompt.

## 3. Formal FLUX.2 comparison: baseline + both proposed initial-noise methods

The final comparison must live in **one fresh run directory**, so all methods have the same immutable manifest, normalization, model/sampler settings, prompt blocks, and metric versions. Do not append proposed methods to `flux2_baseline_pilot`: the run manifest intentionally prevents changing a configuration after generation.

After choosing the aggregate anchor alpha from the pilot, make a copy of the baseline config and freeze it, for example:

```bash
cp configs/flux2_klein.yaml configs/flux2_klein_full_alpha_0p5.yaml
```

Edit the copied YAML as follows (replace `0.5` with the approved aggregate anchor if different, and give `experiment.name` a new final-run name):

```yaml
same_phase_floor:
  enabled: true
  alpha_values: [0.5]
  gamma_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

independent_white:
  enabled: true
  alpha_values: [0.5]
  gamma_values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
```

Then run the **entire** final grid—eight baselines, eleven same-phase conditions, and eleven independent-white conditions—over the frozen formal block manifest:

```bash
python3 run_experiment.py --config configs/flux2_klein_full_alpha_0p5.yaml --stage generate --run-id flux2_full_alpha_0p5
python3 run_experiment.py --config configs/flux2_klein_full_alpha_0p5.yaml --stage metrics --run-id flux2_full_alpha_0p5
python3 run_experiment.py --config configs/flux2_klein_full_alpha_0p5.yaml --stage analyze --run-id flux2_full_alpha_0p5
```

This produces raw condition records, per-image/pair/group metric tables, baseline and method curves, Pareto frontiers, and matched-diversity comparisons under `outputs/flux2_full_alpha_0p5/`. Do not use `--conditions` for this formal run: it is only for controlled smoke subsets.

`--force` can recreate derived metric outputs but cannot replace a cached source-noise batch, generated PNG, or immutable run configuration. Use a new `--run-id` for a deliberately new experiment.

CPU acceptance tests:

```bash
python3 -m unittest discover -s tests -v
```
