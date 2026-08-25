# Initial-noise quality–diversity experiment

This directory is a standalone implementation of the experiment in `docs/INITIAL_NOISE_QD_EXPERIMENT_SPEC.md`; it does not import code, models, or configuration from another checkout.

The implementation was informed by the 2D `rfft2` pink-noise convention and verified FLUX.2 latent-packing behavior in the prior DivGen work, but implements these locally. It adds immutable block/noise provenance, matched white-floor methods, standalone evaluation, and block-level Q–D analysis. Model weights are cached locally in `noise_init/cache/` by default.

`alpha` is the amplitude-spectrum exponent: pink expected PSD is proportional to `(1+r)^(-2 alpha)`. All final conditions use the configured common normalization profile by default.

Run from `Initial_Noise_Optimization/noise_init` using `requirements.txt`. HPSv3 may need its own official metric-stage environment because its published Transformer pin can differ from the Diffusers generation environment; generation artefacts remain immutable and can be scored there later.

```bash
python3 run_experiment.py --config configs/flux2_klein.yaml --stage validate
python3 run_experiment.py --config configs/flux2_klein.yaml --stage generate --prompt "A photo of a red fox in snow" --batch-seed 10000 --conditions white,pink:0.5 --run-id flux_smoke
python3 run_experiment.py --config configs/flux2_klein.yaml --stage metrics --run-id flux_smoke
python3 run_experiment.py --config configs/flux2_klein.yaml --stage analyze --run-id flux_smoke
```

Enable an approved anchor/gamma sweep by setting `same_phase_floor.enabled` and/or `independent_white.enabled` in a new frozen config before generation. `--force` can recreate derived generation/metric outputs but cannot replace a cached source-noise batch or an immutable run configuration.

CPU acceptance tests:

```bash
python3 -m unittest discover -s tests -v
```
