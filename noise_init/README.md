# Initial-noise quality–diversity experiment

This directory implements the experiment in `docs/INITIAL_NOISE_QD_EXPERIMENT_SPEC.md` without changing the adjacent `divgen/` repository.

Audit/reuse map: `divgen/training/noise_utils.py` supplied the existing 2D `rfft2` pink-noise convention and channelwise normalization; `divgen/models/RewardFlux2Klein.py` confirmed that FLUX.2 receives 4D spatial noise and packs it internally; its SDXL loader supplied the VAE/scheduler choices. The implementation here adds immutable block/noise provenance, matched white-floor methods, standalone evaluation, and block-level Q–D analysis.

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
