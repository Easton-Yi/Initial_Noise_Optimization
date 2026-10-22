"""Keep representative synthetic fixtures and figures for visual review."""
import argparse
from pathlib import Path

from analysis import analyze_run
from noise_validation import run_validation
from test_formal_analysis import fixture
from io_utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory to protect existing examples")
    args.output.mkdir(parents=True)
    outputs = {}
    for name, options in (("finer", {"variant": "sdxl_turbo_full_finer.yaml"}),
                          ("geneval", {"variant": "sdxl_turbo_geneval400_1group.yaml"}),
                          ("single", {"independent": False, "single": True}),
                          ("no_baseline", {"baseline": False}),
                          ("baseline_only", {"same": False, "independent": False, "single": True})):
        run, _ = fixture(args.output / name, **options)
        qd = analyze_run(run)
        spectral = run_validation(run_dir=run)
        outputs[name] = {"qd": str(qd), "spectral": spectral["out_dir"], "status": spectral["status"]}
    write_json(args.output / "examples.json", {"data": "synthetic tensors, images and mocked scores; no real-model claims", "outputs": outputs})
    print(args.output / "examples.json")


if __name__ == "__main__": main()
