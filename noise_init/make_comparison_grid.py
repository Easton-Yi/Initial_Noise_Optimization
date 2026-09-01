#!/usr/bin/env python3
"""Standalone generation-quality comparison grid for one (prompt_id, alpha, gamma).

Builds one PNG with two side-by-side galleries from an already-generated run:

  Left grid (6 rows, one seed batch): white, then pink alpha=0.1..0.5.
  Right grid (6 rows, three seed batches): top 3 rows are "ours" same-phase at
  the given alpha/gamma for seeds s000..s002; bottom 3 rows are independent-white
  at the same alpha/gamma for the same three seeds.

Each row reuses the 1x4 gallery PNG already written by run_experiment.py's
generate stage, so this script never loads a model and never samples noise.

run:
python3 make_comparison_grid.py   --config configs/sdxl_turbo_full_finer.yaml   --run-id sdxl_turbo_full_finer   --prompt-id p001   --alpha 0.9 --gamma 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from io_utils import alpha_token, condition_id, read_jsonl

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

REGULAR_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

TITLE_BLUE = (20, 70, 200)
OUTER_MARGIN_H = 40
OUTER_MARGIN_RIGHT_EXTRA = 160
OUTER_MARGIN_V = 100


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _font_regular(size: int) -> ImageFont.FreeTypeFont:
    for path in REGULAR_FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return _font(size)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("YAML root must be a mapping")
    return config


def run_dir_for(config: dict, config_path: Path, run_id: str | None) -> Path:
    root = (config_path.parent.parent / config.get("paths", {}).get("outputs_root", "outputs")).resolve()
    return root / (run_id or config["experiment"]["name"])


def blocks_for_prompt(run_dir: Path, prompt_id: str) -> list[dict]:
    blocks = [block for block in read_jsonl(run_dir / "blocks.jsonl") if block["prompt_id"] == prompt_id]
    if not blocks:
        raise ValueError(f"No blocks with prompt_id={prompt_id!r} in {run_dir / 'blocks.jsonl'}")
    return sorted(blocks, key=lambda block: block["seed_batch_id"])


def row_image_path(run_dir: Path, model_id: str, block_id: str, family: str, alpha: float, gamma: float | None) -> Path:
    return run_dir / "generations" / model_id / block_id / condition_id(family, alpha, gamma) / "grid_1x4.png"


def load_row(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing generated gallery: {path}\n"
            "Run the 'generate' stage for this condition first (see --conditions in run_experiment.py)."
        )
    return Image.open(path).convert("RGB")


def label_row(
    row_image: Image.Image,
    label: str,
    *,
    label_width: int,
    font: ImageFont.FreeTypeFont,
    sub_label: str | None = None,
    sub_font: ImageFont.FreeTypeFont | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", (label_width + row_image.width, row_image.height), "white")
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), label, font=font)
    label_h = bbox[3] - bbox[1]
    if sub_label is not None:
        sub_font = sub_font or font
        sub_bbox = draw.textbbox((0, 0), sub_label, font=sub_font)
        sub_h = sub_bbox[3] - sub_bbox[1]
        gap = 6
        total_h = label_h + gap + sub_h
        top = (row_image.height - total_h) // 2
        draw.text((12, top - bbox[1]), label, fill="black", font=font)
        draw.text((12, top + label_h + gap - sub_bbox[1]), sub_label, fill=(70, 70, 70), font=sub_font)
    else:
        text_y = (row_image.height - label_h) // 2 - bbox[1]
        draw.text((12, text_y), label, fill="black", font=font)
    canvas.paste(row_image, (label_width, 0))
    return canvas


def stack_rows(rows: list[Image.Image], *, row_gap: int, separator_after: int | None = None) -> Image.Image:
    width = rows[0].width
    height = sum(row.height for row in rows) + row_gap * (len(rows) - 1)
    if separator_after is not None:
        height += row_gap  # extra breathing room around the mid-grid separator
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for index, row in enumerate(rows):
        canvas.paste(row, (0, y))
        y += row.height + row_gap
        if index == separator_after:
            y += row_gap
    return canvas


def centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    font: ImageFont.FreeTypeFont,
    box_left: int,
    box_width: int,
    y: int,
    fill,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    x = box_left + (box_width - text_width) // 2 - bbox[0]
    draw.text((x, y - bbox[1]), text, fill=fill, font=font)


def text_height(font: ImageFont.FreeTypeFont) -> int:
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


def build_grid(
    run_dir: Path,
    model_id: str,
    prompt_id: str,
    alpha: float,
    gamma: float,
    baseline_alphas: list[float],
    baseline_seed_batch: str | None,
    label_width: int,
    row_gap: int,
    column_gap: int,
) -> Image.Image:
    blocks = blocks_for_prompt(run_dir, prompt_id)
    baseline_block = next((b for b in blocks if b["seed_batch_id"] == baseline_seed_batch), blocks[0]) if baseline_seed_batch else blocks[0]
    prompt_text = baseline_block["prompt"]

    label_font = _font(44)
    seed_font = _font(34)
    header_font = _font(48)
    main_title_font = _font(76)
    footer_font = main_title_font

    # --- left grid: baseline alpha sweep on one seed batch ---
    left_rows = []
    for a in baseline_alphas:
        label = "white" if a == 0.0 else f"pink α={alpha_token(a).replace('p', '.')}"
        image = load_row(row_image_path(run_dir, model_id, baseline_block["block_id"], "baseline", a, None))
        left_rows.append(label_row(image, label, label_width=label_width, font=label_font))
    left_grid = stack_rows(left_rows, row_gap=row_gap)

    # --- right grid: ours (same-phase / independent-white) across 3 seeds ---
    right_rows = []
    for family, tag in (("same_phase", "same-phase"), ("independent_white", "independent")):
        for block in blocks:
            image = load_row(row_image_path(run_dir, model_id, block["block_id"], family, alpha, gamma))
            right_rows.append(
                label_row(
                    image,
                    tag,
                    label_width=label_width,
                    font=label_font,
                    sub_label=f"seed={block['batch_seed']}",
                    sub_font=seed_font,
                )
            )
    right_grid = stack_rows(right_rows, row_gap=row_gap, separator_after=len(blocks) - 1)

    left_header = "pure white and pink noise init"
    right_header = f"ours: same phase vs. independent white(alpha={alpha}, gamma={gamma})"

    header_height = text_height(header_font) + 20
    grids_width = left_grid.width + column_gap + right_grid.width
    grids_height = max(left_grid.height, right_grid.height)

    headers = Image.new("RGB", (grids_width, header_height), "white")
    header_draw = ImageDraw.Draw(headers)
    centered_text(header_draw, left_header, font=header_font, box_left=0, box_width=left_grid.width, y=10, fill="black")
    centered_text(
        header_draw,
        right_header,
        font=header_font,
        box_left=left_grid.width + column_gap,
        box_width=right_grid.width,
        y=10,
        fill="black",
    )

    grids = Image.new("RGB", (grids_width, grids_height), "white")
    grids.paste(left_grid, (0, 0))
    grids.paste(right_grid, (left_grid.width + column_gap, 0))

    body_gap = 20
    body_height = header_height + body_gap + grids_height
    body = Image.new("RGB", (grids_width, body_height), "white")
    body.paste(headers, (0, 0))
    body.paste(grids, (0, header_height + body_gap))

    # --- main title (real prompt text), bold blue, centered ---
    title_text = f'"{prompt_text}"'
    title_height = text_height(main_title_font) + 90
    footer_text = model_id
    footer_height = text_height(footer_font) + 100

    content_width = grids_width
    content_height = title_height + body_height + footer_height
    content = Image.new("RGB", (content_width, content_height), "white")
    content_draw = ImageDraw.Draw(content)
    centered_text(content_draw, title_text, font=main_title_font, box_left=0, box_width=content_width, y=10, fill=TITLE_BLUE)
    content.paste(body, (0, title_height))
    centered_text(
        content_draw,
        footer_text,
        font=footer_font,
        box_left=0,
        box_width=content_width,
        y=title_height + body_height + 40,
        fill=TITLE_BLUE,
    )

    # --- outer margin so nothing touches the image border (extra breathing room on the right) ---
    canvas_width = content_width + OUTER_MARGIN_H + (OUTER_MARGIN_H + OUTER_MARGIN_RIGHT_EXTRA)
    canvas = Image.new("RGB", (canvas_width, content_height + 2 * OUTER_MARGIN_V), "white")
    canvas.paste(content, (OUTER_MARGIN_H, OUTER_MARGIN_V))
    return canvas


def parse_float_list(value: str) -> list[float]:
    return [float(token) for token in value.split(",") if token.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config for the run, e.g. configs/sdxl_turbo_full_finer.yaml")
    parser.add_argument("--run-id", help="Defaults to experiment.name in the config")
    parser.add_argument("--prompt-id", required=True, help="e.g. p000")
    parser.add_argument("--alpha", required=True, type=float, help="ours alpha (same_phase / independent_white)")
    parser.add_argument("--gamma", required=True, type=float, help="ours gamma (same_phase / independent_white)")
    parser.add_argument("--baseline-alphas", default="0.0,0.1,0.2,0.3,0.4,0.5", help="comma-separated baseline alpha sweep for the left grid")
    parser.add_argument("--baseline-seed-batch", help="seed_batch_id used for the left grid (default: first block for this prompt)")
    parser.add_argument("--label-width", type=int, default=360)
    parser.add_argument("--row-gap", type=int, default=6)
    parser.add_argument("--column-gap", type=int, default=60)
    parser.add_argument("--output", help="Output PNG path (default under <run_dir>/analysis/generation_grids/)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    run_dir = run_dir_for(config, config_path, args.run_id)
    model_id = config["model"]["adapter"]
    baseline_alphas = parse_float_list(args.baseline_alphas)

    grid = build_grid(
        run_dir=run_dir,
        model_id=model_id,
        prompt_id=args.prompt_id,
        alpha=args.alpha,
        gamma=args.gamma,
        baseline_alphas=baseline_alphas,
        baseline_seed_batch=args.baseline_seed_batch,
        label_width=args.label_width,
        row_gap=args.row_gap,
        column_gap=args.column_gap,
    )

    output = Path(args.output) if args.output else run_dir / "analysis" / "generation_grids" / f"{args.prompt_id}_alpha{alpha_token(args.alpha)}_gamma{alpha_token(args.gamma)}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output, format="PNG")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
