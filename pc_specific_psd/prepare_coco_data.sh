#!/usr/bin/env bash
set -euo pipefail

# Prepare a fixed subset of the official COCO 2017 training dataset.
#
# This script:
# - downloads the official COCO 2017 annotations;
# - deterministically selects COUNT train images;
# - downloads the selected images;
# - verifies that every image is readable;
# - computes a SHA-256 checksum for every image;
# - writes the dataset manifest under pc_specific_psd/manifests/.
#
# Example:
#
#   mkdir -p /workspace/datasets/coco_2000
#
#   bash \
#     /workspace/Initial_Noise_Optimization/pc_specific_psd/prepare_coco_2000.sh \
#     /workspace/datasets/coco_2000 \
#     /workspace/Initial_Noise_Optimization

DATA_ROOT="${1:-/workspace/datasets/coco_2000}"
REPO_ROOT="${2:-/workspace/Initial_Noise_Optimization}"

SEED=20260829
COUNT=2000

ANNOTATION_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_ZIP="${DATA_ROOT}/annotations_trainval2017.zip"
ANNOTATION_DIR="${DATA_ROOT}/annotations"
ANNOTATION_JSON="${ANNOTATION_DIR}/instances_train2017.json"

IMAGE_DIR="${DATA_ROOT}/train2017_subset_${COUNT}_seed${SEED}"
SELECTED_JSON="${DATA_ROOT}/selected_images_${COUNT}_seed${SEED}.json"
URL_LIST="${DATA_ROOT}/selected_urls_${COUNT}_seed${SEED}.txt"

MANIFEST_DIR="${REPO_ROOT}/pc_specific_psd/manifests"
MANIFEST_PATH="${MANIFEST_DIR}/coco_train2017_${COUNT}_seed${SEED}.json"

if [ ! -d "${REPO_ROOT}/pc_specific_psd" ]; then
    echo "ERROR: pc_specific_psd directory was not found under:"
    echo "  ${REPO_ROOT}"
    echo
    echo "Pass the repository root as the second argument."
    exit 1
fi

for command_name in wget unzip python3 sha256sum; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "ERROR: required command is unavailable: ${command_name}"
        exit 1
    fi
done

if ! python3 -c "from PIL import Image" >/dev/null 2>&1; then
    echo "ERROR: Pillow is not installed in the current Python environment."
    echo "Install it with:"
    echo "  python3 -m pip install Pillow"
    exit 1
fi

mkdir -p \
    "${DATA_ROOT}" \
    "${IMAGE_DIR}" \
    "${MANIFEST_DIR}"

# Download and extract the official annotations only when the required
# instances_train2017.json file is not already available.
#
# wget --continue resumes an existing partial zip file.
if [ ! -f "${ANNOTATION_JSON}" ]; then
    echo "Downloading official COCO 2017 annotations..."

    wget \
        --continue \
        --directory-prefix="${DATA_ROOT}" \
        "${ANNOTATION_URL}"

    echo "Checking annotation archive..."

    unzip -t "${ANNOTATION_ZIP}" >/dev/null

    echo "Extracting annotations..."

    unzip \
        -q \
        -o \
        "${ANNOTATION_ZIP}" \
        -d "${DATA_ROOT}"
fi

if [ ! -f "${ANNOTATION_JSON}" ]; then
    echo "ERROR: annotation file was not produced:"
    echo "  ${ANNOTATION_JSON}"
    exit 1
fi

echo "Selecting ${COUNT} images using seed ${SEED}..."

python3 - \
    "${ANNOTATION_JSON}" \
    "${SELECTED_JSON}" \
    "${URL_LIST}" \
    "${SEED}" \
    "${COUNT}" <<'PY'
import json
import random
import sys
from pathlib import Path

annotation_path = Path(sys.argv[1])
selected_path = Path(sys.argv[2])
url_path = Path(sys.argv[3])
seed = int(sys.argv[4])
count = int(sys.argv[5])

data = json.loads(annotation_path.read_text(encoding="utf-8"))
images = list(data["images"])

if len(images) < count:
    raise RuntimeError(
        f"Annotation file contains only {len(images)} images; "
        f"{count} are required."
    )

# The official annotation file has a fixed order. Applying a seeded shuffle
# therefore produces a deterministic subset.
random.Random(seed).shuffle(images)

selected = []

for original_record in images[:count]:
    record = dict(original_record)

    coco_url = record.get("coco_url")

    if not coco_url:
        raise RuntimeError(
            f"Image record {record.get('id')} has no coco_url."
        )

    record["download_url"] = coco_url.replace(
        "http://",
        "https://",
        1,
    )

    selected.append(record)

if len(selected) != count:
    raise RuntimeError(
        f"Expected {count} selected records, obtained {len(selected)}."
    )

selected_path.parent.mkdir(parents=True, exist_ok=True)

selected_path.write_text(
    json.dumps(selected, indent=2) + "\n",
    encoding="utf-8",
)

url_path.write_text(
    "\n".join(record["download_url"] for record in selected) + "\n",
    encoding="utf-8",
)

print(f"Selected images: {len(selected)}")
print(f"Selection seed: {seed}")
print(f"Selected-record file: {selected_path}")
print(f"URL list: {url_path}")
PY

echo "Downloading selected COCO images..."

# Existing complete files are retained. Partial files can be resumed by
# rerunning this script because wget uses --continue.
wget \
    --continue \
    --no-verbose \
    --directory-prefix="${IMAGE_DIR}" \
    --input-file="${URL_LIST}"

echo "Verifying images and building the dataset manifest..."

python3 - \
    "${SELECTED_JSON}" \
    "${IMAGE_DIR}" \
    "${MANIFEST_PATH}" \
    "${COUNT}" \
    "${SEED}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

selected_path = Path(sys.argv[1])
image_dir = Path(sys.argv[2]).resolve()
manifest_path = Path(sys.argv[3]).resolve()
expected_count = int(sys.argv[4])
seed = int(sys.argv[5])

selected = json.loads(
    selected_path.read_text(encoding="utf-8")
)

entries = []
missing = []
invalid = []

for record in selected:
    image_path = image_dir / record["file_name"]

    if not image_path.is_file():
        missing.append(str(image_path))
        continue

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as error:
        invalid.append(
            {
                "path": str(image_path),
                "error": str(error),
            }
        )
        continue

    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    entries.append(
        {
            "image_id": (
                f"coco_train2017:"
                f"{int(record['id']):012d}"
            ),
            "path": str(image_path),
            "sha256": digest.hexdigest(),
        }
    )

if missing:
    raise RuntimeError(
        f"{len(missing)} images are missing. "
        f"Rerun the script to resume the downloads. "
        f"First missing image: {missing[0]}"
    )

if invalid:
    first = invalid[0]

    raise RuntimeError(
        f"{len(invalid)} downloaded images are invalid or incomplete. "
        f"Delete the invalid files and rerun the script. "
        f"First invalid image: {first['path']}; "
        f"error: {first['error']}"
    )

if len(entries) != expected_count:
    raise RuntimeError(
        f"Expected {expected_count} manifest entries, "
        f"obtained {len(entries)}."
    )

if len({entry["image_id"] for entry in entries}) != expected_count:
    raise RuntimeError("Duplicate image IDs detected.")

if len({entry["path"] for entry in entries}) != expected_count:
    raise RuntimeError("Duplicate image paths detected.")

if len({entry["sha256"] for entry in entries}) != expected_count:
    raise RuntimeError(
        "Byte-identical duplicate images detected."
    )

manifest = {
    "source": (
        f"coco_train2017_official_subset_"
        f"{expected_count}_seed{seed}"
    ),
    "entries": entries,
}

manifest_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

manifest_path.write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)

print("Dataset preparation: PASS")
print(f"Images: {len(entries)}")
print(f"Image directory: {image_dir}")
print(f"Manifest: {manifest_path}")
PY

echo "Manifest SHA-256:"

sha256sum "${MANIFEST_PATH}"

echo
echo "COCO subset preparation completed successfully."
echo
echo "Next, run:"
echo
echo "  python3 -m pc_specific_psd validate-config \\"
echo "    --config pc_specific_psd/configs/sdxl_turbo_pca_coco_server_v1.yaml"
echo
echo "Then run build-basis with --dry-run before the real basis build."
echo 
echo "Run the real build-basis if all set:"
echo
echo "  python3 -m pc_specific_psd build-basis \\"
echo "    --config pc_specific_psd/configs/sdxl_turbo_pca_coco_server_v1.yaml"