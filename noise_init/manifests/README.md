# Block manifests

`blocks_pilot.jsonl` is the checked-in two-block smoke/pilot manifest.

Before a formal run, create `blocks_formal.jsonl` from the approved prompt ×
seed-batch design and keep it immutable for that run. Each line must be JSON
with `block_id`, `prompt_id`, `prompt`, `seed_batch_id`, and `batch_seed`.
`configs/flux2_klein_full.yaml` deliberately points to this required formal
file. Validation fails if it has not been created or is empty, preventing a
formal run from silently using the pilot blocks.
