# Block manifests

`blocks_pilot.jsonl` is the checked-in two-block smoke/pilot manifest.

`blocks_formal.jsonl` is the checked-in initial formal manifest: the two
approved prompts crossed with `s000`, `s001`, and `s002`, for six complete
prompt × seed-batch blocks. The explicit batch seeds are separated by 100, so
the four required per-block sample seeds (`batch_seed + 0...3`) never overlap.

Each line is JSON with `block_id`, `prompt_id`, `prompt`, `seed_batch_id`, and
`batch_seed`. Review this prompt coverage before a costly run; extend it with
additional prompts or seed batches only by committing a new frozen manifest.
`configs/flux2_klein_full.yaml` deliberately points to this formal file,
preventing a formal run from silently using the pilot blocks.
