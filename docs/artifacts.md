# Artifacts

| File | Made by | Notes |
|---|---|---|
| `runs/final/v2/raw.jsonl.gz` | `specdec final` | Every timed request. The only final run quoted |
| `runs/final/v2/cold.jsonl.gz` | `specdec final` | First request in a fresh process |
| `runs/final/v2/manifest.json` | `specdec final` | Commit, input hashes, config, hardware, command |
| `runs/final/v2/report/` | `specdec report` | Tables, plots, decision and page text |
| `runs/final/v1/` | `specdec final` | Discarded for a seed bug. Kept as a record |
| `runs/validation/v1/` | `specdec validate` | Sweep and selection |
| `config/frozen.json` | `benchmarks/freeze_plan.py` | The frozen arms and rule |
| `runs/law_audit/mps.json` | `benchmarks/law_audit.py` | Exact law on real weights |
| `runs/numeric_gap/mps.json` | `benchmarks/numeric_gap.py` | float16 and float32 distributions |
| `runs/profile/` | `benchmarks/profile_passes.py` | Cost of one model pass |
| `runs/distill/` | `specdec teacher`, `specdec distill` | Teacher and training cost |
| `site/index.html` | `specdec demo` | The offline replay |

Trained draft weights are not in Git. `specdec distill` retrains them, and
`runs/final/v2/manifest.json` records the SHA-256 of the weights the run used.
