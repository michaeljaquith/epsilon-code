# Backup: Original Scenario Runs (2026-04-16)

## What these files are

Three scenario benchmark result files from the first complete pass of the
scenarios A–E benchmark, run during initial paper development:

- `scenarios_gpt-4o.json` — GPT-4o (gpt-4o-2024-08-06), OpenAI API
- `scenarios_gpt-4o-mini.json` — GPT-4o-mini, OpenAI API
- `scenarios_deepseek-ai_DeepSeek-V3.json` — DeepSeek V3, Together AI API

## How they were made

Each file was produced by `benchmark_scenarios.py` — 36 prompts total:
- 10 LOW (pure logic, FP test)
- 20 HIGH (API/library version-split tasks)
- 6 Scenario E (auth module, per-function body mode)

Prompts high_01–high_04 correspond to paper Scenarios A–D respectively.
Temperature=0, logprobs=True, top_logprobs=5 throughout.

## Why they were made

These were the primary scenario runs used to write §4.1 of the paper.
The §4.1 qualitative narrative (ε values and tier labels for Scenarios A–D)
was based on a separate earlier exploratory run, not directly from these files.
When we cross-checked in April 2026, Scenarios C and D showed a discrepancy:

| Scenario | Paper §4.1 (GPT-4o) | This file (GPT-4o) |
|---|---|---|
| C (SQLAlchemy) | ε=0.435, FLAGGED | ε=0.902, PAUSED |
| D (FastAPI)    | ε=0.812, PAUSED  | ε=0.528, FLAGGED |

The tiers for C and D are swapped relative to what the paper states. The
paper values appear to come from an earlier exploratory session whose raw
output was not saved as a structured file.

## Why we made new runs

To resolve the discrepancy and get a single authoritative set of numbers
across all four models before finalising §4.1, we ran fresh benchmarks:

- Added GPT-4-turbo (present in the calibration table but missing from
  scenario files)
- Re-ran all 36 prompts × 4 models to get consistent provenance
- New files live in `results/` as `scenarios_<model>.json`

The new runs are what the final paper's §4.1 multi-model table is based on.
These backups are retained for audit trail only.
