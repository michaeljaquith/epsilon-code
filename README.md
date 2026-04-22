# epsilon — Token-Level Uncertainty for LLM Code Generation

Implementation accompanying the paper:

> **Beyond Hallucination Detection: Measuring Consequential Uncertainty in LLM-Generated Code**
> Michael Jaquith, arXiv 2026

## What it does

`epsilon` computes a per-token uncertainty score (ε) from the log-probability
distribution returned by the LLM during a single generation. It filters out
cosmetic naming uncertainty (using AST analysis) and surfaces the decisions
where the model was genuinely split — API version choices, library selection,
architectural patterns — that determine whether generated code is correct and
compatible.

```
ε = 0.00  →  COMPLETE   model was certain; no review needed
ε = 0.35  →  FLAGGED    model had a notable decision; review flagged tokens
ε = 0.73  →  PAUSED     significant uncertainty; review before proceeding
ε = 0.96  →  ABORTED    model was nearly random; re-prompt
```

## Quick start

```bash
pip install openai python-dotenv rich
export OPENAI_API_KEY=sk-...
python demos/demo_scenario_a.py
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.11+. The `epsilon` package is in the `epsilon/` directory;
import it directly or add the repo root to your `PYTHONPATH`.

## Library usage

```python
from openai import OpenAI
from epsilon.core import EpsilonWrapper

client  = OpenAI()
wrapper = EpsilonWrapper(client, log_path="epsilon.log")

result = wrapper.generate_code(
    prompt="Write a Python function to charge a customer via Stripe",
    context="Python 3.11, Stripe SDK installed",
)

print(result.status)          # PAUSED
print(result.epsilon_file)    # 0.914
print(result.flags[0])        # ".Ch" (line 4) ε=0.73 — alternative ".Payment" was nearly as likely (P=0.47)
```

## Demos

Five calibrated scenarios demonstrating different failure classes:

| Script | Failure class | Expected status |
|---|---|---|
| `demos/demo_scenario_a.py` | Deprecated API (Stripe Charge vs PaymentIntent) | PAUSED |
| `demos/demo_scenario_b.py` | Confident wrongness (OpenAI SDK v0 vs v1) | PAUSED |
| `demos/demo_scenario_c.py` | Syntax split (SQLAlchemy 1.x vs 2.0) | FLAGGED |
| `demos/demo_scenario_d.py` | Compatibility mismatch (FastAPI async/sync) | PAUSED |
| `demos/demo_scenario_e.py` | Multi-decision module (auth, 6 functions) | PAUSED |

## Benchmark

```bash
# 30-prompt benchmark (10 LOW + 20 HIGH) across models
python benchmark/benchmark_calibration.py --model gpt-4o
python benchmark/benchmark_calibration.py --model gpt-4o-mini
python benchmark/benchmark_calibration.py --model gpt-4-turbo

# UQLM comparison baseline (requires: pip install uqlm langchain-openai)
python benchmark/benchmark_uqlm.py --skip-epsilon

# Scenario A–E qualitative runs
python benchmark/benchmark_scenarios.py --model gpt-4o

# Scenario E combined-prompt collapse experiment
python benchmark/benchmark_scenario_e_collapse.py

# exec()-based ground truth verification for LOW prompts
python benchmark/ground_truth_low.py

# Precision/recall curve analysis (combines LOW + HIGH ground truth)
python benchmark/analyze_epsilon_pr.py

# Two-stage review loop (2A: no context, 2B: with learned FP patterns)
python benchmark/review_loop.py
```

Results from the paper are in `benchmark/results/`.

## How it works

1. **Token-level ε**: normalized Shannon entropy over top-5 logprob alternatives at each token position — `ε_t = -(1/log k) Σ p_i log p_i`
2. **AST filtering**: function names, parameter names, and local variable names excluded (Type 1 cosmetic uncertainty); only API selectors, import targets, and keyword arguments retained
3. **Max aggregation**: file/function ε = max token ε above floor (0.30) — localizes uncertainty to a specific actionable token
4. **Two-stage system**: ε filter at recall=1.00 (catches everything) + parallel LLM reviewer (clears false positives, precision → ~100%)

See the paper for full details including the three-type uncertainty taxonomy and review loop evaluation.

## Repository structure

```
epsilon/                    Core library
  __init__.py
  core.py                   EpsilonWrapper, EpsilonResult, TokenEpsilon
  renderer.py               Terminal output with rich
  logger.py                 Persistent JSONL session log

demos/                      Calibrated scenario scripts (A–E)
benchmark/                  Benchmark and analysis scripts
  benchmark_calibration.py  30-prompt calibration benchmark
  benchmark_uqlm.py         UQLM comparison baseline
  benchmark_scenarios.py    Qualitative scenario runner (A–E)
  benchmark_scenario_e_collapse.py  Combined vs per-function generation
  ground_truth_low.py       exec()-based LOW ground truth verification
  analyze_epsilon_pr.py     Precision/recall curve analysis
  review_loop.py            Two-stage filter + LLM review loop
  results/                  Pre-computed benchmark outputs

paper/                      LaTeX source (arXiv submission)
```

## License

MIT
