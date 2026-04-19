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
# ε false positive / detection rate benchmark (20 prompts)
python benchmark/benchmark_calibration.py

# UQLM comparison (requires: pip install uqlm langchain-openai)
python benchmark/benchmark_uqlm.py --skip-epsilon

# Domain threshold calibration (runs Scenario E repeatedly)
python benchmark/calibrate_thresholds.py --runs 12
```

## How it works

1. **Token-level ε**: normalized Shannon entropy over top-5 logprob alternatives at each token position
2. **AST filtering**: function names, parameter names, and local variable names excluded (cosmetic naming uncertainty, not consequential)
3. **Max aggregation**: file/function ε = max token ε above noise floor (0.30) — bounded, interpretable, doesn't explode with length
4. **Adaptive threshold**: K-NN neighborhood in embedding space; MAD domain after ~13 similar runs, conformal after ~75

See the paper for full details.

## Repository structure

```
epsilon/                Core library
  __init__.py
  core.py               EpsilonWrapper, EpsilonResult, TokenEpsilon
  renderer.py           Terminal output with rich
  logger.py             Persistent JSONL session log + K-NN retrieval

demos/                  Calibrated scenario scripts (A–E)
benchmark/              Calibration and comparison scripts

paper/                  LaTeX source (arXiv submission)
```

## License

MIT
