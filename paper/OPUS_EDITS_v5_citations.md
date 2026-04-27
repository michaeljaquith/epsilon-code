# Opus Edit Instructions — Citation & Positioning Updates (v5 → v6)

Target files:
- `paper/sections_v5/related_work.tex`
- `paper/references.bib`

**Goal:** Add 5 citation clusters to tighten related-work positioning without restructuring the paper. Do not change section names, labels, or any content outside the specified locations.

**Verification note:** Before inserting any new `\cite{}` key, confirm the arXiv ID exists. Self-Refine (2303.17651) is certain. Flag any others that 404 and leave a `% VERIFY:` comment inline.

---

## Edit 1 — Gap 1: AdaDec closing sentence

**Location:** `\subsection{Generation-time intervention and adaptive decoding}` — append one sentence at the end of the subsection (after "...could not resolve.").

**Current last sentence of that subsection (lines 143–145):**
```
...with residual high-\eps tokens concentrating on consequential
decisions that even a lookahead re-rank could not resolve.
```

**Append after that sentence:**
```latex
Consequently, applying AdaDec upstream of our pipeline is expected to
reduce average \eps values but not eliminate high-entropy API-selector
tokens, which remain the primary target of the review loop.
```

---

## Edit 2 — Gap 2: Code uncertainty ↔ correctness (calibration subsection)

**Location:** `\subsection{Token-level and sequence-level calibration}` — insert a new paragraph after the paragraph ending `...which we report in \S\ref{sec:comparison}.` (the paragraph ending with "a direct head-to-head comparison is possible; we report it in §comparison.").

**Insert this paragraph:**
```latex
Two concurrent code-specific studies further establish that token-level
uncertainty localizes to the same regions \eps targets. Sharma and
David~\cite{sharma2025multisample} adapt entropy and mutual information
to code generation and show, using symbolic execution as ground truth,
that uncertainty-based abstention reduces incorrect outputs to near
zero; their finding grounds the core premise of \eps in prior empirical
evidence without requiring symbolic execution at inference time. Gros
and Devanbu~\cite{gros2025localized} assign calibrated uncertainty to
individual code regions using a supervisor model, demonstrating that
sub-function localization is both feasible and informative. Our
contribution relative to both is that we infer token-level risk
directly from the generator's logprob distribution---no auxiliary model
or test executor required---and route the result to a downstream review
loop rather than to abstention.
```

**New bib entry to add to `references.bib`:**
```bibtex
@misc{gros2025localized,
  title         = {Localized Calibrated Uncertainty in Code Language Models},
  author        = {Gros, David and Devanbu, Prem},
  year          = {2025},
  eprint        = {2512.24560},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SE},
  note          = {arXiv:2512.24560},
}
```

**Also:** In the existing multi-sample subsection, the current cite `\cite{sharma2025multisample}` is used for "stronger equivalence checks." That usage is fine to keep as-is; the new reference above is an *additional* mention in the calibration subsection. No removal needed.

---

## Edit 3 — Gap 3: Token-level > sequence-level (length bias)

**Location:** `\subsection{Token-level and sequence-level calibration}` — extend the paragraph that ends with `...Conformal prediction approaches...provide set-valued predictions with coverage guarantees but do not localize which token within a generation is uncertain.`

**Find this sentence:**
```latex
Earlier token-level calibration studies (Kadavath et
al.~\cite{kadavath2022language}; Varshney et
al.~\cite{varshney2023stitch}) establish that LLMs are reasonably
calibrated at the token level for factual questions and that
selective generation can exploit this.
```

**Replace with:**
```latex
Earlier token-level calibration studies (Kadavath et
al.~\cite{kadavath2022language}; Varshney et
al.~\cite{varshney2023stitch}) establish that LLMs are reasonably
calibrated at the token level for factual questions and that
selective generation can exploit this. Gupta et
al.~\cite{gupta2024cascades} prove that sequence-level aggregation of
token probabilities suffers from length bias---longer outputs are
systematically over- or under-penalised by mean logprob---and show
that token-level deferral rules substantially outperform na\"{i}ve
sequence-level means; this provides a principled explanation for why
\papg is a challenging baseline to beat despite sharing the same raw
logprob signal. Santilli et al.~\cite{santilli2025length} further
demonstrate that length bias compromises \emph{evaluation} of
uncertainty methods across eight signals and four models, validating
our choice to compare \eps and \papg at matched operating points
rather than by raw AUC.
```

**New bib entries to add to `references.bib`:**
```bibtex
@misc{gupta2024cascades,
  title         = {Language Model Cascades: Token-Level Uncertainty and Beyond},
  author        = {Gupta, Neha and Narasimhan, Harikrishna and Jitkrittum, Wittawat
                   and Rawat, Ankit Singh and Menon, Aditya Krishna and Kumar, Sanjiv},
  year          = {2024},
  eprint        = {2404.10136},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {arXiv:2404.10136},
}

@inproceedings{santilli2025length,
  title     = {Revisiting Uncertainty Quantification Evaluation in Language Models:
               Spurious Interactions with Response Length Bias Results},
  author    = {Santilli, Andrea and Golinski, Adam and Kirchhof, Michael and
               Danieli, Federico and Blaas, Arno and Xiong, Miao and Zappella, Luca
               and Williamson, Sinead},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for
               Computational Linguistics (ACL)},
  year      = {2025},
  note      = {arXiv:2504.13677},
}
```

---

## Edit 4 — Gap 4: Entropy-guided refinement loops (new subsection)

**Location:** Insert a new subsection **between** `\subsection{Generation-time intervention and adaptive decoding}` and `\subsection{Static analysis, linters, and execution-based evaluation}`.

**Insert this entire block:**
```latex
\subsection{Entropy-guided refinement and self-repair}
\label{subsec:related-refinement}

A separate line of work uses uncertainty signals to trigger iterative
repair rather than review. Self-Refine~\cite{madaan2023selfrefine}
establishes the canonical pattern: a single model generates, critiques
its own output, and refines iteratively, achieving $\approx 20\%$
absolute improvement across diverse tasks without supervision. The
refinement loop is unconditional---every generation is sent back
regardless of confidence.

Conditional entropy-triggered variants tighten this by gating
second-pass work on measured uncertainty. Correa and de
Matos~\cite{correa2025entropyloop} compute Shannon entropy over top-$k$
alternatives at each step, trigger refinement when entropy, perplexity,
or low-confidence token counts exceed learned thresholds, and feed an
uncertainty report back to the originating model; they report
$16$\,pp accuracy gains at one-third the cost of reasoning-class
models. ERGO~\cite{khalid2025ergo} applies the same principle to
multi-turn generation, resetting the prompt context when entropy
spikes sharply across turns.

Our review loop shares the entropy-triggered gating with these methods
but differs in two ways. First, the second-pass agent is a separate
reviewer with domain-specific prompting, not a self-critique call to
the originating model; this removes the correlation between generator
failure and reviewer failure. Second, the review is over
\emph{code-specific risk}---the flagged token and its alternatives,
not the full response---which concentrates reviewer attention and
enables the precision improvement from $27\%$ to $94\%$
(\S\ref{sec:comparison}).
```

**New bib entries to add to `references.bib`:**
```bibtex
@misc{madaan2023selfrefine,
  title         = {Self-Refine: Iterative Refinement with Self-Feedback},
  author        = {Madaan, Aman and Tandon, Niket and Gupta, Prakhar and Hallinan, Skyler
                   and Gao, Luyu and Wiegreffe, Sarah and Alon, Uri and Dziri, Nouha
                   and Prabhumoye, Shrimai and Yang, Yiming and Gupta, Shashank and
                   Majumder, Bodhisattwa Prasad and Hermann, Katherine and Welleck, Sean
                   and Yazdanbakhsh, Amir and Clark, Peter},
  year          = {2023},
  eprint        = {2303.17651},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {NeurIPS 2023; arXiv:2303.17651},
}

@misc{correa2025entropyloop,
  title         = {Entropy-Guided Loop: Achieving Reasoning through Uncertainty-Aware Generation},
  author        = {Correa, Andrew G. A. and de Matos, Ana C. H.},
  year          = {2025},
  eprint        = {2509.00079},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  note          = {arXiv:2509.00079 — VERIFY ID before submission},
}

@inproceedings{khalid2025ergo,
  title     = {{ERGO}: Entropy-guided Resetting for Generation Optimization
               in Multi-turn Language Models},
  author    = {Khalid, Haziq Mohammad and Jeyaganthan, Athikash and Do, Timothy and
               Fu, Yicheng and O'Brien, Sean and Sharma, Vasu and Zhu, Kevin},
  booktitle = {Proceedings of the 2nd Workshop on Uncertainty Aware {NLP}
               ({UncertaiNLP} 2025)},
  year      = {2025},
  note      = {arXiv:2510.14077 — VERIFY ID before submission},
}
```

---

## Edit 5 — Gap 5: TECP conformal prediction (one sentence addition)

**Location:** `\subsection{Token-level and sequence-level calibration}` — find the conformal prediction sentence:

```latex
Conformal prediction
approaches~\cite{quach2023conformal,angelopoulos2022conformal}
provide set-valued predictions with coverage guarantees but do not
localize which token within a generation is uncertain.
```

**Replace with:**
```latex
Conformal prediction
approaches~\cite{quach2023conformal,angelopoulos2022conformal}
provide set-valued predictions with coverage guarantees but do not
localize which token within a generation is uncertain; TECP~\cite{xu2025tecp}
applies split conformal prediction directly to token-entropy scores,
providing per-output coverage guarantees that could in future work be
used to calibrate the \eps threshold with statistical guarantees.
```

**New bib entry to add to `references.bib`:**
```bibtex
@misc{xu2025tecp,
  title         = {{TECP}: Token-Entropy Conformal Prediction for {LLMs}},
  author        = {Xu, Beining and Lu, Yongming},
  year          = {2025},
  eprint        = {2509.00461},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {arXiv:2509.00461 — VERIFY ID before submission},
}
```

---

## Summary of new bib keys

| Key | ArXiv | Confidence |
|-----|-------|------------|
| `gros2025localized` | 2512.24560 | Verify |
| `gupta2024cascades` | 2404.10136 | Verify |
| `santilli2025length` | 2504.13677 | Verify |
| `madaan2023selfrefine` | 2303.17651 | **High — NeurIPS 2023** |
| `correa2025entropyloop` | 2509.00079 | Verify |
| `khalid2025ergo` | 2510.14077 | Verify |
| `xu2025tecp` | 2509.00461 | Verify |

Self-Refine is the only one with high confidence. All others should have their arXiv IDs confirmed at arxiv.org before the final LaTeX compile.

---

## What NOT to change

- Section labels and numbering
- The `sharma2025multisample` cite in the multi-sample subsection (keep as-is; new mention is additive)
- Any content in other `.tex` files
- Figure references, table numbers, or anything in the evaluation/comparison sections
