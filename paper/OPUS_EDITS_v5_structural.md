# Opus Edit Instructions — Structural & Data Updates (v5 → v6)

## BEFORE YOU BEGIN — Context and Direction

**The paper currently has a really good story.** The purpose of these edits is not to do a massive change to the story, but to fix a few alignments and add a bit more context and history. Please also review the current LLM feedback (summarized below) and revise this paper in such a way that continues the history of edits we've done.

Do not restructure sections, change the argument flow, or introduce new ideas beyond what is specified below. Treat this as a precision calibration pass, not a rewrite.

---

### History of paper versions

The paper has gone through a deliberate series of passes, each tightening the story:

- **v2 (2026-04-21, Opus 4.7):** Systemization pass. Established the two-stage architecture as the main contribution. Locked in "ε as a filter, not a detector" framing. First clean narrative flow.

- **v3 / v4 (2026-04-22, Opus 4.7):** Major rewrite. Added full Related Work section (Wang et al., Spiess et al. p_avg, AdaDec, Sharma & David multi-sample, UQLM). Introduced explicit intra/inter-generational distinction. New figures: `fig_comparison.pdf`, `fig_intra_inter.pdf`. Confirmed the p_avg concession (ε ties p_avg on function-level detection for 3 of 4 OpenAI models) as a high-integrity scientific move rather than a weakness.

- **v5 (2026-04-23–24, Opus 4.7, two passes):** Two-level signal narrative crystallized: ε is a function-level filter (shared with p_avg) **plus** a token-level localizer (unique to ε). Token-focus analysis added (89% reduction in review surface; average flagged function narrows to 10.4 tokens). Review loop reframed: reviewer is directed to the 1–2 high-ε tokens, not asked to re-read the function. Threshold scrub: corrected 0.65 labeling throughout.

- **v6 (this pass):** The specific edits below. No structural changes — alignment corrections and data updates only.

---

### LLM peer review of v4 — what reviewers said

Two independent LLM reviewers evaluated v4 before v5 was written. Their feedback is reproduced here so you understand what was already addressed and what remains.

**ChatGPT (GPT-4o) review of v4:**

Overall verdict: *"arXiv: Strong, polished, citation-aware. Top-tier: Now legitimately competitive, but still vulnerable on external validation + novelty positioning. The remaining work is positioning, not invention."*

What improved in v4:
- Paper now reads as a coherent research narrative: clear problem → limitation of prior work → new primitive → system → evaluation
- "ε as a filter, not detector" is fully locked in — aligns with empirical results, avoids overclaiming, supports the system architecture
- Two-stage system is convincingly the main contribution

Four remaining concerns after v4:
1. **Recall claim is the biggest liability** (→ addressed in Category A of this document). Even v4 implies "misses nothing / perfect recall." The fix: replace everywhere with "recall-maximizing with zero observed misses in the evaluated set."
2. **Evaluation is partially indirect** — "cleared" ≠ verified correct; "confirmed" is partly LLM-judged. Acknowledged in limitations; not addressed by this pass.
3. **Type 3 uncertainty: "rare in production" is unsupported** (→ addressed in Category C of this document, C1).
4. **Intra vs. inter distinction is argued intuitively, not formalized** — biggest missed opportunity for elevation at top-tier venues. Not addressed in this pass.

Five literature gaps flagged (→ all addressed in `OPUS_EDITS_v5_citations.md`):
1. Token-level entropy + decoding work (AdaDec) — need explicit distinction: they use entropy to improve generation; we use it post-generation for risk localization
2. Code-generation uncertainty ↔ correctness (Sharma & David, Gros & Devanbu)
3. Token-level uncertainty frameworks (length bias theory: Gupta et al.)
4. Token entropy + conformal prediction (TECP)
5. Entropy-guided refinement loops (Self-Refine, Entropy-Guided Loop, ERGO)

**Gemini review of v4:**

Overall verdict: *"Ready for submission. This is a high-impact paper that effectively challenges the current dominance of multi-sample uncertainty methods. Submit to arXiv under cs.LG and cs.SE."*

Specific highlights:
- The p_avg concession (acknowledging ties on function-level detection) is *"a high-integrity scientific move — it validates the recall-first design while making the localization claim more persuasive."*
- Uncertainty collapse (Scenario E: joint-prompting resolves uncertainty before decoding) is the most profound finding
- The 89% reduction in review surface (narrowing 98 tokens to 10.4) is *"the money shot — your strongest practical claim"*
- The 27% → 94% precision improvement through the review loop is *"a compelling product story"*

One open thread from Gemini: *"You should actually run the p_avg baseline through the full 201-entry review loop. You hypothesize ε is better because of token attribution, but a reviewer might demand to see if a 'blind' p_avg reviewer (given the whole function) is actually slower or less precise. It's the only loose thread left."* → **This has now been run** (see Category B of this document, B1). The result: p_avg achieves 44% API recall and 50.5% end-to-end precision vs. ε's 93% and 94%, closing this thread definitively.

---

### Summary of what this v6 pass does

This document covers everything EXCEPT citations (see `OPUS_EDITS_v5_citations.md`).
Three categories of edits: (A) recall language, (B) stale "open question" passages replaced
with actual data, (C) minor cleanup.

---

## CATEGORY A — Recall overclaims (7 instances across 5 files)

The correct framing: ε is a recall-maximizing filter that observed zero misses on the
evaluated set. It is NOT a theoretical guarantee. The SQLAlchemy GPT-4o-mini miss
(ε=0.000, already disclosed in §limitations) is the known counter-example.

Replace "recall = 1.00" / "100% recall" / "catches 100% of true failures" / "misses nothing"
with language that is accurate for the evaluated set without claiming universality.

---

### A1 — abstract.tex, line 19

**Current:**
```
recall $=1.00$ and
precision $=0.272$ on a 200-entry evaluated set.
```

**Replace with:**
```
zero observed misses on the 200-entry evaluated set (recall-maximizing design) and
precision $=0.272$.
```

---

### A2 — introduction.tex, lines 68–70

**Current:**
```
The function-level
score is a recall-maximizing filter: at threshold $\eps \ge 0.30$ it
flags $97\%$ of HIGH prompts and catches $100\%$ of true failures on
our benchmark.
```

**Replace with:**
```
The function-level
score is a recall-maximizing filter: at threshold $\eps \ge 0.30$ it
flags $97\%$ of HIGH prompts with zero observed misses on
our evaluated benchmark set.
```

---

### A3 — introduction.tex, contribution 3 (lines 144–148)

**Current:**
```
    \item A two-stage filter-plus-review architecture evaluated on
    201 entries across three models (GPT-4o, GPT-4o-mini, DeepSeek
    V3): near-$100\%$ end-to-end precision at $100\%$ recall, with
    the reviewer directed to \eps's high-entropy tokens rather than
    asked to re-read the function.
```

**Replace with:**
```
    \item A two-stage filter-plus-review architecture evaluated on
    201 entries across three models (GPT-4o, GPT-4o-mini, DeepSeek
    V3): $94\%$ end-to-end precision (context-primed, zero late false
    positives) with zero observed misses at stage 1, the reviewer
    directed to \eps's high-entropy tokens rather than asked to
    re-read the function.
```

---

### A4 — evaluation.tex, Table~\ref{tab:pr} caption (lines 308–319)

The table cell itself showing `$\mathbf{1.000}$` is factually correct for the evaluated
set and can stay. Add one sentence to the caption acknowledging it is an observed
result, not a guarantee.

**Current caption ends with:**
```
(\S\ref{subsec:gt-disclosure}); \textsc{low} ground truth ($30/200$)
is \texttt{exec()}-verified.
```

**Append to caption (before closing brace):**
```
Recall $=1.000$ is an observed result on this set; the known false
negative (GPT-4o-mini on SQLAlchemy, $\eps=0.000$) falls outside
this evaluated set and is discussed in \S\ref{subsec:limitations}.
```

---

### A5 — evaluation.tex, lines 383–386

**Current:**
```
End-to-end failure rates: GPT-4o-mini $40\%$, GPT-4o $29\%$,
DeepSeek V3 $26\%$. Because stage-$1$ recall is $100\%$, end-to-end
precision is determined entirely by the reviewer's precision on the
\textsc{flagged+} set;
```

**Replace with:**
```
End-to-end failure rates: GPT-4o-mini $40\%$, GPT-4o $29\%$,
DeepSeek V3 $26\%$. Because stage-$1$ recall is recall-maximizing
(zero observed misses on this set), end-to-end
precision is determined primarily by the reviewer's precision on the
\textsc{flagged+} set;
```

---

### A6 — conclusion.tex, lines 8–9

**Current:**
```
the function-level score is a recall-maximizing filter: on the evaluated
set, it flags $97\%$ of \textsc{high} API-integration prompts and
catches $100\%$ of true failures at precision $0.272$.
```

**Replace with:**
```
the function-level score is a recall-maximizing filter: on the evaluated
set, it flags $97\%$ of \textsc{high} API-integration prompts with
zero observed misses, at precision $0.272$.
```

---

### A7 — conclusion.tex, lines 48–49

**Current:**
```
A filter that misses nothing, plus a
reviewer that is told exactly where to look, moves review off the
developer's critical path.
```

**Replace with:**
```
A recall-maximizing filter, plus a
reviewer that is told exactly where to look, moves review off the
developer's critical path.
```

---

## CATEGORY B — Stale "open question" language + new experimental data

The p_avg → review loop experiment has now been run. Key results:
- At the most aggressive p_avg threshold (−mean_logprob ≥ 0.05), p_avg achieves
  only **44% API recall** (76/174 entries), vs ε's **93%** (162/174). mean_logprob
  stays near zero in production — the model is globally confident even when uncertain
  at specific tokens, so the average is swamped.
- Through the review loop WITHOUT token attribution, end-to-end precision is **50.5%**
  (50/99 confirmed), vs ε's **94%**. A 43.5pp gap.
- All 50 suspect entries were CONFIRMED (0 downgrades), meaning the reviewer correctly
  found real issues when it found anything — but it cleared 49 genuine problems it
  couldn't localize.

---

### B1 — comparison.tex, replace the "open question" paragraph

**Location:** §subsec:papg-comp, the closing paragraph (after the min_logprob table).
The current text (lines 97–103 of comparison.tex) says:

**Current:**
```
This is the distinguishing claim. It is not a higher rate; it is a
different kind of output. \papg is a competitive function-level
signal; \eps is a function-level signal plus a token-level attribution.
Whether the attribution translates to better end-to-end precision in
a \papg-routed review loop is an open empirical question --- the
\papg baseline has not been extended to the full $201$-entry review-%
loop evaluation --- but at the signal level the token-level output
is strictly more informative at the same cost.
```

**Replace with:**
```
This is the distinguishing claim. It is not a higher rate; it is a
different kind of output. \papg is a competitive function-level
signal; \eps is a function-level signal plus a token-level attribution.
We have now run the \papg baseline through the full review loop to
measure whether that attribution gap produces a precision gap at the
system level.

\paragraph{\papg through the review loop.}
We re-ran all six production and scenario result files with
\texttt{mean\_logprob} captured, swept \papg thresholds to find the
one whose API-entry recall best matches \eps's $93\%$, and ran the
identical review loop without providing token attribution to the reviewer.
The threshold sweep surfaces a structural constraint first: at every
threshold, \papg's maximum recall on API entries is $44\%$ ($76/174$).
Even at the most aggressive threshold, \texttt{mean\_logprob} stays
near zero across production completions --- the model is globally
confident, and averaging over the full token sequence absorbs the
localized uncertainty that \eps reads at the decision token.

At the matched threshold (\texttt{-mean\_logprob} $\ge 0.05$),
\papg flags $99$ entries. Through the review loop, end-to-end
precision is $50.5\%$ ($50/99$ confirmed true failures), compared
to \eps's $94\%$ on $200$ entries. All $50$ suspect entries were
confirmed by the consolidation pass ($0$ downgrades): the reviewer
correctly identified issues when it found them, but cleared $49$
genuine problems it could not localize to specific tokens.

\begin{table}[h]
\centering
\small
\begin{tabular}{lrrrr}
\toprule
Signal & Entries reviewed & API recall & End-to-end precision & True errors found \\
\midrule
\eps (cascaded)               & $200$ & $93\%$ ($162/174$) & $\mathbf{94\%}$ & ${\sim}188$ \\
\papg (${\ge}0.05$ threshold) & $\phantom{0}99$ & $44\%$ ($\phantom{0}76/174$) & $50.5\%$ & $\phantom{{\sim}0}50$ \\
\bottomrule
\end{tabular}
\caption{End-to-end comparison: \eps vs.\ \papg through the full review loop on the
same 228-entry production-plus-scenario set (three models). \papg's recall ceiling is
structural: \texttt{mean\_logprob} averages over all tokens including confident boilerplate,
absorbing localized API-decision uncertainty. The precision gap ($43.5$pp) reflects the
reviewer's inability to localize without token attribution; errors were confirmed when
found but $49$ genuine failures were cleared because the reviewer had no token target to
examine.}
\label{tab:pavg-endtoend}
\end{table}

The signal-level conclusion from \S\ref{subsec:papg-comp} stands: at
$1\times$ API cost, \papg is competitive with \eps on raw
function-level detection for three of four OpenAI models. The
system-level conclusion is different. Token attribution is not a
refinement on top of the filter; it is the mechanism that enables
$94\%$ end-to-end precision. Without it, the same review loop
achieves $50.5\%$.
```

---

### B2 — comparison.tex, summary paragraph (§subsec:comp-summary, lines 239–250)

The current summary paragraph lists two empirical observations (i) and (ii). Add a
third now that we have the review loop data.

**Current last sentence of that paragraph:**
```
The review loop in
\S\ref{subsec:review-loop} is what turns the second answer into an
end-to-end system: the reviewer is not asked to re-read $98$ tokens;
it is asked to evaluate the high-\eps tokens \eps attributed.
```

**Replace with:**
```
The review loop in
\S\ref{subsec:review-loop} is what turns the second answer into an
end-to-end system: the reviewer is not asked to re-read $98$ tokens;
it is asked to evaluate the high-\eps tokens \eps attributed.
(iii) Through the review loop, the token-attribution gap becomes a
precision gap: \eps achieves $94\%$ end-to-end precision;
\papg-routed review, without token localization, achieves $50.5\%$
on the same infrastructure (Table~\ref{tab:pavg-endtoend}).
```

---

### B3 — discussion.tex, replace the §limitations "p_avg comparison scope" paragraph

**Current (lines 55–65):**
```
\paragraph{\papg comparison scope.} The \papg evaluation uses the
calibration benchmark ($30$ prompts per model, $120$ total) and has
not been extended to the $201$-entry production-plus-scenario set
used for the review-loop results. Whether \papg's function-level
competitiveness translates to a working review loop when the
reviewer is not given token attribution is an open question. A
natural experiment would run the same review loop twice, once with
\eps-routed token targets and once with \papg-routed whole-function
review; we expect the former to be materially faster per entry and
at least as precise, but have not run it.
```

**Replace with:**
```
\paragraph{\papg comparison scope.} The signal-level \papg evaluation
(\S\ref{subsec:papg-comp}) uses the calibration benchmark ($30$
prompts per model, $120$ total). The end-to-end comparison through
the review loop (\S\ref{subsec:papg-comp}, Table~\ref{tab:pavg-endtoend})
uses the full $228$-entry production-plus-scenario set. The recall
ceiling finding --- \papg's maximum API-entry recall is $44\%$ at
any threshold --- rests on the production set and is not subject to
the small-sample caveat that bounds the calibration results.
```

---

## CATEGORY C — Minor cleanup

### C1 — epsilon.tex, lines 137–138 (unsupported "rare in production" claim)

**Current (in Table~\ref{tab:filtering} caption):**
```
Residual $50\%$ is Type 3, rare in production API-integration
code.
```

**Replace with:**
```
Residual $50\%$ is Type 3; the fraction of Type 3 fires
in production API-integration code depends on how constrained
the function's algorithm space is.
```

---

### C2 — discussion.tex, future work paragraph "Combined p_avg + ε routing" (lines 157–160)

**Current:**
```
\paragraph{Combined \papg $+$ \eps routing.}
\papg's function-level specificity could be combined with \eps's
token localization --- a two-signal gate that fires when both agree,
with \eps supplying the token target for the reviewer.
```

**Delete this paragraph entirely.** The review loop experiment showed p_avg's
API recall caps at 44% vs ε's 93%; a joint gate requiring both to agree would
halve ε's recall without a compensating precision benefit that isn't already
delivered by the review loop.

---

## New data reference (for Opus context)

The experiment results are at:
`repo/benchmark/results/p_avg_review_results.json`

Key numbers to use verbatim:
- p_avg entries reviewed: **99**
- p_avg API recall: **44%** (76/174)
- p_avg end-to-end precision: **50.5%** (50/99)
- ε entries reviewed: **200** (from paper's existing results)
- ε API recall: **93%** (162/174)
- ε end-to-end precision: **94%** (context-primed)
- Precision delta: **−43.5pp** (ε better)

---

## What NOT to change

- Any table data values (ε detection rates, FP rates, etc.)
- The SQLAlchemy false-negative discussion in §limitations — it is correctly
  disclosed and should remain as-is
- Section labels or numbering
- Any content in the citation edit document (handled separately)
- The ε section discussion of Type 3 being "irreducible at the signal level" — that claim
  is correct and should stay
