# Redundancy Audit — v6 Paper

## Summary

I identified **9 redundancy clusters** ranging from full near-duplicate paragraphs to repeated framing sentences. Severity is moderate-to-high: the paper is functional and the redundancy doesn't damage clarity, but several core claims (token-focus reduction, p_avg-vs-ε framing, intra/inter distinction, the 52/47 Stripe example) are stated 3–5 times each in nearly the same words. Recommended approach: cut the most aggressive duplications in the conclusion and discussion, condense the introduction's mid-section paragraphs, and pick a single canonical home for each repeated framing. The taxonomy/labels intro repetition (HIGH/LOW) is the cleanest single fix.

---

## Cluster 1 — HIGH/LOW label introduction stated twice

**Appears in:**
- `introduction.tex` lines 156–162 (last paragraph before the github URL)
- `epsilon.tex` lines 24–31 (the "Terminology" paragraph in subsec:per-token)

**Type:** Full near-duplicate (concept re-introduced as if reader hasn't seen it)

**Canonical location:** `introduction.tex` (it serves as forward declaration before evaluation references HIGH/LOW)

**Proposed resolution:** Cut the `epsilon.tex` "Terminology" paragraph entirely. Replace with a one-line parenthetical at first use, or just remove. The introduction already establishes the labels and explicitly says "A detailed calibration table appears in §subsec:calibration."

**Rationale:** Two full paragraphs introducing the same labels with the same five-library list (Stripe, OpenAI, SQLAlchemy, FastAPI, Pydantic) and the same four pure-logic tasks (sort, palindrome, Fibonacci, two-sum). The author flagged this exact duplication.

**Instance A — `introduction.tex` 156–162:**
> We use the labels \textsc{high} and \textsc{low} throughout: \textsc{high} prompts are API/version-split tasks across Stripe, OpenAI, SQLAlchemy, FastAPI, and Pydantic where an incorrect choice causes functional failure; \textsc{low} prompts are pure-logic tasks (sort, palindrome, Fibonacci, two-sum) where any correct algorithm is functionally equivalent. A detailed calibration table appears in \S\ref{subsec:calibration}.

**Instance B — `epsilon.tex` 24–31:**
> \paragraph{Terminology: \textsc{high} and \textsc{low} prompts.}
> We use the labels \textsc{high} and \textsc{low} throughout. \textsc{high} prompts are API/version-split tasks (Stripe, OpenAI, SQLAlchemy, FastAPI, Pydantic) where an incorrect choice causes functional failure. \textsc{low} prompts are pure-logic tasks (sort, palindrome, Fibonacci, two-sum) where any correct algorithm is functionally equivalent. The split and detection rates per category appear in \S\ref{subsec:calibration}.

---

## Cluster 2 — Token-focus reduction claim ("89% / ~10 tokens / 98-token function")

**Appears in:**
- `abstract.tex` 21–26
- `introduction.tex` 71–79 ("two levels at which it operates" paragraph)
- `introduction.tex` 138–143 (Contributions item 2)
- `epsilon.tex` 222–229 (paragraph after Table 4)
- `evaluation.tex` 60–67 (subsec:token-eval recap)
- `comparison.tex` 84–94 ("Where p_avg stops")
- `conclusion.tex` 17–32

**Type:** Same headline statistic repeated 7 times across the paper

**Canonical location:** `epsilon.tex` §subsec:token-focus is the formal definition + table; `introduction.tex` paragraph "two levels at which it operates" and `conclusion.tex` are appropriate echoes.

**Proposed resolution:**
- Keep: abstract, the §epsilon definition + table, one tight conclusion sentence.
- Cut from `introduction.tex` Contributions item 2 the redundant numerical recap ("$\sim 98$-token functions reduce to an average of $10.4$ actionable tokens per function ($10.5\%$ of the function) --- an $89\%$ reduction"); the prior paragraph already gave it. Reduce to: "Token-level localization as ε's distinguishing claim: the function-level flag agrees with p_avg; the token-level attribution is what no sequence-level signal can produce."
- Cut from `evaluation.tex` 60–63 the prose recap "The aggregate numbers are: 10.4 tokens per flagged function..." — the table already shows this; one sentence suffices.
- Cut from `comparison.tex` 92–94 the "$89\%$" callback — the earlier intro and §epsilon already own it; replace with "(see Table~\ref{tab:token-focus})."
- Trim conclusion paragraph 17–32: keep the framing, drop the second sentence's full numerical re-statement (already given in lead sentence).

**Rationale:** The "$\sim 10$ of $98$ tokens / $89\%$ reduction" formula appears verbatim or near-verbatim 5+ times. A reader hits the same number-set in nearly every section. Some echo is appropriate (abstract, conclusion, definition); but the introduction states it twice on its own (paragraph 71–79 AND contributions item 2), and evaluation re-states what its own table just showed.

**Instance A — `abstract.tex` 21–26:**
> The contribution is that the same score localizes: of the $\sim 98$ tokens in a typical flagged function, only $10.5\%$ exceed the flag threshold. \eps narrows the developer's review surface from a whole function to an average of $10.4$ tokens per function ($10.5\%$ of the function) --- an $89\%$ reduction in within-function review burden.

**Instance B — `introduction.tex` 71–79:**
> But the function-level view discards most of what \eps knows. Of the $\sim 98$ tokens in a typical flagged function, only $10.5\%$ exceed the flag threshold. The same score that flags the function also tells the reviewer that the relevant uncertainty lives in the method selector on line $4$, not in the parameter block on lines $7$--$12$.

**Instance C — `introduction.tex` 138–143 (Contributions):**
> Token-level localization as \eps's distinguishing claim: $\sim 98$-token functions reduce to an average of $10.4$ actionable tokens per function ($10.5\%$ of the function) --- an $89\%$ reduction in within-function review burden.

**Instance D — `epsilon.tex` 222–229:**
> A developer reviewing an AI-generated function does not face a $98$-token problem. They face an $\sim 10$-token problem at the flag threshold...

**Instance E — `evaluation.tex` 60–63:**
> The aggregate numbers are: $10.4$ tokens per flagged function at the flag threshold ($10.5\%$ of an average $\sim 98$-token function). The token review burden reduction relative to reading the whole function is $89\%$ at the flag threshold.

**Instance F — `comparison.tex` 92–94:**
> Table~\ref{tab:token-focus} showed that this localization reduces the within-function review surface by $89\%$ at the flag threshold.

**Instance G — `conclusion.tex` 17–22:**
> Of the $\sim 98$ semantic tokens in a typical flagged function, $10.5\%$ exceed the flag threshold. A reviewer given \eps's output is not asked to re-read $98$ tokens; they are asked to evaluate the $\sim 10$ tokens that cross the flag threshold --- an $89\%$ reduction in within-function review surface.

---

## Cluster 3 — "ε ties p_avg on function-level, adds token localization" framing

**Appears in:**
- `abstract.tex` 27–30
- `introduction.tex` 99–117 ("A three-way comparison" paragraph)
- `comparison.tex` 75–101 ("p_avg's real strength" + "Where p_avg stops")
- `comparison.tex` 286–300 (Summary subsection)
- `conclusion.tex` 10–32

**Type:** Same conceptual framing stated four times

**Canonical location:** `comparison.tex` §subsec:papg-comp owns this argument with the table.

**Proposed resolution:**
- Abstract: keep — abstracts repeat the headline.
- Introduction "A three-way comparison" paragraph: condense; drop the model-by-model breakdown ("tying ε on GPT-4o, beating it on GPT-4o-mini and GPT-4-turbo, and losing to it on DeepSeek V3") since the comparison section gives the table. Keep the one-sentence framing.
- Comparison §subsec:comp-summary lines 286–300: shorten substantially. The two-axis structure ("function-level" vs "token-level") is already stated in the section's opening (lines 11–15), the two p_avg paragraphs (75–101), and the Through-the-Review-Loop paragraphs. The Summary subsection currently re-states this for a fourth time.
- Conclusion: pick either the p_avg framing or the multi-sample framing — currently both are restated in full.

**Rationale:** The "p_avg matches/beats ε at function level; ε wins at token level" framing is the paper's central argument and benefits from one strong statement. Repeating it four times feels defensive.

**Instance A — `abstract.tex` 27–30:**
> \papg, at the same $1\times$ API cost, is competitive on raw detection ($65$--$90\%$ vs.\ \eps's $70$--$85\%$ across four models) and in fact beats \eps on two of four OpenAI models; what it cannot do is say \emph{where in the function} to look.

**Instance B — `introduction.tex` 99–117:**
> At $1\times$: the sequence-level mean token probability \papg... detects $65$--$90\%$ of HIGH prompts --- tying \eps on GPT-4o, beating it on GPT-4o-mini and GPT-4-turbo, and losing to it on DeepSeek V3. \papg deserves credit: on raw function-level detection it is competitive with \eps and at the same cost. What it cannot do is identify the specific token at which the model was uncertain.

**Instance C — `comparison.tex` 75–82:**
> On three of four OpenAI models, \papg matches or beats \eps's \textsc{high} detection rate at the same or lower \textsc{low} false-positive rate, using the same API call and the same logprobs.

**Instance D — `comparison.tex` 286–300 (Summary):**
> (i) At the function-level question --- ``is this function risky?'' --- \papg matches or outperforms \eps on three of four models at the same API cost. \papg is a strong baseline... (ii) At the token-level question --- ``where in this function is the risk?'' --- \papg is silent...

**Instance E — `conclusion.tex` 10–15:**
> The sequence-level mean token probability \papg of Spiess et al.\ reads the same logprobs, costs the same $1\times$ in API budget, and is competitive with or better than \eps on function-level detection for three of four OpenAI models.

---

## Cluster 4 — Stripe Charge-vs-PaymentIntent / 0.52-vs-0.47 example

**Appears in:**
- `abstract.tex` 1–7
- `introduction.tex` 31–42 ("A concrete instance" paragraph)
- `introduction.tex` 44–59 ("Intra- versus inter-generational" paragraph)
- `epsilon.tex` 50–54 (Type 2 paragraph)
- `evaluation.tex` 107–115 (Scenario A paragraph)
- `comparison.tex` 213–232 (Figure intra-inter caption + paragraph)
- `conclusion.tex` 27–28

**Type:** Same numerical example used as illustration in 6+ places

**Canonical location:** `evaluation.tex` Scenario A is the formal place; `introduction.tex` "A concrete instance" is the motivating use; `comparison.tex` Figure intra-inter is the empirical proof.

**Proposed resolution:** This is mostly acceptable repetition — the example is the paper's spine and repeats serve different purposes. **But:**
- The introduction states $0.52$ vs $0.47$ twice within itself: lines 4–5 of abstract, line 38–39 ("A concrete instance"), and line 53 ("a $0.52$ prior on the deprecated `Charge` API"). Within the introduction, consolidate "A concrete instance" and "Intra- versus inter-generational" so the numbers are stated once.
- `epsilon.tex` Type 2 paragraph 50–54 can shorten — the Type 1/2/3 table doesn't need to re-quote the probabilities since the prose has already used them.
- Conclusion line 27–28 ("a $0.52/0.47$ internal split produces a unanimous output, as GPT-4o on Stripe Scenario A empirically demonstrates") — fine as a callback, leave it.

**Rationale:** Recurring use of the canonical example is the right pattern. The redundancy is the *probability split* itself being re-quoted in 6 locations. State the split fully in the abstract and §evaluation, callback by name elsewhere.

**Instance A — `abstract.tex` 4–5:**
> when the underlying token distribution is a near-50/50 split ($P=0.52$ vs.\ $0.47$).

**Instance B — `introduction.tex` 38–39:**
> the two choices split the probability mass near 50/50 ($P=0.52$ vs.\ $0.47$); the output exposes none of this.

**Instance C — `introduction.tex` 52–55:**
> A model with a $0.52$ prior on the deprecated \texttt{Charge} API samples that branch nearly every time...

**Instance D — `epsilon.tex` 50–54:**
> Example: \texttt{stripe.Charge.create()} (prob.\ $0.52$) vs.\ \texttt{stripe.PaymentIntent.create()} (prob.\ $0.47$), $\eps = 0.73$ at the attribute token.

**Instance E — `evaluation.tex` 110–112:**
> The peak token is the method selector after \texttt{stripe}: \texttt{.Ch} (\texttt{Charge}, prob.\ $0.52$) vs.\ \texttt{.Payment} (\texttt{PaymentIntent}, prob.\ $0.47$).

**Instance F — `comparison.tex` 215–217:**
> A model with probability $0.52$ on \texttt{Charge} and $0.47$ on \texttt{PaymentIntent} samples \texttt{Charge} on roughly $52\%$ of re-rolls...

**Instance G — `conclusion.tex` 27–28:**
> a $0.52/0.47$ internal split produces a unanimous output, as GPT-4o on Stripe Scenario A empirically demonstrates ($\eps = 0.878$, sample diversity $=0.00$ across $N=5$ at $T=0.7$).

---

## Cluster 5 — Multi-sample structural blindness explanation

**Appears in:**
- `abstract.tex` 10–13 ("multi-sample methods measure output disagreement that simply does not occur for peaked code-token distributions")
- `introduction.tex` 47–59 ("Intra- versus inter-generational uncertainty" paragraph)
- `introduction.tex` 112–117 (multi-sample at $5\times$)
- `comparison.tex` 211–245 (subsec:multisample-comp body)
- `related_work.tex` 134–149 (subsec:related-multisample second paragraph)
- `conclusion.tex` 26–30

**Type:** Conceptual explanation re-derived 4 times

**Canonical location:** `comparison.tex` §subsec:multisample-comp owns the empirical case + Figure intra-inter; `related_work.tex` covers the theoretical positioning.

**Proposed resolution:**
- The introduction's "Intra- versus inter-generational" paragraph (44–59) and the "three-way comparison" paragraph (99–117) both re-derive the same point. Cut the second derivation in 99–117; leave the framing once.
- `related_work.tex` 134–149 and `comparison.tex` 211–245 each re-state "$0.52$-peaked distribution picks its peak nearly every time at $T=0.7$." Pick one home — recommend `comparison.tex` since it has the figure. Have `related_work.tex` reference it: "we provide the empirical instance in §subsec:multisample-comp."
- Conclusion line 26–30: keep — appropriate one-sentence callback.

**Rationale:** The "peaked distribution → unanimous samples → multi-sample blind" reasoning is conceptual and reads as a fresh derivation each time. Pick one place to derive it; everywhere else, name it ("the structural blindspot").

**Instance A — `introduction.tex` 50–57:**
> A model with a $0.52$ prior on the deprecated \texttt{Charge} API samples that branch nearly every time; once the early tokens commit, the rest of the function is forced by context. The uncertainty exists, but it lives \emph{inside a single generation}, in the token distribution at each decoding step, not \emph{across} generations.

**Instance B — `comparison.tex` 213–219:**
> A model with probability $0.52$ on \texttt{Charge} and $0.47$ on \texttt{PaymentIntent} samples \texttt{Charge} on roughly $52\%$ of re-rolls at $T=1.0$, and nearly always at $T=0.7$, where the sharpening exponent concentrates mass on the peak.

**Instance C — `related_work.tex` 134–139:**
> Intra-generational uncertainty does not surface as output variation when the distribution is peaked (as it typically is for code tokens: a $0.52$-peaked distribution picks its peak nearly every time at temperature $0.7$, and always at temperature $0$).

---

## Cluster 6 — Static analysis complementarity

**Appears in:**
- `comparison.tex` 256–269 (subsec:static-cov)
- `discussion.tex` 21–25 ("a primary motivation for pairing ε with static analysis")
- `related_work.tex` 213–227 (subsec:related-static)

**Type:** Same point made three times

**Canonical location:** `related_work.tex` subsec:related-static is the natural home.

**Proposed resolution:**
- Cut `comparison.tex` subsec:static-cov entirely OR shrink to one sentence pointing to related_work. The current 14-line subsection adds nothing the related_work paragraph doesn't say. The note about API access (OpenAI logprobs / DeepSeek via Together / vLLM / Anthropic missing) is a useful operational detail but belongs in the methods or limitations section, not under a "static analysis" header.
- The duplicate sentence in `comparison.tex` 268–269 ("The method requires only top-$k$ log-probabilities... no model-internal access") is itself a duplicate of an earlier sentence in the same paragraph (267 and 269 both say "requires only top-$k$ logprobs").

**Rationale:** Three sub-sections in three different sections all making "static analysis is the floor under ε" is overkill. Consolidate.

**Instance A — `comparison.tex` 256–269 (verbatim near-duplicate of itself):**
> Static analysis is the floor under \eps; \eps operates in the gap where rules do not yet exist. The method requires only top-$k$ log-probabilities, available via OpenAI's \texttt{logprobs} parameter, DeepSeek via Together AI, and vLLM for open-source models. Anthropic's API does not currently expose per-token log-probabilities. The method requires only top-$k$ logprobs --- no model-internal access.

**Instance B — `related_work.tex` 213–221:**
> Static analysis and linter rules catch \emph{known}-bad patterns for specific libraries once deprecation has been encoded into a ruleset. These approaches are complementary to \eps and represent the floor of detection... \eps operates in the gap...

**Note:** The literal duplication within `comparison.tex` 267–269 ("The method requires only top-$k$ log-probabilities" / "The method requires only top-$k$ logprobs") is a clear copy-paste artifact and should be fixed regardless.

---

## Cluster 7 — Two-stage architecture / review loop framing

**Appears in:**
- `abstract.tex` 32–42
- `introduction.tex` 118–130 ("The two-stage system")
- `introduction.tex` 144–149 (Contributions item 3)
- `evaluation.tex` 331–342 (review-loop intro)
- `comparison.tex` 293–300 (Summary point iii)
- `conclusion.tex` 34–41

**Type:** Architecture re-described in nearly identical terms

**Canonical location:** `evaluation.tex` §subsec:review-loop has the data; `introduction.tex` "The two-stage system" is the right narrative home.

**Proposed resolution:**
- Trim Contributions item 3 (lines 144–149): it duplicates "The two-stage system" paragraph (118–130) within the same section. Either drop it or shorten to one line: "A two-stage filter-plus-review architecture: end-to-end $94\%$ precision on $201$ entries, three models, with the reviewer directed to high-ε tokens."
- The phrase "The reviewer is not asked to re-read the function" / "The reviewer is not asked to re-read $98$ tokens" appears in 4 places (introduction 122, intro 149, evaluation 339, conclusion 21, conclusion 36, comparison 295). Pick two homes (introduction + conclusion) and remove from the others.
- The "$68.7\%$ cleared / zero late false positives / $201$ entries / three models" stat appears in abstract, introduction, evaluation, and conclusion. Acceptable in three of those; the introduction states it twice (lines 128–130 and 144–149).

**Rationale:** This is the paper's secondary architecture story. The introduction states it twice (paragraph + contribution bullet), and the conclusion re-derives it. Two statements is enough.

**Instance A — `introduction.tex` 118–130 ("The two-stage system"):**
> Stage $2$ is a parallel LLM review loop: every flagged entry is dispatched, in parallel, to a reviewer model... It is asked to examine the specific high-\eps tokens... On 201 entries across three models, a context-primed reviewer clears $68.7\%$ of flagged entries with zero late false positives.

**Instance B — `introduction.tex` 144–149 (Contributions 3):**
> A two-stage filter-plus-review architecture evaluated on 201 entries across three models (GPT-4o, GPT-4o-mini, DeepSeek V3): $94\%$ end-to-end precision (context-primed, zero late false positives) with zero observed misses at stage 1, the reviewer directed to \eps's high-entropy tokens rather than asked to re-read the function.

**Instance C — `conclusion.tex` 34–41:**
> Stage $1$ is the recall-maximizing filter; stage $2$ is a parallel LLM reviewer directed to the specific high-\eps tokens rather than asked to re-read the function. On $201$ entries across three models, a context-primed reviewer clears $68.7\%$ of flagged entries with zero late false positives.

---

## Cluster 8 — "Recall-maximizing filter" framing sentence

**Appears in:**
- `abstract.tex` 18–20
- `introduction.tex` 67–70
- `evaluation.tex` 335 ("the recall-maximizing filter into an end-to-end system")
- `conclusion.tex` 7–9
- `conclusion.tex` 35

**Type:** Same one-line framing sentence

**Canonical location:** Introduction or abstract — wherever the phrase is first established.

**Proposed resolution:**
- Five uses of "recall-maximizing filter" verbatim. The phrase is good; using it as a noun phrase is fine. The redundancy is the *full sentence* "the function-level score is a recall-maximizing filter: it flags X% with zero misses at precision Y" appearing in 3 places. Keep the abstract version and the introduction version; trim conclusion 7–9 to a shorter callback.

**Rationale:** Repeating a short label is fine and aids cohesion. Repeating the full numerical statement attached to it is not.

**Instance A — `abstract.tex` 18–20:**
> At the function level \eps is a recall-maximizing filter: on four models it flags $97.1\%$ of HIGH API-integration prompts, with zero observed misses on the 200-entry evaluated set (recall-maximizing design) and precision $=0.272$.

**Instance B — `introduction.tex` 67–70:**
> The function-level score is a recall-maximizing filter: at threshold $\eps \ge 0.30$ it flags $97\%$ of HIGH prompts with zero observed misses on our evaluated benchmark set.

**Instance C — `conclusion.tex` 7–9:**
> The function-level score is a recall-maximizing filter: on the evaluated set, it flags $97\%$ of \textsc{high} API-integration prompts with zero observed misses, at precision $0.272$.

---

## Cluster 9 — UQLM 5% one-liner

**Appears in:**
- `abstract.tex` (no explicit UQLM mention but the structural multi-sample claim)
- `introduction.tex` 115–117
- `comparison.tex` 247–253
- `related_work.tex` 126–128
- `conclusion.tex` 29–30

**Type:** Same 5%-on-our-benchmark fact stated 4 times

**Canonical location:** `comparison.tex` subsec:multisample-comp has the full evaluation context.

**Proposed resolution:** Keep introduction (sets the bar), keep comparison (full evaluation), trim conclusion to "UQLM detects 5% on the same benchmark" → already short, fine. Cut the related_work mention down: it currently introduces UQLM as "the closest off-the-shelf tool" and says "we report its performance in §sec:comparison in one sentence" — fine to reference; do not re-quote the 5% there.

**Rationale:** Minor; mostly already controlled. The introduction → comparison → conclusion path is normal forward-reference structure.

---

## Cluster 10 — "Type 3 floor / reviewer clears them fast"

**Appears in:**
- `epsilon.tex` 71–74 (Type 3 paragraph)
- `epsilon.tex` 233–241 ("Why low looks similar")
- `evaluation.tex` 279–286 (subsec:low-gt)
- `discussion.tex` 73–82 (Type 3 floor limitation)

**Type:** Same Type 3 reasoning re-stated

**Canonical location:** `epsilon.tex` Type 2/3 paragraph (50–74) is the formal definition; the rest are echoes.

**Proposed resolution:**
- `epsilon.tex` "Why low looks similar" (lines 231–241) repeats the Type 3 floor explanation already given in the Type 3 paragraph (71–74) plus the Table 1 caption — three places in the same section.
- `discussion.tex` Type 3 floor (73–82) repeats again. Cut down to: "Type 3 false-positive floor remains; the mitigation is reviewer speed (both branches pass)."

**Rationale:** The Type 2 / Type 3 distinction is genuinely subtle and benefits from one careful explanation. The pattern of "definitional, not structural; reviewer clears fast because both branches pass" is now stated four times in nearly identical words.

**Instance A — `epsilon.tex` 71–74:**
> Type 3 false positives are fast for the stage-2 reviewer to clear because both alternatives pass inspection.

**Instance B — `epsilon.tex` 240–241:**
> the stage-$2$ reviewer clears them fast because both branches pass.

**Instance C — `evaluation.tex` 281–286:**
> The $25$ non-\textsc{complete} fires are all Type 3 false positives: correct code where \eps observed hesitation between algorithm structures that are both correct...

**Instance D — `discussion.tex` 80–82:**
> a Type 3 fire localizes to a small number of tokens and the reviewer clears it fast because both alternatives pass inspection.

---

## Cluster 11 — Token attribution as "the mechanism"

**Appears in:**
- `introduction.tex` 126–128
- `comparison.tex` 144–150 ("Token attribution is not a refinement on top of the filter; it is the mechanism")
- `comparison.tex` 296–300 (Summary point iii)
- `conclusion.tex` 39–41

**Type:** Same metaphor + framing repeated

**Canonical location:** `comparison.tex` 144–150 makes it most strongly tied to the data.

**Proposed resolution:** The intro and conclusion both say "the token-level attribution is not decoration on the filter; it is the mechanism that makes the reviewer fast" in nearly identical words. Pick one. Recommend cutting from conclusion (less load-bearing there) or from introduction (since introduction already has Contributions item 3 making the same point). Keep `comparison.tex` 144–150 since it's tied to the 94% vs 50.5% comparison.

**Instance A — `introduction.tex` 126–128:**
> The token-level attribution is not decoration on the filter; it is the mechanism that makes the reviewer fast and precise.

**Instance B — `comparison.tex` 147–150:**
> Token attribution is not a refinement on top of the filter; it is the mechanism that enables $94\%$ end-to-end precision. Without it, the same review loop achieves $50.5\%$.

**Instance C — `conclusion.tex` 39–41:**
> The token-level attribution is not decoration; it is the mechanism that makes the reviewer fast enough to keep up with machine-speed code generation.

---

## Cluster 12 — Comparison.tex internal duplication ("the function-level question vs the token-level question")

**Appears in (within `comparison.tex` alone):**
- Lines 11–15 (section opening)
- Lines 96–101 ("This is the distinguishing claim. It is not a higher rate; it is a different kind of output.")
- Lines 286–300 (Summary subsection)

**Type:** Section opens with the framing, repeats it mid-section, repeats it again in summary

**Canonical location:** Section opening is fine.

**Proposed resolution:** Cut the Summary subsection's first paragraph (286–300, points i–iii). The Through-the-Review-Loop discussion (103–150) and the multi-sample subsection have already established everything the Summary "concludes." Replace the Summary with the figure caption alone, or one tight closing paragraph that points forward to §evaluation review-loop.

**Rationale:** A section that opens AND closes with the same two-axis framing, and re-states it once in the middle, is over-structured. The figure (`fig:comparison`) plus one closing sentence is enough.

---

## Not redundant (flagged for completeness)

The following look like repetition but are doing genuinely distinct work — leave alone:

- **Wang et al. deprecated-API statistics ($25$–$38\%$)** — appears in `introduction.tex` 11–15, `discussion.tex` 19–25, `related_work.tex` 16–26. Each use is positioning the citation differently (motivation / limitation / formal related-work review). Keep all three.

- **Stripe Scenario A as named example** — being referenced by name in multiple sections is normal cross-referencing. The redundancy lives in re-quoting probabilities (Cluster 4), not in re-naming the scenario.

- **Figure references to fig_token_focus / fig_token_focus_detail** — both figures are intentionally surfaced in the introduction (teaser) and the body (detail). Standard practice.

- **Confident wrongness limitation** — appears in `evaluation.tex` Scenario B+C and `discussion.tex` "Confident wrongness". Discussion explicitly references the scenarios; this is the right structure (results → discussion analyzes them).

- **Recall = 1.000 / "zero observed misses"** — repeated, but each instance specifies a different evaluated set ($n=200$ benchmark vs $n=201$ review-loop set vs the "across this set" qualifier). Keep all; the precision matters for honesty.

- **"At $1\times$ API cost" cost framing** — comparison and abstract both make this point but in different argumentative positions. Acceptable.

- **AdaDec connection (~10% token rate)** — only appears once in `related_work.tex`; not a redundancy.

- **AST filtering description** — appears in `epsilon.tex` subsec:ast (full) and `comparison.tex` (one-line callback) and `related_work.tex` (one-line callback). The full description is in only one place; the callbacks are appropriate.
