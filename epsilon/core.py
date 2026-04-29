import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from openai import OpenAI


@dataclass
class TokenEpsilon:
    token:            str
    position:         int
    line:             int
    char_offset:      int
    col_offset:       int    # column within line (0-indexed); used for AST declaration matching
    logprob:          float
    probability:      float
    epsilon:          float
    top_alternatives: list   # [(token_str, probability), ...]
    is_code_token:    bool = True  # False for noise tokens (comments, whitespace, declarations)
    # Fine-grained filter provenance — used by filtering-stage replay analysis
    is_noise_ws:   bool = False   # True if whitespace / comment / backtick fence
    is_ast_decl:   bool = False   # True if AST declaration (function name, param, local var)
    is_fence_fmt:  bool = False   # True if output-format uncertainty (first ≤2 tokens w/ backtick alt)


@dataclass
class EpsilonResult:
    code:               str
    epsilon_file:       float
    epsilon_by_line:    dict         # {line_number: float}
    epsilon_by_func:    dict         # {function_name: float}
    token_epsilons:     list         # [TokenEpsilon, ...]
    peak_tokens:        list         # top N highest-ε tokens
    flags:              list         # human-readable flag strings
    status:             str          # COMPLETE | FLAGGED | PAUSED | ABORTED
    model:              str
    prompt_tokens:      int
    completion_tokens:  int
    n_code_tokens:      int          = 0     # ε-contributing code tokens (is_code_token and above floor)
    ensemble_threshold: float | None = None  # adaptive threshold; None = cold start
    trigger:            str          = "absolute"  # "absolute" | "ensemble" | "none"


def _find_comment_start(line: str) -> int:
    """Return the index of the first '#' that begins a comment (not inside a string).

    Handles single and double quoted strings. Returns -1 if no comment found.
    """
    in_string: str | None = None
    for i, ch in enumerate(line):
        if in_string:
            if ch == in_string:
                in_string = None
        elif ch in ('"', "'"):
            in_string = ch
        elif ch == "#":
            return i
    return -1


def _is_noise_token(token_str: str, line_idx: int, token_col: int, code_lines: list) -> bool:
    """Return True for tokens that should be excluded from ε accumulation.

    Excluded:
      - Whitespace-only tokens (spaces, newlines, tabs)
      - Markdown fence tokens (backtick-only: `, ``, ```)
      - Full-line comments  (line's first non-space char is #)
      - Inline comment text (token appears after # on a code line)
    """
    stripped = token_str.strip()
    if not stripped:
        return True
    if stripped == stripped.replace("`", ""):  # token is only backticks if stripping ` gives ""
        pass  # handled below
    if all(c == "`" for c in stripped):
        return True                              # markdown fence token
    if 0 <= line_idx < len(code_lines):
        line = code_lines[line_idx]
        if line.lstrip().startswith("#"):
            return True                          # full-line comment
        comment_pos = _find_comment_start(line)
        if comment_pos >= 0 and token_col >= comment_pos:
            return True                          # inline comment
    return False


def _find_declaration_positions(code: str) -> set[tuple[int, int, int]]:
    """Return (1-indexed line, start_col, end_col) ranges for cosmetic naming tokens.

    These are tokens where the model is choosing a *name*, not an API or library.
    The choice has no effect on correctness, compatibility, or behavior — any
    valid identifier produces equally correct code.

    Marked cosmetic (excluded from ε aggregation):
      - Function / method names   (the identifier after 'def ')
      - Parameter names           (ast.arg nodes)
      - Local variable names      (ast.Name in Store context: assignments,
                                   for-loop targets, with-as targets, etc.)

    NOT marked cosmetic (consequential — left in ε aggregation):
      - Import targets  (ast.alias) — module / symbol selection matters
      - Attribute access (ast.Attribute) — method / property selection matters
      - Keywords (async, def, return, …) — structural choices matter
      - Type annotations — may carry compatibility information

    Returns an empty set if code cannot be parsed (SyntaxError is silenced).

    Returns full identifier ranges (start inclusive, end exclusive) so that
    multi-token identifiers (e.g. 'find' + '_max' for 'find_max') are all
    covered, not just the first BPE token.
    """
    import ast as _ast

    try:
        tree = _ast.parse(code)
    except SyntaxError:
        return set()

    ranges: set[tuple[int, int, int]] = set()
    code_lines = code.split("\n")

    for node in _ast.walk(tree):
        # Function / async-function names — position is right after 'def '
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            if 1 <= node.lineno <= len(code_lines):
                line = code_lines[node.lineno - 1]
                def_idx = line.find("def ")
                if def_idx >= 0:
                    start_col = def_idx + 4
                    ranges.add((node.lineno, start_col, start_col + len(node.name)))

        # Parameter names (positional, keyword, *args, **kwargs)
        elif isinstance(node, _ast.arg):
            ranges.add((node.lineno, node.col_offset, node.col_offset + len(node.arg)))

        # All local name bindings: x=…, for x in …, with … as x, [x for x in …]
        elif isinstance(node, _ast.Name) and isinstance(node.ctx, _ast.Store):
            ranges.add((node.lineno, node.col_offset, node.col_offset + len(node.id)))

    return ranges


class EpsilonWrapper:

    THRESHOLD_SOFT  = 0.30
    THRESHOLD_HARD  = 0.65
    THRESHOLD_ABORT = 0.95

    def __init__(self, client: OpenAI, config: dict = None, log_path=None):
        self.client   = client
        self.log_path = log_path   # path to JSONL log; enables ensemble detection
        self.config = {
            "threshold_soft":    self.THRESHOLD_SOFT,
            "threshold_hard":    self.THRESHOLD_HARD,
            "threshold_abort":   self.THRESHOLD_ABORT,
            "accumulation_floor": 0.30,   # tokens below this are excluded from ε aggregation
            "intervention":      "pause_prompt",
            "logging":           True,
            "log_detail":        "full",
            "thread_continuity": "auto",
            "peak_min_epsilon":  0.40,
            "peak_max_count":    8,
            "embedding_model":       "text-embedding-3-small",
            "knn_k":                 20,    # neighbors to retrieve per query
            "knn_cold_start_tokens": 500,   # below this: absolute thresholds only
            "knn_conformal_tokens":  3000,  # above this: empirical 95th-percentile
            "knn_min_n_mad":         5,     # minimum neighbor count for MAD to activate
            "knn_min_n_conformal":   20,    # minimum neighbor count for conformal to activate
        }
        if config:
            self.config.update(config)

        self._session_epsilon = 0.0
        self._session_log: list[dict] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate_code(
        self,
        prompt:  str,
        context: str = "",
        system:  str = (
            "You are an expert software engineer. "
            "Respond with raw Python code only — no markdown fences, "
            "no explanation, no prose. Output only the function(s) requested. "
            "Use concise conventional parameter names (n for integers, s for strings, "
            "lst for lists, d for dicts, f for floats). "
            "Name functions using the most direct verb-noun form from the prompt."
        ),
        model:   str = "gpt-4o",
    ) -> EpsilonResult:
        """Call the OpenAI API with logprobs enabled and return an EpsilonResult."""
        messages = self._build_messages(prompt, context, system)

        # Embedding and generation are independent — start both simultaneously.
        # text-embedding-3-small returns in ~80ms; generation takes 1–5s.
        # The embedding is ready long before the generation response arrives,
        # so this adds zero latency on the critical path.
        embedding: list = [None]
        embed_thread = None
        if self.log_path and self.config.get("logging", True):
            def _do_embed():
                embedding[0] = self._embed_prompt(prompt, context)
            embed_thread = threading.Thread(target=_do_embed, daemon=True)
            embed_thread.start()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            logprobs=True,
            top_logprobs=5,
        )

        if embed_thread is not None:
            embed_thread.join(timeout=10.0)

        result = self._process_response(response, prompt, context, embedding[0])
        if self.config["logging"]:
            self._log_result(prompt, context, result, embedding[0])
        return result

    # ------------------------------------------------------------------ #
    # Internal: message building
    # ------------------------------------------------------------------ #

    def _build_messages(self, prompt: str, context: str, system: str) -> list:
        system_text = system
        if context:
            system_text = f"{system}\n\nContext: {context}"
        return [
            {"role": "system",  "content": system_text},
            {"role": "user",    "content": prompt},
        ]

    # ------------------------------------------------------------------ #
    # Internal: embedding
    # ------------------------------------------------------------------ #

    def _embed_prompt(self, prompt: str, context: str) -> list[float] | None:
        """Return a text-embedding-3-small vector for (prompt, context).

        Returns None on any API or network failure — callers treat None as
        'no embedding available' and fall back to absolute thresholds.
        """
        text = f"{prompt}\n\nContext: {context}" if context else prompt
        try:
            resp = self.client.embeddings.create(
                model=self.config["embedding_model"],
                input=text,
            )
            return resp.data[0].embedding
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Internal: per-token epsilon
    # ------------------------------------------------------------------ #

    def _compute_token_epsilon(self, token_logprob) -> TokenEpsilon:
        top   = token_logprob.top_logprobs
        probs = [math.exp(t.logprob) for t in top]

        # Partial Shannon entropy over top-N alternatives
        entropy     = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
        normalized  = entropy / max_entropy if max_entropy > 0 else 0.0

        return TokenEpsilon(
            token=token_logprob.token,
            position=0,       # filled in by _map_tokens_to_lines
            line=0,           # filled in by _map_tokens_to_lines
            char_offset=0,    # filled in by _map_tokens_to_lines
            col_offset=0,     # filled in by _map_tokens_to_lines
            logprob=token_logprob.logprob,
            probability=math.exp(token_logprob.logprob),
            epsilon=min(1.0, normalized),
            top_alternatives=[(t.token, math.exp(t.logprob)) for t in top],
        )

    # ------------------------------------------------------------------ #
    # Internal: aggregation
    # ------------------------------------------------------------------ #

    def _aggregate_epsilon(self, token_epsilons: list) -> float:
        """File/function-level ε = max code-token ε above the noise floor.

        The compound model 1 − ∏(1 − εᵢ) is borrowed from reliability theory
        and converges to 1.0 for any sufficiently long token sequence, regardless
        of actual uncertainty. That is the wrong question.

        The right question is: "how uncertain was the model at its single worst
        code decision?" The max answers this directly, is bounded by construction,
        and never blows up. A file with one genuinely split decision scores the
        same as a file with fifty comma-placement uncertainties — which is correct,
        because one wrong API choice matters; punctuation style choices do not.
        """
        floor = self.config.get("accumulation_floor", 0.30)
        candidates = [
            te.epsilon for te in token_epsilons
            if te.is_code_token and te.epsilon > floor
        ]
        return max(candidates) if candidates else 0.0

    def _compound_epsilon(self, token_epsilons: list) -> float:
        """Retained for research/comparison. Not used for status determination."""
        floor  = self.config.get("accumulation_floor", 0.30)
        result = 1.0
        for te in token_epsilons:
            if te.is_code_token and te.epsilon > floor:
                result *= (1.0 - te.epsilon)
        return 1.0 - result

    # ------------------------------------------------------------------ #
    # Internal: position mapping
    # ------------------------------------------------------------------ #

    def _map_tokens_to_lines(self, token_epsilons: list, full_text: str) -> list:
        """Assign line numbers, char offsets, and code/noise classification to each token."""
        code_lines = full_text.split("\n")

        # Precompute absolute char offset of each line's start
        line_starts = [0]
        for line in code_lines:
            line_starts.append(line_starts[-1] + len(line) + 1)  # +1 for \n

        char_pos = 0
        line_num = 1
        for i, te in enumerate(token_epsilons):
            te.position    = i
            te.char_offset = char_pos
            te.line        = line_num
            # Column within the line (used for inline comment detection)
            line_idx  = line_num - 1
            line_start = line_starts[line_idx] if line_idx < len(line_starts) else 0
            token_col  = char_pos - line_start
            te.col_offset    = token_col
            noise            = _is_noise_token(te.token, line_idx, token_col, code_lines)
            te.is_noise_ws   = noise
            te.is_code_token = not noise
            char_pos  += len(te.token)
            line_num  += te.token.count("\n")
        return token_epsilons

    # ------------------------------------------------------------------ #
    # Internal: function boundary detection
    # ------------------------------------------------------------------ #

    def _extract_function_boundaries(self, code: str) -> dict:
        """Return {function_name: (start_line, end_line)} for def statements."""
        boundaries   = {}
        lines        = code.split("\n")
        current_func = None
        current_start = 0

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                if current_func:
                    boundaries[current_func] = (current_start, i - 1)
                raw_name     = stripped.split("(")[0]
                func_name    = raw_name.replace("async def ", "").replace("def ", "").strip()
                current_func  = func_name
                current_start = i + 1

        if current_func:
            boundaries[current_func] = (current_start, len(lines))

        return boundaries

    # ------------------------------------------------------------------ #
    # Internal: response processing
    # ------------------------------------------------------------------ #

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove leading/trailing markdown code fences (```python ... ```)."""
        import re
        # Remove ```python or ``` at start and ``` at end
        stripped = re.sub(r"^\s*```[a-zA-Z]*\n?", "", text.strip())
        stripped = re.sub(r"\n?```\s*$", "", stripped)
        return stripped.strip()

    def _process_response(self, response, prompt: str, context: str = "", embedding: list | None = None) -> EpsilonResult:
        raw_content      = response.choices[0].message.content
        logprobs_content = response.choices[0].logprobs.content

        # Strip markdown fences before any analysis — fence tokens are high-entropy
        # noise that inflate ε without representing genuine code uncertainty.
        content = self._strip_code_fences(raw_content)

        # Per-token epsilon
        token_epsilons = [self._compute_token_epsilon(t) for t in logprobs_content]
        # Map against raw_content so token positions and code_lines are in sync.
        # Tokens reference the original byte stream; raw_content is the ground truth.
        self._map_tokens_to_lines(token_epsilons, raw_content)
        # Compute how many lines the fence prefix occupies (0 if no fences stripped).
        fence_line_offset = 0
        if content and raw_content != content:
            snippet = content[:40].rstrip()
            first_line = snippet.split("\n")[0]
            idx = raw_content.find(first_line)
            if idx > 0:
                fence_line_offset = raw_content[:idx].count("\n")
        if fence_line_offset:
            for te in token_epsilons:
                te.line = max(1, te.line - fence_line_offset)

        # Mark declaration-context tokens as cosmetic (naming uncertainty, not API uncertainty).
        # Function names, parameter names, and local variable names are excluded from ε
        # aggregation — the model's split between 'fetch_weather' vs 'get_weather' is real
        # token-level uncertainty but has zero effect on correctness or compatibility.
        #
        # Uses range-overlap matching (not exact col) because BPE tokens include a leading
        # space: token ' find' starts at col 3 but the identifier 'find_max' starts at
        # col 4. Range overlap catches both the leading-space offset and multi-token
        # identifiers (e.g. 'find' + '_max' in 'find_max' are both cosmetic).
        #
        # GUARD: only filter pure-identifier tokens (after stripping whitespace).
        # Mixed tokens like '(d' (open-paren + first char of param name) or '):'
        # combine structural syntax with naming — they should NOT be filtered because
        # the structural part is consequential.
        import keyword as _keyword
        decl_ranges = _find_declaration_positions(content)
        if decl_ranges:
            decl_by_line: dict[int, list[tuple[int, int]]] = {}
            for (dcl_line, dcl_start, dcl_end) in decl_ranges:
                decl_by_line.setdefault(dcl_line, []).append((dcl_start, dcl_end))
            for te in token_epsilons:
                if te.is_code_token and te.line in decl_by_line:
                    stripped = te.token.strip()
                    is_naming_token = False
                    # Case 1: pure identifier — 'find', '_max', 'result'
                    if stripped.isidentifier() and not _keyword.iskeyword(stripped):
                        is_naming_token = True
                    # Case 2: delimiter-prefixed identifier — '(n', '(c', '*args'
                    # If ALL top alternatives share the same leading delimiter, the
                    # delimiter is certain and only the identifier name is uncertain.
                    elif (len(stripped) >= 2
                            and not stripped[0].isalnum() and stripped[0] != '_'
                            and stripped[1:].isidentifier()
                            and not _keyword.iskeyword(stripped[1:])):
                        prefix = stripped[0]
                        if all(a.strip() and a.strip()[0] == prefix
                               for a, _ in te.top_alternatives if a.strip()):
                            is_naming_token = True
                    if not is_naming_token:
                        continue
                    tok_start = te.col_offset
                    tok_end   = te.col_offset + len(te.token)
                    for (dcl_start, dcl_end) in decl_by_line[te.line]:
                        if tok_start < dcl_end and tok_end > dcl_start:
                            te.is_code_token = False
                            te.is_ast_decl   = True
                            break

        # Mark fence-format uncertainty as cosmetic noise.
        # At the very start of a response, the model is sometimes split between
        # beginning with raw code (e.g. 'def') and beginning with a markdown fence
        # ('```python'). This is output-FORMAT uncertainty, not code uncertainty —
        # the code content is identical either way. Detect this by checking whether
        # any top alternative for the first few tokens is a backtick sequence.
        _FENCE_PREFIXES = ('`',)
        for te in token_epsilons:
            if te.is_code_token and te.position <= 2:
                if any(alt.strip().startswith(_FENCE_PREFIXES) for alt, _ in te.top_alternatives):
                    te.is_code_token = False
                    te.is_fence_fmt  = True

        # Count ε-contributing code tokens — used for the token-budget domain switch.
        # Uses the same floor as _aggregate_epsilon so the count reflects exactly
        # what enters the aggregation, not all filtered tokens.
        floor = self.config.get("accumulation_floor", 0.30)
        n_code_tokens = sum(
            1 for te in token_epsilons
            if te.is_code_token and te.epsilon > floor
        )

        # File-level epsilon — max code-token ε
        epsilon_file = self._aggregate_epsilon(token_epsilons)

        # Per-line epsilon (max token epsilon per line — code tokens only)
        epsilon_by_line: dict[int, float] = {}
        for te in token_epsilons:
            if not te.is_code_token:
                continue
            epsilon_by_line[te.line] = max(epsilon_by_line.get(te.line, 0.0), te.epsilon)

        # Per-function epsilon — max code-token ε within function boundaries
        func_boundaries = self._extract_function_boundaries(content)
        epsilon_by_func: dict[str, float] = {}
        for func, (start, end) in func_boundaries.items():
            func_tokens = [te for te in token_epsilons if start <= te.line <= end]
            epsilon_by_func[func] = (
                self._aggregate_epsilon(func_tokens) if func_tokens else 0.0
            )

        # Peak tokens — code tokens only; comments excluded from flags
        min_eps    = self.config["peak_min_epsilon"]
        max_count  = self.config["peak_max_count"]
        peak_tokens = sorted(
            [te for te in token_epsilons if te.is_code_token and te.epsilon > min_eps],
            key=lambda t: t.epsilon,
            reverse=True,
        )[:max_count]

        flags                        = self._generate_flags(peak_tokens)
        neighborhood_eps, nbr_tokens = self._load_neighborhood(embedding)
        n_tokens_total               = nbr_tokens + n_code_tokens
        ensemble_threshold           = self._compute_ensemble_threshold(neighborhood_eps, n_tokens_total)
        status, trigger              = self._determine_status(epsilon_file, ensemble_threshold)

        result = EpsilonResult(
            code=content,
            epsilon_file=epsilon_file,
            epsilon_by_line=epsilon_by_line,
            epsilon_by_func=epsilon_by_func,
            token_epsilons=token_epsilons,
            peak_tokens=peak_tokens,
            flags=flags,
            status=status,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            n_code_tokens=n_code_tokens,
            ensemble_threshold=ensemble_threshold,
            trigger=trigger,
        )

        return result

    # ------------------------------------------------------------------ #
    # Internal: flags
    # ------------------------------------------------------------------ #

    def _generate_flags(self, peak_tokens: list) -> list:
        flags = []
        for te in peak_tokens:
            # Deduplicate alternatives by stripped display value — filters same-glyph
            # tokenizer artifacts where the same character appears multiple times.
            seen_labels: set[str] = set()
            deduped_alts = []
            for tok, prob in te.top_alternatives:
                display = tok.strip() or repr(tok)
                if display not in seen_labels:
                    seen_labels.add(display)
                    deduped_alts.append((tok, prob))

            label = te.token.strip() or repr(te.token)
            flag  = f'"{label}" (line {te.line}) ε={te.epsilon:.2f}'
            if len(deduped_alts) > 1:
                chosen = te.token.strip()
                other  = next(
                    (a for a in deduped_alts if a[0].strip() != chosen),
                    None,
                )
                if other:
                    flag += f' — alternative "{other[0].strip()}" was nearly as likely (P={other[1]:.2f})'
            flags.append(flag)
        return flags

    # ------------------------------------------------------------------ #
    # Internal: neighborhood loading and ensemble threshold
    # ------------------------------------------------------------------ #

    def _load_neighborhood(self, embedding: list[float] | None) -> tuple[list[float], int]:
        """Return (epsilon_values, total_n_code_tokens) for the K-NN neighborhood.

        When embedding is None (embedding call failed or logging disabled),
        returns ([], 0) — which triggers cold start / absolute thresholds.
        This is Option A: honest cold start rather than a potentially misleading
        fallback to a heterogeneous flat pool.
        """
        if not self.log_path or embedding is None:
            return [], 0
        from .logger import EpsilonLogger
        return EpsilonLogger(self.log_path).get_neighborhood(
            embedding, self.config["knn_k"]
        )

    def _compute_ensemble_threshold(self, epsilons: list[float], n_tokens_total: int) -> float | None:
        """Compute the adaptive ensemble threshold from neighborhood ε values.

        Domain switching uses BOTH the token budget AND the run count because
        neither alone is sufficient:
          - Many low-count runs can hit the token budget with too few ε values
            to estimate the distribution reliably.
          - A few long runs may have abundant tokens but not enough samples for
            a stable tail estimate.

        Stages (both conditions must hold for a stage to activate):
          conformal : n_tokens_total >= knn_conformal_tokens  AND n >= knn_min_n_conformal
                      → empirical 95th-percentile (conformal coverage guarantee)
          MAD       : n_tokens_total >= knn_cold_start_tokens AND n >= knn_min_n_mad
                      → modified Z-score (Iglewicz & Hoaglin 1993):
                        threshold = median + 3.0 * MAD / 0.6745
          cold start: otherwise → returns None (absolute thresholds take over)
        """
        import statistics

        n                 = len(epsilons)
        cold_start_tokens = self.config["knn_cold_start_tokens"]
        conformal_tokens  = self.config["knn_conformal_tokens"]
        min_n_mad         = self.config["knn_min_n_mad"]
        min_n_conformal   = self.config["knn_min_n_conformal"]

        if n_tokens_total >= conformal_tokens and n >= min_n_conformal:
            sorted_eps = sorted(epsilons)
            idx        = min(int(0.95 * n), n - 1)
            return sorted_eps[idx]

        if n_tokens_total >= cold_start_tokens and n >= min_n_mad:
            median = statistics.median(epsilons)
            mad    = statistics.median([abs(e - median) for e in epsilons])
            if mad == 0:
                return None   # degenerate: all neighbors returned identical ε
            return min(1.0, median + 3.0 * mad / 0.6745)

        return None   # cold start — not enough token budget or neighbor count

    # ------------------------------------------------------------------ #
    # Internal: status
    # ------------------------------------------------------------------ #

    def _determine_status(
        self, epsilon_file: float, ensemble_threshold: float | None = None
    ) -> tuple[str, str]:
        """Return (status, trigger).

        trigger is one of: "absolute" | "ensemble" | "none"

        Absolute thresholds are always active and determine severity.
        The ensemble threshold adds a path to FLAGGED for values that are
        statistically anomalous for this project even if below threshold_soft.
        """
        if epsilon_file > self.config["threshold_abort"]:
            return "ABORTED", "absolute"
        if epsilon_file > self.config["threshold_hard"]:
            return "PAUSED", "absolute"
        if epsilon_file > self.config["threshold_soft"]:
            return "FLAGGED", "absolute"
        # Below all absolute thresholds — check ensemble
        if ensemble_threshold is not None and epsilon_file > ensemble_threshold:
            return "FLAGGED", "ensemble"
        return "COMPLETE", "none"

    # ------------------------------------------------------------------ #
    # Internal: logging
    # ------------------------------------------------------------------ #

    def _log_result(self, prompt: str, context: str, result: EpsilonResult, embedding: list | None):
        entry = {
            "timestamp":          datetime.now().isoformat(timespec="seconds"),
            "prompt":             prompt[:120] + "..." if len(prompt) > 120 else prompt,
            "model":              result.model,
            "epsilon":            round(result.epsilon_file, 4),
            "status":             result.status,
            "trigger":            result.trigger,
            "ensemble_threshold": round(result.ensemble_threshold, 4) if result.ensemble_threshold is not None else None,
            "completion_tokens":  result.completion_tokens,
            "n_code_tokens":      result.n_code_tokens,
            "peak_flags":         result.flags[:3],
        }
        self._session_log.append(entry)
        if self.log_path:
            from .logger import EpsilonLogger
            EpsilonLogger(self.log_path).append(prompt, result, context=context, embedding=embedding)
