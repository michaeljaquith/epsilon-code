"""
Terminal rendering for EpsilonResult objects.
Uses `rich` for color output. Falls back to plain text if rich is not installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import EpsilonResult, EpsilonWrapper

try:
    from rich.console import Console
    from rich.panel   import Panel
    from rich.text    import Text
    from rich.table   import Table
    from rich         import box
    _RICH = True
except ImportError:
    _RICH = False


# ------------------------------------------------------------------ #
# Colour helpers
# ------------------------------------------------------------------ #

def _epsilon_color(eps: float) -> str:
    if eps >= 0.65:
        return "bold red"
    if eps >= 0.30:
        return "bold yellow"
    return "green"


def _status_color(status: str) -> str:
    return {
        "COMPLETE": "green",
        "FLAGGED":  "yellow",
        "PAUSED":   "bold red",
        "ABORTED":  "bold red",
    }.get(status, "white")


# ------------------------------------------------------------------ #
# Public rendering functions
# ------------------------------------------------------------------ #

def render_result(result: "EpsilonResult") -> None:
    """Print a full, human-readable summary of an EpsilonResult."""
    if _RICH:
        _render_rich(result)
    else:
        _render_plain(result)


def render_token_map(result: "EpsilonResult", min_epsilon: float = 0.40) -> None:
    """Print a filtered token-level ε table."""
    tokens = [te for te in result.token_epsilons if te.epsilon >= min_epsilon]
    if not tokens:
        print(f"No tokens with ε ≥ {min_epsilon:.2f}")
        return

    if _RICH:
        console = Console()
        table = Table(
            title=f"Token map (ε ≥ {min_epsilon:.2f})",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("Pos",   style="dim",       width=6)
        table.add_column("Line",  style="dim",       width=6)
        table.add_column("Token", style="bold white", width=20)
        table.add_column("P",     width=8)
        table.add_column("ε",     width=8)
        table.add_column("Top alternatives", width=50)

        for te in sorted(tokens, key=lambda t: t.epsilon, reverse=True):
            alts_str = "  ".join(
                f"{tok.strip()!r}={p:.2f}" for tok, p in te.top_alternatives[:3]
            )
            table.add_row(
                str(te.position),
                str(te.line),
                repr(te.token),
                f"{te.probability:.3f}",
                Text(f"{te.epsilon:.3f}", style=_epsilon_color(te.epsilon)),
                alts_str,
            )
        console.print(table)
    else:
        print(f"\nToken map (ε ≥ {min_epsilon:.2f}):")
        print(f"{'Pos':>5}  {'Line':>5}  {'Token':20}  {'P':>7}  {'ε':>7}  Alternatives")
        print("-" * 80)
        for te in sorted(tokens, key=lambda t: t.epsilon, reverse=True):
            alts_str = "  ".join(
                f"{tok.strip()!r}={p:.2f}" for tok, p in te.top_alternatives[:3]
            )
            print(
                f"{te.position:>5}  {te.line:>5}  {repr(te.token):20}  "
                f"{te.probability:>7.3f}  {te.epsilon:>7.3f}  {alts_str}"
            )


def render_session_summary(wrapper: "EpsilonWrapper") -> None:
    """Print a summary of all queries in the current session."""
    log = wrapper._session_log
    if not log:
        print("No queries in this session yet.")
        return

    if _RICH:
        console = Console()
        table = Table(title="Session summary", box=box.SIMPLE_HEAVY)
        table.add_column("Time",    style="dim",  width=10)
        table.add_column("ε",       width=8)
        table.add_column("Status",  width=10)
        table.add_column("Tokens",  width=8)
        table.add_column("Prompt",  width=60)

        for entry in log:
            ts     = entry["timestamp"].split("T")[1]
            eps    = entry["epsilon"]
            status = entry["status"]
            table.add_row(
                ts,
                Text(f"{eps:.3f}", style=_epsilon_color(eps)),
                Text(status, style=_status_color(status)),
                str(entry["completion_tokens"]),
                entry["prompt"][:60],
            )
        console.print(table)
    else:
        print("\nSession summary:")
        print(f"{'Time':10}  {'ε':>7}  {'Status':10}  {'Tokens':>6}  Prompt")
        print("-" * 80)
        for entry in log:
            ts = entry["timestamp"].split("T")[1]
            print(
                f"{ts:10}  {entry['epsilon']:>7.3f}  {entry['status']:10}  "
                f"{entry['completion_tokens']:>6}  {entry['prompt'][:50]}"
            )


# ------------------------------------------------------------------ #
# Rich renderer
# ------------------------------------------------------------------ #

def _render_rich(result: "EpsilonResult") -> None:
    console = Console()
    SEP = "━" * 55

    # Header
    status_text = Text(f"⚠  {result.status}", style=_status_color(result.status))

    console.print()
    console.print(SEP)
    console.print(" ε CODE GENERATION RESULT", style="bold white")
    console.print(SEP)
    console.print(f" Status:  ", end=""); console.print(status_text)
    if result.trigger == "ensemble" and result.ensemble_threshold is not None:
        console.print(
            f" Trigger: distributional anomaly  "
            f"(ε={result.epsilon_file:.3f} > ensemble p95={result.ensemble_threshold:.3f})",
            style="yellow",
        )
    console.print(f" Model:   {result.model}")
    if result.epsilon_by_func:
        for fname, feps in result.epsilon_by_func.items():
            console.print(
                f" Function: {fname}()  ",
                end="",
            )
            console.print(Text(f"ε={feps:.3f}", style=_epsilon_color(feps)))
    console.print(SEP)
    console.print()

    # Code with inline annotations
    lines = result.code.split("\n")
    for i, line in enumerate(lines, start=1):
        line_eps = result.epsilon_by_line.get(i, 0.0)
        num      = Text(f"{i:>3}  ", style="dim")
        content  = Text(line)
        if line_eps >= 0.65:
            annotation = Text(f"  ◄ ⚠ ε={line_eps:.2f}", style="bold red")
        elif line_eps >= 0.30:
            annotation = Text(f"  ◄ ε={line_eps:.2f}", style="yellow")
        else:
            annotation = Text("")
        console.print(num, end="")
        console.print(content, end="")
        console.print(annotation)

    console.print()

    # Uncertainty flags
    if result.peak_tokens:
        console.print(SEP)
        console.print(" UNCERTAINTY FLAGS", style="bold white")
        console.print(SEP)
        for te in result.peak_tokens:
            label = te.token.strip() or repr(te.token)
            console.print(
                f"\n Line {te.line} — ",
                end="",
            )
            console.print(
                Text(f'"{label}" ε={te.epsilon:.2f}', style=_epsilon_color(te.epsilon)),
                end="",
            )
            console.print()
            for tok, prob in te.top_alternatives[:3]:
                tok_label = tok.strip() or repr(tok)
                console.print(f"   {tok_label:30}  P={prob:.3f}")
        console.print()
        console.print(SEP)

    # Action prompt for PAUSED/ABORTED
    if result.status in ("PAUSED", "ABORTED"):
        console.print()
        console.print(
            "  [C] Continue    [R] Regenerate    [I] Token detail    [A] Abort",
            style="bold white",
        )
        console.print(SEP)


# ------------------------------------------------------------------ #
# Plain-text fallback
# ------------------------------------------------------------------ #

def _render_plain(result: "EpsilonResult") -> None:
    SEP = "=" * 55
    print(f"\n{SEP}")
    print(" ε CODE GENERATION RESULT")
    print(SEP)
    print(f" Status:  {result.status}  (ε = {result.epsilon_file:.3f})")
    if result.trigger == "ensemble" and result.ensemble_threshold is not None:
        print(f" Trigger: distributional anomaly  (ensemble p95={result.ensemble_threshold:.3f})")
    print(f" Model:   {result.model}")
    if result.epsilon_by_func:
        for fname, feps in result.epsilon_by_func.items():
            print(f" Function: {fname}()  ε={feps:.3f}")
    print(SEP)
    print()

    lines = result.code.split("\n")
    for i, line in enumerate(lines, start=1):
        line_eps = result.epsilon_by_line.get(i, 0.0)
        if line_eps >= 0.65:
            print(f"{i:>3}  {line}  # << ε={line_eps:.2f} HIGH")
        elif line_eps >= 0.30:
            print(f"{i:>3}  {line}  # < ε={line_eps:.2f}")
        else:
            print(f"{i:>3}  {line}")
    print()

    if result.peak_tokens:
        print(SEP)
        print(" UNCERTAINTY FLAGS")
        print(SEP)
        for te in result.peak_tokens:
            label = te.token.strip() or repr(te.token)
            print(f'\n  Line {te.line} — "{label}" ε={te.epsilon:.2f}')
            for tok, prob in te.top_alternatives[:3]:
                tok_label = tok.strip() or repr(tok)
                print(f"    {tok_label:30}  P={prob:.3f}")
        print()
        print(SEP)

    if result.status in ("PAUSED", "ABORTED"):
        print()
        print("  [C] Continue    [R] Regenerate    [I] Token detail    [A] Abort")
        print(SEP)
