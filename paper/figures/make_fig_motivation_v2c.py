"""
Figure 1 — Variant 2C.

Rhetorical-arc layout: a vertical reading order that mirrors the paper's
argument structure.

  Row 1  PROBLEM       — the fluent code the reviewer sees, with caret
                         and the "Fluent. Compiles. Deprecated." verdict
  Row 2  HIDDEN SPLIT  — a single dramatic stacked bar (Charge 52 vs
                         PaymentIntent 47) under one headline number ε=0.878
                         that exposes the within-distribution near-tie
  Row 3  SOLUTION      — the 98-token grid + the -89% review-burden payoff

Wider single panels per row mean each element gets the full figure width and
fonts can be substantially larger. Designed for ~11in width at 0.85\\textwidth.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_FLAG     = "#C0392B"
C_ALT      = "#2C7FB8"
C_TAIL     = "#9AA0A6"
C_CONF     = "#BFBFBF"
C_INK      = "#1A1A1A"
C_MUTED    = "#5A6470"
C_BG_PANEL = "#FAFAFA"

# ---------------------------------------------------------------------------
# Figure scaffold — three horizontal bands
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11.0, 7.6))
gs = fig.add_gridspec(
    nrows=3, ncols=1,
    height_ratios=[1.05, 0.85, 1.30],
    left=0.045, right=0.955, top=0.955, bottom=0.05,
    hspace=0.24,
)

ax_code   = fig.add_subplot(gs[0, 0])
ax_split  = fig.add_subplot(gs[1, 0])
ax_pay    = fig.add_subplot(gs[2, 0])

for ax in (ax_code, ax_split, ax_pay):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# ---------------------------------------------------------------------------
# Row 1 — PROBLEM
# ---------------------------------------------------------------------------
ax_code.add_patch(FancyBboxPatch(
    (0.005, 0.02), 0.99, 0.96,
    boxstyle="round,pad=0.005,rounding_size=0.012",
    facecolor=C_BG_PANEL, edgecolor="#E2E2E2", linewidth=0.8,
    transform=ax_code.transAxes, zorder=0,
))

# Stage tag
ax_code.text(0.020, 0.95, "1.  PROBLEM",
             fontsize=10.0, color=C_MUTED, ha="left", va="top",
             fontweight="bold", family="serif")
ax_code.text(0.020, 0.85, "What the reviewer sees",
             fontsize=14.5, color=C_INK, ha="left", va="top",
             fontweight="bold")

code_font = {"family": "monospace", "size": 13.0}

line1_segs = [
    ("stripe.", C_INK, False),
    ("Charge", C_FLAG, True),
    (".create(amount=5000, currency=\"usd\", source=token)", C_INK, False),
]

def render_line(ax, segs, x0, y, lw=2.0):
    x = x0
    for txt, col, hl in segs:
        if hl:
            t = ax.text(x, y, txt, color=col, fontweight="bold",
                        ha="left", va="center", **code_font)
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = ax.transData.inverted()
            (x_left, _), (x_right, _) = inv.transform(bb)[[0, 1]] if bb.width else ((x, 0), (x, 0))
            ax.plot([x_left, x_right], [y - 0.085, y - 0.085],
                    color=C_FLAG, lw=lw, solid_capstyle="butt")
            x = x_right
        else:
            t = ax.text(x, y, txt, color=col,
                        ha="left", va="center", **code_font)
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = ax.transData.inverted()
            (x_left, _), (x_right, _) = inv.transform(bb)[[0, 1]] if bb.width else ((x, 0), (x, 0))
            x = x_right
    return x

render_line(ax_code, line1_segs, 0.04, 0.50)

# Caret label ABOVE the flagged token, vertical drop line, no curved arrow
caret_x = 0.123
charge_top_y = 0.585     # just above the glyph at y=0.50
label_y      = 0.69      # safely below the header at y=0.85
ax_code.plot([caret_x, caret_x], [label_y - 0.025, charge_top_y],
             color=C_FLAG, lw=0.9, solid_capstyle="butt", zorder=5)
ax_code.plot([caret_x - 0.010, caret_x, caret_x + 0.010],
             [charge_top_y + 0.025, charge_top_y, charge_top_y + 0.025],
             color=C_FLAG, lw=0.9, solid_capstyle="round",
             solid_joinstyle="round", zorder=5)
ax_code.text(caret_x, label_y, "load-bearing token",
             color=C_FLAG, fontsize=11.0, ha="center", va="bottom")

# Verdict line (italic) and log-prob
ax_code.text(0.04, 0.25,
             "Fluent.   Compiles.   Deprecated since 2019.",
             color=C_INK, style="italic", fontsize=12.5, ha="left", va="center",
             fontweight="bold")
ax_code.text(0.04, 0.10,
             "Single-sample log-prob of this output:  0.91",
             color=C_MUTED, fontsize=11.5, ha="left", va="center")

# ---------------------------------------------------------------------------
# Row 2 — HIDDEN SPLIT (one dramatic stacked bar + ε headline)
# ---------------------------------------------------------------------------
ax_split.add_patch(FancyBboxPatch(
    (0.005, 0.02), 0.99, 0.96,
    boxstyle="round,pad=0.005,rounding_size=0.012",
    facecolor=C_BG_PANEL, edgecolor="#E2E2E2", linewidth=0.8,
    transform=ax_split.transAxes, zorder=0,
))

ax_split.text(0.020, 0.91, "2.  HIDDEN SPLIT",
              fontsize=10.0, color=C_MUTED, ha="left", va="top",
              fontweight="bold")
ax_split.text(0.020, 0.79, "What the model nearly did instead",
              fontsize=14.5, color=C_INK, ha="left", va="top",
              fontweight="bold")

# Stacked horizontal bar across most of the width
bar_left   = 0.040
bar_right  = 0.700
bar_y      = 0.36
bar_h      = 0.20
total      = 1.0
charge_p   = 0.52
payment_p  = 0.47
tail_p     = 0.01

w = bar_right - bar_left
seg_charge_w  = w * charge_p
seg_payment_w = w * payment_p
seg_tail_w    = w * tail_p

ax_split.add_patch(mpatches.Rectangle(
    (bar_left, bar_y), seg_charge_w, bar_h,
    facecolor=C_FLAG, edgecolor="none"))
ax_split.add_patch(mpatches.Rectangle(
    (bar_left + seg_charge_w, bar_y), seg_payment_w, bar_h,
    facecolor=C_ALT, edgecolor="none"))
ax_split.add_patch(mpatches.Rectangle(
    (bar_left + seg_charge_w + seg_payment_w, bar_y), seg_tail_w, bar_h,
    facecolor=C_TAIL, edgecolor="none", alpha=0.7))

# Segment labels INSIDE the bar (white text)
ax_split.text(bar_left + seg_charge_w / 2, bar_y + bar_h / 2,
              "Charge  0.52", color="white", fontsize=12.0,
              fontweight="bold", ha="center", va="center",
              family="monospace")
ax_split.text(bar_left + seg_charge_w + seg_payment_w / 2, bar_y + bar_h / 2,
              "PaymentIntent  0.47", color="white", fontsize=12.0,
              fontweight="bold", ha="center", va="center",
              family="monospace")

# Below-bar role tags
ax_split.text(bar_left + seg_charge_w / 2, bar_y - 0.06,
              "deprecated", color=C_FLAG, fontsize=10.5,
              style="italic", ha="center", va="top")
ax_split.text(bar_left + seg_charge_w + seg_payment_w / 2, bar_y - 0.06,
              "current API", color=C_ALT, fontsize=10.5,
              style="italic", ha="center", va="top")

# Tail callout (arrow + label) — point to the tiny grey sliver
tail_cx = bar_left + seg_charge_w + seg_payment_w + seg_tail_w / 2
ax_split.annotate(
    "+ 3 long-tail tokens (0.010)",
    xy=(tail_cx, bar_y + bar_h),
    xytext=(tail_cx + 0.005, bar_y + bar_h + 0.20),
    fontsize=10.0, color=C_MUTED, ha="left", va="bottom",
    arrowprops=dict(arrowstyle="-", color=C_MUTED, lw=0.7,
                    connectionstyle="arc3,rad=0"),
)

# Headline ε number on the right side of the row
ax_split.text(0.965, 0.62, r"$\varepsilon = 0.878$",
              fontsize=22.0, fontweight="bold", color=C_INK,
              ha="right", va="center")
ax_split.text(0.965, 0.40,
              "top-k normalised entropy",
              fontsize=11.0, color=C_MUTED, ha="right", va="center",
              style="italic")
ax_split.text(0.965, 0.27,
              "at this single token",
              fontsize=11.0, color=C_MUTED, ha="right", va="center",
              style="italic")

# Bottom narrative aligned with bar
ax_split.text(0.040, 0.09,
              "A near-tie that single-sample likelihood (0.91) hides entirely.",
              fontsize=11.5, color=C_INK, style="italic", ha="left", va="center")

# ---------------------------------------------------------------------------
# Row 3 — SOLUTION (token grid + payoff)
# ---------------------------------------------------------------------------
ax_pay.add_patch(FancyBboxPatch(
    (0.005, 0.02), 0.99, 0.96,
    boxstyle="round,pad=0.005,rounding_size=0.012",
    facecolor=C_BG_PANEL, edgecolor="#E2E2E2", linewidth=0.8,
    transform=ax_pay.transAxes, zorder=0,
))

ax_pay.text(0.020, 0.95, "3.  SOLUTION",
            fontsize=10.0, color=C_MUTED, ha="left", va="top",
            fontweight="bold")
ax_pay.text(0.020, 0.87, "Where ε directs the review",
            fontsize=14.5, color=C_INK, ha="left", va="top",
            fontweight="bold")

# Left half: token grid 7x14
n_tokens = 98
flag_idx = {3, 17, 31, 32, 33, 48, 60, 71, 84, 92}
rows, cols = 7, 14
cell_w = 0.034
cell_h = 0.062
grid_x0 = 0.040
grid_y0 = 0.20
gap_x = 0.005
gap_y = 0.010

for i in range(n_tokens):
    r = i // cols
    c = i % cols
    x = grid_x0 + c * (cell_w + gap_x)
    y = grid_y0 + (rows - 1 - r) * (cell_h + gap_y)
    flagged = i in flag_idx
    face = C_FLAG if flagged else C_CONF
    rect = mpatches.FancyBboxPatch(
        (x, y), cell_w, cell_h,
        boxstyle="round,pad=0.0,rounding_size=0.005",
        facecolor=face, edgecolor="none", linewidth=0,
        alpha=1.0 if flagged else 0.55,
    )
    ax_pay.add_patch(rect)

ax_pay.text(grid_x0, 0.74, "98 generated tokens",
            fontsize=11.0, color=C_MUTED, ha="left", va="center")

# Legend below grid
lx = grid_x0
ly = 0.085
ax_pay.add_patch(mpatches.FancyBboxPatch(
    (lx, ly), 0.020, 0.045,
    boxstyle="round,pad=0,rounding_size=0.004",
    facecolor=C_FLAG, edgecolor="none"))
ax_pay.text(lx + 0.030, ly + 0.022,
            r"$\varepsilon \geq 0.30$  (10 tokens, 10.2%)",
            fontsize=10.5, color=C_INK, va="center", ha="left")

ax_pay.add_patch(mpatches.FancyBboxPatch(
    (lx + 0.30, ly), 0.020, 0.045,
    boxstyle="round,pad=0,rounding_size=0.004",
    facecolor=C_CONF, edgecolor="none", alpha=0.55))
ax_pay.text(lx + 0.30 + 0.030, ly + 0.022,
            "confident boilerplate (88 tokens)",
            fontsize=10.5, color=C_MUTED, va="center", ha="left")

# Right half: big payoff number
ax_pay.text(0.78, 0.55, "−89%",
            fontsize=46.0, fontweight="bold", color=C_FLAG,
            ha="center", va="center")
ax_pay.text(0.78, 0.27,
            "review surface",
            fontsize=13.0, color=C_INK, ha="center", va="center",
            fontweight="bold")
ax_pay.text(0.78, 0.18,
            "vs. flagging the whole function",
            fontsize=11.0, color=C_MUTED, ha="center", va="center",
            style="italic")

# Vertical separator between grid block and payoff
ax_pay.plot([0.62, 0.62], [0.10, 0.80], color="#DDDDDD", lw=0.8, zorder=1)

# ---------------------------------------------------------------------------
# Bottom narrative
# ---------------------------------------------------------------------------
fig.text(0.5, 0.018,
         "ε measures the within-distribution split that single-sample likelihood hides — and names the few tokens that need review.",
         ha="center", va="center", fontsize=11.5, color=C_INK, style="italic")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = r"d:\Language\paper\figures"
os.makedirs(out_dir, exist_ok=True)
out_pdf = os.path.join(out_dir, "fig_motivation_v2c.pdf")
out_png = os.path.join(out_dir, "fig_motivation_v2c.png")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.05)
fig.savefig(out_png, format="png", dpi=220, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")
