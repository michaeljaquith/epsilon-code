"""
Generate Figure 1 for "Beyond Hallucination Detection: Measuring Consequential
Uncertainty in LLM-Generated Code".

Three-panel anatomy of a flagged token (Stripe Scenario A, GPT-4o):
  Left   - the fluent output a reviewer sees, with the load-bearing token marked
  Middle - the hidden top-k distribution at that one token: 52/47 split, eps=0.878
  Right  - the downstream review surface: 98 tokens -> 10 flagged (-89%)

Designed for 0.85\\textwidth in a single-column layout.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Palette: muted, print-safe, colour-blind friendly
C_FLAG     = "#C0392B"   # the deprecated / high-eps token
C_ALT      = "#2C7FB8"   # the alternative branch
C_TAIL     = "#9AA0A6"   # other top-k entries
C_CONF     = "#BFBFBF"   # confident-boilerplate tokens
C_INK      = "#1A1A1A"
C_MUTED    = "#5A6470"
C_BG_PANEL = "#FAFAFA"
C_THRESH   = "#444444"

# ---------------------------------------------------------------------------
# Figure scaffold
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11.0, 3.6))
# Three panels with a tiny gap between; bottom strip reserved for caption-tier labels
gs = fig.add_gridspec(
    nrows=1, ncols=3,
    width_ratios=[1.55, 1.10, 1.05],
    left=0.035, right=0.985, top=0.86, bottom=0.10,
    wspace=0.32,
)

ax_code   = fig.add_subplot(gs[0, 0])
ax_dist   = fig.add_subplot(gs[0, 1])
ax_burden = fig.add_subplot(gs[0, 2])

for ax in (ax_code, ax_dist, ax_burden):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

# Master suptitle-style header on each panel via fig.text so we control kerning
def panel_header(x, y, num, title):
    fig.text(x, y, num, fontsize=9.5, fontweight="bold", color=C_INK,
             ha="left", va="bottom")
    fig.text(x + 0.022, y, title, fontsize=9.5, color=C_INK,
             ha="left", va="bottom")

panel_header(0.045, 0.905, "1.", "What the reviewer sees")
panel_header(0.405, 0.905, "2.", "What the model nearly did instead")
panel_header(0.715, 0.905, "3.", "Where ε directs the review")

# ---------------------------------------------------------------------------
# Panel 1 - the fluent output
# ---------------------------------------------------------------------------
ax_code.set_xlim(0, 1)
ax_code.set_ylim(0, 1)

# Soft panel background
ax_code.add_patch(FancyBboxPatch(
    (0.005, 0.04), 0.99, 0.92,
    boxstyle="round,pad=0.005,rounding_size=0.012",
    facecolor=C_BG_PANEL, edgecolor="#E2E2E2", linewidth=0.7,
    transform=ax_code.transAxes, zorder=0,
))

# Render the generated line of code as separate spans so we can colour the
# flagged token. Using a pseudo monospace via Courier serif fallback.
code_font = {"family": "monospace", "size": 9.2}

# Two lines so the call fits comfortably at panel width
line1_segs = [
    ("stripe.", C_INK, False),
    ("Charge", C_FLAG, True),       # the flagged, load-bearing token
    (".create(", C_INK, False),
]
line2_segs = [
    ("    amount=5000, currency=\"usd\", source=token", C_INK, False),
]
line3_segs = [
    (")", C_INK, False),
]

def render_line(ax, segs, x0, y, highlight_box=False):
    x = x0
    for txt, col, hl in segs:
        if hl:
            # Subtle red underline + bold for the flagged token
            t = ax.text(x, y, txt, color=col, fontweight="bold",
                        ha="left", va="center", **code_font)
            # measure approximate width via renderer
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
            inv = ax.transData.inverted()
            (x_left, _), (x_right, _) = inv.transform(bb)[[0, 1]] if bb.width else ((x, 0), (x, 0))
            # underline
            ax.plot([x_left, x_right], [y - 0.075, y - 0.075],
                    color=C_FLAG, lw=1.4, solid_capstyle="butt")
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

render_line(ax_code, line1_segs, 0.05, 0.62)
render_line(ax_code, line2_segs, 0.05, 0.50)
render_line(ax_code, line3_segs, 0.05, 0.38)

# Annotation ABOVE the flagged token — uses the empty header space, never
# crosses other code lines.
caret_x = 0.165
charge_top_y = 0.685   # just above the "Charge" glyph (line at y=0.62)
label_y = 0.85
# vertical line from label to glyph top
ax_code.plot([caret_x, caret_x], [label_y - 0.045, charge_top_y],
             color=C_FLAG, lw=0.7, solid_capstyle="butt", zorder=5)
# Downward arrowhead chevron just above the glyph
ax_code.plot([caret_x - 0.012, caret_x, caret_x + 0.012],
             [charge_top_y + 0.030, charge_top_y, charge_top_y + 0.030],
             color=C_FLAG, lw=0.7, solid_capstyle="round",
             solid_joinstyle="round", zorder=5)
ax_code.text(caret_x, label_y, "load-bearing token",
             color=C_FLAG, fontsize=8.2, ha="center", va="bottom")

# Verdict line under the code
ax_code.text(0.05, 0.22,
             "Fluent.  Compiles.  Deprecated since 2019.",
             color=C_MUTED, style="italic", fontsize=8.6, ha="left", va="center")
ax_code.text(0.05, 0.11,
             "Single-sample log-prob of this output:  0.91",
             color=C_MUTED, fontsize=8.0, ha="left", va="center")

# ---------------------------------------------------------------------------
# Panel 2 - the hidden top-k distribution at the flagged token
# ---------------------------------------------------------------------------
ax_dist.set_xlim(0, 1)
ax_dist.set_ylim(0, 1)

ax_dist.add_patch(FancyBboxPatch(
    (0.005, 0.04), 0.99, 0.92,
    boxstyle="round,pad=0.005,rounding_size=0.018",
    facecolor=C_BG_PANEL, edgecolor="#E2E2E2", linewidth=0.7,
    transform=ax_dist.transAxes, zorder=0,
))

# Inset axes for a horizontal bar chart of top-k probs
inset = ax_dist.inset_axes([0.07, 0.22, 0.88, 0.62])
labels = ["Charge",
          "PaymentIntent",
          "checkout",
          "Subscription",
          "Customer"]
probs = [0.52, 0.47, 0.006, 0.003, 0.001]
colors = [C_FLAG, C_ALT, C_TAIL, C_TAIL, C_TAIL]

ypos = np.arange(len(labels))[::-1]
inset.barh(ypos, probs, color=colors, edgecolor="none", height=0.62)

# Probability labels at end of each bar; italic role tag on the two contenders
role_tag = {0: "deprecated", 1: "current"}
for i, (y, p, col) in enumerate(zip(ypos, probs, colors)):
    inset.text(p + 0.010, y, f"{p:.2f}",
               va="center", ha="left", fontsize=8.2, color=col,
               fontweight="bold" if col != C_TAIL else "normal")
    if i in role_tag:
        inset.text(p + 0.10, y, role_tag[i],
                   va="center", ha="left", fontsize=7.4, color=col,
                   style="italic")

inset.set_yticks(ypos)
inset.set_yticklabels(labels, fontsize=8.4,
                      family="monospace")
inset.set_xlim(0, 0.82)
inset.set_xticks([0, 0.25, 0.5])
inset.set_xticklabels(["0", "0.25", "0.5"], fontsize=7.5, color=C_MUTED)
inset.tick_params(axis="y", length=0, pad=2)
inset.tick_params(axis="x", length=2, color=C_MUTED, labelcolor=C_MUTED)

# Highlight the y-tick text for the two contenders
for tick, col in zip(inset.get_yticklabels(), [C_FLAG, C_ALT, C_TAIL, C_TAIL, C_TAIL]):
    tick.set_color(col)
    if col != C_TAIL:
        tick.set_fontweight("bold")

for spine_name, spine in inset.spines.items():
    if spine_name in ("top", "right"):
        spine.set_visible(False)
    else:
        spine.set_color("#CCCCCC")
        spine.set_linewidth(0.6)
inset.set_axisbelow(True)
inset.grid(axis="x", color="#E8E8E8", lw=0.5, zorder=0)

# ε readout — single centred line, well clear of the bars
ax_dist.text(0.50, 0.07,
             r"$\varepsilon = 0.878$  (top-$k$ normalised entropy at this token)",
             fontsize=8.6, color=C_INK,
             ha="center", va="bottom")

# (italic role tags now drawn inside the inset axes; see above)

# ---------------------------------------------------------------------------
# Panel 3 - reviewer burden reduction
# ---------------------------------------------------------------------------
ax_burden.set_xlim(0, 1)
ax_burden.set_ylim(0, 1)

ax_burden.add_patch(FancyBboxPatch(
    (0.005, 0.04), 0.99, 0.92,
    boxstyle="round,pad=0.005,rounding_size=0.018",
    facecolor=C_BG_PANEL, edgecolor="#E2E2E2", linewidth=0.7,
    transform=ax_burden.transAxes, zorder=0,
))

# Token strip: 98 tokens, 10 flagged. Render as a compact 7x14 grid.
n_tokens = 98
n_flag = 10
# Indices of flagged tokens chosen to look organic (clustered around method-selector)
flag_idx = {3, 17, 31, 32, 33, 48, 60, 71, 84, 92}
# We'll arrange in 7 rows x 14 cols
rows, cols = 7, 14
cell_w = 0.045
cell_h = 0.055
grid_x0 = 0.08
grid_y0 = 0.45
gap = 0.006

for i in range(n_tokens):
    r = i // cols
    c = i % cols
    x = grid_x0 + c * (cell_w + gap)
    y = grid_y0 + (rows - 1 - r) * (cell_h + gap)
    flagged = i in flag_idx
    face = C_FLAG if flagged else C_CONF
    edge = "none"
    rect = mpatches.FancyBboxPatch(
        (x, y), cell_w, cell_h,
        boxstyle="round,pad=0.0,rounding_size=0.006",
        facecolor=face, edgecolor=edge, linewidth=0,
        alpha=1.0 if flagged else 0.55,
    )
    ax_burden.add_patch(rect)

# Header above grid
ax_burden.text(grid_x0, 0.90, "98 generated tokens",
               fontsize=8.4, color=C_INK, ha="left", va="center")

# Legend dots
lx = grid_x0
ly = 0.355
ax_burden.add_patch(mpatches.FancyBboxPatch(
    (lx, ly), 0.022, 0.030,
    boxstyle="round,pad=0,rounding_size=0.004",
    facecolor=C_FLAG, edgecolor="none"))
ax_burden.text(lx + 0.030, ly + 0.015, "ε ≥ 0.30  (10 tokens, 10.2%)",
               fontsize=8.0, color=C_INK, va="center", ha="left")

ax_burden.add_patch(mpatches.FancyBboxPatch(
    (lx, ly - 0.060), 0.022, 0.030,
    boxstyle="round,pad=0,rounding_size=0.004",
    facecolor=C_CONF, edgecolor="none", alpha=0.55))
ax_burden.text(lx + 0.030, ly - 0.060 + 0.015,
               "confident boilerplate (88 tokens)",
               fontsize=8.0, color=C_MUTED, va="center", ha="left")

# Big payoff number
ax_burden.text(0.5, 0.165, "−89%",
               fontsize=18.0, fontweight="bold", color=C_FLAG,
               ha="center", va="center")
ax_burden.text(0.5, 0.07, "review surface vs. flagging the whole function",
               fontsize=7.8, color=C_MUTED,
               ha="center", va="center", style="italic")

# Inter-panel connectors removed — the numbered headers already cue reading order.

# ---------------------------------------------------------------------------
# Bottom strip narrative
# ---------------------------------------------------------------------------
fig.text(0.5, 0.025,
         "ε measures the within-distribution split that single-sample likelihood hides — and names the few tokens that need review.",
         ha="center", va="center", fontsize=8.6, color=C_INK, style="italic")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_dir = r"d:\Language\paper\figures"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig_token_focus_new.pdf")
fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
# Also drop a PNG for quick visual review
fig.savefig(out_path.replace(".pdf", ".png"), format="png", dpi=220, bbox_inches="tight", pad_inches=0.05)
print(f"wrote {out_path}")
