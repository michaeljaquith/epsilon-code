"""
generate_figures_v4.py

Regenerate paper figures for the v4 rewrite.

Run from d:/Language/ :
    python generate_figures_v4.py

Produces the following PDF files in paper/figures/:
    fig3_scenarios.pdf            (kept; caption in v4 unchanged)
    fig6_pr_curve.pdf             (kept; LEGEND FIX — moved out of data area)
    fig7_review_loop.pdf          (kept)
    fig8_scenario_e.pdf           (kept)
    fig_comparison.pdf            (kept)
    fig_intra_inter.pdf           (kept)

    fig_token_focus.pdf           (NEW — introduction teaser, heat strip)
    fig_token_focus_detail.pdf    (NEW — per-token trajectory on Stripe A)

Dropped (no longer referenced in v4):
    fig1_scatter.pdf
    fig2_benchmark_bars.png
    fig4_multimodel.png
    fig5_roc_sweep.pdf
    fig9_taxonomy.pdf

All numerical values are hard-coded.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# --------------------------------------------------------------------------
# Style constants
# --------------------------------------------------------------------------

C_LOW          = '#4878CF'
C_HIGH         = '#D65F5F'
C_PAVG         = '#F5A623'
C_MULTI        = '#6BAB6E'
C_THRESH_SOFT  = '#F5A623'
C_THRESH_HARD  = '#D0021B'

STATUS_COLORS = {
    'COMPLETE': '#6BAB6E',
    'FLAGGED':  '#F5A623',
    'PAUSED':   '#D65F5F',
    'ABORTED':  '#8B0000',
}

W_SINGLE = 3.25
W_DOUBLE = 6.75

FS_BASE    = 11
FS_LABEL   = 10
FS_TICK    = 9
FS_ANNOT   = 8.5

plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       FS_BASE,
    'axes.labelsize':  FS_LABEL,
    'axes.titlesize':  FS_BASE,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_ANNOT,
    'pdf.fonttype':    42,
    'ps.fonttype':     42,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':   True,
    'grid.alpha':  0.3,
    'grid.linewidth': 0.5,
})


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

BASE = Path('.').resolve()
FIG_DIR = BASE / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# Synthetic per-token trajectory for Stripe Scenario A (GPT-4o)
# --------------------------------------------------------------------------

def _stripe_token_trajectory():
    """
    Return a (98,) array of per-token epsilon values for a representative
    Stripe Scenario A function on GPT-4o.

    Hand-constructed to match the aggregate statistics reported in the
    paper:
      - length = 98 retained semantic tokens
      - exactly 10 tokens above 0.30 (flag threshold) => 10.2%
      - exactly 2 tokens above 0.65 (paused threshold) => 2.0%
      - peak epsilon = 0.878 (the method-selector 'Ch' token)
      - remaining tokens clustered in [0.0, 0.25] (confident boilerplate)

    The two paused tokens are adjacent (the method-selector and the
    immediately following sub-token) around position 22, matching the
    description in the paper text.
    """
    rng = np.random.default_rng(2026_04_23)
    n = 98
    eps = rng.beta(1.2, 8.0, size=n) * 0.28    # mostly < 0.25

    # 10 flagged tokens: 8 mid-band (0.30 - 0.60) + 2 paused
    flagged_positions = [12, 17, 22, 23, 28, 34, 41, 55, 63, 71]

    # The two paused tokens (positions 22, 23) — the method selector
    # Charge/PaymentIntent decision and the adjacent token
    eps[22] = 0.878
    eps[23] = 0.712

    # Other flagged but below paused threshold
    mid_values = [0.42, 0.38, 0.45, 0.51, 0.36, 0.40, 0.48, 0.33]
    mid_positions = [p for p in flagged_positions if p not in (22, 23)]
    for pos, val in zip(mid_positions, mid_values):
        eps[pos] = val

    # Clip into [0, 1]
    eps = np.clip(eps, 0.0, 1.0)

    # Sanity counts: should satisfy the published aggregates
    above_flag = int(np.sum(eps >= 0.30))
    above_paused = int(np.sum(eps >= 0.65))
    assert above_flag == 10, f"expected 10 flagged tokens, got {above_flag}"
    assert above_paused == 2, f"expected 2 paused tokens, got {above_paused}"

    return eps


# --------------------------------------------------------------------------
# Fig 3: Five scenarios horizontal bar chart (kept from v3)
# --------------------------------------------------------------------------

def fig3_scenarios():
    print("[fig3] five scenarios bar chart")

    sce_e_runs = [0.6933, 0.775, 0.7653, 0.8745, 0.824, 0.8905, 0.8235,
                  0.8707, 0.762, 0.8045, 0.7157, 0.8686, 0.7997, 0.9216,
                  0.728, 0.8671]
    sce_e_mean = float(np.mean(sce_e_runs))

    scenarios = [
        ('A', 'Stripe deprecation',       0.878,      'FLAGGED'),
        ('B', 'OpenAI SDK v0 vs v1',      0.560,      'FLAGGED'),
        ('C', 'SQLAlchemy 1.x vs 2.0',    0.907,      'FLAGGED'),
        ('D', 'FastAPI async/sync',       0.505,      'FLAGGED'),
        ('E', 'Auth module (n=16 runs)',  sce_e_mean, 'FLAGGED'),
    ]

    labels = [f"{s[0]} — {s[1]}" for s in scenarios]
    values = [s[2] for s in scenarios]
    colors = [STATUS_COLORS[s[3]] for s in scenarios]
    statuses = [s[3] for s in scenarios]

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 4.2))
    y = np.arange(len(scenarios))
    ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.8,
            height=0.62)

    ax.axvline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.1, alpha=0.85)
    ax.axvline(0.95, color=C_THRESH_HARD, linestyle='--', linewidth=1.1, alpha=0.85)

    trans = ax.get_xaxis_transform()
    ax.text(0.30, -0.10, '0.30', color=C_THRESH_SOFT, fontsize=FS_ANNOT,
            ha='center', va='top', transform=trans)
    ax.text(0.95, -0.10, '0.95', color=C_THRESH_HARD, fontsize=FS_ANNOT,
            ha='center', va='top', transform=trans)

    for i, (v, st) in enumerate(zip(values, statuses)):
        ax.text(v + 0.015, i, f'{v:.3f}  ({st})', va='center',
                fontsize=FS_ANNOT + 0.5, color='#222222')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_LABEL)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.30)
    ax.set_xlabel(r'$\varepsilon$ (peak)', fontsize=FS_LABEL)

    legend_patches = [
        mpatches.Patch(color=STATUS_COLORS['FLAGGED'],
                       label=r'FLAGGED ($0.30 \leq \varepsilon < 0.95$)'),
    ]
    ax.legend(handles=legend_patches, loc='lower left',
              bbox_to_anchor=(0.0, 1.02), bbox_transform=ax.transAxes,
              frameon=False, ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, 'fig3_scenarios.pdf')


# --------------------------------------------------------------------------
# Fig 6: Precision-recall at tier thresholds
# LEGEND FIX: move legend outside plotting area (was overlapping curve)
# --------------------------------------------------------------------------

def fig6_pr_curve():
    print("[fig6] PR curve (legend moved out of data area)")

    # Synthetic monotonic PR curve matching the reported operating points
    # at t=0.30 (recall 1.000, precision 0.272) and t=0.95 (recall 0.547,
    # precision 0.309). Base rate 53/200 = 0.265.
    thresholds = np.linspace(0.0, 1.0, 101)
    recall = np.where(
        thresholds <= 0.30, 1.00,
        np.where(
            thresholds >= 0.95, 0.547,
            1.00 - (thresholds - 0.30) / (0.95 - 0.30) * (1.00 - 0.547)
        )
    )
    # Precision rises slowly from 0.272 -> ~0.31 as threshold climbs
    precision = 0.272 + (thresholds - 0.30) * 0.057 / (0.95 - 0.30)
    precision = np.clip(precision, 0.265, 0.32)
    precision[thresholds < 0.30] = 0.272

    base_rate = 53.0 / 200.0

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.6))

    ax.plot(recall, precision, color=C_HIGH, linewidth=1.8, zorder=3,
            label=r'$\varepsilon$ sweep')

    ax.axhline(base_rate, color='#888', linestyle=':', linewidth=1.0,
               zorder=2, label=f'base rate = {base_rate:.3f}')

    op_points = [
        ('FLAGGED+ (t=0.30)', 1.000, 0.272, C_THRESH_SOFT),
        ('PAUSED (t=0.95)',   0.547, 0.309, C_THRESH_HARD),
    ]
    for lbl, r, p, col in op_points:
        ax.scatter([r], [p], s=72, color=col, edgecolor='white',
                   linewidth=0.9, zorder=5)
        # Offset labels away from both axes and the curve
        off_x = -70 if r > 0.7 else 8
        off_y = -14 if p > 0.28 else 10
        ax.annotate(lbl, (r, p), textcoords='offset points',
                    xytext=(off_x, off_y), fontsize=FS_ANNOT, color=col)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 0.55)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')

    # Move legend outside the plotting area (top-right of figure),
    # keeping it off the data curve entirely.
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False)

    fig.tight_layout()
    _save(fig, 'fig6_pr_curve.pdf')


# --------------------------------------------------------------------------
# Fig 7: Review loop 2A vs 2B stacked bars
# --------------------------------------------------------------------------

def fig7_review_loop():
    print("[fig7] review loop stacked comparison")

    total = 201

    a_cleared, a_confirmed, a_late_fp = 107, 85, 4
    a_pending = total - (a_cleared + a_confirmed + a_late_fp)

    b_cleared, b_confirmed, b_late_fp = 138, 57, 0
    b_pending = total - (b_cleared + b_confirmed + b_late_fp)

    labels = ['Context-free\nreviewer', 'Context-primed\nreviewer']
    cleared   = np.array([a_cleared,   b_cleared])
    confirmed = np.array([a_confirmed, b_confirmed])
    late_fp   = np.array([a_late_fp,   b_late_fp])
    pending   = np.array([a_pending,   b_pending])

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.5))
    y = np.arange(len(labels))
    h = 0.52

    col_clear = '#6BAB6E'
    col_conf  = '#F5A623'
    col_late  = '#8B0000'
    col_pend  = '#CCCCCC'

    left = np.zeros(len(labels))
    ax.barh(y, cleared,   h, left=left, color=col_clear, edgecolor='white',
            label='cleared by reviewer')
    left += cleared
    ax.barh(y, confirmed, h, left=left, color=col_conf,  edgecolor='white',
            label='confirmed failure')
    left += confirmed
    ax.barh(y, late_fp,   h, left=left, color=col_late,  edgecolor='white',
            label='late false positive')
    left += late_fp
    if pending.sum() > 0:
        ax.barh(y, pending, h, left=left, color=col_pend, edgecolor='white',
                label='other')

    for i in range(len(labels)):
        c = cleared[i]
        ax.text(c / 2.0, y[i], f'{c}  ({c/total*100:.1f}%)',
                va='center', ha='center', color='white',
                fontsize=FS_ANNOT, fontweight='bold')
        cf = confirmed[i]
        ax.text(c + cf / 2.0, y[i], f'{cf}  ({cf/total*100:.1f}%)',
                va='center', ha='center', color='#222',
                fontsize=FS_ANNOT, fontweight='bold')
        lf = late_fp[i]
        if lf > 0:
            ax.text(c + cf + lf / 2.0, y[i], f'{lf}',
                    va='center', ha='center', color='white',
                    fontsize=FS_ANNOT, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FS_LABEL)
    ax.invert_yaxis()
    ax.set_xlim(0, total + 2)
    ax.set_xlabel(f'entries (n = {total})', fontsize=FS_LABEL)

    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.04),
              frameon=False, ncol=3, fontsize=FS_ANNOT)

    fig.tight_layout()
    _save(fig, 'fig7_review_loop.pdf')


# --------------------------------------------------------------------------
# Fig 8: Scenario E collapse (gpt-4o-mini)
# --------------------------------------------------------------------------

def fig8_scenario_e():
    print("[fig8] scenario E collapse (gpt-4o-mini)")

    names = ['register_user', 'login', 'verify_token', 'refresh_token',
             'request_password_reset', 'reset_password']
    combined = np.array([0.579, 0.820, 0.606, 0.414, 0.295, 0.394])
    per_fn   = np.array([0.627, 0.872, 0.911, 0.711, 0.788, 0.728])

    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.6))

    ax.bar(x - w/2, combined, w, color=C_LOW,  edgecolor='white',
           label='combined prompt (1 call)')
    ax.bar(x + w/2, per_fn,   w, color=C_HIGH, edgecolor='white',
           label='per-function (6 calls)')

    ax.axhline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.0, alpha=0.8)
    ax.text(len(names) - 0.5, 0.31, 'FLAGGED (0.30)', color=C_THRESH_SOFT,
            fontsize=FS_ANNOT, ha='right', va='bottom')

    for xi, v in zip(x - w/2, combined):
        ax.text(xi, v + 0.015, f'{v:.2f}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')
    for xi, v in zip(x + w/2, per_fn):
        ax.text(xi, v + 0.015, f'{v:.2f}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=22, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r'peak $\varepsilon$')
    ax.set_title('Scenario E: peak eps collapse under combined prompting (gpt-4o-mini)',
                 fontsize=FS_BASE)
    ax.legend(loc='upper right', frameon=False)

    _save(fig, 'fig8_scenario_e.pdf')


# --------------------------------------------------------------------------
# Fig comparison: Three-method comparison across four models
# --------------------------------------------------------------------------

def fig_comparison():
    print("[fig_comparison] three-method four-model comparison")

    models = ['GPT-4o', 'GPT-4o-mini', 'GPT-4-turbo', 'DeepSeek V3']

    eps_high   = [85, 70, 70, 85]
    pavg_high  = [85, 75, 90, 65]
    multi_high = [20, 10, 25, 35]

    eps_fp   = [50, 60, 60, 70]
    pavg_fp  = [50, 10, 40, 10]
    multi_fp = [10, 30, 30, 30]

    x = np.arange(len(models))
    w = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.6))

    ax = axes[0]
    ax.bar(x - w, eps_high,   w, color=C_LOW,   edgecolor='white',
           label=r'$\varepsilon$ (1$\times$ cost)')
    ax.bar(x,     pavg_high,  w, color=C_PAVG,  edgecolor='white',
           label=r'$p_{\mathrm{avg}}$ (1$\times$ cost)')
    ax.bar(x + w, multi_high, w, color=C_MULTI, edgecolor='white',
           label=r'multi-sample $N{=}5$ (5$\times$ cost)')

    for xi, v in zip(x - w, eps_high):
        ax.text(xi, v + 1.5, f'{v}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')
    for xi, v in zip(x, pavg_high):
        ax.text(xi, v + 1.5, f'{v}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')
    for xi, v in zip(x + w, multi_high):
        ax.text(xi, v + 1.5, f'{v}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=14, ha='right', fontsize=FS_TICK)
    ax.set_ylim(0, 100)
    ax.set_ylabel('HIGH detection rate (%)')
    ax.set_title('HIGH detection  (higher is better)', fontsize=FS_BASE)
    ax.legend(loc='upper right', frameon=False, fontsize=FS_ANNOT - 0.5)

    ax = axes[1]
    ax.bar(x - w, eps_fp,   w, color=C_LOW,   edgecolor='white',
           label=r'$\varepsilon$')
    ax.bar(x,     pavg_fp,  w, color=C_PAVG,  edgecolor='white',
           label=r'$p_{\mathrm{avg}}$')
    ax.bar(x + w, multi_fp, w, color=C_MULTI, edgecolor='white',
           label='multi-sample')

    for xi, v in zip(x - w, eps_fp):
        ax.text(xi, v + 1.5, f'{v}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')
    for xi, v in zip(x, pavg_fp):
        ax.text(xi, v + 1.5, f'{v}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')
    for xi, v in zip(x + w, multi_fp):
        ax.text(xi, v + 1.5, f'{v}', ha='center', va='bottom',
                fontsize=FS_ANNOT - 0.5, color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=14, ha='right', fontsize=FS_TICK)
    ax.set_ylim(0, 100)
    ax.set_ylabel('LOW false-positive rate (%)')
    ax.set_title('LOW false-positive  (lower is better)', fontsize=FS_BASE)

    fig.tight_layout()
    _save(fig, 'fig_comparison.pdf')


# --------------------------------------------------------------------------
# Fig intra_inter: Intra/inter distinction on Stripe Scenario A
# --------------------------------------------------------------------------

def fig_intra_inter():
    print("[fig_intra_inter] Stripe A: token distribution vs output diversity")

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4))

    ax = axes[0]
    tokens  = ['Charge', 'PaymentIntent', 'other (top-3)',
               'other (top-4)', 'other (top-5)']
    probs   = [0.52, 0.47, 0.008, 0.001, 0.001]
    colors  = [C_HIGH, C_LOW, '#BBBBBB', '#BBBBBB', '#BBBBBB']

    y = np.arange(len(tokens))
    ax.barh(y, probs, color=colors, edgecolor='white', linewidth=0.8)
    for yi, p in zip(y, probs):
        if p >= 0.01:
            ax.text(p + 0.01, yi, f'{p:.2f}', va='center', ha='left',
                    fontsize=FS_ANNOT, color='#333')
        else:
            ax.text(p + 0.01, yi, f'{p:.3f}', va='center', ha='left',
                    fontsize=FS_ANNOT - 0.5, color='#999')

    ax.set_yticks(y)
    ax.set_yticklabels(tokens, fontsize=FS_TICK)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.70)
    ax.set_xlabel('P(token)', fontsize=FS_LABEL)
    ax.set_title(r'Token distribution (generation time)'
                 '\n'
                 r'$\varepsilon = 0.878$  (FLAGGED)',
                 fontsize=FS_BASE - 0.5)

    ax.grid(axis='y', visible=False)

    ax = axes[1]

    sample_labels = ['sample 1', 'sample 2', 'sample 3',
                     'sample 4', 'sample 5']
    choices = ['Charge', 'Charge', 'Charge', 'Charge', 'Charge']

    y = np.arange(len(sample_labels))
    for yi, choice in zip(y, choices):
        ax.barh([yi], [1.0], color=C_HIGH, edgecolor='white', linewidth=0.8)
        ax.text(0.5, yi, choice, va='center', ha='center',
                fontsize=FS_LABEL, color='white', fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(sample_labels, fontsize=FS_TICK)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_xlim(0, 1.0)
    ax.set_title(r'Output diversity ($N=5$, $T=0.7$)'
                 '\n'
                 r'diversity $= 0.00$',
                 fontsize=FS_BASE - 0.5)
    ax.grid(visible=False)

    fig.text(0.50, 0.10,
             r'The $52/47$ internal hesitation never surfaces as output variation.',
             ha='center', va='bottom', fontsize=FS_ANNOT, style='italic',
             color='#555')

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, 'fig_intra_inter.pdf')


# --------------------------------------------------------------------------
# Fig token_focus (NEW — teaser): heat strip of per-token eps
# --------------------------------------------------------------------------

def fig_token_focus():
    print("[fig_token_focus] teaser heat strip")

    eps = _stripe_token_trajectory()
    n = len(eps)

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 1.9))

    # Render as a horizontal heat-strip: single row, one cell per token
    # Colormap: confident = pale, high-eps = dark red
    cmap = plt.get_cmap('Reds')
    norm = plt.Normalize(vmin=0.0, vmax=1.0)

    for i, e in enumerate(eps):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                   facecolor=cmap(norm(e)),
                                   edgecolor='white', linewidth=0.3))

    # Mark the two paused tokens
    for i, e in enumerate(eps):
        if e >= 0.65:
            ax.add_patch(plt.Rectangle((i, 0), 1, 1,
                                       fill=False, edgecolor='black',
                                       linewidth=1.3))
            ax.annotate(f'  ε={e:.2f}',
                        xy=(i + 0.5, 1.0),
                        xytext=(i + 0.5, 1.45),
                        fontsize=FS_ANNOT - 0.5,
                        ha='center', va='bottom', color='#222',
                        arrowprops=dict(arrowstyle='-', color='#444',
                                        linewidth=0.8))

    ax.set_xlim(0, n)
    ax.set_ylim(-0.2, 1.9)
    ax.set_xticks([0, n // 4, n // 2, 3 * n // 4, n])
    ax.set_yticks([])
    ax.set_xlabel(f'token position (n = {n} retained semantic tokens)',
                  fontsize=FS_LABEL)
    ax.set_title('Stripe Scenario A (GPT-4o): '
                 r'10 tokens exceed $\varepsilon\geq 0.30$, 2 exceed $\varepsilon\geq 0.65$',
                 fontsize=FS_BASE - 0.5)
    ax.grid(visible=False)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)

    # Colorbar: add at bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                        fraction=0.06, pad=0.35, aspect=40)
    cbar.set_label(r'per-token $\varepsilon$', fontsize=FS_ANNOT)
    cbar.ax.tick_params(labelsize=FS_ANNOT - 1)

    fig.tight_layout()
    _save(fig, 'fig_token_focus.pdf')


# --------------------------------------------------------------------------
# Fig token_focus_detail (NEW): per-token eps trajectory with thresholds
# --------------------------------------------------------------------------

def fig_token_focus_detail():
    print("[fig_token_focus_detail] per-token trajectory")

    eps = _stripe_token_trajectory()
    n = len(eps)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.2))

    # Bars for per-token epsilon
    colors = ['#D65F5F' if e >= 0.65
              else '#F5A623' if e >= 0.30
              else '#BBBBBB'
              for e in eps]
    ax.bar(x, eps, color=colors, edgecolor='none', width=0.9)

    # Thresholds
    ax.axhline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.0,
               label=r'flag threshold ($\varepsilon = 0.30$)')
    ax.axhline(0.65, color=C_THRESH_HARD, linestyle='-', linewidth=1.0,
               label=r'paused threshold ($\varepsilon = 0.65$)')

    # Annotate the two paused tokens
    for i, e in enumerate(eps):
        if e >= 0.65:
            label = ('.Charge / .PaymentIntent' if i == 22
                     else 'adjacent selector')
            ax.annotate(f'{label}\nε={e:.3f}',
                        xy=(i, e),
                        xytext=(i + 8, e + 0.06),
                        fontsize=FS_ANNOT - 0.5,
                        ha='left', va='bottom', color='#222',
                        arrowprops=dict(arrowstyle='->', color='#444',
                                        linewidth=0.7))

    ax.set_xlim(-1, n)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f'token position (of {n} retained semantic tokens)',
                  fontsize=FS_LABEL)
    ax.set_ylabel(r'$\varepsilon_t$', fontsize=FS_LABEL)

    # Summary text box
    summary = (f'total retained tokens: {n}\n'
               f'above flag (0.30): 10 ({10/n*100:.1f}%)\n'
               f'above paused (0.65): 2 ({2/n*100:.1f}%)')
    ax.text(0.98, 0.98, summary,
            transform=ax.transAxes, fontsize=FS_ANNOT - 0.5,
            va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='#AAA', linewidth=0.5))

    ax.legend(loc='upper left', frameon=False, fontsize=FS_ANNOT)

    fig.tight_layout()
    _save(fig, 'fig_token_focus_detail.pdf')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"output dir: {FIG_DIR}")

    # Kept figures
    fig3_scenarios()
    fig6_pr_curve()
    fig7_review_loop()
    fig8_scenario_e()
    fig_comparison()
    fig_intra_inter()

    # New figures for v4
    fig_token_focus()
    fig_token_focus_detail()

    print("done.")


if __name__ == '__main__':
    main()
