"""
generate_figures_final.py

Regenerate all paper figures for arXiv submission.

Run from d:/Language/ :
    python generate_figures_final.py

Produces 6 PDF files in paper/figures/:
    fig1_scatter.pdf
    fig3_scenarios.pdf
    fig5_roc_sweep.pdf
    fig6_pr_curve.pdf
    fig7_review_loop.pdf
    fig8_scenario_e.pdf

Note: fig9_taxonomy.pdf is no longer produced; the taxonomy is now
a LaTeX table (tab:taxonomy) in epsilon.tex. The fig9_taxonomy()
function is preserved below for reproducibility but not called.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np


# --------------------------------------------------------------------------
# Style constants
# --------------------------------------------------------------------------

C_LOW          = '#4878CF'   # blue  - LOW prompts
C_HIGH         = '#D65F5F'   # warm red - HIGH prompts
C_THRESH_SOFT  = '#F5A623'   # orange - FLAGGED threshold (0.30)
C_THRESH_HARD  = '#D0021B'   # dark red - PAUSED threshold (0.95)

STATUS_COLORS = {
    'COMPLETE': '#6BAB6E',
    'FLAGGED':  '#F5A623',
    'PAUSED':   '#D65F5F',
    'ABORTED':  '#8B0000',
}

# Widths (in inches) matching standard LaTeX article (10pt/11pt, ~6.75in text)
W_SINGLE = 3.25
W_DOUBLE = 6.75

# Font sizes
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
    'pdf.fonttype':    42,   # TrueType (editable in illustrator)
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
RESULTS_DIR = BASE / 'repo' / 'benchmark' / 'results'
FIG_DIR     = BASE / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

BENCH_PATH   = RESULTS_DIR / 'benchmark_gpt-4o.json'
UQLM_PATH    = RESULTS_DIR / 'benchmark_uqlm.json'
PR_PATH      = RESULTS_DIR / 'epsilon_pr_analysis.json'
SCE_E_PATH   = RESULTS_DIR / 'scenario_e_collapse.json'


def _load(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out, format='pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------
# Fig 1: eps vs UQLM confidence scatter
# --------------------------------------------------------------------------

def fig1_scatter():
    print("[fig1] eps vs UQLM confidence scatter")
    bench = _load(BENCH_PATH)
    uqlm  = _load(UQLM_PATH)

    bench_by_id = {r['id']: r for r in bench['results']}
    uqlm_rows   = uqlm['uqlm_results']

    xs_low, ys_low = [], []
    xs_high, ys_high = [], []
    for u in uqlm_rows:
        b = bench_by_id.get(u['id'])
        if b is None:
            continue
        conf = float(u['confidence'])
        eps  = float(b['epsilon'])
        if b['category'] == 'LOW':
            xs_low.append(conf);  ys_low.append(eps)
        else:
            xs_high.append(conf); ys_high.append(eps)

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.6))

    ax.scatter(xs_low,  ys_low,  s=42, c=C_LOW,  alpha=0.82,
               edgecolors='white', linewidth=0.6, label=f'LOW (n={len(xs_low)})',
               zorder=3)
    ax.scatter(xs_high, ys_high, s=42, c=C_HIGH, alpha=0.82,
               edgecolors='white', linewidth=0.6, label=f'HIGH (n={len(xs_high)})',
               marker='D', zorder=3)

    ax.axhline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.1,
               alpha=0.85, zorder=2)
    ax.axhline(0.95, color=C_THRESH_HARD, linestyle='--', linewidth=1.1,
               alpha=0.85, zorder=2)
    ax.text(0.015, 0.31, 'FLAGGED (eps >= 0.30)',
            color=C_THRESH_SOFT, fontsize=FS_ANNOT, va='bottom')
    ax.text(0.015, 0.96, 'PAUSED (eps >= 0.95)',
            color=C_THRESH_HARD, fontsize=FS_ANNOT, va='bottom')

    ax.set_xlabel('UQLM confidence (min_probability)')
    ax.set_ylabel(r'$\varepsilon$ (token-level entropy score)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.invert_xaxis()  # high-confidence-left is intuitive for a failure signal
    ax.set_xlim(1.02, -0.02)

    ax.legend(loc='upper right', frameon=False)

    _save(fig, 'fig1_scatter.pdf')


# --------------------------------------------------------------------------
# Fig 3: Five scenarios horizontal bar chart
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

    labels = [f"{s[0]} \u2014 {s[1]}" for s in scenarios]
    values = [s[2] for s in scenarios]
    colors = [STATUS_COLORS[s[3]] for s in scenarios]
    statuses = [s[3] for s in scenarios]

    # Taller figure so each bar has breathing room
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 4.2))
    y = np.arange(len(scenarios))
    ax.barh(y, values, color=colors, edgecolor='white', linewidth=0.8,
            height=0.62)

    # thresholds (3-tier scheme: FLAGGED 0.30-0.95, PAUSED >= 0.95)
    ax.axvline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.1, alpha=0.85)
    ax.axvline(0.95, color=C_THRESH_HARD, linestyle='--', linewidth=1.1, alpha=0.85)

    # Threshold labels placed below the x-axis using xaxis transform
    # (x in data coords, y in axes coords: 0=bottom, negative=below axis)
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
    ax.set_xlim(0.0, 1.30)   # extra room for end labels
    ax.set_xlabel(r'$\varepsilon$ (peak)', fontsize=FS_LABEL)

    # Legend placed fully outside the axes area, above the plot
    legend_patches = [
        mpatches.Patch(color=STATUS_COLORS['FLAGGED'], label=r'FLAGGED ($0.30 \leq \varepsilon < 0.95$)'),
    ]
    ax.legend(handles=legend_patches, loc='lower left',
              bbox_to_anchor=(0.0, 1.02), bbox_transform=ax.transAxes,
              frameon=False, ncol=1)

    fig.tight_layout(rect=[0, 0, 1, 0.96])   # leave space above for legend
    _save(fig, 'fig3_scenarios.pdf')


# --------------------------------------------------------------------------
# Fig 5: Dual-panel threshold sweep ROC (eps | UQLM)
# --------------------------------------------------------------------------

def _sweep(signal_high, signal_low, direction='ge'):
    """
    Return (thresholds, tpr, fpr).  direction='ge' means fire when signal>=t
    (for eps).  direction='le' means fire when signal<=t (for UQLM confidence
    - low confidence implies failure).
    """
    thr = np.linspace(0.0, 1.0, 101)
    tpr, fpr = [], []
    for t in thr:
        if direction == 'ge':
            tp = sum(1 for s in signal_high if s >= t)
            fp = sum(1 for s in signal_low  if s >= t)
        else:
            tp = sum(1 for s in signal_high if s <= t)
            fp = sum(1 for s in signal_low  if s <= t)
        tpr.append(tp / max(len(signal_high), 1))
        fpr.append(fp / max(len(signal_low),  1))
    return thr, np.array(tpr), np.array(fpr)


def fig5_roc_sweep():
    print("[fig5] dual-panel ROC sweep (eps | UQLM)")
    bench = _load(BENCH_PATH)
    uqlm  = _load(UQLM_PATH)

    bench_by_id = {r['id']: r for r in bench['results']}

    eps_low, eps_high = [], []
    uq_low,  uq_high  = [], []
    for u in uqlm['uqlm_results']:
        b = bench_by_id.get(u['id'])
        if b is None:
            continue
        eps  = float(b['epsilon'])
        conf = float(u['confidence'])
        if b['category'] == 'LOW':
            eps_low.append(eps);  uq_low.append(conf)
        else:
            eps_high.append(eps); uq_high.append(conf)

    thr_e, tpr_e, fpr_e = _sweep(eps_high, eps_low, direction='ge')
    thr_u, tpr_u, fpr_u = _sweep(uq_high,  uq_low,  direction='le')

    # Slightly larger fonts for this figure (user reported the labels looked
    # a touch small). Axis labels get FS_ANNOT+2, ticks get FS_TICK+1, legend +1.
    axis_fs   = FS_ANNOT + 2
    tick_fs   = FS_TICK + 1
    legend_fs = FS_ANNOT + 1
    title_fs  = FS_BASE + 1

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.2), sharey=True)

    ax = axes[0]
    ax.plot(fpr_e, tpr_e, color=C_HIGH, linewidth=1.8, label=r'$\varepsilon$')
    ax.plot([0, 1], [0, 1], color='#888', linewidth=0.7, linestyle=':',
            label='chance')
    # annotate operating thresholds 0.30 and 0.95
    for t_mark, color, lbl in ((0.30, C_THRESH_SOFT, 't=0.30'),
                               (0.95, C_THRESH_HARD, 't=0.95')):
        idx = int(round(t_mark * (len(thr_e) - 1)))
        ax.scatter([fpr_e[idx]], [tpr_e[idx]], s=46, color=color,
                   edgecolor='white', linewidth=0.8, zorder=5)
        ax.annotate(lbl, (fpr_e[idx], tpr_e[idx]),
                    textcoords='offset points', xytext=(6, -10),
                    fontsize=legend_fs, color=color)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('false-positive rate (LOW flagged)', fontsize=axis_fs)
    ax.set_ylabel('detection rate (HIGH caught)', fontsize=axis_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.set_title(r'$\varepsilon$ threshold sweep', fontsize=title_fs)
    ax.legend(loc='lower right', frameon=False, fontsize=legend_fs)

    ax = axes[1]
    ax.plot(fpr_u, tpr_u, color=C_LOW, linewidth=1.8, label='UQLM')
    ax.plot([0, 1], [0, 1], color='#888', linewidth=0.7, linestyle=':',
            label='chance')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel('false-positive rate (LOW flagged)', fontsize=axis_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.set_title('UQLM threshold sweep', fontsize=title_fs)
    ax.legend(loc='lower right', frameon=False, fontsize=legend_fs)

    fig.tight_layout()
    _save(fig, 'fig5_roc_sweep.pdf')


# --------------------------------------------------------------------------
# Fig 6: Precision-recall at tier thresholds
# --------------------------------------------------------------------------

def fig6_pr_curve():
    print("[fig6] PR curve with tier operating points")
    pr = _load(PR_PATH)
    curve = pr['pr_curve']

    thresholds = np.array([c['threshold'] for c in curve])
    precision  = np.array([c['precision'] for c in curve])
    recall     = np.array([c['recall']    for c in curve])

    base_rate = 53.0 / 200.0  # 0.265

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.6))

    ax.plot(recall, precision, color=C_HIGH, linewidth=1.8, zorder=3,
            label=r'$\varepsilon$ sweep')

    ax.axhline(base_rate, color='#888', linestyle=':', linewidth=1.0,
               zorder=2, label=f'base rate = {base_rate:.3f}')

    # operating points (3-tier scheme: FLAGGED 0.30-0.95, PAUSED >= 0.95)
    op_points = [
        ('FLAGGED+ (t=0.30)', 1.000, 0.272, C_THRESH_SOFT),
        ('PAUSED (t=0.95)',   0.547, 0.309, C_THRESH_HARD),
    ]
    for lbl, r, p, col in op_points:
        ax.scatter([r], [p], s=72, color=col, edgecolor='white',
                   linewidth=0.9, zorder=5)
        ax.annotate(lbl, (r, p), textcoords='offset points',
                    xytext=(8, 6), fontsize=FS_ANNOT, color=col)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 0.55)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.legend(loc='upper right', frameon=False)

    _save(fig, 'fig6_pr_curve.pdf')


# --------------------------------------------------------------------------
# Fig 7: Review loop 2A vs 2B stacked bars
# --------------------------------------------------------------------------

def fig7_review_loop():
    print("[fig7] review loop stacked comparison")

    total = 201

    # Context-free reviewer
    a_cleared   = 107
    a_confirmed = 85
    a_late_fp   = 4
    a_pending = total - (a_cleared + a_confirmed + a_late_fp)

    # Context-primed reviewer
    b_cleared   = 138
    b_confirmed = 57
    b_late_fp   = 0
    b_pending = total - (b_cleared + b_confirmed + b_late_fp)

    labels = ['Context-free\nreviewer', 'Context-primed\nreviewer']
    cleared   = np.array([a_cleared,   b_cleared])
    confirmed = np.array([a_confirmed, b_confirmed])
    late_fp   = np.array([a_late_fp,   b_late_fp])
    pending   = np.array([a_pending,   b_pending])

    # Extra top margin for the external legend
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.5))
    y = np.arange(len(labels))
    h = 0.52

    col_clear = '#6BAB6E'
    col_conf  = '#F5A623'
    col_late  = '#8B0000'
    col_pend  = '#CCCCCC'

    left = np.zeros(len(labels))
    b1 = ax.barh(y, cleared,   h, left=left, color=col_clear, edgecolor='white',
                 label='cleared by reviewer')
    left += cleared
    b2 = ax.barh(y, confirmed, h, left=left, color=col_conf,  edgecolor='white',
                 label='confirmed failure')
    left += confirmed
    b3 = ax.barh(y, late_fp,   h, left=left, color=col_late,  edgecolor='white',
                 label='late false positive')
    left += late_fp
    b4 = None
    if pending.sum() > 0:
        b4 = ax.barh(y, pending, h, left=left, color=col_pend, edgecolor='white',
                     label='other')

    # annotations inside segments
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

    # Legend above the plot, outside the data area
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.04),
              frameon=False, ncol=3, fontsize=FS_ANNOT)

    fig.tight_layout()
    _save(fig, 'fig7_review_loop.pdf')


# --------------------------------------------------------------------------
# Fig 8: Scenario E collapse (gpt-4o-mini) — combined vs per-function peak eps
# --------------------------------------------------------------------------

def fig8_scenario_e():
    print("[fig8] scenario E collapse (gpt-4o-mini)")
    data = _load(SCE_E_PATH)
    mini = None
    for row in data['results']:
        if row.get('model') == 'gpt-4o-mini':
            mini = row
            break
    if mini is None:
        raise RuntimeError("gpt-4o-mini not found in scenario_e_collapse.json")

    fns = mini['functions']
    names     = [f['fn'] for f in fns]
    combined  = np.array([f['combined_peak_eps'] for f in fns])
    per_fn    = np.array([f['per_fn_peak_eps']   for f in fns])

    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.6))

    ax.bar(x - w/2, combined, w, color=C_LOW,  edgecolor='white',
           label='combined prompt (1 call)')
    ax.bar(x + w/2, per_fn,   w, color=C_HIGH, edgecolor='white',
           label='per-function (6 calls)')

    # FLAGGED threshold (0.30) is the load-bearing boundary for scenario E;
    # PAUSED (0.95) is too high to be informative for these values.
    ax.axhline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.0, alpha=0.8)
    ax.text(len(names) - 0.5, 0.31, 'FLAGGED (0.30)', color=C_THRESH_SOFT,
            fontsize=FS_ANNOT, ha='right', va='bottom')

    # annotate bar heights
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
# Fig 9: Three-type uncertainty taxonomy (styled table figure)
# --------------------------------------------------------------------------

def fig9_taxonomy():
    print("[fig9] three-type uncertainty taxonomy")

    rows = [
        {
            'type':   '1',
            'label':  'Cosmetic',
            'source': 'Naming choices\n(sort_list vs sort_numbers)',
            'fires':  'Yes\n(0.7\u20130.9)',
            'risk':   'None',
            'filter': 'AST declaration\nfilter removes it',
            'bg':     '#F4F7FB',
            'accent': C_LOW,
        },
        {
            'type':   '2',
            'label':  'Consequential',
            'source': 'API / library decisions\n(Charge vs PaymentIntent)',
            'fires':  'Yes\n(0.5\u20130.9)',
            'risk':   'High',
            'filter': 'Target signal\n(this system)',
            'bg':     '#FBEFEF',
            'accent': C_HIGH,
        },
        {
            'type':   '3',
            'label':  'Implementation',
            'source': 'Algorithm structure\n(recursive vs iterative)',
            'fires':  'Yes\n(0.3\u20130.7)',
            'risk':   'None',
            'filter': 'Irreducible\nFP floor',
            'bg':     '#FBF6EC',
            'accent': C_THRESH_SOFT,
        },
    ]

    # Column layout (figure coords 0..1) — give Source more room
    col_edges = [0.0, 0.10, 0.23, 0.50, 0.64, 0.75, 1.0]
    col_names = ['Type', 'Label', 'Source', r'$\varepsilon$ fires?', 'Risk', 'Filter / Note']

    # Taller figure so rows have room for two-line text
    fig_w = W_DOUBLE
    fig_h = 4.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.grid(False)

    header_h = 0.14
    row_h    = (1.0 - header_h) / len(rows)

    # Header band
    ax.add_patch(Rectangle((0, 1 - header_h), 1, header_h,
                            facecolor='#2A2F3A', edgecolor='none'))
    for i, name in enumerate(col_names):
        x0, x1 = col_edges[i], col_edges[i + 1]
        ax.text((x0 + x1) / 2, 1 - header_h / 2, name,
                ha='center', va='center', color='white',
                fontsize=FS_BASE, fontweight='bold')

    # Rows
    for r_idx, row in enumerate(rows):
        y_top = 1 - header_h - r_idx * row_h
        y_bot = y_top - row_h

        ax.add_patch(Rectangle((0, y_bot), 1, row_h,
                               facecolor=row['bg'], edgecolor='none'))
        ax.add_patch(Rectangle((0, y_bot), 0.006, row_h,
                               facecolor=row['accent'], edgecolor='none'))

        y_mid = (y_top + y_bot) / 2

        cells = [row['type'], row['label'], row['source'],
                 row['fires'], row['risk'], row['filter']]

        for i, val in enumerate(cells):
            x0, x1 = col_edges[i], col_edges[i + 1]
            cx = (x0 + x1) / 2
            if i == 0:
                ax.text(cx, y_mid, val, ha='center', va='center',
                        fontsize=FS_BASE + 6, fontweight='bold',
                        color=row['accent'])
            elif i == 1:
                ax.text(cx, y_mid, val, ha='center', va='center',
                        fontsize=FS_BASE + 1, fontweight='bold',
                        color='#222')
            elif i == 4:
                color  = C_HIGH if val.strip().lower() == 'high' else '#555'
                weight = 'bold' if val.strip().lower() == 'high' else 'normal'
                ax.text(cx, y_mid, val, ha='center', va='center',
                        fontsize=FS_LABEL + 1, fontweight=weight, color=color)
            elif i == 5:
                ax.text(cx, y_mid, val, ha='center', va='center',
                        fontsize=FS_LABEL, style='italic', color='#333',
                        linespacing=1.4)
            else:
                ax.text(cx, y_mid, val, ha='center', va='center',
                        fontsize=FS_LABEL, color='#222', linespacing=1.4)

        if r_idx < len(rows) - 1:
            ax.plot([0.006, 1.0], [y_bot, y_bot],
                    color='#CCC', linewidth=0.8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    _save(fig, 'fig9_taxonomy.pdf')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"output dir: {FIG_DIR}")
    fig1_scatter()
    fig3_scenarios()
    fig5_roc_sweep()
    fig6_pr_curve()
    fig7_review_loop()
    fig8_scenario_e()
    # fig9_taxonomy() is intentionally not called: the taxonomy is now
    # rendered as a LaTeX table (tab:taxonomy) in epsilon.tex. The function
    # is retained below for reproducibility but is no longer invoked.
    print("done.")


if __name__ == '__main__':
    main()
