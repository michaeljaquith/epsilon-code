"""
generate_figures_v3.py

Regenerate paper figures for v4 arXiv submission.

Run from d:/Language/ :
    python generate_figures_v3.py

Produces the following PDF files in paper/figures/:
    fig3_scenarios.pdf       (kept from v2)
    fig6_pr_curve.pdf        (kept from v2)
    fig7_review_loop.pdf     (kept from v2)
    fig8_scenario_e.pdf      (kept from v2)
    fig_comparison.pdf       (NEW - three-method four-model comparison)
    fig_intra_inter.pdf      (NEW - Stripe scenario intra/inter visualization)

Not produced (removed from v4):
    fig1_scatter.pdf        (UQLM demoted to one sentence)
    fig5_roc_sweep.pdf      (same reason)
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import json
import math
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

C_LOW          = '#4878CF'   # blue  - LOW prompts / epsilon
C_HIGH         = '#D65F5F'   # warm red - HIGH prompts / multisample
C_PAVG         = '#F5A623'   # orange - p_avg
C_MULTI        = '#6BAB6E'   # green - multi-sample
C_THRESH_SOFT  = '#F5A623'   # orange - FLAGGED threshold (0.30)
C_THRESH_HARD  = '#D0021B'   # dark red - PAUSED threshold (0.95)

STATUS_COLORS = {
    'COMPLETE': '#6BAB6E',
    'FLAGGED':  '#F5A623',
    'PAUSED':   '#D65F5F',
    'ABORTED':  '#8B0000',
}

# Widths (in inches) matching standard LaTeX article (~6.75in text)
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
RESULTS_DIR = BASE / 'repo' / 'benchmark' / 'results'
FIG_DIR     = BASE / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Per-model calibration / benchmark files (epsilon + mean_logprob data)
BENCH_PATHS = {
    'gpt-4o':        RESULTS_DIR / 'benchmark_gpt-4o.json',
    'gpt-4o-mini':   RESULTS_DIR / 'benchmark_gpt-4o-mini.json',
    'gpt-4-turbo':   RESULTS_DIR / 'benchmark_gpt-4-turbo.json',
    'DeepSeek V3':   RESULTS_DIR / 'calibration_deepseek-ai_DeepSeek-V3.json',
}

# Multi-sample files
MULTI_PATHS = {
    'gpt-4o':        RESULTS_DIR / 'multisample_gpt_4o.json',
    'gpt-4o-mini':   RESULTS_DIR / 'multisample_gpt_4o_mini.json',
    'gpt-4-turbo':   RESULTS_DIR / 'multisample_gpt_4_turbo.json',
    'DeepSeek V3':   RESULTS_DIR / 'multisample_deepseek_ai_DeepSeek_V3.json',
}

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
# Fig 3: Five scenarios horizontal bar chart (kept from v2)
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
# Fig 6: Precision-recall at tier thresholds (kept from v2)
# --------------------------------------------------------------------------

def fig6_pr_curve():
    print("[fig6] PR curve with tier operating points")
    pr = _load(PR_PATH)
    curve = pr['pr_curve']

    thresholds = np.array([c['threshold'] for c in curve])
    precision  = np.array([c['precision'] for c in curve])
    recall     = np.array([c['recall']    for c in curve])

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
        ax.annotate(lbl, (r, p), textcoords='offset points',
                    xytext=(8, 6), fontsize=FS_ANNOT, color=col)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 0.55)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    ax.legend(loc='upper right', frameon=False)

    _save(fig, 'fig6_pr_curve.pdf')


# --------------------------------------------------------------------------
# Fig 7: Review loop 2A vs 2B stacked bars (kept from v2)
# --------------------------------------------------------------------------

def fig7_review_loop():
    print("[fig7] review loop stacked comparison")

    total = 201

    a_cleared   = 107
    a_confirmed = 85
    a_late_fp   = 4
    a_pending = total - (a_cleared + a_confirmed + a_late_fp)

    b_cleared   = 138
    b_confirmed = 57
    b_late_fp   = 0
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
# Fig 8: Scenario E collapse (gpt-4o-mini) — kept from v2
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
# Fig comparison (NEW): Three-method comparison across four models
# --------------------------------------------------------------------------

def fig_comparison():
    print("[fig_comparison] three-method four-model comparison")

    models = ['GPT-4o', 'GPT-4o-mini', 'GPT-4-turbo', 'DeepSeek V3']

    # HIGH detection rates (%)
    eps_high   = [85, 70, 70, 85]
    pavg_high  = [85, 75, 90, 65]
    multi_high = [20, 10, 25, 35]

    # LOW FP rates (%)
    eps_fp   = [50, 60, 60, 70]
    pavg_fp  = [50, 10, 40, 10]
    multi_fp = [10, 30, 30, 30]

    x = np.arange(len(models))
    w = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.6))

    # Left: HIGH detection
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

    # Right: LOW FP
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
# Fig intra_inter (NEW): Intra/inter distinction on Stripe Scenario A
# --------------------------------------------------------------------------

def fig_intra_inter():
    print("[fig_intra_inter] Stripe A: token distribution vs output diversity")

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.4))

    # --- Left: token distribution (intra-generational view) ---
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

    # Remove grid on y-axis
    ax.grid(axis='y', visible=False)

    # --- Right: output diversity (inter-generational view) ---
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

    # Annotation bar between panels
    fig.text(0.50, 0.10,
             r'The $52/47$ internal hesitation never surfaces as output variation.',
             ha='center', va='bottom', fontsize=FS_ANNOT, style='italic',
             color='#555')

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, 'fig_intra_inter.pdf')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    print(f"output dir: {FIG_DIR}")

    # Kept figures (regenerated to ensure reproducibility)
    fig3_scenarios()
    fig6_pr_curve()
    fig7_review_loop()
    fig8_scenario_e()

    # New figures for v4
    fig_comparison()
    fig_intra_inter()

    print("done.")


if __name__ == '__main__':
    main()
