#!/usr/bin/env python3
"""
Generate paper figures for the epsilon-code paper.
Outputs PNG (300 DPI) to paper/figures/.

Figures produced:
  fig1_scatter.png         -- ε vs UQLM confidence scatter (30 prompts, annotate pandas)
  fig2_benchmark_bars.png  -- dual-panel bar chart across 30 prompts
  fig3_scenarios.png       -- 5 scenarios with threshold lines
  fig4_multimodel.png      -- multi-model grouped bar chart
  fig5_roc_sweep.png       -- dual-panel threshold sweep (ε ROC + UQLM ROC)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ------------------------------------------------------------------ #
# Style
# ------------------------------------------------------------------ #

plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  9,
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'grid.linestyle':   '--',
})

OUT = Path('paper/figures')
OUT.mkdir(parents=True, exist_ok=True)

# Color palette — colorblind-friendly
C_LOW    = '#4878CF'   # blue
C_HIGH   = '#D65F5F'   # red
C_SOFT   = '#f0f0f0'
C_THRESH_SOFT = '#F5A623'
C_THRESH_HARD = '#D0021B'

STATUS_COLORS = {
    'COMPLETE': '#6BAB6E',
    'FLAGGED':  '#F5A623',
    'PAUSED':   '#D65F5F',
    'ABORTED':  '#8B0000',
}

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #

gpt4o_data = json.loads(Path('benchmark_gpt-4o.json').read_text(encoding='utf-8'))['results']
uqlm_file  = json.loads(Path('benchmark_uqlm.json').read_text(encoding='utf-8'))
uqlm_data  = uqlm_file['uqlm_results']

# Merge gpt-4o epsilon results with UQLM confidence by id
uqlm_by_id = {r['id']: r for r in uqlm_data}
records = [
    {
        'id':        r['id'],
        'label':     r['label'],
        'category':  r['category'],
        'epsilon':   r['epsilon'],
        'status':    r['status'],
        'min_logprob': r.get('min_logprob'),
        'uqlm_conf': uqlm_by_id[r['id']]['confidence'] if r['id'] in uqlm_by_id else None,
    }
    for r in gpt4o_data
]

LOW  = [r for r in records if r['category'] == 'LOW']
HIGH = [r for r in records if r['category'] == 'HIGH']

# Short labels — 30 prompts
labels_short = [
    'sort', 'palindrome', 'fibonacci', 'max', 'word-freq',
    'prime', 'celsius', 'dedup', 'flatten', 'bin-search',
    'stripe', 'openai-sdk', 'sqla', 'fastapi', 'pw-hash',
    'jwt', 'utc-dt', 'http-get', 'redis', 'postgres',
    'pydantic', 'pandas', 'boto3', 'celery', 'token-dt',
    'pillow', 'django-rf', 'async-http', 'schema', 'pytest',
]


# ================================================================== #
# Figure 1: Scatter — ε vs UQLM confidence (30 prompts)
# ================================================================== #

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# Slight jitter on UQLM to separate overlapping points (UQLM is very discrete)
rng = np.random.default_rng(42)

for rec in LOW:
    if rec['uqlm_conf'] is None:
        continue
    jx = rng.uniform(-0.012, 0.012)
    jy = rng.uniform(-0.015, 0.015)
    ax.scatter(rec['epsilon'] + jx, rec['uqlm_conf'] + jy,
               color=C_LOW, s=75, zorder=3, alpha=0.85,
               edgecolors='white', linewidths=0.5)

pandas_rec = None
for rec in HIGH:
    if rec['uqlm_conf'] is None:
        continue
    if rec['id'] == 'high_12':
        pandas_rec = rec
        continue
    jx = rng.uniform(-0.012, 0.012)
    jy = rng.uniform(-0.015, 0.015)
    ax.scatter(rec['epsilon'] + jx, rec['uqlm_conf'] + jy,
               color=C_HIGH, s=75, marker='^', zorder=3, alpha=0.85,
               edgecolors='white', linewidths=0.5)

# Highlight the single UQLM detection (pandas concat)
if pandas_rec is not None:
    ax.scatter(pandas_rec['epsilon'], pandas_rec['uqlm_conf'],
               color='#FFD93D', s=260, marker='*', zorder=5,
               edgecolors='#333', linewidths=1.0,
               label='UQLM detection (pandas concat)')
    ax.annotate(
        'UQLM detection\n(pandas concat)',
        xy=(pandas_rec['epsilon'], pandas_rec['uqlm_conf']),
        xytext=(pandas_rec['epsilon'] - 0.30, pandas_rec['uqlm_conf'] - 0.18),
        fontsize=7.5, color='#333',
        arrowprops=dict(arrowstyle='->', color='#555', lw=0.8),
        bbox=dict(boxstyle='round,pad=0.25', facecolor='#fffbe6',
                  edgecolor='#bbb', alpha=0.9),
    )

# Threshold lines
ax.axvline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.2,
           label=r'$\varepsilon$ soft threshold (0.30)', zorder=2)
ax.axvline(0.65, color=C_THRESH_HARD, linestyle='--', linewidth=1.2,
           label=r'$\varepsilon$ hard threshold (0.65)', zorder=2)
ax.axhline(0.50, color='#666666', linestyle=':', linewidth=1.0,
           label='UQLM fire threshold (0.50)', zorder=2)

# Annotations
ax.text(0.37, 0.52, 'UQLM fire threshold (0.50)', fontsize=7.5,
        color='#555555', va='bottom')
ax.text(0.31, 0.04, r'$\varepsilon$ soft', fontsize=7.5,
        color=C_THRESH_SOFT, va='bottom')
ax.text(0.66, 0.04, r'$\varepsilon$ hard', fontsize=7.5,
        color=C_THRESH_HARD, va='bottom')

# Legend
low_patch  = mpatches.Patch(color=C_LOW,  label='LOW prompts (simple algorithms)')
high_patch = mpatches.Patch(color=C_HIGH, label='HIGH prompts (version-split APIs)')
star_marker = plt.Line2D([0], [0], marker='*', color='w',
                         markerfacecolor='#FFD93D', markeredgecolor='#333',
                         markersize=12, label='UQLM detection (high_12)')
ax.legend(handles=[low_patch, high_patch, star_marker], loc='upper left',
          framealpha=0.9, edgecolor='#cccccc')

ax.set_xlabel(r'$\varepsilon$ (token-level uncertainty score)')
ax.set_ylabel('UQLM confidence score\n(semantic negentropy; higher = more confident)')
ax.set_xlim(-0.03, 1.0)
ax.set_ylim(0.0,   1.12)
ax.set_title(
    r'$\varepsilon$ discriminates HIGH from LOW prompts; UQLM does not'
    '\n(30 benchmark prompts)',
    pad=8
)

# Quadrant annotation
ax.text(0.82, 0.06,
        'HIGH ε\nUQLM confident\n(missed detections)',
        fontsize=7, color='#888', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff8f8',
                  edgecolor='#ddd', alpha=0.8))

fig.tight_layout()
fig.savefig(OUT / 'fig1_scatter.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig1_scatter.png done')


# ================================================================== #
# Figure 2: Dual-panel bar chart — 30 prompts
# ================================================================== #

x = np.arange(len(records))
eps_vals  = [r['epsilon']   for r in records]
uqlm_vals = [1.0 - r['uqlm_conf'] if r['uqlm_conf'] is not None else 0.0
             for r in records]
bar_colors = [STATUS_COLORS[r['status']] for r in records]

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(12, 5.8), sharex=True,
    gridspec_kw={'height_ratios': [3, 2], 'hspace': 0.08}
)

# ---- Top panel: ε ----
bars = ax_top.bar(x, eps_vals, color=bar_colors, edgecolor='white',
                  linewidth=0.5, zorder=3)

ax_top.axhline(0.30, color=C_THRESH_SOFT, linestyle='--',
               linewidth=1.2, label='soft (0.30)', zorder=4)
ax_top.axhline(0.65, color=C_THRESH_HARD, linestyle='--',
               linewidth=1.2, label='hard (0.65)', zorder=4)

# LOW / HIGH divider
ax_top.axvline(9.5, color='#aaaaaa', linewidth=1.0, zorder=2)
ax_top.text(4.5,  0.97, 'LOW prompts',  ha='center', fontsize=8.5,
            color='#555', transform=ax_top.get_xaxis_transform())
ax_top.text(19.5, 0.97, 'HIGH prompts', ha='center', fontsize=8.5,
            color='#555', transform=ax_top.get_xaxis_transform())

ax_top.set_ylim(0, 1.05)
ax_top.set_ylabel(r'$\varepsilon$ score')

status_patches = [
    mpatches.Patch(color=STATUS_COLORS['COMPLETE'], label='COMPLETE'),
    mpatches.Patch(color=STATUS_COLORS['FLAGGED'],  label='FLAGGED'),
    mpatches.Patch(color=STATUS_COLORS['PAUSED'],   label='PAUSED'),
]
threshold_lines_top = [
    plt.Line2D([0], [0], color=C_THRESH_SOFT, linestyle='--', linewidth=1.2,
               label='soft (0.30)'),
    plt.Line2D([0], [0], color=C_THRESH_HARD, linestyle='--', linewidth=1.2,
               label='hard (0.65)'),
]
ax_top.legend(handles=status_patches + threshold_lines_top, loc='upper left',
              framealpha=0.9, edgecolor='#ccc', fontsize=8, ncol=2)

# ---- Bottom panel: UQLM ----
uqlm_colors = [C_LOW if r['category'] == 'LOW' else C_HIGH for r in records]
bars_bot = ax_bot.bar(x, uqlm_vals, color=uqlm_colors, alpha=0.75,
                      edgecolor='white', linewidth=0.5, zorder=3)

ax_bot.axhline(0.50, color='#666666', linestyle=':', linewidth=1.2,
               label='UQLM fire threshold\n(uncertainty > 0.50)', zorder=4)
ax_bot.axvline(9.5,  color='#aaaaaa', linewidth=1.0, zorder=2)

# Annotate the pandas UQLM detection — high_12 is at index 21 (10 LOW + 12th HIGH = index 21)
pandas_idx = 21
if pandas_idx < len(uqlm_vals):
    ax_bot.annotate(
        'UQLM↑',
        xy=(pandas_idx, uqlm_vals[pandas_idx]),
        xytext=(pandas_idx, uqlm_vals[pandas_idx] + 0.10),
        fontsize=7.5, color='#333', ha='center', fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#555', lw=0.7),
    )

ax_bot.set_ylim(0, 0.80)
ax_bot.set_ylabel('UQLM uncertainty\n(1 − confidence)')
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=7.5)
ax_bot.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc', fontsize=8)

# Shared title
fig.suptitle(
    r'Per-prompt $\varepsilon$ (top) and UQLM confidence (bottom) across 30-prompt benchmark',
    fontsize=10.5, y=1.01
)

fig.tight_layout()
fig.savefig(OUT / 'fig2_benchmark_bars.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig2_benchmark_bars.png done')


# ================================================================== #
# Figure 3: 5 scenarios horizontal bar chart
# ================================================================== #

scenarios = [
    {'label': 'A — Stripe\ndeprecation',    'eps': 0.914, 'status': 'PAUSED',   'class': 'Deprecated API'},
    {'label': 'B — OpenAI SDK\nv0 vs v1',   'eps': 0.551, 'status': 'PAUSED',   'class': 'Confident wrongness'},
    {'label': 'C — SQLAlchemy\n1.x vs 2.0', 'eps': 0.435, 'status': 'FLAGGED',  'class': 'Syntax split'},
    {'label': 'D — FastAPI\nasync/sync',     'eps': 0.812, 'status': 'PAUSED',   'class': 'Compatibility mismatch'},
    {'label': 'E — Auth module\n(mean)',      'eps': 0.812, 'status': 'PAUSED',   'class': 'Multi-decision'},
]
# Scenario E mean across 16 runs
e_eps = [0.6933, 0.775, 0.7653, 0.8745, 0.824, 0.8905, 0.8235, 0.8707,
         0.762, 0.8045, 0.7157, 0.8686, 0.7997, 0.9216, 0.728, 0.8671]
scenarios[4]['eps'] = sum(e_eps) / len(e_eps)

fig, ax = plt.subplots(figsize=(7, 3.8))

y      = np.arange(len(scenarios))
widths = [s['eps'] for s in scenarios]
colors = [STATUS_COLORS[s['status']] for s in scenarios]

bars = ax.barh(y, widths, color=colors, edgecolor='white',
               linewidth=0.6, height=0.55, zorder=3)

# Threshold lines
ax.axvline(0.30, color=C_THRESH_SOFT, linestyle='--', linewidth=1.3,
           label='soft (0.30)', zorder=4)
ax.axvline(0.65, color=C_THRESH_HARD, linestyle='--', linewidth=1.3,
           label='hard (0.65)', zorder=4)

# Value labels on bars
for bar, s in zip(bars, scenarios):
    w = bar.get_width()
    ax.text(w + 0.012, bar.get_y() + bar.get_height() / 2,
            f'{w:.3f}', va='center', fontsize=8.5, color='#333')

# Failure class labels inside or beside bars
for i, s in enumerate(scenarios):
    ax.text(0.01, i, s['class'], va='center', fontsize=7.5,
            color='white', fontweight='bold')

ax.set_yticks(y)
ax.set_yticklabels([s['label'] for s in scenarios], fontsize=9)
ax.set_xlim(0, 1.08)
ax.set_xlabel(r'$\varepsilon$ score')
ax.set_title('Five evaluation scenarios with failure class and ' + r'$\varepsilon$ result', pad=8)

# Status legend
status_patches = [
    mpatches.Patch(color=STATUS_COLORS['FLAGGED'], label='FLAGGED'),
    mpatches.Patch(color=STATUS_COLORS['PAUSED'],  label='PAUSED'),
]
threshold_lines = [
    plt.Line2D([0], [0], color=C_THRESH_SOFT, linestyle='--', linewidth=1.2,
               label='soft threshold (0.30)'),
    plt.Line2D([0], [0], color=C_THRESH_HARD, linestyle='--', linewidth=1.2,
               label='hard threshold (0.65)'),
]
ax.legend(handles=status_patches + threshold_lines,
          loc='lower right', framealpha=0.9, edgecolor='#ccc', fontsize=8)

# Scenario E range annotation
e_bar = bars[4]
ax.annotate(
    f'range: [{min(e_eps):.2f}–{max(e_eps):.2f}]\n(16 runs)',
    xy=(max(e_eps), e_bar.get_y() + e_bar.get_height() / 2),
    xytext=(max(e_eps) + 0.01, e_bar.get_y() + e_bar.get_height() / 2 + 0.25),
    fontsize=7, color='#555',
    arrowprops=dict(arrowstyle='->', color='#888', lw=0.8),
)

fig.tight_layout()
fig.savefig(OUT / 'fig3_scenarios.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig3_scenarios.png done')


# ================================================================== #
# Figure 4: Multi-model grouped bar chart
# ================================================================== #

models = ['GPT-4o', 'GPT-4o-mini', 'GPT-4-Turbo']
high_detection = [0.85, 0.70, 0.70]
low_fp         = [0.50, 0.60, 0.60]

C_DETECT = '#2E5C8A'   # darker blue
C_FP     = '#E8A87C'   # lighter / warm

fig, ax = plt.subplots(figsize=(7, 4.2))

x_m = np.arange(len(models))
w = 0.36

bars_d = ax.bar(x_m - w / 2, high_detection, w, color=C_DETECT,
                edgecolor='white', linewidth=0.6, zorder=3,
                label='HIGH detection rate')
bars_f = ax.bar(x_m + w / 2, low_fp, w, color=C_FP,
                edgecolor='white', linewidth=0.6, zorder=3,
                label='LOW false-positive rate')

# Value labels
for bar in bars_d:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
            f'{int(round(h * 100))}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#222')
for bar in bars_f:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
            f'{int(round(h * 100))}%', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#222')

ax.set_xticks(x_m)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel('rate')
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([f'{int(v*100)}%' for v in np.arange(0, 1.01, 0.2)])
ax.set_title(r'$\varepsilon$ across three GPT models (30-prompt benchmark)', pad=8)

ax.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc', fontsize=9)

ax.text(0.5, -0.18,
        'Detection scales with model calibration quality',
        transform=ax.transAxes, ha='center', fontsize=8.5,
        color='#555', style='italic')

fig.tight_layout()
fig.savefig(OUT / 'fig4_multimodel.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig4_multimodel.png done')


# ================================================================== #
# Figure 5: Threshold sweep — ε ROC and UQLM ROC (dual panel)
# ================================================================== #

eps_low  = np.array([r['epsilon'] for r in LOW])
eps_high = np.array([r['epsilon'] for r in HIGH])

conf_low  = np.array([r['uqlm_conf'] for r in LOW  if r['uqlm_conf'] is not None])
conf_high = np.array([r['uqlm_conf'] for r in HIGH if r['uqlm_conf'] is not None])

thresholds = np.linspace(0.0, 1.0, 201)

# ε: fire when epsilon >= t
eps_detect = np.array([(eps_high >= t).mean() for t in thresholds])
eps_fp     = np.array([(eps_low  >= t).mean() for t in thresholds])

# UQLM: fire when confidence < t
uqlm_detect = np.array([(conf_high < t).mean() for t in thresholds])
uqlm_fp     = np.array([(conf_low  < t).mean() for t in thresholds])

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.3))

# ---- Left panel: ε sweep ----
ax_l.plot(thresholds, eps_detect, color=C_HIGH, linewidth=2.0,
          label='HIGH detection rate', zorder=4)
ax_l.plot(thresholds, eps_fp, color=C_LOW, linewidth=2.0,
          label='LOW false-positive rate', zorder=4)

# Operating point at t=0.30
op = 0.30
op_idx = np.argmin(np.abs(thresholds - op))
op_detect = eps_detect[op_idx]
op_fp     = eps_fp[op_idx]

ax_l.axvline(op, color='#333', linestyle='--', linewidth=1.1, zorder=3)
ax_l.scatter([op, op], [op_detect, op_fp],
             color=['#333', '#333'], s=45, zorder=5,
             edgecolors='white', linewidths=0.8)

ax_l.annotate(
    f'operating point\nt=0.30\ndetection={op_detect*100:.0f}%, FP={op_fp*100:.0f}%',
    xy=(op, op_detect),
    xytext=(op + 0.12, op_detect - 0.02),
    fontsize=8, color='#333',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffbe6',
              edgecolor='#bbb', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#555', lw=0.8),
)

# Shade the "useful" range where detection > FP
useful_mask = eps_detect > eps_fp
ax_l.fill_between(thresholds, 0, 1, where=useful_mask,
                  color='#d5f5d5', alpha=0.35, zorder=1,
                  label='detection > FP')

ax_l.set_xlim(0, 1)
ax_l.set_ylim(0, 1.05)
ax_l.set_xlabel(r'$\varepsilon$ threshold $t$ (fire if $\varepsilon \geq t$)')
ax_l.set_ylabel('rate')
ax_l.set_title(r'$\varepsilon$ threshold sweep (gpt-4o)')
ax_l.legend(loc='upper right', framealpha=0.9, edgecolor='#ccc', fontsize=8.5)

# ---- Right panel: UQLM sweep ----
ax_r.plot(thresholds, uqlm_detect, color=C_HIGH, linewidth=2.0,
          label='HIGH detection rate', zorder=4)
ax_r.plot(thresholds, uqlm_fp, color=C_LOW, linewidth=2.0,
          label='LOW false-positive rate', zorder=4)

# Shade the "useful" range (where detection > FP)
uqlm_useful = uqlm_detect > uqlm_fp
ax_r.fill_between(thresholds, 0, 1, where=uqlm_useful,
                  color='#d5f5d5', alpha=0.35, zorder=1,
                  label='detection > FP (narrow/absent)')

# Annotate the inversion zone — where FP rises faster than detection
# Find where FP first equals or exceeds detection
cross_idx = None
for i in range(1, len(thresholds)):
    if uqlm_fp[i] >= uqlm_detect[i] and uqlm_fp[i] > 0:
        cross_idx = i
        break

if cross_idx is not None:
    t_cross = thresholds[cross_idx]
    ax_r.axvline(t_cross, color='#666', linestyle=':', linewidth=1.0, zorder=3)
    ax_r.annotate(
        f'FP catches detection\nat t≈{t_cross:.2f}\n(raising threshold\nhurts more than helps)',
        xy=(t_cross, uqlm_fp[cross_idx]),
        xytext=(t_cross + 0.05, 0.75),
        fontsize=7.5, color='#333',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff0f0',
                  edgecolor='#d99', alpha=0.9),
        arrowprops=dict(arrowstyle='->', color='#555', lw=0.8),
    )

ax_r.set_xlim(0, 1)
ax_r.set_ylim(0, 1.05)
ax_r.set_xlabel('UQLM confidence threshold $t$ (fire if conf < $t$)')
ax_r.set_ylabel('rate')
ax_r.set_title('UQLM threshold sweep (gpt-4o)')
ax_r.legend(loc='upper left', framealpha=0.9, edgecolor='#ccc', fontsize=8.5)

fig.suptitle(
    r'Threshold sweep: $\varepsilon$ separates HIGH from LOW; UQLM does not',
    fontsize=11.5, y=1.02
)

fig.tight_layout()
fig.savefig(OUT / 'fig5_roc_sweep.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('fig5_roc_sweep.png done')


print(f'\nAll figures saved to {OUT}/')
