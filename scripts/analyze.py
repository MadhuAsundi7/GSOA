#!/usr/bin/env python3
"""
Reproduce every statistic, table and figure for a CEC2022 dimension.

    python scripts/analyze.py --dim 10 --results results/cec2022/d10 --out analysis_out/d10
    python scripts/analyze.py --dim 20 --results results/cec2022/d20 --out analysis_out/d20

Produces: summary CSV (min/max/mean/std), per-function ranks, Friedman with
Iman-Davenport, Wilcoxon at function level with Holm correction, per-function
Wilcoxon over the runs, win/loss/tie, a critical difference diagram and a
convergence figure.
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, rankdata, f as fdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from gsoa.core import results_io as rio                    # noqa: E402

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']

DEFAULT_ALGS = ['GSOA', 'EA4eig', 'S-LSHADE-DP', 'NL-SHADE-LBC',
                'NL-SHADE-RSP-MID', 'GSA', 'PSO', 'DE', 'GA']
GROUPS = {'Unimodal': [1], 'Basic': [2, 3, 4, 5],
          'Hybrid': [6, 7, 8], 'Composition': [9, 10, 11, 12]}
COL = {'GSOA': '#d62728', 'EA4eig': '#17becf', 'S-LSHADE-DP': '#8c564b',
       'NL-SHADE-LBC': '#7f7f7f', 'NL-SHADE-RSP-MID': '#bcbd22',
       'GSA': '#1f77b4', 'PSO': '#2ca02c', 'DE': '#9467bd', 'GA': '#ff7f0e'}
STY = {'GSOA': '-', 'EA4eig': '--', 'S-LSHADE-DP': '-.', 'NL-SHADE-LBC': ':',
       'NL-SHADE-RSP-MID': (0, (5, 1)), 'GSA': (0, (3, 1, 1, 1)),
       'PSO': (0, (1, 1)), 'DE': (0, (4, 2)), 'GA': (0, (2, 2, 6, 2))}
#: Nemenyi critical values q_0.05 by number of algorithms
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
       7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def cd_diagram(mr, algs, CD, path):
    order = sorted(algs, key=lambda a: mr[a])
    rk = [mr[a] for a in order]
    cliques = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and rk[j + 1] - rk[i] <= CD:
            j += 1
        if j > i:
            cliques.append((i, j))
    maximal = [c for c in cliques
               if not any(c != o and o[0] <= c[0] and c[1] <= o[1] for o in cliques)]
    lo, hi = 1, len(algs)
    pad = 2.6 if len(algs) > 6 else 1.1
    fig, ax = plt.subplots(figsize=(9.4, 1.2 + 0.34 * len(algs)))
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.35 - 0.16 * len(algs), 1.05)
    ax.axis('off')
    ax.plot([lo, hi], [0, 0], 'k-', lw=1.5)
    for v in np.arange(lo, hi + 1e-9, 0.5):
        big = abs(v - round(v)) < 1e-9
        ax.plot([v, v], [0, 0.10 if big else 0.055], 'k-', lw=1.5 if big else 1.0)
        if big:
            ax.text(v, 0.15, f'{int(v)}', ha='center', va='bottom', fontsize=10)
    ax.plot([lo, lo + CD], [0.55, 0.55], 'k-', lw=1.8)
    for xx in (lo, lo + CD):
        ax.plot([xx, xx], [0.49, 0.61], 'k-', lw=1.8)
    ax.text(lo + CD / 2, 0.65, f'CD = {CD:.3f}', ha='center', va='bottom', fontsize=10)
    nL = (len(order) + 1) // 2
    for i, a in enumerate(order):
        r = mr[a]
        if i < nL:
            yv = -0.22 - 0.15 * i; xt = lo - pad + 0.15; ha = 'right'; xl = xt - 0.06
        else:
            yv = -0.22 - 0.15 * (len(order) - 1 - i); xt = hi + pad - 0.15
            ha = 'left'; xl = xt + 0.06
        ax.plot([r, r], [0, yv], 'k-', lw=1.0)
        ax.plot([r, xt], [yv, yv], 'k-', lw=1.0)
        ax.text(xl, yv, f'{a} ({r:.3f})', ha=ha, va='center', fontsize=10,
                fontweight='bold' if a == 'GSOA' else 'normal')
    for m, (i, j) in enumerate(maximal):
        ax.plot([rk[i] - 0.05, rk[j] + 0.05], [-0.075 - 0.075 * m] * 2,
                'k-', lw=4.2, solid_capstyle='butt')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(f'{path}.{ext}', dpi=600, bbox_inches='tight')
    plt.close('all')


def conv_figure(curves, chk, algs, path):
    fig, axes = plt.subplots(4, 3, figsize=(13, 14))
    axes = axes.ravel()
    for k, f in enumerate(range(1, 13)):
        ax = axes[k]
        for a in algs:
            ax.loglog(chk, np.maximum(curves[a][f], 1e-9), color=COL.get(a),
                      ls=STY.get(a, '-'), lw=2.2 if a == 'GSOA' else 1.2,
                      label=a, alpha=0.92)
        ax.set_title(f'F{f}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Function evaluations', fontsize=8.5)
        ax.set_ylabel('Mean error', fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, which='major', ls=':', alpha=0.35)
        ax.set_xlim(chk[0], chk[-1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.012))
    plt.tight_layout(rect=[0, 0.035, 1, 1])
    for ext in ('png', 'pdf'):
        plt.savefig(f'{path}.{ext}', dpi=600 if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close('all')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, choices=[10, 20], required=True)
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--algorithms', nargs='*', default=DEFAULT_ALGS)
    ap.add_argument('--reference', default='GSOA')
    args = ap.parse_args()

    D, algs, ref = args.dim, args.algorithms, args.reference
    os.makedirs(args.out, exist_ok=True)
    F = list(range(1, 13))

    problems = rio.validate(args.results, algs, D)
    if problems:
        print('validation problems:')
        for p in problems:
            print('  ', p)
    else:
        print('result files validated: shapes, monotone checkpoints, budget')

    raw, curves = rio.load_all(args.results, algs, D)
    means = {a: np.array([raw[a][f].mean() for f in F]) for a in algs}
    M = np.array([rankdata([means[a][i] for a in algs]) for i in range(12)])
    mr = dict(zip(algs, M.mean(0)))

    chi2, p = friedmanchisquare(*[means[a] for a in algs])
    k, n = len(algs), 12
    Fi = (n - 1) * chi2 / (n * (k - 1) - chi2)
    pi = 1 - fdist.cdf(Fi, k - 1, (k - 1) * (n - 1))
    CD = Q05[k] * np.sqrt(k * (k + 1) / (6 * n))

    print(f'\nFriedman: chi2={chi2:.4f} (df={k-1}) p={p:.4g}')
    print(f'Iman-Davenport: F={Fi:.4f} p={pi:.4g}   Nemenyi CD={CD:.4f}\n')
    for r, a in enumerate(sorted(algs, key=lambda x: mr[x]), 1):
        print(f'  {r}. {a:<18} {mr[a]:.3f}')

    # summary csv
    with open(os.path.join(args.out, 'summary.csv'), 'w') as fh:
        fh.write('function,metric,' + ','.join(algs) + '\n')
        for f in F:
            for lab, fn in [('min', np.min), ('max', np.max), ('mean', np.mean),
                            ('std', lambda v: np.std(v, ddof=1))]:
                fh.write(f'F{f},{lab},' +
                         ','.join(f'{fn(raw[a][f]):.6e}' for a in algs) + '\n')

    # ranks csv
    with open(os.path.join(args.out, 'ranks.csv'), 'w') as fh:
        fh.write('function,' + ','.join(algs) + '\n')
        for i, f in enumerate(F):
            fh.write(f'F{f},' + ','.join(f'{M[i, j]:g}' for j in range(k)) + '\n')
        fh.write('mean_rank,' + ','.join(f'{v:.3f}' for v in M.mean(0)) + '\n')

    # wilcoxon, function level, Holm
    others = [a for a in algs if a != ref]
    rows = []
    for a in others:
        d = means[ref] - means[a]
        W, pv = ((np.nan, 1.0) if np.all(d == 0)
                 else wilcoxon(d, zero_method='wilcox')[:2])
        rows.append((a, W, pv, int((d < 0).sum()), int((d > 0).sum())))
    rows.sort(key=lambda r: r[2])
    print(f'\nWilcoxon ({ref} vs each, Holm over {len(rows)}):')
    with open(os.path.join(args.out, 'wilcoxon_function_level.csv'), 'w') as fh:
        fh.write('vs,W,p_value,wins,losses,holm_threshold,decision\n')
        for i, (a, W, pv, b, w) in enumerate(rows, 1):
            ht = 0.05 / (len(rows) - i + 1)
            dec = (f'{ref} better' if pv < ht and b > w
                   else f'{ref} worse' if pv < ht else 'not significant')
            fh.write(f'{a},{W:.1f},{pv:.6g},{b},{w},{ht:.6f},{dec}\n')
            print(f'  vs {a:<18} W={W:>6.1f} p={pv:<11.4g} {b:2d}/{w:2d}  {dec}')

    # per-function wilcoxon
    with open(os.path.join(args.out, 'wilcoxon_per_function.csv'), 'w') as fh:
        fh.write('function,vs,p_value,direction\n')
        for f in F:
            for a in others:
                d = raw[ref][f] - raw[a][f]
                if np.all(d == 0):
                    pv, side = 1.0, 'tie'
                else:
                    pv = wilcoxon(d, zero_method='wilcox')[1]
                    side = 'better' if np.median(d) < 0 else 'worse'
                fh.write(f'F{f},{a},{pv:.6g},{side if pv < 0.05 else "no_diff"}\n')

    # win/loss/tie
    with open(os.path.join(args.out, 'win_loss_tie.csv'), 'w') as fh:
        fh.write('vs,wins,losses,ties\n')
        for a, W, pv, b, w in rows:
            fh.write(f'{a},{b},{w},{12 - b - w}\n')

    # group ranks
    with open(os.path.join(args.out, 'group_ranks.csv'), 'w') as fh:
        fh.write('group,' + ','.join(algs) + '\n')
        for g, fs in GROUPS.items():
            idx = [F.index(x) for x in fs]
            Rg = np.array([rankdata([means[a][j] for a in algs]) for j in idx]).mean(0)
            fh.write(f'{g},' + ','.join(f'{v:.3f}' for v in Rg) + '\n')

    cd_diagram(mr, algs, CD, os.path.join(args.out, f'cd_diagram_d{D}'))
    conv_figure(curves, rio.checkpoints(D), algs,
                os.path.join(args.out, f'convergence_d{D}'))
    print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    main()
