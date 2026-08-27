"""
GSOA v6 on the mechanical design problems B1-B3.

Phase 1: alpha grid search per problem, scored as the mean over held-out
         seeds that are never used for evaluation.
Phase 2: 30 independent runs at alpha*, seeds 0-29.

native search in
[-100,100]^D with affine rescaling to true bounds inside the evaluator,
static penalty 1e6, MaxFES = 200,000, output as a 16 x 30 matrix of
best-so-far values at the 16 CEC checkpoints.
"""

import os
import sys
import time
import pickle

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from gsoa.core.mechanical import PROBLEMS, penalised, feasible          # noqa: E402
from gsoa import Budget, gsoa

N = 30
MAXFES = 200_000
N_RUNS = 30
BASE_SEED = 0
TUNE_SEED = 1000
N_TUNE = 5
LB, UB = -100.0, 100.0
ALPHA_GRID = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

OUT = '/mnt/user-data/outputs/mechanical'
os.makedirs(OUT, exist_ok=True)


def checkpoints(D):
    return np.array([D ** (k / 5.0 - 3.0) * MAXFES for k in range(16)])


def run_once(pid, alpha, seed):
    p = PROBLEMS[pid]
    B = Budget(lambda X: penalised(pid, X), MAXFES, 0.0)
    gsoa(B, p['dim'], N, LB, UB, alpha, seed)
    return B


if __name__ == '__main__':
    t0 = time.time()

    # ── Phase 1 ──────────────────────────────────────────────────────
    print('=' * 78)
    print(f'PHASE 1  alpha search   MAXFES={MAXFES}  '
          f'{N_TUNE} runs on held-out seeds {TUNE_SEED}..{TUNE_SEED+N_TUNE-1}')
    print('=' * 78)
    best_alpha, alpha_scores = {}, {}
    for pid in ['B1', 'B2', 'B3']:
        sc = {}
        for a in ALPHA_GRID:
            vals = [run_once(pid, a, TUNE_SEED + k).gbest for k in range(N_TUNE)]
            sc[a] = float(np.mean(vals))
        alpha_scores[pid] = sc
        best_alpha[pid] = min(ALPHA_GRID, key=lambda a: sc[a])
        row = '  '.join(f'{a:+.1f}:{sc[a]:.6g}' for a in ALPHA_GRID)
        print(f"  {PROBLEMS[pid]['name']}")
        print(f"    alpha* = {best_alpha[pid]:+.1f}   mean = {sc[best_alpha[pid]]:.6f}")
        print(f"    grid: {row}   [{time.time()-t0:.0f}s]", flush=True)

    # ── Phase 2 ──────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print(f'PHASE 2  {N_RUNS} runs, seeds {BASE_SEED}..{BASE_SEED+N_RUNS-1}')
    print('=' * 78)
    results = {}
    for pid in ['B1', 'B2', 'B3']:
        p = PROBLEMS[pid]
        chk = checkpoints(p['dim'])
        finals, curves, feas, best_x = [], [], [], None
        best_val = np.inf
        for r in range(N_RUNS):
            B = run_once(pid, best_alpha[pid], BASE_SEED + r)
            finals.append(B.gbest)
            curves.append(B.curve(chk))
            if B.gbest < best_val:
                best_val = B.gbest
        finals = np.array(finals)
        # feasibility of each run's best value: penalised == objective only
        # when no constraint is violated, so a run is feasible iff its value
        # is below the penalty scale
        results[pid] = dict(alpha=best_alpha[pid], finals=finals,
                            curves=np.array(curves), chk=chk)
        print(f"  {pid}  alpha*={best_alpha[pid]:+.1f}  "
              f"best={finals.min():.6f}  mean={finals.mean():.6f}  "
              f"std={finals.std(ddof=1):.3e}  ref={p['ref']:.6f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    pickle.dump(dict(results=results, alpha_scores=alpha_scores,
                     best_alpha=best_alpha),
                open(f'{OUT}/gsoa_mechanical.pkl', 'wb'))

    # write result files in the repository's format
    for pid, r in results.items():
        M = r['curves'].T                       # 16 x 30
        assert M.shape == (16, N_RUNS)
        np.savetxt(f'{OUT}/GSOA_{pid}.txt', M, fmt='%.10e', delimiter='\t')
    print(f'\nsaved to {OUT}')
    print(f'Total {time.time()-t0:.0f}s')
