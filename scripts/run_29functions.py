#!/usr/bin/env python3
"""
Run GSOA and the baseline algorithms on the 29-function suite.

    python scripts/run_29functions.py --runs 30 --out results/f29

Phase 1 selects alpha per function on held-out seeds; phase 2 executes the
evaluation runs. All algorithms receive the same evaluation budget.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from gsoa import Budget, ALGORITHMS, search_alpha            # noqa: E402
from gsoa.core import benchmark29 as bench                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=30)
    ap.add_argument('--pop', type=int, default=30)
    ap.add_argument('--maxfes', type=int, default=15000)
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--base-seed', type=int, default=0)
    ap.add_argument('--tune-seed', type=int, default=1000)
    ap.add_argument('--tune-runs', type=int, default=5)
    ap.add_argument('--functions', type=int, nargs='*', default=bench.ALL_FIDS)
    ap.add_argument('--algorithms', nargs='*', default=list(ALGORITHMS))
    ap.add_argument('--out', default='results/f29')
    args = ap.parse_args()

    if args.dim != bench.D:
        raise SystemExit(f'benchmark29 is configured for D={bench.D}')

    os.makedirs(args.out, exist_ok=True)
    ck = os.path.join(args.out, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    alpha_grid = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    tune_seeds = range(args.tune_seed, args.tune_seed + args.tune_runs)
    D, N, LB, UB = args.dim, args.pop, bench.LB, bench.UB
    t0 = time.time()

    for fid in args.functions:
        path = os.path.join(ck, f'F{fid}.pkl')
        if os.path.exists(path):
            print(f'  F{fid:02d} cached', flush=True)
            continue

        def make_budget(_fid=fid):
            return Budget(lambda X: bench.evaluate(_fid, X), args.maxfes, 0.0)

        alpha, scores = search_alpha(make_budget, D, N, LB, UB,
                                     alpha_grid, tune_seeds)
        rec = {'alpha': alpha, 'alpha_scores': scores, 'algorithms': {}}
        for name in args.algorithms:
            fn = ALGORITHMS[name]
            finals, curves = [], []
            for r in range(args.runs):
                B = make_budget()
                seed = args.base_seed + r
                if name == 'GSOA':
                    fn(B, D, N, LB, UB, alpha, seed)
                else:
                    fn(B, D, N, LB, UB, seed)
                finals.append(B.gbest)
                curves.append(B.curve(np.linspace(N, args.maxfes, 400)))
            rec['algorithms'][name] = {'finals': np.array(finals),
                                       'curves': np.array(curves)}
        with open(path, 'wb') as fh:
            pickle.dump(rec, fh)
        line = '  '.join(f'{a}={rec["algorithms"][a]["finals"].mean():.3e}'
                         for a in args.algorithms)
        print(f'  F{fid:02d} (alpha*={alpha:+.1f})  {line}  '
              f'[{time.time() - t0:.0f}s]', flush=True)

    print(f'done in {time.time() - t0:.0f}s -> {ck}')


if __name__ == '__main__':
    main()
