#!/usr/bin/env python3
"""
Run GSOA (and optionally the baselines) on the official CEC2022 suite.

    bash scripts/build_cec2022.sh            # once
    python scripts/run_cec2022.py --dim 10 --runs 30 --out results/cec2022_d10
    python scripts/run_cec2022.py --dim 20 --runs 30 --out results/cec2022_d20

Errors are recorded at the 16 standard CEC checkpoints and written one
pickle per function, so an interrupted run resumes where it stopped.

Use --algorithms GSOA to run GSOA alone, for example when the competing
algorithms' published result files are used instead.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from gsoa import Budget, ALGORITHMS, search_alpha            # noqa: E402
from gsoa.core.cec2022 import CEC2022, LB, UB                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, choices=[10, 20], required=True)
    ap.add_argument('--pop', type=int, default=30)
    ap.add_argument('--maxfes', type=int, default=None,
                    help='default: 200000 for D=10, 1000000 for D=20')
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--base-seed', type=int, default=0)
    ap.add_argument('--tune-seed', type=int, default=1000)
    ap.add_argument('--tune-runs', type=int, default=3)
    ap.add_argument('--functions', type=int, nargs='*',
                    default=list(range(1, 13)))
    ap.add_argument('--algorithms', nargs='*', default=['GSOA'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--skip-verify', action='store_true')
    args = ap.parse_args()

    suite = CEC2022(args.dim)
    maxfes = args.maxfes or suite.maxfes
    out = args.out or f'results/cec2022_d{args.dim}'

    if not args.skip_verify:
        print('verifying suite against published optima ...')
        print('  OK' if suite.verify() else '  MISMATCH - check input_data/')

    out = os.path.abspath(out)          # CEC2022 changes the working directory
    ck = os.path.join(out, 'ckpt')
    os.makedirs(ck, exist_ok=True)

    alpha_grid = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    tune_seeds = range(args.tune_seed, args.tune_seed + args.tune_runs)
    D, N = args.dim, args.pop
    t0 = time.time()

    for fid in args.functions:
        path = os.path.join(ck, f'F{fid}.pkl')
        if os.path.exists(path):
            print(f'  F{fid:02d} cached', flush=True)
            continue

        def make_budget(_fid=fid):
            return Budget(lambda X: suite.evaluate(_fid, X),
                          maxfes, suite.f_star(_fid))

        rec = {'algorithms': {}}
        if 'GSOA' in args.algorithms:
            alpha, scores = search_alpha(make_budget, D, N, LB, UB,
                                         alpha_grid, tune_seeds)
            rec['alpha'] = alpha
            rec['alpha_scores'] = scores

        for name in args.algorithms:
            fn = ALGORITHMS[name]
            errs, curves = [], []
            for r in range(args.runs):
                B = make_budget()
                seed = args.base_seed + r
                if name == 'GSOA':
                    fn(B, D, N, LB, UB, rec['alpha'], seed)
                else:
                    fn(B, D, N, LB, UB, seed)
                errs.append(B.error)
                curves.append(B.cec_checkpoints(D))
            rec['algorithms'][name] = {'errors': np.array(errs),
                                       'curves': np.array(curves)}
        with open(path, 'wb') as fh:
            pickle.dump(rec, fh)
        line = '  '.join(f'{a}={rec["algorithms"][a]["errors"].mean():.3e}'
                         for a in args.algorithms)
        tag = f' (alpha*={rec["alpha"]:+.1f})' if 'alpha' in rec else ''
        print(f'  F{fid:02d}{tag}  {line}  [{time.time() - t0:.0f}s]', flush=True)

    print(f'done in {time.time() - t0:.0f}s -> {ck}')


if __name__ == '__main__':
    main()
