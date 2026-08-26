"""
Reader for the standard CEC result-file format.

Each file holds a 17 x 30 matrix for one algorithm on one function:

    rows 0-15   best-so-far error at the 16 checkpoints
                FES_k = D^(k/5 - 3) * MaxFES
    row 16      the number of evaluations consumed by each run
    columns     the 30 independent runs

Row 15 is therefore the final error of each run.
"""

import os
import re

import numpy as np

CHECKPOINT_ROWS = 16
N_RUNS = 30

#: Filename templates. NL-SHADE-RSP-MID encodes its population size.
PATTERNS = {
    'GSOA':             '{alg}_{f}_{D}.txt',
    'GSA':              '{alg}_{f}_{D}.txt',
    'PSO':              '{alg}_{f}_{D}.txt',
    'DE':               '{alg}_{f}_{D}.txt',
    'GA':               '{alg}_{f}_{D}.txt',
    'EA4eig':           '{alg}_{f}_{D}.txt',
    'S-LSHADE-DP':      '{alg}_{f}_{D}.txt',
    'NL-SHADE-LBC':     '{alg}_{f}_{D}.txt',
    'NL-SHADE-RSP-MID': 'NL-SHADE-RSP_MID{f}_{D}_pop_{pop}.txt',
}

#: population size used by NL-SHADE-RSP-MID at each dimension
RSP_MID_POP = {10: 50, 20: 100}

F_STAR = [300, 400, 600, 800, 900, 1800,
          2000, 2200, 2300, 2400, 2600, 2700]
MAXFES = {10: 200_000, 20: 1_000_000}


def filename(alg, fid, D):
    return PATTERNS[alg].format(alg=alg, f=fid, D=D, pop=RSP_MID_POP.get(D))


def load_matrix(directory, alg, fid, D):
    path = os.path.join(directory, filename(alg, fid, D))
    M = np.loadtxt(path)
    if M.shape != (17, N_RUNS):
        raise ValueError(f'{path}: expected (17, {N_RUNS}), got {M.shape}')
    return M


def load_algorithm(directory, alg, D, functions=range(1, 13)):
    """Returns (finals, curves) keyed by function id."""
    finals, curves = {}, {}
    for f in functions:
        M = load_matrix(directory, alg, f, D)
        finals[f] = M[CHECKPOINT_ROWS - 1]
        curves[f] = M[:CHECKPOINT_ROWS].mean(axis=1)
    return finals, curves


def load_all(directory, algorithms, D, functions=range(1, 13)):
    raw, curves = {}, {}
    for a in algorithms:
        raw[a], curves[a] = load_algorithm(directory, a, D, functions)
    return raw, curves


def checkpoints(D, maxfes=None):
    maxfes = maxfes or MAXFES[D]
    return np.array([D ** (k / 5.0 - 3.0) * maxfes for k in range(CHECKPOINT_ROWS)])


def validate(directory, algorithms, D, functions=range(1, 13)):
    """Sanity checks: shape, monotone checkpoints, non-negative errors."""
    problems = []
    for a in algorithms:
        for f in functions:
            try:
                M = load_matrix(directory, a, f, D)
            except Exception as exc:                       # noqa: BLE001
                problems.append(f'{a} F{f}: {exc}')
                continue
            conv = M[:CHECKPOINT_ROWS]
            if not np.all(np.diff(conv, axis=0) <= 1e-6):
                problems.append(f'{a} F{f}: checkpoints not non-increasing')
            if np.any(conv < -1e-12):
                problems.append(f'{a} F{f}: negative error')
            if np.any(M[16] > MAXFES[D] * 1.001):
                problems.append(f'{a} F{f}: FES exceeds MaxFES')
    return problems
