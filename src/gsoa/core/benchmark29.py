"""
The 29-function shifted-and-rotated benchmark suite used in this study.

Groups: unimodal F1-F3, multimodal F4-F10, hybrid F11-F20 (genuine hybrids
with variable partitioning), composition F21-F29. Search range [-100, 100]^D,
additive bias 100*i on function i.

The landscape (shift vectors and rotation matrices) is generated
pseudo-randomly under a fixed seed so the suite is reproducible; it is NOT
the CEC2017 competition data. See README for the distinction.
"""

import numpy as np

EPS = 1e-12
D = 30                    # set before import-time use; see set_dimension()
LB, UB = -100.0, 100.0
ALL_FIDS = list(range(1, 30))
GROUPS = {
    'Unimodal':    list(range(1, 4)),
    'Multimodal':  list(range(4, 11)),
    'Hybrid':      list(range(11, 21)),
    'Composition': list(range(21, 30)),
}

# ══════════════════════════════════════════════════════════════════════
# LANDSCAPE  (unchanged: shifts ~ U(-80,80), rotations from QR, seed 2017)
# ══════════════════════════════════════════════════════════════════════
np.random.seed(2017)
_SHIFTS = {d: np.random.uniform(-80, 80, (29, d)) for d in [10, 30]}
_ROTS = {d: np.array([np.linalg.qr(np.random.randn(d, d))[0]
                      for _ in range(29)]) for d in [10, 30]}
BIAS = np.arange(1, 30) * 100.0


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS  (identical to the supplied implementation)
# ══════════════════════════════════════════════════════════════════════
def _z(X, fid):  return (X - _SHIFTS[D][fid]) @ _ROTS[D][fid]
def _zs(X, fid): return X - _SHIFTS[D][fid]

def _sphere(Z): return np.sum(Z**2, axis=1)
def _bent(Z):   return Z[:, 0]**2 + 1e6 * np.sum(Z[:, 1:]**2, axis=1)

def _rosen(Z):
    Y = Z + 1
    return np.sum(100*(Y[:, 1:] - Y[:, :-1]**2)**2 + (Y[:, :-1] - 1)**2, axis=1)

def _rast(Z):   return np.sum(Z**2 - 10*np.cos(2*np.pi*Z) + 10, axis=1)

def _ackley(Z):
    return (-20*np.exp(-0.2*np.sqrt(np.mean(Z**2, axis=1)))
            - np.exp(np.mean(np.cos(2*np.pi*Z), axis=1)) + 20 + np.e)

def _weier(Z):
    nk = Z.shape[1]
    k = np.arange(21); ak = 0.5**k; bk = 3.0**k
    return (np.sum(ak * np.cos(2*np.pi*bk * (Z[:, :, None] + 0.5)), axis=(1, 2))
            - nk * np.sum(ak * np.cos(np.pi * bk)))

def _griew(Z):
    nk = Z.shape[1]
    return 1 + np.sum(Z**2, axis=1)/4000 - np.prod(
        np.cos(Z / np.sqrt(np.arange(1, nk+1))), axis=1)

def _schwe(Z):
    nk = Z.shape[1]
    Z2 = Z * 10
    return 418.9829*nk - np.sum(Z2 * np.sin(np.sqrt(np.abs(Z2))), axis=1)

def _scaff(Z):
    nk = Z.shape[1]
    r = np.zeros(Z.shape[0])
    for d in range(nk - 1):
        x, y = Z[:, d], Z[:, d+1]; s = x**2 + y**2
        r += 0.5 + (np.sin(np.sqrt(s))**2 - 0.5) / (1 + 0.001*s)**2
    x, y = Z[:, -1], Z[:, 0]; s = x**2 + y**2
    r += 0.5 + (np.sin(np.sqrt(s))**2 - 0.5) / (1 + 0.001*s)**2
    return r

def _hgbat(Z):
    s = np.sum(Z**2, axis=1); t = np.sum(Z, axis=1)
    return np.abs(s**2 - t**2)**0.5 + (0.5*s + t)/Z.shape[1] + 0.5

def _egr(Z):
    Y = Z + 1
    ros = 100*(Y[:, 1:] - Y[:, :-1]**2)**2 + (Y[:, :-1] - 1)**2
    return np.sum(ros**2/4000 - np.cos(ros) + 1, axis=1)


# ══════════════════════════════════════════════════════════════════════
# HYBRID FUNCTIONS  F11-F20
# Each is a genuine hybrid: the shifted vector is permuted by a fixed
# shuffle vector, split into N consecutive blocks whose sizes follow the
# proportions p, and each block is rotated by its own orthogonal matrix
# and passed to its own basic kernel. The block values are summed.
#
# A dedicated RandomState is used so the existing _SHIFTS / _ROTS draw
# order is untouched and F1-F10, F21-F29 are bit-identical to before.
# ══════════════════════════════════════════════════════════════════════
HYBRID_SPEC = {
    11: ([_bent,  _rosen,  _rast],                        [0.3, 0.3, 0.4]),
    12: ([_ackley, _schwe, _sphere],                       [0.3, 0.3, 0.4]),
    13: ([_rosen, _griew,  _rast],                        [0.2, 0.4, 0.4]),
    14: ([_bent,  _ackley, _rast,  _griew],               [0.2, 0.2, 0.3, 0.3]),
    15: ([_rosen, _hgbat,  _rast,  _schwe],               [0.2, 0.2, 0.3, 0.3]),
    16: ([_scaff, _hgbat,  _rosen, _schwe],               [0.2, 0.2, 0.3, 0.3]),
    17: ([_weier, _ackley, _egr,   _schwe, _rast],        [0.1, 0.2, 0.2, 0.2, 0.3]),
    18: ([_bent,  _ackley, _rast,  _hgbat, _sphere],      [0.2, 0.2, 0.2, 0.2, 0.2]),
    19: ([_bent,  _rast,   _egr,   _weier, _scaff],       [0.2, 0.2, 0.2, 0.2, 0.2]),
    20: ([_hgbat, _weier,  _ackley, _rast, _schwe, _griew],
         [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]),
}

KERNEL_NAME = {
    _sphere: 'Sphere', _bent: 'Bent Cigar', _rosen: 'Rosenbrock',
    _rast: 'Rastrigin', _ackley: 'Ackley', _weier: 'Weierstrass',
    _griew: 'Griewank', _schwe: 'Schwefel', _scaff: 'Expanded Schaffer F6',
    _hgbat: 'HGBat', _egr: 'Expanded Griewank plus Rosenbrock',
}


def _block_sizes(props, dim):
    """n_k = ceil(p_k * dim) for k < N, last block takes the remainder."""
    n = [int(np.ceil(p * dim)) for p in props[:-1]]
    n.append(dim - int(np.sum(n)))
    return n


_HRS = np.random.RandomState(2018)          # dedicated stream, seed 2018
_SHUFFLE = {d: np.array([_HRS.permutation(d) for _ in range(29)])
            for d in [10, 30]}
_HROTS = {}
for _d in [10, 30]:
    for _fid, (_fns, _props) in HYBRID_SPEC.items():
        _sizes = _block_sizes(_props, _d)
        _HROTS[(_d, _fid)] = [np.linalg.qr(_HRS.randn(nk, nk))[0]
                              for nk in _sizes]


def _hybrid(X, fid):
    """fid is the 1-based function id (11..20)."""
    fns, props = HYBRID_SPEC[fid]
    Y = X - _SHIFTS[D][fid - 1]              # shift
    Y = Y[:, _SHUFFLE[D][fid - 1]]           # permute the variables
    sizes = _block_sizes(props, D)
    rots = _HROTS[(D, fid)]
    out = np.zeros(X.shape[0])
    a = 0
    for k, nk in enumerate(sizes):
        Zk = Y[:, a:a + nk] @ rots[k]        # rotate this block
        out += fns[k](Zk)                    # own kernel on the block
        a += nk
    return out


# The probe-based normalisation constants are deterministic (RandomState(0)
# and fixed lambda/R), so caching them is a pure speed-up: identical values.
_NORM_CACHE = {}

def _comp(X, fid, fns, sigs, lams):
    S, R = _SHIFTS[D], _ROTS[D]
    Np = X.shape[0]; n = len(fns)

    W = np.zeros((Np, n))
    for i in range(n):
        dz = X - S[min(fid + i, 28)]
        W[:, i] = np.exp(-np.sum(dz**2, axis=1) / (2 * D * sigs[i]**2))
    Ws = np.sum(W, axis=1, keepdims=True)
    W = np.where(Ws > 0, W / Ws, 1.0 / n)

    if fid not in _NORM_CACHE:
        scales = []
        rng_probe = np.random.RandomState(0)
        probe = rng_probe.uniform(-5, 5, (5, D))
        for i in range(n):
            zp = probe / lams[i] @ R[fid]
            val = np.abs(fns[i](zp))
            val = np.where(val < EPS, 1.0, val)
            scales.append(2000.0 / np.mean(val))
        _NORM_CACHE[fid] = scales
    scales = _NORM_CACHE[fid]

    res = np.zeros(Np)
    for i in range(n):
        zi = (X - S[min(fid + i, 28)]) / lams[i] @ R[fid]
        res += W[:, i] * (fns[i](zi) * scales[i])
    return res


def cec_f(fid, X):
    b = BIAS[fid - 1]
    if   fid ==  1: return _bent(_zs(X, fid-1)) + b
    elif fid ==  2: return _sphere(_zs(X, fid-1)) + b
    elif fid ==  3: return _sphere(_zs(X, fid-1))*1e-4 + b
    elif fid ==  4: return _rosen(_z(X, fid-1)) + b
    elif fid ==  5: return _rast(_z(X, fid-1)) + b
    elif fid ==  6: return _ackley(_z(X, fid-1)) + b
    elif fid ==  7: return _weier(_z(X, fid-1)) + b
    elif fid ==  8: return _griew(_z(X, fid-1)) + b
    elif fid ==  9: return _schwe(_z(X, fid-1)) + b
    elif fid == 10: return _rast(_z(X, fid-1)) + b
    elif fid == 11: return _hybrid(X, fid) + b
    elif fid == 12: return _hybrid(X, fid) + b
    elif fid == 13: return _hybrid(X, fid) + b
    elif fid == 14: return _hybrid(X, fid) + b
    elif fid == 15: return _hybrid(X, fid) + b
    elif fid == 16: return _hybrid(X, fid) + b
    elif fid == 17: return _hybrid(X, fid) + b
    elif fid == 18: return _hybrid(X, fid) + b
    elif fid == 19: return _hybrid(X, fid) + b
    elif fid == 20: return _hybrid(X, fid) + b
    elif fid == 21: return _comp(X, fid-1, [_rosen,  _rast,   _schwe], [10,20,30], [1,10,1]) + b
    elif fid == 22: return _comp(X, fid-1, [_rast,   _griew,  _schwe], [10,20,30], [1,10,1]) + b
    elif fid == 23: return _comp(X, fid-1, [_rosen,  _ackley, _schwe], [10,20,30], [1,10,1]) + b
    elif fid == 24: return _comp(X, fid-1, [_ackley, _rast,   _griew], [10,20,30], [1,10,1]) + b
    elif fid == 25: return _comp(X, fid-1, [_rast,   _schwe,  _rosen], [10,20,30], [1,10,1]) + b
    elif fid == 26: return _comp(X, fid-1, [_scaff,  _schwe,  _griew], [10,20,30], [1,10,1]) + b
    elif fid == 27: return _comp(X, fid-1, [_hgbat,  _rast,   _schwe], [10,20,30], [1,10,1]) + b
    elif fid == 28: return _comp(X, fid-1, [_ackley, _griew,  _rosen], [10,20,30], [1,10,1]) + b
    elif fid == 29: return _comp(X, fid-1, [_rast,   _ackley, _schwe], [10,20,30], [1,10,1]) + b




def evaluate(fid, X):
    """Objective value of function `fid` (1..29) at each row of X."""
    return cec_f(fid, np.atleast_2d(X))
