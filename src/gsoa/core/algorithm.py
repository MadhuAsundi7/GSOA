"""
Gravitational Slingshot Optimization Algorithm (GSOA) and the baseline
algorithms used for comparison.

Every optimiser terminates on a shared evaluation budget rather than a fixed
iteration count, so algorithms with different per-iteration evaluation costs
are compared fairly.
"""

import numpy as np

EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────
# Evaluation budget meter
# ──────────────────────────────────────────────────────────────────────
class Budget:
    """Wraps an objective function, counts every evaluation and records the
    best-so-far trace so convergence can be plotted against evaluations.

    Parameters
    ----------
    fn : callable
        Takes an (m, D) array, returns an (m,) array of objective values.
    maxfes : int
        Evaluation budget. ``done`` becomes True once it is reached.
    f_star : float
        Known optimum, subtracted when reporting ``error``. Use 0.0 if the
        raw objective value should be reported instead.
    """

    def __init__(self, fn, maxfes, f_star=0.0):
        self.fn = fn
        self.maxfes = maxfes
        self.f_star = f_star
        self.fes = 0
        self.gbest = np.inf
        self.trace_fes = []
        self.trace_best = []

    def __call__(self, X):
        X = np.atleast_2d(X)
        f = np.asarray(self.fn(X), dtype=float)
        self.fes += X.shape[0]
        m = float(np.min(f))
        if m < self.gbest:
            self.gbest = m
        self.trace_fes.append(self.fes)
        self.trace_best.append(self.gbest)
        return f

    @property
    def done(self):
        return self.fes >= self.maxfes

    @property
    def error(self):
        return max(0.0, self.gbest - self.f_star)

    def curve(self, grid):
        """Best-so-far error sampled on an arbitrary grid of FES values."""
        tf = np.asarray(self.trace_fes)
        tb = np.maximum(np.asarray(self.trace_best) - self.f_star, 0.0)
        idx = np.clip(np.searchsorted(tf, grid, side='right') - 1, 0, len(tb) - 1)
        return tb[idx]

    def cec_checkpoints(self, D):
        """Error at the 16 standard CEC checkpoints D^(k/5-3) * MaxFES."""
        chk = np.array([D ** (k / 5.0 - 3.0) * self.maxfes for k in range(16)])
        return self.curve(chk)


# ──────────────────────────────────────────────────────────────────────
# Gravitational step, shared by GSOA and GSA
# ──────────────────────────────────────────────────────────────────────
def gravitational_velocity(X, V, f, t, t_span, g_decay, rng):
    """One velocity update of the gravitational search step.

    Masses are normalised linearly from fitness. The force on agent i is
    attractive, directed from x_i toward x_j:

        a_i = sum_j  G(t) * M_j * (x_j - x_i) / (||x_j - x_i|| + eps)
        v_i = r * v_i + a_i,    r ~ U(0, 1)

    with G(t) = 100 * exp(-g_decay * t / t_span).
    """
    N, D = X.shape
    f_best, f_worst = np.min(f), np.max(f)
    m = (f_worst - f) / (f_worst - f_best + EPS)
    M = m / (np.sum(m) + EPS)

    G = 100.0 * np.exp(-g_decay * t / t_span)

    # diff[i, j] = x_j - x_i  (attraction)
    diff = X[None, :, :] - X[:, None, :]
    dist = np.linalg.norm(diff, axis=2) + EPS
    np.fill_diagonal(dist, np.inf)
    w = G * M[None, :] / dist
    np.fill_diagonal(w, 0.0)

    F = np.einsum('ij,ijk->ik', w * M[:, None], diff)
    A = F / (M[:, None] + EPS)
    return rng.uniform(0, 1, (N, D)) * V + A


# ──────────────────────────────────────────────────────────────────────
# GSOA
# ──────────────────────────────────────────────────────────────────────
def gsoa(B, D, N, lb, ub, alpha, seed, d_ref=10):
    """Gravitational Slingshot Optimization Algorithm.

    Each iteration performs a gravitational step to produce a trial
    population. Agents whose trial point fails to improve are relocated by
    the slingshot operator toward the population mean:

        x_i <- x_i + r_i * sign(alpha) * |alpha| * (x_mean - x_i),
        v_i <- 0

    alpha > 0 contracts toward the centroid, alpha < 0 expands away from it,
    and alpha = 0 disables the operator, reducing GSOA to GSA.

    Evaluation cost is one population call for the trial points plus a
    partial call covering only the relocated agents; the fitness of the
    current population is carried across iterations rather than recomputed.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, (N, D))
    V = np.zeros((N, D))
    f = B(X)

    # expected iteration count, used only for the G decay schedule
    per_iter = N * (1.5 if alpha != 0.0 else 1.0)
    t_span = max(1, int(B.maxfes // per_iter))
    g_decay = 20.0 * d_ref / D

    t = 0
    while not B.done:
        t += 1
        V = gravitational_velocity(X, V, f, t, t_span, g_decay, rng)
        Xc = np.clip(X + V, lb, ub)
        fc = B(Xc)
        if B.done:
            break

        if alpha != 0.0:
            x_mean = np.mean(X, axis=0)
            failed = fc >= f
            Xn, fn_ = Xc.copy(), fc.copy()
            if failed.any():
                r = rng.uniform(0, 1, (int(failed.sum()), D))
                Xn[failed] = np.clip(
                    X[failed] + r * np.sign(alpha) * abs(alpha)
                    * (x_mean - X[failed]), lb, ub)
                V[failed] = 0.0                     # velocity reset
                fn_[failed] = B(Xn[failed])         # only movers re-evaluated
            X, f = Xn, fn_
        else:
            X, f = Xc, fc
    return B


def search_alpha(make_budget, D, N, lb, ub, grid, seeds, d_ref=10):
    """Grid search for alpha, scored as the mean over held-out seeds.

    ``make_budget`` must return a fresh Budget for each call.
    """
    scores = {}
    for a in grid:
        scores[a] = float(np.mean([gsoa(make_budget(), D, N, lb, ub, a, s, d_ref).error
                                   for s in seeds]))
    return min(grid, key=lambda a: scores[a]), scores


# ──────────────────────────────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────────────────────────────
def gsa(B, D, N, lb, ub, seed, d_ref=10):
    """Gravitational Search Algorithm (Rashedi et al.)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, (N, D))
    V = np.zeros((N, D))
    f = B(X)
    t_span = max(1, B.maxfes // N)
    g_decay = 20.0 * d_ref / D
    t = 0
    while not B.done:
        t += 1
        V = gravitational_velocity(X, V, f, t, t_span, g_decay, rng)
        X = np.clip(X + V, lb, ub)
        f = B(X)
    return B


def pso(B, D, N, lb, ub, seed, w=0.729, c1=1.494, c2=1.494):
    """Particle Swarm Optimisation with constriction coefficients."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, (N, D))
    V = rng.uniform(-(ub - lb), (ub - lb), (N, D))
    pbest = X.copy()
    f_pbest = B(X)
    gbest = pbest[np.argmin(f_pbest)].copy()
    f_gbest = np.min(f_pbest)
    while not B.done:
        r1, r2 = rng.rand(N, D), rng.rand(N, D)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = np.clip(X + V, lb, ub)
        f = B(X)
        improved = f < f_pbest
        pbest[improved] = X[improved]
        f_pbest[improved] = f[improved]
        b = np.argmin(f_pbest)
        if f_pbest[b] < f_gbest:
            f_gbest = f_pbest[b]
            gbest = pbest[b].copy()
    return B


def de(B, D, N, lb, ub, seed, F=0.8, CR=0.9):
    """Differential Evolution, DE/rand/1/bin."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(lb, ub, (N, D))
    fX = B(X)
    ar = np.arange(N)
    while not B.done:
        R = rng.rand(N, N)
        R[ar, ar] = np.inf                       # never select self
        pick = np.argsort(R, axis=1)[:, :3]
        a, b, c = X[pick[:, 0]], X[pick[:, 1]], X[pick[:, 2]]
        mutant = np.clip(a + F * (b - c), lb, ub)
        cross = rng.rand(N, D) < CR
        cross[ar, rng.randint(0, D, N)] = True   # at least one gene
        trial = np.where(cross, mutant, X)
        ft = B(trial)
        sel = ft <= fX
        X[sel] = trial[sel]
        fX[sel] = ft[sel]
    return B


def ga(B, D, N, lb, ub, seed, pc=0.9, eta_c=20, eta_m=20):
    """Real-coded GA: binary tournament, SBX crossover, polynomial
    mutation, one-elite replacement."""
    rng = np.random.RandomState(seed)
    pm = 1.0 / D
    X = rng.uniform(lb, ub, (N, D))
    fX = B(X)
    half = N // 2
    while not B.done:
        i1, i2 = rng.randint(N, size=(N, 2)).T
        sel = np.where(fX[i1] < fX[i2], i1, i2)
        nX = X[sel].copy()

        p1, p2 = nX[0::2].copy(), nX[1::2].copy()
        do = rng.rand(half) < pc
        u = rng.rand(half, D)
        beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta_c + 1)),
                        (1 / (2 * (1 - u + EPS))) ** (1 / (eta_c + 1)))
        c1 = np.clip(0.5 * ((1 + beta) * p1 + (1 - beta) * p2), lb, ub)
        c2 = np.clip(0.5 * ((1 - beta) * p1 + (1 + beta) * p2), lb, ub)
        nX[0::2] = np.where(do[:, None], c1, p1)
        nX[1::2] = np.where(do[:, None], c2, p2)

        mut = rng.rand(N, D) < pm
        if mut.any():
            u = rng.rand(N, D)
            delta = np.minimum(nX - lb, ub - nX) / (ub - lb + EPS)
            dq = np.where(
                u < 0.5,
                (2 * u + (1 - 2 * u) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1)) - 1,
                1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - delta) ** (eta_m + 1)) ** (1 / (eta_m + 1)))
            nX = np.clip(nX + mut * dq * (ub - lb), lb, ub)

        fnX = B(nX)
        best, worst = np.argmin(fX), np.argmax(fnX)
        if fX[best] < fnX[worst]:                 # elitism
            nX[worst] = X[best]
            fnX[worst] = fX[best]
        X, fX = nX, fnX
    return B


ALGORITHMS = {'GSOA': gsoa, 'GSA': gsa, 'PSO': pso, 'DE': de, 'GA': ga}
