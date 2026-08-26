"""B1-B3 mechanical design problems, ported exactly from the repository's
C++ evaluators (src/problem_evaluators/*.cpp).

Search is native in [-100,100]^D; each dimension is affinely rescaled to the
problem's true bounds inside the evaluator. Constraints use a static penalty
of 1e6, with the normalisations documented in the C++ sources.
"""
import numpy as np
PENALTY = 1.0e6

def _rescale(Xn, lb, ub):
    t = np.clip((Xn + 100.0) / 200.0, 0.0, 1.0)
    return lb + t * (ub - lb)

# ---------------- B1 pressure vessel ----------------
B1_LB = np.array([0.0, 0.0, 10.0, 10.0]); B1_UB = np.array([99.0, 99.0, 200.0, 200.0])
def b1_parts(Xn):
    x = _rescale(np.atleast_2d(Xn), B1_LB, B1_UB)
    x1, x2, x3, x4 = x[:,0], x[:,1], x[:,2], x[:,3]
    f = 0.6224*x1*x3*x4 + 1.7781*x2*x3**2 + 3.1661*x1**2*x4 + 19.84*x1**2*x3
    g = np.column_stack([
        -x1 + 0.0193*x3,
        -x2 + 0.00954*x3,
        (-np.pi*x3**2*x4 - (4.0/3.0)*np.pi*x3**3 + 1296000.0)/1.0e6,   # normalised
        x4 - 240.0])
    return f, g, x

# ---------------- B2 tension/compression spring ----------------
B2_LB = np.array([0.05, 0.25, 2.0]); B2_UB = np.array([2.0, 1.3, 15.0])
def b2_parts(Xn):
    x = _rescale(np.atleast_2d(Xn), B2_LB, B2_UB)
    d, Dc, N = x[:,0], x[:,1], x[:,2]
    f = (N + 2.0)*Dc*d**2
    g = np.column_stack([
        1.0 - (Dc**3*N)/(71785.0*d**4),
        (4.0*Dc**2 - d*Dc)/(12566.0*(Dc*d**3 - d**4)) + 1.0/(5108.0*d**2) - 1.0,
        1.0 - (140.45*d)/(Dc**2*N),
        (Dc + d)/1.5 - 1.0])
    return f, g, x

# ---------------- B3 welded beam ----------------
B3_LB = np.array([0.1, 0.1, 0.1, 0.1]); B3_UB = np.array([2.0, 10.0, 10.0, 2.0])
_P, _L, _E, _G = 6000.0, 14.0, 30e6, 12e6
TAU_MAX, SIGMA_MAX, DELTA_MAX = 13600.0, 30000.0, 0.25
def b3_parts(Xn):
    x = _rescale(np.atleast_2d(Xn), B3_LB, B3_UB)
    x1, x2, x3, x4 = x[:,0], x[:,1], x[:,2], x[:,3]
    f = 1.10471*x1**2*x2 + 0.04811*x3*x4*(14.0 + x2)
    tp = _P/(np.sqrt(2.0)*x1*x2)
    M = _P*(_L + x2/2.0)
    R = np.sqrt(x2**2/4.0 + ((x1 + x3)/2.0)**2)
    J = 2.0*(np.sqrt(2.0)*x1*x2*(x2**2/12.0 + ((x1 + x3)/2.0)**2))
    tpp = M*R/J
    tau = np.sqrt(tp**2 + 2.0*tp*tpp*(x2/(2.0*R)) + tpp**2)
    sigma = 6.0*_P*_L/(x4*x3**2)
    delta = 4.0*_P*_L**3/(_E*x3**3*x4)
    Pc = (4.013*_E*np.sqrt(x3**2*x4**6/36.0)/_L**2)*(1.0 - (x3/(2.0*_L))*np.sqrt(_E/(4.0*_G)))
    g = np.column_stack([
        (tau - TAU_MAX)/1000.0,       # normalised
        sigma - SIGMA_MAX,
        x1 - x4,
        0.10471*x1**2 + 0.04811*x3*x4*(14.0 + x2) - 5.0,
        0.125 - x1,
        delta - DELTA_MAX,
        (_P - Pc)/1000.0])            # normalised
    return f, g, x

PROBLEMS = {
    'B1': dict(name='B1 Pressure vessel', dim=4, parts=b1_parts, ref=5885.3328,
               lb=B1_LB, ub=B1_UB, vnames=['Ts','Th','R','L']),
    'B2': dict(name='B2 Tension/compression spring', dim=3, parts=b2_parts, ref=0.012665,
               lb=B2_LB, ub=B2_UB, vnames=['d','D','N']),
    'B3': dict(name='B3 Welded beam', dim=4, parts=b3_parts, ref=1.724852,
               lb=B3_LB, ub=B3_UB, vnames=['h','l','t','b']),
}

def penalised(pid, Xn):
    f, g, _ = PROBLEMS[pid]['parts'](Xn)
    with np.errstate(all='ignore'):
        viol = np.sum(np.maximum(0.0, g), axis=1)
    val = f + PENALTY*viol
    return np.nan_to_num(val, nan=1e30, posinf=1e30, neginf=1e30)

def feasible(pid, Xn, tol=0.0):
    _, g, _ = PROBLEMS[pid]['parts'](Xn)
    return np.all(g <= tol, axis=1)
