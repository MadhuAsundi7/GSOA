"""
Wrapper around the official CEC2022 benchmark.

The competition C source is not redistributed here. Run
``scripts/build_cec2022.sh`` once to download it from the official
repository and compile it into a shared library, then use ``load()``.

    from gsoa.cec2022 import CEC2022
    suite = CEC2022(D=10)
    f = suite.evaluate(1, X)         # X is (m, 10)
    suite.f_star(1)                  # 300.0
"""

import ctypes
import os

import numpy as np

F_STAR = [300, 400, 600, 800, 900, 1800,
          2000, 2200, 2300, 2400, 2600, 2700]
MAXFES = {10: 200_000, 20: 1_000_000}
LB, UB = -100.0, 100.0
GROUPS = {'Unimodal': [1], 'Basic': [2, 3, 4, 5],
          'Hybrid': [6, 7, 8], 'Composition': [9, 10, 11, 12]}

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'cec2022_src')


class CEC2022:
    """ctypes binding to ``cec22_test_func``.

    The C implementation reads its shift, rotation and shuffle data from a
    relative ``input_data/`` path, so the working directory is changed to
    the source directory on construction.
    """

    def __init__(self, D, src_dir=None):
        if D not in (10, 20):
            raise ValueError('CEC2022 is defined here for D = 10 or 20')
        self.D = D
        self.src = src_dir or DEFAULT_DIR
        lib_path = os.path.join(self.src, 'libcec22.so')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f'{lib_path} not found. Run scripts/build_cec2022.sh first.')
        os.chdir(self.src)                 # required by the C source
        self._lib = ctypes.CDLL('./libcec22.so')
        self._lib.cec22_eval.argtypes = (
            [ctypes.POINTER(ctypes.c_double)] * 2 + [ctypes.c_int] * 3)

    def evaluate(self, fid, X):
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        m, n = X.shape
        if n != self.D:
            raise ValueError(f'expected {self.D} columns, got {n}')
        f = np.zeros(m)
        self._lib.cec22_eval(
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), n, m, fid)
        return f

    @staticmethod
    def f_star(fid):
        return float(F_STAR[fid - 1])

    @property
    def maxfes(self):
        return MAXFES[self.D]

    def verify(self):
        """Each function must return its published F* at its shift vector."""
        ok = True
        for fid in range(1, 13):
            o = np.atleast_2d(np.loadtxt(
                os.path.join(self.src, 'input_data',
                             f'shift_data_{fid}.txt')))[0][:self.D]
            err = abs(self.evaluate(fid, o[None, :])[0] - self.f_star(fid))
            if err > 1e-6:
                ok = False
                print(f'  F{fid}: mismatch {err:.3e}')
        return ok
