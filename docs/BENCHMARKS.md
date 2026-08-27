# Benchmarks

## 1. CEC2022 (D = 10 and 20)

The official IEEE CEC2022 single-objective bound-constrained suite: twelve
functions on [−100, 100]^D — one unimodal (F1), four basic (F2–F5), three
hybrid (F6–F8) and four composition (F9–F12).

| Function | Name | F* |
|---|---|---|
| F1 | Shifted and full Rotated Zakharov | 300 |
| F2 | Shifted and full Rotated Rosenbrock | 400 |
| F3 | Shifted and full Rotated Expanded Schaffer F6 | 600 |
| F4 | Shifted and full Rotated Non-Continuous Rastrigin | 800 |
| F5 | Shifted and full Rotated Levy | 900 |
| F6 | Hybrid Function 1 (N = 3) | 1800 |
| F7 | Hybrid Function 2 (N = 6) | 2000 |
| F8 | Hybrid Function 3 (N = 5) | 2200 |
| F9 | Composition Function 1 (N = 5) | 2300 |
| F10 | Composition Function 2 (N = 4) | 2400 |
| F11 | Composition Function 3 (N = 5) | 2600 |
| F12 | Composition Function 4 (N = 6) | 2700 |

The reference C source is fetched and compiled by `scripts/build_cec2022.sh`;
it is not redistributed here. Every run first verifies the suite by evaluating
each function at its own shift vector and checking that the published F* is
returned exactly. Reported values are errors, f(x) − F*.

Results on this suite **are** directly comparable to published competition
figures.

---

## 2. The 29-function suite (D = 30)

A shifted-and-rotated benchmark of 29 functions: unimodal F1–F3, multimodal
F4–F10, hybrid F11–F20, composition F21–F29. Range [−100, 100]^30, with an
additive bias of 100·i on function i.

Implemented in `src/gsoa/benchmark29.py`.

### This suite is CEC-style but is not CEC2017

Its shift vectors are drawn from U(−80, 80) and its rotation matrices from the
QR factorisation of Gaussian matrices under a fixed seed, rather than taken
from the CEC2017 competition data files. No domain shrink rate is applied, and
several kernels differ from their CEC2017 counterparts. **Results on it are
not comparable to published CEC2017 figures** and it is a
custom benchmark.

### Construction

* **Landscape** — `o_kj ~ U(−80, 80)`; rotations are the orthogonal factor of
  the QR decomposition of a standard normal matrix. Seed 2017 for shifts and
  rotations, seed 2018 for shuffle vectors and hybrid block rotations.
* **Transformations** — F1–F3 are shifted only; F4–F20 are shifted and
  rotated, `z = Rᵀ(x − o)`.
* **Hybrids (F11–F20)** — the shifted vector is permuted by a fixed shuffle
  vector, split into blocks of size `⌈p_k·D⌉` (the last block taking the
  remainder), and each block is rotated by its own orthogonal matrix and
  passed to its own kernel. Block values are summed.
* **Compositions (F21–F29)** — three components with σ = (10, 20, 30) and
  λ = (1, 10, 1), Gaussian weights, and a probe-based normalisation constant.



---

## 3. Mechanical design problems B1–B3

Three classical constrained problems.

| ID | Problem | Variables | Constraints | Best known |
|---|---|---|---|---|
| B1 | Pressure vessel | 4 | 4 | 5885.3328 |
| B2 | Tension/compression spring | 3 | 4 | 0.012665 |
| B3 | Welded beam | 4 | 7 | 1.724852 |

### Constraint handling

Static penalty:

```
F(x) = f(x) + 1e6 · Σ_j max(0, g_j(x))
```

Three constraints are normalised before penalising because their raw
magnitudes differ from the rest by orders of magnitude: B1's `g3` by 1e6, and
B3's `g1` and `g7` by 1e3. Without this a single constraint would dominate the
penalty term.

### Search space

Every algorithm searches natively in [−100, 100]^D, and each dimension is
affinely rescaled to the problem's true bounds inside the evaluator. This
keeps each algorithm's search, mutation and selection code identical to its
CEC2022 form, so only the fitness landscape differs. Box constraints hold by
construction.

The feasible region is reachable by sampling: of one million uniform random
points, 75.9 % are feasible on B1, 0.76 % on B2 and 2.7 % on B3.

`src/gsoa/mechanical.py` is a Python port of the C++ evaluators in
`src/problem_evaluators/` and reproduces all three published optima exactly.

### Statistical caveat

With only three problems the Friedman test has three blocks and very little
power, and the Nemenyi critical difference (6.94 for nine algorithms) exceeds
the full range of observed ranks, so the post-hoc separates no pair. The raw
best/mean/std values are the primary evidence; the ranking is context.
