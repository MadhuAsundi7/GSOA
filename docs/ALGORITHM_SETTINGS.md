# Algorithm settings

Every algorithm compared in this work, with its parameters and the source
file that implements it.

All CEC2022 experiments use 30 independent runs and a shared evaluation
budget: MaxFES = 200,000 at *D* = 10 and 1,000,000 at *D* = 20. Runs
terminate on the budget or when the error falls below 10⁻⁸, whichever comes
first.

---

## Proposed algorithm

### GSOA — Gravitational Slingshot Optimization Algorithm

`src/gsoa/algorithm.py` (reference implementation)

| Parameter | Value |
|---|---|
| Population size *N* | 30 |
| Slingshot coefficient α | selected per function from {−2.0, −1.5, −1.0, −0.5, 0, 0.5, 1.0, 1.5, 2.0} |
| α selection | mean of held-out runs, seeds 1000+ (never used for evaluation) |
| Gravitational constant | G(t) = 100·exp(−g_decay·t/t_span) |
| g_decay | 20·D_ref/D, with D_ref = 10 → 20.0 at *D*=10, 10.0 at *D*=20, 6.667 at *D*=30 |
| Force direction | attractive, (x_j − x_i) |
| Slingshot target | population mean |
| Velocity reset | yes, for relocated agents |
| ε | 10⁻¹² |
| Evaluation seeds | 0–29 |

Selected α per function is in `results/cec2022/gsoa_alpha_star.csv`.

---

## Classical baselines

`src/baselines/cec2022/gsa_pso_de_ga_cec2022.cpp` (C++)
`src/gsoa/algorithm.py` (Python, used for the reported GSA results)

| Algorithm | Parameters |
|---|---|
| **GSA** | *N* = 30; same G(t) schedule and g_decay as GSOA; attractive force; no slingshot |
| **PSO** | *N* = 30; inertia *w* = 0.729; c₁ = c₂ = 1.494; velocity initialised U(−(ub−lb), ub−lb) |
| **DE** | *N* = 30; DE/rand/1/bin; *F* = 0.8; *CR* = 0.9; at least one gene inherited from the mutant |
| **GA** | *N* = 30; binary tournament selection; SBX crossover p_c = 0.9, η_c = 20; polynomial mutation p_m = 1/*D*, η_m = 20; one-elite replacement |

---

## CEC2022 competition entries

These are the authors' own implementations, used unmodified. Parameters are
those of the competition submissions.

| Algorithm | Source | Key settings |
|---|---|---|
| **EA4eig** | `src/competitors/EA4eig/ea4eig.py` | 4 adaptive sub-strategies selected by roulette; memory size *H* = 5; CMA-ES component; linear population-size reduction |
| **S-LSHADE-DP** | `src/competitors/S-LSHADE-DP/` | success-history adaptation; p-best mutation; external archive; linear population-size reduction; diversity-preservation component |
| **NL-SHADE-LBC** | `src/competitors/NL-SHADE-LBC/nl_shade_lbc.cpp` | non-linear population-size reduction; linear bias change in parameter adaptation; Cauchy/normal parameter sampling; external archive |
| **NL-SHADE-RSP-MID** | `src/competitors/NL-SHADE-RSP-MID/nl_shade_rsp_mid.cpp` | rank-based selective pressure; adaptive archive; population size 50 at *D* = 10 and 100 at *D* = 20 (reflected in the result filenames) |

Because these entries adapt their own control parameters at runtime, they
have no fixed per-function settings beyond the initial population size.

---

## Benchmark suites

| Suite | Dimension | Range | Budget | Source |
|---|---|---|---|---|
| CEC2022 | 10, 20 | [−100, 100] | 200,000 / 1,000,000 | official C source, fetched by `scripts/build_cec2022.sh` |
| 29-function custom suite | 30 | [−100, 100] | 15,000 | `src/gsoa/benchmark29.py` |
| Mechanical design B1–B3 | 3–4 | search in [−100, 100], rescaled to true bounds | 200,000 | `src/problem_evaluators/`, `src/gsoa/mechanical.py` |

The 29-function suite is CEC-*style* but is **not** CEC2017: its shift
vectors and rotation matrices are generated pseudo-randomly under a fixed
seed rather than taken from competition data files, and no domain shrink rate
is applied. Its results are not comparable to published CEC2017 figures.


---

## Mechanical design problems

| Item | Value |
|---|---|
| Constraint handling | static penalty, `f + 1e6 · Σ max(0, g_j)` |
| Normalised constraints | B1 `g3` / 1e6; B3 `g1`, `g7` / 1e3 |
| Search space | `[-100, 100]^D`, affinely rescaled to true bounds in the evaluator |
| Budget | MaxFES = 200,000 |
| Runs | 30 (seeds 0–29) |
| GSOA α* | +0.5 on B1, B2 and B3 |
