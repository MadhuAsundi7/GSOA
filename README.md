# GSOA — Gravitational Slingshot Optimization Algorithm

Code and data accompanying *"GSOA: Gravitational Slingshot Optimization
Algorithm"*.

This repository contains the algorithm, every baseline and competitor
implementation used for comparison, all raw result files, and scripts that
regenerate every table and figure in the paper from those files.

---

## The algorithm

GSOA extends the Gravitational Search Algorithm (GSA) with a **slingshot
operator**. Each iteration performs a gravitational step to produce a trial
population. Agents whose trial point fails to improve are relocated toward the
population mean, and their velocity is cleared:

```
x_i  ←  x_i + r_i · sign(α)·|α| · (x̄ − x_i),     r_i ~ U(0,1)
v_i  ←  0
```

where `x̄` is the population centroid. The coefficient α controls the
direction and strength:

* `α > 0` — contraction toward the centroid
* `α < 0` — expansion away from it
* `α = 0` — operator disabled; GSOA reduces exactly to GSA

The underlying gravitational force is attractive, directed from `x_i` toward
`x_j`, as in Rashedi's formulation:

```
a_i = Σ_j  G(t) · M_j · (x_j − x_i) / (‖x_j − x_i‖ + ε)
G(t) = 100 · exp(−g_decay · t / t_span),    g_decay = 20 · D_ref / D,  D_ref = 10
```

Implementation: `src/gsoa/algorithm.py`.

### Two details that materially affect results

**Velocity reset.** Clearing `v_i` for relocated agents is not cosmetic.
Without it, stale momentum drags the agent back along the trajectory that just
failed. On the 29-function suite the reset is significantly better on 18 of 29
functions and worse on 2 (Wilcoxon, *p* = 7.7 × 10⁻⁴), with per-function
differences of up to five orders of magnitude.

**Cached evaluation.** The fitness of the current population is carried across
iterations, and after the slingshot only the relocated agents are
re-evaluated. This changes no decision the algorithm makes; it avoids spending
budget on values already known.

---

## Evaluation protocol

Every algorithm terminates on a **shared evaluation budget**, not a fixed
iteration count. This matters because the algorithms differ in evaluations per
iteration — comparing at equal iterations would silently grant some of them
several times the search effort. The `Budget` class in `algorithm.py` enforces
it: every optimiser loops on `while not B.done`.

| Study | *D* | Population | MaxFES | Runs |
|---|---|---|---|---|
| CEC2022 | 10 | 30 | 200,000 | 30 (seeds 0–29) |
| CEC2022 | 20 | 30 | 1,000,000 | 30 (seeds 0–29) |
| 29-function suite | 30 | 30 | 15,000 | 30 (seeds 0–29) |
| Mechanical design B1–B3 | 3–4 | 30 | 200,000 | 30 (seeds 0–29) |

α is selected per problem by grid search over −2.0 … +2.0 in steps of 0.5,
scored as the mean over **held-out** seeds (1000 onward) that are never used
for evaluation. Selected values are in `results/*/gsoa_alpha_star.csv`.

---

## Quick start

```bash
pip install -r requirements.txt

# regenerate all statistics, tables and figures from the stored results
python scripts/analyze.py --dim 10 --results results/cec2022/d10 --out out/d10
python scripts/analyze.py --dim 20 --results results/cec2022/d20 --out out/d20

# re-run from scratch (optional)
bash scripts/build_cec2022.sh                 # fetch + compile the official suite
python scripts/run_cec2022.py --dim 10 --runs 30 --out out/run_d10
python scripts/run_29functions.py --runs 30 --out out/run_f29
python scripts/run_mechanical.py
```

`bash reproduce.sh analyze` regenerates the analysis for every study; `bash reproduce.sh rerun` re-executes GSOA from scratch.

---

## Layout

```
src/
  gsoa/
    core/                     the algorithm and benchmarks (Python)
      algorithm.py              GSOA, GSA, PSO, DE, GA, Budget meter
      benchmark29.py            29-function suite
      cec2022.py                binding to the official CEC2022 suite
      mechanical.py             B1–B3 constrained design problems
      results_io.py             reader for the CEC result-file format
    cec2022/run.py            entry point for the CEC2022 study
    mechanical/B*/run.py      entry points for the mechanical problems
  baselines/cec2022/        GSA, PSO, DE, GA (C++)
  competitors/              EA4eig, S-LSHADE-DP, NL-SHADE-LBC, NL-SHADE-RSP-MID
                            (third-party — see src/competitors/README.md)
  cec2022_evaluator/        official CEC2022 evaluator
  problem_evaluators/       B1–B3 evaluators (C++)

scripts/
  build_cec2022.sh          download and compile the official CEC2022 source
  run_cec2022.py            run GSOA (and optionally baselines) on CEC2022
  run_29functions.py        run the 29-function suite
  run_mechanical.py         run GSOA on B1–B3
  analyze.py                regenerate every statistic, table and figure
  (see build_all.sh and reproduce.sh at the repository root)

results/
  cec2022/d10, d20          17×30 matrices, one file per algorithm and function
  f29/                      30 runs per algorithm and function, plus curves
  mechanical/B1, B2, B3     16×30 matrices

analysis/
  plot_results.py           figure helper carried over from the original repository

docs/
  ALGORITHM_SETTINGS.md     hyperparameters for every algorithm
  BENCHMARKS.md             benchmark definitions and caveats

build_all.sh                compile the C++ components
reproduce.sh                regenerate the analysis, or rerun GSOA
```

---

## Result file formats

**CEC2022** — `{ALG}_{function}_{D}.txt`, a 17 × 30 matrix:
rows 0–15 are the best-so-far error at the checkpoints
`FES_k = D^(k/5−3)·MaxFES`; row 16 is the evaluations consumed by each run;
columns are the 30 runs. Row 15 is the final error.
`NL-SHADE-RSP-MID` encodes its population size in the filename
(`_pop_50` at *D* = 10, `_pop_100` at *D* = 20).

**29-function suite** — `{ALG}_{function}_30.txt` holds the 30 final values;
`{ALG}_{function}_30_curve.txt` holds the mean convergence curve on the FES
grid in `_fes_grid.txt`. Values are raw objective values including the bias
`100·i`, not errors (see `docs/BENCHMARKS.md`).

**Mechanical** — `{ALG}_{problem}.txt`, a 16 × 30 matrix of best-so-far
values at the same checkpoints. The last row is the final value of each run.

`scripts/analyze.py` validates every CEC2022 file on load: shape, monotone
non-increasing checkpoints, non-negative errors and budget compliance.

---

## Provenance

| Algorithms | Source |
|---|---|
| GSOA | this work — `src/gsoa/algorithm.py` |
| GSA, PSO, DE, GA | this work — `src/gsoa/algorithm.py` (Python) and `src/baselines/` (C++) |
| EA4eig, S-LSHADE-DP, NL-SHADE-LBC, NL-SHADE-RSP-MID | the authors' own competition implementations, used unmodified, run under the same protocol |

GSOA and the classical baselines are implemented in Python; the four
competition entries are the original C++ and Python submissions and are
included for reproducibility. They are not our implementations and retain
their authors' terms.

---

## Reproducibility

Evaluation runs use seeds 0–29. α is tuned on held-out seeds from 1000 onward,
never used for evaluation. The 29-function landscape is generated under seed
2017 (shift vectors, rotation matrices) and seed 2018 (shuffle vectors, block
rotations).

Given the same seeds and budget, every number in the paper follows from these
scripts and result files.

---

## License

MIT for `src/gsoa`, `src/baselines`, `src/problem_evaluators` and `scripts`.
Third-party competition implementations under `src/competitors` retain their
original authors' terms. The CEC2022 benchmark source is not redistributed;
`scripts/build_cec2022.sh` fetches it from the official repository.
