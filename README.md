# GSOA — Gravitational Slingshot Optimization Algorithm

| Study | *D* | Population | MaxFES | Runs |
|---|---|---|---|---|
| CEC2022 | 10 | 30 | 200,000 | 30 (seeds 0–29) |
| CEC2022 | 20 | 30 | 1,000,000 | 30 (seeds 0–29) |
| 29-function suite | 30 | 30 | 15,000 | 30 (seeds 0–29) |
| Mechanical design B1–B3 | 3–4 | 30 | 200,000 | 30 (seeds 0–29) |

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

## License

 competition implementations under `src/competitors` retain their
original authors' terms. 
