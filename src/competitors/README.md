# Competitor implementations

The four CEC2022 competition entries in this directory are **third-party
code**, included unmodified so the comparison in this work can be reproduced.
They are not our implementations, and they remain the property of their
original authors under their original terms.

| Directory | Algorithm | Original authors |
|---|---|---|
| `EA4eig/` | EA4eig | P. Bujok and P. Kolenovsky |
| `S-LSHADE-DP/` | S-LSHADE-DP | submitted to the CEC2022 competition |
| `NL-SHADE-LBC/` | NL-SHADE-LBC | V. Stanovov, S. Akhmedova and E. Semenkin |
| `NL-SHADE-RSP-MID/` | NL-SHADE-RSP-MID | submitted to the CEC2022 competition |

All four were submitted to the IEEE CEC2022 Competition on Single Objective
Bound Constrained Numerical Optimization. Please cite the corresponding
competition papers when using their results, and consult the original
repositories for licensing terms:
<https://github.com/P-N-Suganthan/2022-SO-BO>.

Hyperparameters as used here are documented in
`docs/ALGORITHM_SETTINGS.md`.

## Note on `EA4eig/cec22_test_func.cpp`

This file is the official CEC2022 benchmark source, duplicated from the
competition release. Elsewhere this repository does not redistribute that
source: `scripts/build_cec2022.sh` fetches it from the official repository
instead. The copy is retained here only because the EA4eig wrapper expects it
in this location.
