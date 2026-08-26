#!/usr/bin/env bash
# build_all.sh — compile the C++ components.
#
# GSOA itself is Python (src/gsoa) and needs no compilation; this script
# builds the CEC2022 evaluator, the C++ baselines and the competitor entries.
#
# Usage: bash build_all.sh [eval|baselines|competitors|all]   (default: all)
set -e
TARGET=${1:-all}
CEC_EVAL="src/cec2022_evaluator/cec22_test_func.cpp"
CXXFLAGS="-O2 -std=c++11"

echo "=== Build ==="

if [[ "$TARGET" == "eval" || "$TARGET" == "all" ]]; then
    echo "[1/3] CEC2022 shared library (for the Python runners)"
    bash scripts/build_cec2022.sh
fi

if [[ "$TARGET" == "baselines" || "$TARGET" == "all" ]]; then
    echo "[2/3] C++ baselines (GSA, PSO, DE, GA)"
    mkdir -p build
    g++ $CXXFLAGS -o build/baselines_cec2022 \
        "$CEC_EVAL" src/baselines/cec2022/gsa_pso_de_ga_cec2022.cpp
fi

if [[ "$TARGET" == "competitors" || "$TARGET" == "all" ]]; then
    echo "[3/3] competitor entries (third-party, see src/competitors/README.md)"
    mkdir -p build
    g++ $CXXFLAGS -o build/nl_shade_lbc \
        "$CEC_EVAL" src/competitors/NL-SHADE-LBC/nl_shade_lbc.cpp || \
        echo "    NL-SHADE-LBC: skipped"
    g++ $CXXFLAGS -o build/nl_shade_rsp_mid \
        "$CEC_EVAL" src/competitors/NL-SHADE-RSP-MID/nl_shade_rsp_mid.cpp || \
        echo "    NL-SHADE-RSP-MID: skipped"
    g++ $CXXFLAGS -o build/s_lshade_dp \
        "$CEC_EVAL" src/competitors/S-LSHADE-DP/s_lshade_dp.cpp \
        src/competitors/S-LSHADE-DP/main.cpp \
        src/competitors/S-LSHADE-DP/search_algorithm.cpp || \
        echo "    S-LSHADE-DP: skipped"
fi

echo "done."
