#!/usr/bin/env bash
# reproduce.sh — regenerate the reported analysis, and optionally rerun GSOA.
#
#   bash reproduce.sh analyze   regenerate tables and figures from stored results (fast)
#   bash reproduce.sh rerun     rerun GSOA on all three studies (hours)
set -e
MODE=${1:-analyze}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p out

if [[ "$MODE" == "analyze" || "$MODE" == "all" ]]; then
    echo "=== CEC2022 D=10 ==="
    python3 scripts/analyze.py --dim 10 --results results/cec2022/d10 --out out/cec2022_d10
    echo "=== CEC2022 D=20 ==="
    python3 scripts/analyze.py --dim 20 --results results/cec2022/d20 --out out/cec2022_d20
fi

if [[ "$MODE" == "rerun" || "$MODE" == "all" ]]; then
    echo "This overwrites result files. Ctrl+C within 5 seconds to cancel."
    sleep 5
    bash scripts/build_cec2022.sh
    python3 scripts/run_cec2022.py --dim 10 --runs 30 --out out/run_d10
    python3 scripts/run_cec2022.py --dim 20 --runs 30 --out out/run_d20
    python3 scripts/run_29functions.py --runs 30 --out out/run_f29
    python3 scripts/run_mechanical.py
fi

echo "done -> out/"
