#!/usr/bin/env bash
# Download and compile the official CEC2022 benchmark.
# The competition source is not redistributed with this repository.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/cec2022_src"

if [ -f "$DEST/libcec22.so" ]; then
  echo "already built: $DEST/libcec22.so"; exit 0
fi

TMP="$(mktemp -d)"
echo "downloading official CEC2022 suite ..."
git clone --depth 1 https://github.com/P-N-Suganthan/2022-SO-BO.git "$TMP/repo"
unzip -q -o "$TMP/repo/CEC2022.zip" -d "$TMP/x"
mkdir -p "$DEST"
cp -r "$TMP/x/CEC2022/C-Code/." "$DEST/"

cd "$DEST"
# the released source includes a Windows-only header
sed -i 's/#include <WINDOWS.H>//' cec22_test_func.cpp

cat > wrap.cpp <<'WRAP'
#include <cstdio>
#include <cstdlib>
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);
extern "C" {
  void cec22_eval(double *x, double *f, int nx, int mx, int func_num) {
    cec22_test_func(x, f, nx, mx, func_num);
  }
  void cec22_reset() { ini_flag = 0; }
}
WRAP

g++ -O2 -fPIC -shared -o libcec22.so cec22_test_func.cpp wrap.cpp
rm -rf "$TMP"
echo "built: $DEST/libcec22.so"
