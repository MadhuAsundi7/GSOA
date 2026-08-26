#!/usr/bin/env python3
"""GSOA entry point for the CEC2022 suite.

Equivalent to ``scripts/run_cec2022.py``; kept here so each study has an
entry point alongside its counterparts under ``src/``.

    python src/gsoa/cec2022/run.py --dim 10 --runs 30
"""
import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.argv[0] = os.path.join(ROOT, 'scripts', 'run_cec2022.py')
runpy.run_path(sys.argv[0], run_name='__main__')
