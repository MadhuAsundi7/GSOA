#!/usr/bin/env python3
"""GSOA entry point for the mechanical design problems (runs B1-B3).

Equivalent to scripts/run_mechanical.py.
"""
import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.argv[0] = os.path.join(ROOT, 'scripts', 'run_mechanical.py')
runpy.run_path(sys.argv[0], run_name='__main__')
