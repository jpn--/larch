#!/bin/bash
conda info
python _scripts/hide_test_cells.py
jb build .
