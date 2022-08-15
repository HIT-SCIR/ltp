#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH -t 7-00:00:00

export TOKENIZERS_PARALLELISM=false
PYTHONPATH=. python ltp_core/eval.py "$@"
