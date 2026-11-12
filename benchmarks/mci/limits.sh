#!/bin/bash
# One heavy job at a time: finish alg5 n=30, then exact n=28, alg5 n=32, exact n=30. Each step is resumable.
cd "$(dirname "$0")"
export PYTHONPATH=/home/joan/PycharmProjects/ESIpy-dev-mciaprox OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
ulimit -v 6000000
until grep -q '"n": 30' scan_alg5.jsonl 2>/dev/null; do sleep 60; done
pkill -f "scan_alg5.py 6 40" ; sleep 5
NCORES=8 nice python exact_extend.py 28 2>&1 | grep --line-buffered -v -i warn >> scan_alg5.log
NCORES=8 nice python scan_alg5.py 32 32 100 2>&1 | grep --line-buffered -v -i warn >> scan_alg5.log
NCORES=8 nice python exact_extend.py 30 2>&1 | grep --line-buffered -v -i warn >> scan_alg5.log
echo LIMITS_DONE >> scan_alg5.log
