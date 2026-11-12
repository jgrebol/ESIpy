#!/bin/bash
# Waits for the test suite, then the rotation accuracy test on 2 cores.
cd "$(dirname "$0")"
until grep -q TESTS_DONE tests_branch.log 2>/dev/null; do sleep 30; done
export PYTHONPATH=/home/joan/PycharmProjects/ESIpy-dev-mciaprox OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NCORES=2
ulimit -v 5000000
nice python accuracy.py 12 16 20 22 24 2>&1 | grep --line-buffered -v -i warn > accuracy.log
echo ACCURACY_DONE >> accuracy.log
