#!/bin/bash
# Full test suite on this branch, then on the untouched dev-mciaprox checkout as a baseline (same tests).
ulimit -v 6000000
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
W=/home/joan/PycharmProjects/ESIpy-dev-mciaprox
cd $W && PYTHONPATH=$W nice python -m pytest $(ls tests/test*.py) -q -p no:cacheprovider > $W/benchmarks/mci/tests_branch.log 2>&1
echo TESTS_DONE >> $W/benchmarks/mci/tests_branch.log
