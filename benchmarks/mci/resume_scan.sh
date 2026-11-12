#!/bin/bash
# Continues the annulene mcialg=5 scan (rings already in scan_alg5.jsonl are skipped).
cd "$(dirname "$0")"
export PYTHONPATH=/home/joan/PycharmProjects/ESIpy-dev-mciaprox OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NCORES=${NCORES:-6}
ulimit -v 6000000
nice python scan_alg5.py 6 40 30 2>&1 | grep --line-buffered -v -i warn >> scan_alg5.log
echo SCAN_DONE >> scan_alg5.log
