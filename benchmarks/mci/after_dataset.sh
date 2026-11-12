#!/bin/bash
# Waits for the dataset run to finish, then repairs the mcialg=5 rows and rebuilds the workbook copy.
cd "$(dirname "$0")"
until grep -q DATASET_DONE dataset_run.log 2>/dev/null; do sleep 60; done
export PYTHONPATH=/home/joan/PycharmProjects/ESIpy-dev-mciaprox OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -v 6000000
nice python fix_alg5.py 2>&1 | grep --line-buffered -v -i warn >> dataset_run.log
nice python build_xlsx.py >> dataset_run.log 2>&1
echo AFTER_DONE >> dataset_run.log
