#!/bin/bash
# Resumes the dataset run (one folder at a time, finished rings skipped), then builds the workbook copy.
cd "$(dirname "$0")"
export PYTHONPATH=/home/joan/PycharmProjects/ESIpy-dev-mciaprox OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -v 6000000
for f in DATASET_S DATASET_T ANNULENES FULLERENES SURFACES; do
  nice python run_dataset.py 2 $f 2>&1 | grep --line-buffered -v -i warn >> dataset_run.log
done
nice python build_xlsx.py >> dataset_run.log 2>&1
echo DATASET_DONE >> dataset_run.log
