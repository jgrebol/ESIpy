#!/bin/bash
procs=1
slurm_mem="30G"
gauss_mem="3GB" # Slightly less than SLURM mem to prevent OOM errors
queue="regular" # regular(1d), long(2d), xlong(8d)
mail="joan.grebol@dipc.org"

# --- Control Flags ---
esipy=true   # Set to true to run ESIpy
aimall=true  # Set to true to run AIMAll
# ---------------------

for f in *.gjf; do
    # Skip _run.com files if they exist to avoid duplication
    [[ "$f" == *_run.gjf ]] && continue
    name="${f%.gjf}"

    cat << EOF > ${name}.job
#!/bin/bash
#SBATCH --job-name=$name
#SBATCH --partition=general
#SBATCH --qos=$queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=$procs
#SBATCH --mem=$slurm_mem
#SBATCH --output=${name}_%j.out
#SBATCH --error=${name}_%j.err
#SBATCH --mail-user=$mail
#SBATCH --mail-type=FAIL

echo "==================================="
echo "Job started at \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Running on cluster: \$SLURM_CLUSTER_NAME"
echo "Running on host: \$SLURM_NODELIST"
echo "Job name: \$SLURM_JOB_NAME"
echo "Working directory: \$SLURM_SUBMIT_DIR"
echo "Partition: \$SLURM_JOB_PARTITION"
echo "Nodes: \$SLURM_JOB_NUM_NODES"
echo "Tasks: \$SLURM_NTASKS"
echo "CPUs per task: \$SLURM_CPUS_PER_TASK"
echo "==================================="

# Load Modules and Environment
module load Gaussian/16

# Set scratch
export SCRATCH_DIR=\$SLURM_SUBMIT_DIR/g16_\$SLURM_JOB_ID
mkdir -p \$SCRATCH_DIR
export GAUSS_SCRDIR=\$SCRATCH_DIR

# Prepare Gaussian input
(
  echo "%nproc=\$SLURM_CPUS_PER_TASK"
  echo "%mem=$gauss_mem"
  echo "%chk=${name}.chk"
  cat ${name}.gjf
) > ${name}_run.gjf

# Run Gaussian
g16 < ${name}_run.gjf > ${name}.log

# Convert checkpoint file if available
if [ -f "${name}.chk" ]; then
    formchk ${name}.chk ${name}.fchk
fi

# Clean scratch
rm -rf \$SCRATCH_DIR
rm ${name}_run.gjf

echo "==================================="
echo "Job finished at \$(date)"
echo "==================================="
EOF

    # Sleep to prevent overwhelming the SLURM scheduler
    sleep 0.5
    sbatch ${name}.job
done


