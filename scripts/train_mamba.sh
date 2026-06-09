#!/bin/bash
# ============================================================
# SLURM Job Script — Mamba SSM Training
# Cluster: GCluster (gcluster.dmi.unict.it)
# Sottometti con: sbatch scripts/train_mamba.sh
#
# NOTA: Questo script è il wrapper SLURM+Apptainer.
# La logica di training vera è in scripts/run_train_mamba.sh,
# che può essere eseguito direttamente dentro mamba-docker.
# ============================================================

#SBATCH --job-name=mamba_ssm
#SBATCH --output=experiments/logs/mamba_%j.out
#SBATCH --error=experiments/logs/mamba_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# ---------- Ambiente ----------
echo "================================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "Working   : $(pwd)"
echo "================================================"

# ---------- Logging dir ----------
mkdir -p experiments/logs
mkdir -p experiments/checkpoints

# ---------- Lancio nel container ----------
APPTAINER_IMAGE="mamba_env.sif"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    $APPTAINER_IMAGE \
    bash /workspace/scripts/run_train_mamba.sh \
    --wandb_run_name "mamba-ssm-wholevid-$SLURM_JOB_ID"

echo "================================================"
echo "End: $(date)"
echo "================================================"
