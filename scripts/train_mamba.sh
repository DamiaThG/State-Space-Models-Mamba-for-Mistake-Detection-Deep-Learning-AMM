#!/bin/bash
# ============================================================
# SLURM Job Script — Mamba SSM Training
# Cluster: GCluster (gcluster.dmi.unict.it)
# Sottometti con: sbatch scripts/train_mamba.sh
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

# Attiva l'ambiente conda/venv del progetto (modifica il nome se necessario)
# source activate mistake-detection
# oppure:
# source .venv/bin/activate

# ---------- Logging dir ----------
mkdir -p experiments/logs
mkdir -p experiments/checkpoints

# ---------- Training ----------
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# Riduce la frammentazione della VRAM (consigliato da PyTorch per OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Utilizzo Apptainer per isolare l'ambiente (sostituisci mamba_env.sif col nome corretto)
APPTAINER_IMAGE="mamba_env.sif"

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    $APPTAINER_IMAGE \
    python src/training/train_mamba_whole_video.py \
    --processed_dir   data/processed \
    --batch_size      4 \
    --num_workers     4 \
    --accumulate_grad_batches 2 \
    --d_model         512 \
    --n_layers        6 \
    --dropout         0.2 \
    --max_seq_len     25000 \
    --use_checkpointing \
    --epochs          50 \
    --lr              5e-5 \
    --weight_decay    1e-3 \
    --seed            42 \
    --wandb_project   mistake-detection \
    --wandb_run_name  "mamba-ssm-wholevid-$SLURM_JOB_ID" \
    --ckpt_dir        experiments/checkpoints



echo "================================================"
echo "End: $(date)"
echo "================================================"
