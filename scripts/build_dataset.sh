#!/bin/bash
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-large
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:5000
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/home/mssdmn01t05c351v/assembly-mistake-detection/logs/build_dataset-%j.log

echo "Inizio build dataset..."
apptainer run --nv /shared/sifs/latest.sif python \
    /home/mssdmn01t05c351v/assembly-mistake-detection/scripts/build_dataset.py
echo "Fine."
