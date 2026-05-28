#!/bin/bash
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-medium
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:5000
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/home/mssdmn01t05c351v/assembly-mistake-detection/logs/get_read_lmdb-%j.log

apptainer run --nv /shared/sifs/latest.sif python \
    /home/mssdmn01t05c351v/assembly-mistake-detection/scripts/get_read_lmdb.py
