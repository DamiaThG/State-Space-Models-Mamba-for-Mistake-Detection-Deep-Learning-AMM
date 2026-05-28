#!/bin/bash
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-medium
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:5000
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --output=/home/mssdmn01t05c351v/assembly-mistake-detection/logs/extract_tsm-%j.log

echo "Inizio estrazione C10119_rgb.zip..."
apptainer run --nv /shared/sifs/latest.sif python -c "
import zipfile, os
zip_path = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/TSM_features/C10119_rgb.zip'
out_dir = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/TSM_features'
print('Apertura zip...')
with zipfile.ZipFile(zip_path, 'r') as z:
    members = z.namelist()
    print(f'File totali nello zip: {len(members)}')
    print(f'Primi 5: {members[:5]}')
    z.extractall(out_dir)
print('Estrazione completata.')
"
echo "Fine."
