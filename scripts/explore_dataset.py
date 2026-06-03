# pyrefly: ignore [missing-import]
from huggingface_hub import HfFileSystem
import os

# Usa la variabile d'ambiente HF_TOKEN invece di un token hardcoded
fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))

print('=== Tutte le voci in TSM_features ===')
tsm = fs.ls('datasets/cvml-nus/assembly101/TSM_features', detail=True)
total_gb = 0
for f in tsm:
    size = f.get('size', 0)
    size_gb = size / 1e9 if size else 0
    total_gb += size_gb
    name = f['name'].split('/')[-1]
    print(f'{name:50s}  {size_gb:.2f} GB')
print(f'\nTotale TSM_features: {total_gb:.2f} GB')

print()
print('=== Cerca cartella annotations ===')
try:
    ann = fs.ls('datasets/cvml-nus/assembly101/annotations', detail=True)
    for f in ann[:20]:
        size = f.get('size', 0)
        size_mb = size / 1e6 if size else 0
        name = f['name'].split('/')[-1]
        print(f'{name:50s}  {size_mb:.1f} MB')
except Exception as e:
    print(f'Nessuna cartella annotations: {e}')

print()
print('=== Cerca file CSV/JSON nella root ===')
root = fs.ls('datasets/cvml-nus/assembly101', detail=True)
for f in root:
    name = f['name'].split('/')[-1]
    size = f.get('size', 0)
    size_mb = size / 1e6 if size else 0
    if any(name.endswith(ext) for ext in ['.csv', '.json', '.zip', '.txt']):
        print(f'{name:50s}  {size_mb:.1f} MB')
