"""
build_dataset.py

Per ogni sequenza video in Assembly101:
1. Apre l'LMDB e carica tutti i frame esistenti con le loro feature (T, 2048)
2. Legge il CSV delle annotazioni e assegna una label a ogni frame:
   - 1 se il frame ricade nel range [start, end] di una coarse action 'mistake'
   - 2 se ricade in una 'correction'
   - 0 altrimenti ('correct' o nessuna coarse action)
3. Salva un file .pt per sequenza con features e labels

Output: data/processed/<sequence_name>.pt
"""

import os
import lmdb
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# ── Percorsi ──────────────────────────────────────────────────────────────────
LMDB_PATH      = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/TSM_features/C10119_rgb'
ANNOTATIONS_DIR = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/annotations/assembly101-mistake-detection/annots'
OUTPUT_DIR     = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/processed'
VIEW_NAME      = 'C10119_rgb'

# ── Mapping label ──────────────────────────────────────────────────────────────
LABEL_MAP = {'correct': 0, 'mistake': 1, 'correction': 2}


def load_sequence_from_lmdb(env, sequence_name):
    """
    Carica tutti i frame di una sequenza dall'LMDB.
    Restituisce:
        frames:      dict {frame_no (int) -> feature (np.array 2048)}
        frame_nos:   lista ordinata di frame number esistenti
    """
    prefix = f'{sequence_name}/{VIEW_NAME}/{VIEW_NAME}_'.encode('utf-8')
    frames = {}

    with env.begin() as txn:
        cursor = txn.cursor()
        # posiziona il cursore alla prima chiave che inizia con il prefix
        cursor.set_range(prefix)

        for key, value in cursor:
            if not key.startswith(prefix):
                break  # usciti dalla sequenza

            # estrai il numero del frame dalla chiave
            # formato: sequence/view/view_0000000001.jpg
            key_str = key.decode('utf-8')
            filename = key_str.split('/')[-1]          # view_0000000001.jpg
            frame_str = filename.replace(VIEW_NAME + '_', '').replace('.jpg', '')
            frame_no = int(frame_str)

            feat = np.frombuffer(value, dtype=np.float32).copy()
            if feat.shape[0] == 2048:
                frames[frame_no] = feat

    frame_nos = sorted(frames.keys())
    return frames, frame_nos


def build_label_array(frame_nos, annotations_df):
    """
    Assegna una label a ogni frame.
    Default: 0 (correct).
    Se il frame ricade in [start, end] di una coarse action → eredita la label.
    In caso di sovrapposizione, mistake e correction hanno priorità su correct.
    """
    # dizionario frame_no -> label, default 0
    label_dict = {f: 0 for f in frame_nos}

    for _, row in annotations_df.iterrows():
        try:
            start     = int(row['start'])
            end       = int(row['end'])
            label_str = str(row['label']).strip()
            label     = LABEL_MAP.get(label_str, 0)
        except (ValueError, KeyError):
            continue

        # assegna la label a tutti i frame nel range
        for frame_no in frame_nos:
            if start <= frame_no <= end:
                # priorità: mistake/correction sovrascrivono correct
                if label > label_dict[frame_no]:
                    label_dict[frame_no] = label

    labels = [label_dict[f] for f in frame_nos]
    return labels


def process_sequence(env, sequence_name, csv_path, output_dir):
    """
    Processa una singola sequenza e salva il file .pt.
    Restituisce True se OK, False se skippata.
    """
    output_path = output_dir / f'{sequence_name}.pt'
    if output_path.exists():
        return True  # già processata

    # 1. carica i frame dall'LMDB
    frames, frame_nos = load_sequence_from_lmdb(env, sequence_name)

    if len(frame_nos) == 0:
        print(f'  [SKIP] Nessun frame trovato per {sequence_name}')
        return False

    # 2. leggi le annotazioni
    try:
        df = pd.read_csv(csv_path, header=None)
        
        df = df.iloc[:, [0, 1, 5]]
        
        df.columns = ['start', 'end', 'label']
        
    except Exception as e:
        print(f'  [SKIP] Errore lettura CSV {csv_path}: {e}')
        return False

    # 3. costruisci features e labels
    features = np.stack([frames[f] for f in frame_nos])  # (T, 2048)
    labels   = build_label_array(frame_nos, df)           # (T,)

    features_t = torch.tensor(features, dtype=torch.float32)
    labels_t   = torch.tensor(labels,   dtype=torch.long)

    # 4. salva
    torch.save({
        'features':      features_t,    # (T, 2048)
        'labels':        labels_t,      # (T,)
        'frame_nos':     frame_nos,     # lista frame number originali
        'sequence_name': sequence_name,
    }, output_path)

    return True


def main():
    output_dir      = Path(OUTPUT_DIR)
    annotations_dir = Path(ANNOTATIONS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # apri LMDB una sola volta per tutto lo script
    env = lmdb.open(
        LMDB_PATH,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    csv_files = sorted(annotations_dir.glob('*.csv'))
    print(f'Sequenze trovate: {len(csv_files)}')

    ok = 0
    skip = 0
    label_counts = {0: 0, 1: 0, 2: 0}

    for csv_path in tqdm(csv_files, desc='Processando sequenze'):
        sequence_name = csv_path.stem
        success = process_sequence(env, sequence_name, csv_path, output_dir)

        if success:
            ok += 1
            # leggi il file appena salvato per le statistiche
            output_path = output_dir / f'{sequence_name}.pt'
            if output_path.exists():
                data = torch.load(output_path, weights_only=True)
                for lbl in [0, 1, 2]:
                    label_counts[lbl] += (data['labels'] == lbl).sum().item()
        else:
            skip += 1

    env.close()

    total_frames = sum(label_counts.values())
    print(f'\n=== Completato ===')
    print(f'Sequenze processate: {ok}')
    print(f'Sequenze skippate:   {skip}')
    print(f'Frame totali:        {total_frames:,}')
    print(f'\nDistribuzione label:')
    names = {0: 'correct', 1: 'mistake', 2: 'correction'}
    for lbl, count in label_counts.items():
        pct = 100 * count / total_frames if total_frames else 0
        print(f'  {names[lbl]:12s}: {count:8,} ({pct:.1f}%)')
    print(f'\nFile salvati in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
