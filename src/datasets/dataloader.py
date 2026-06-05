"""
Generic Mistake Detection DataLoader
====================================
Dataset + DataLoader universale per Mistake Detection.

Legge i .pt e i CSV, e restituisce per ogni azione la storia causale
[0 : end_frame] senza applicare alcun pooling o manipolazione architetturale.
Il batching gestisce il padding dinamico.
"""

import glob
import random
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
LABEL_MAP = {"correct": 0, "mistake": 1, "correction": 2}
PAD_LABEL = -1  # ignorato dalla CrossEntropyLoss con ignore_index=-1

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MistakeDetectionDataset(Dataset):
    """
    Dataset generico per Mistake Detection su Assembly101.
    Ogni sample è una sequenza causale completa da frame 0 a end_frame.

    Args:
        max_seq_len: se specificato, tronca le sequenze più lunghe ai
            *max_seq_len* frame più recenti (preserva il contesto causale
            fino a end_idx evitando tensori enormi in GPU).
    """

    def __init__(
        self,
        processed_dir: str,
        annotations_dir: str,
        csv_pattern: str = "*.csv",
        max_seq_len: Optional[int] = None,
    ):
        self.processed_dir = Path(processed_dir)
        self.max_seq_len   = max_seq_len
        self.samples = []

        csv_files = sorted(Path(annotations_dir).glob(csv_pattern))
        if not csv_files:
            raise FileNotFoundError(
                f"Nessun CSV in '{annotations_dir}' con pattern '{csv_pattern}'"
            )

        missing_pt = []

        for csv_path in csv_files:
            sequence_name = csv_path.stem
            pt_path = self.processed_dir / f"{sequence_name}.pt"

            if not pt_path.exists():
                missing_pt.append(sequence_name)
                continue

            try:
                # Legge il CSV ignorando l'header mancante (come scoperto prima)
                df = pd.read_csv(csv_path, header=None)
                df = df.iloc[:, [0, 1, 5]]
                df.columns = ['start', 'end', 'label']
            except Exception as e:
                print(f"[WARN] Errore lettura CSV {csv_path}: {e} — skip")
                continue

            # Carica frame_nos per mappare frame_no assoluto -> indice nel tensore
            data = torch.load(pt_path, weights_only=True)
            frame_nos = data["frame_nos"]  
            frame_no_to_idx = {fn: i for i, fn in enumerate(frame_nos)}

            for _, row in df.iterrows():
                try:
                    end_frame_no = int(row["end"])
                except (ValueError, KeyError):
                    continue

                # Mappa end_frame_no -> indice nel tensore
                if end_frame_no in frame_no_to_idx:
                    end_idx = frame_no_to_idx[end_frame_no]
                else:
                    # Frame esatto mancante: prendi il più vicino disponibile
                    available = [fn for fn in frame_nos if fn <= end_frame_no]
                    if not available:
                        continue
                    end_idx = frame_no_to_idx[max(available)]

                if end_idx < 0:
                    continue

                self.samples.append({
                    "sequence_name": sequence_name,
                    "pt_path": str(pt_path),
                    "end_idx": end_idx,
                })

        if missing_pt:
            print(f"[WARN] {len(missing_pt)} sequenze senza .pt: {missing_pt[:5]}...")

        print(f"[INFO] Dataset pronto: {len(self.samples)} sample da {len(csv_files) - len(missing_pt)} sequenze")
        if max_seq_len is not None:
            print(f"[INFO] Sequenze troncate a max_seq_len={max_seq_len} frame")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        info    = self.samples[idx]
        end_idx = info["end_idx"]

        # Estrae i tensori e taglia la storia causale
        data     = torch.load(info["pt_path"], weights_only=True)
        features = data["features"][: end_idx + 1]   # [T_storia, 2048]
        labels   = data["labels"][: end_idx + 1]     # [T_storia]

        # Tronca ai max_seq_len frame più recenti se specificato
        if self.max_seq_len is not None and features.shape[0] > self.max_seq_len:
            features = features[-self.max_seq_len :]
            labels   = labels[-self.max_seq_len :]

        return {
            "features": features,
            "labels": labels,
            "sequence_name": info["sequence_name"],
            "original_length": features.shape[0],
        }


# ---------------------------------------------------------------------------
# Collate Function & Factory
# ---------------------------------------------------------------------------

def generic_collate_fn(batch: list) -> dict:
    """
    Padding dinamico a T_max del batch corrente.
    """
    features_list = [item["features"] for item in batch]
    labels_list   = [item["labels"]   for item in batch]
    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)

    # Pad con 0.0 per le features, -1 per le label (ignore_index)
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    labels_padded   = pad_sequence(labels_list,   batch_first=True, padding_value=PAD_LABEL)

    B, T_max = features_padded.shape[:2]
    attention_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = True

    return {
        "features": features_padded,          # [B, T_max, 2048]
        "labels": labels_padded,              # [B, T_max]
        "attention_mask": attention_mask,     # [B, T_max]
        "lengths": lengths,                   # [B]
        "sequence_names": [item["sequence_name"] for item in batch],
    }


def build_dataloader(
    processed_dir: str,
    annotations_dir: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_seq_len: Optional[int] = None,
) -> DataLoader:

    dataset = MistakeDetectionDataset(
        processed_dir=processed_dir,
        annotations_dir=annotations_dir,
        max_seq_len=max_seq_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=generic_collate_fn,
        persistent_workers=num_workers > 0,
    )


def build_split_dataloaders(
    processed_dir: str,
    annotations_dir: str,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_seq_len: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Crea train / val / test DataLoader con split a livello di **sequenza**.

    Perché split per sequenza e non per sample?
    Ogni sample del dataset è definito come [0 : end_frame] di una sequenza.
    Se si splittasse per sample, frame della stessa sequenza finirebbero sia
    in train che in val → data leakage e metriche di validazione falsate.
    Splittando per sequenza si garantisce che l'intero video vada in un solo
    split.

    Returns:
        (train_loader, val_loader, test_loader)
        test_loader è None se test_split == 0.
    """
    dataset = MistakeDetectionDataset(
        processed_dir=processed_dir,
        annotations_dir=annotations_dir,
        max_seq_len=max_seq_len,
    )

    # ── Raggruppa indici per sequenza ────────────────────────────────────────
    seq_to_indices: dict = {}
    for idx, sample in enumerate(dataset.samples):
        seq = sample["sequence_name"]
        seq_to_indices.setdefault(seq, []).append(idx)

    sequences = list(seq_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(sequences)

    n       = len(sequences)
    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    train_seqs = sequences[:n_train]
    val_seqs   = sequences[n_train : n_train + n_val]
    test_seqs  = sequences[n_train + n_val :]

    def _indices(seqs):
        idxs = []
        for s in seqs:
            idxs.extend(seq_to_indices[s])
        return idxs

    train_idx = _indices(train_seqs)
    val_idx   = _indices(val_seqs)
    test_idx  = _indices(test_seqs)

    print(
        f"[INFO] Split sequenze → train: {len(train_seqs)}, "
        f"val: {len(val_seqs)}, test: {len(test_seqs)}"
    )
    print(
        f"[INFO] Split sample   → train: {len(train_idx)}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )

    # ── Crea i Subset e i DataLoader ─────────────────────────────────────────
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=generic_collate_fn,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,  **loader_kwargs
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),   batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = (
        DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, **loader_kwargs)
        if test_idx else None
    )

    return train_loader, val_loader, test_loader
