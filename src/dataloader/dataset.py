"""
Assembly101 Mistake Detection — Dataset

Ogni campione corrisponde a un segmento coarse (una riga CSV).
Le feature TSM vengono caricate dall'LMDB per intero sequenza (lazy, con cache)
e poi estratte per slice [start:end].

Struttura chiavi LMDB:
    {sequence_name}/{view_name}/{view_name}_{frame_no:010d}.jpg
Esempio:
    nusar-2021_action_both_9011-a01/C10119_rgb/C10119_rgb_0000000001.jpg

Valore: float32 array di 2048 dimensioni.
"""

import os
import csv
import struct
import numpy as np
from pathlib import Path
from typing import Optional

import lmdb
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

LABEL_MAP = {"correct": 0, "mistake": 1, "correction": 2}
VIEW_NAME = "C10119_rgb"  # v4, migliore performance dal paper
FEATURE_DIM = 2048


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AssemblyMistakeDataset(Dataset):
    """
    Dataset per mistake detection su Assembly101.

    Ogni item è un segmento coarse con:
        - features: tensor (T_seg, 2048) — frame da start a end
        - label:    int in {0, 1, 2}
        - meta:     dizionario con sequence_name, segment_idx, start, end

    Il meta['segment_idx'] permette ai modelli (es. TempAgg) di recuperare
    i segmenti precedenti della stessa sequenza per costruire lo storico.

    Args:
        annotations_dir: cartella con i 328 CSV delle annotazioni
        lmdb_dir:        cartella LMDB estratta (contiene data.mdb e lock.mdb)
        split:           'train', 'val', o 'test' (filtra per sequenze dello split)
        split_file:      path al file .txt con i nomi delle sequenze dello split
                         (se None, carica tutte le sequenze trovate nei CSV)
        view:            camera view da usare (default: C10119_rgb = v4)
        min_frames:      scarta segmenti con meno di N frame (default: 1)
    """

    def __init__(
        self,
        annotations_dir: str,
        lmdb_dir: str,
        split: Optional[str] = None,
        split_file: Optional[str] = None,
        view: str = VIEW_NAME,
        min_frames: int = 1,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.lmdb_dir = Path(lmdb_dir)
        self.view = view
        self.min_frames = min_frames

        # Cache: sequence_name → np.ndarray (T_total, 2048)
        self._cache: dict[str, np.ndarray] = {}

        # Apre l'ambiente LMDB in sola lettura (una sola connessione condivisa)
        # lock=False perché è read-only e potremmo avere più worker
        self._lmdb_env = lmdb.open(
            str(self.lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # Carica la lista di sequenze dello split (se fornita)
        self._split_sequences: Optional[set] = None
        if split_file is not None:
            with open(split_file, "r") as f:
                self._split_sequences = {line.strip() for line in f if line.strip()}

        # Costruisce la lista di campioni da tutti i CSV
        self.samples = self._parse_annotations()

    # -----------------------------------------------------------------------
    # Parsing annotazioni
    # -----------------------------------------------------------------------

    def _parse_annotations(self) -> list[dict]:
        """
        Legge tutti i CSV e costruisce la lista di campioni.

        Ogni campione è un dizionario:
            sequence_name, segment_idx, start, end, label (int)
        """
        samples = []
        csv_files = sorted(self.annotations_dir.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"Nessun CSV trovato in {self.annotations_dir}"
            )

        for csv_path in csv_files:
            # Il nome della sequenza è il nome del file senza estensione
            sequence_name = csv_path.stem

            # Filtra per split se richiesto
            if self._split_sequences is not None:
                if sequence_name not in self._split_sequences:
                    continue

            seg_idx = 0
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label_str = row["label"].strip().lower()
                    if label_str not in LABEL_MAP:
                        # Riga non riconosciuta, salta
                        continue

                    try:
                        start = int(row["start"])
                        end = int(row["end"])
                    except (ValueError, KeyError):
                        continue

                    # Gestisce il caso raro start > end nel CSV
                    # (presente in almeno una riga dell'esempio del paper)
                    if end <= start:
                        start, end = end, start
                    if (end - start) < self.min_frames:
                        continue

                    samples.append({
                        "sequence_name": sequence_name,
                        "segment_idx": seg_idx,
                        "start": start,
                        "end": end,
                        "label": LABEL_MAP[label_str],
                    })
                    seg_idx += 1

        if len(samples) == 0:
            raise RuntimeError(
                "Nessun campione valido trovato. "
                "Controlla annotations_dir e split_file."
            )

        return samples

    # -----------------------------------------------------------------------
    # Caricamento LMDB per sequenza intera (lazy + cache)
    # -----------------------------------------------------------------------

    def _load_sequence(self, sequence_name: str) -> np.ndarray:
        """
        Carica tutti i frame di una sequenza dall'LMDB in un array numpy.

        Le chiavi LMDB hanno formato:
            {sequence_name}/{view}/{view}_{frame_no:010d}.jpg

        I frame vengono ordinati per numero e impilati in (T, 2048).
        Il risultato viene salvato in self._cache[sequence_name].

        Returns:
            np.ndarray di shape (T_total, 2048), dtype float32
        """
        if sequence_name in self._cache:
            return self._cache[sequence_name]

        prefix = f"{sequence_name}/{self.view}/{self.view}_".encode()

        frame_dict: dict[int, np.ndarray] = {}

        with self._lmdb_env.begin(write=False) as txn:
            cursor = txn.cursor()
            # Posiziona il cursore sulla prima chiave con questo prefisso
            if not cursor.set_range(prefix):
                raise KeyError(
                    f"Nessuna chiave trovata per sequenza '{sequence_name}' "
                    f"view '{self.view}' nell'LMDB."
                )

            for key_bytes, value_bytes in cursor.iternext_dup() if False else cursor:
                if not key_bytes.startswith(prefix):
                    break  # Usciti dal range della sequenza

                # Estrae il numero di frame dalla chiave
                # Es: "...C10119_rgb_0000000042.jpg" → 42
                key_str = key_bytes.decode()
                frame_part = key_str.rsplit("_", 1)[-1]       # "0000000042.jpg"
                frame_no = int(frame_part.split(".")[0])       # 42

                # Deserializza il vettore float32
                # I valori sono raw bytes di float32
                n_floats = len(value_bytes) // 4
                vec = np.frombuffer(value_bytes, dtype=np.float32, count=n_floats)
                frame_dict[frame_no] = vec

        if len(frame_dict) == 0:
            raise KeyError(
                f"Nessun frame trovato per '{sequence_name}' / '{self.view}'."
            )

        # Ordina per frame number e impila
        max_frame = max(frame_dict.keys())
        # Usa frame number come indice (1-based nel LMDB → offset di 1)
        # Creiamo un array denso; i frame mancanti restano zero
        T = max_frame  # frame vanno da 1 a max_frame (1-based)
        features = np.zeros((T, FEATURE_DIM), dtype=np.float32)
        for frame_no, vec in frame_dict.items():
            features[frame_no - 1] = vec  # converti a 0-based

        self._cache[sequence_name] = features
        return features

    # -----------------------------------------------------------------------
    # Dataset interface
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        sequence_name = sample["sequence_name"]
        start = sample["start"]
        end = sample["end"]

        # Carica (o recupera dalla cache) la sequenza intera
        seq_features = self._load_sequence(sequence_name)

        # Converti frame number (1-based) a indici 0-based
        # start e end nel CSV sono frame number a 30fps (1-based)
        start_idx = max(0, start - 1)
        end_idx = min(end, len(seq_features))  # end è esclusivo dopo -1+1

        segment = seq_features[start_idx:end_idx]  # (T_seg, 2048)

        return {
            "features": torch.from_numpy(segment.copy()),  # (T_seg, 2048)
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "meta": {
                "sequence_name": sequence_name,
                "segment_idx": sample["segment_idx"],
                "start": start,
                "end": end,
            },
        }

    def __del__(self):
        # Chiude l'ambiente LMDB alla distruzione dell'oggetto
        if hasattr(self, "_lmdb_env") and self._lmdb_env is not None:
            self._lmdb_env.close()

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def get_class_weights(self) -> torch.Tensor:
        """
        Calcola i pesi inversi delle classi per la weighted cross-entropy.
        Utile perché il dataset è fortemente sbilanciato (~77% correct).

        Returns:
            tensor di shape (3,) con pesi per [correct, mistake, correction]
        """
        counts = np.zeros(3, dtype=np.float32)
        for s in self.samples:
            counts[s["label"]] += 1

        # Peso = N_totale / (N_classi * N_campioni_classe)
        total = counts.sum()
        weights = total / (3.0 * counts)
        return torch.from_numpy(weights)

    def get_sequence_samples(self, sequence_name: str) -> list[dict]:
        """
        Restituisce tutti i campioni di una sequenza, in ordine.
        Utile per TempAgg che deve accedere allo storico.
        """
        return [
            s for s in self.samples
            if s["sequence_name"] == sequence_name
        ]

    def clear_cache(self):
        """Libera la memoria della cache (utile se la RAM è limitata)."""
        self._cache.clear()


# ---------------------------------------------------------------------------
# Collate function per DataLoader
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """
    Gestisce batch con sequenze di lunghezza variabile.

    Padding con zeri alla lunghezza massima nel batch.
    Restituisce anche una maschera booleana per ignorare il padding.

    Returns:
        features:      (B, T_max, 2048)
        labels:        (B,)
        padding_mask:  (B, T_max) — True dove ci sono dati reali
        meta:          lista di dizionari
    """
    features_list = [item["features"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    meta = [item["meta"] for item in batch]

    # Trova lunghezza massima nel batch
    lengths = [f.shape[0] for f in features_list]
    T_max = max(lengths)
    B = len(batch)

    # Alloca tensore con padding
    features_padded = torch.zeros(B, T_max, FEATURE_DIM, dtype=torch.float32)
    padding_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, (feat, length) in enumerate(zip(features_list, lengths)):
        features_padded[i, :length] = feat
        padding_mask[i, :length] = True

    return {
        "features": features_padded,       # (B, T_max, 2048)
        "labels": labels,                  # (B,)
        "padding_mask": padding_mask,      # (B, T_max)
        "lengths": torch.tensor(lengths),  # (B,)
        "meta": meta,
    }