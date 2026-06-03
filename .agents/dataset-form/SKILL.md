# 📦 Context Skill: Dataset Format & DataLoader Output

## 1. Origine dei Dati — Assembly101

Il dataset è **Assembly101** (benchmark per procedural mistake detection).
Le feature visive sono estratte con **TSM (Temporal Shift Module)** su video multi-view.

| Parametro | Valore |
|---|---|
| View usata | `C10119_rgb` (vista fissa frontale) |
| Feature dim | **2048** float32 per frame |
| Sorgente raw | LMDB (`data/TSM_features/C10119_rgb`) |
| Annotazioni | CSV per sequenza in `data/annotations/assembly101-mistake-detection/annots/` |
| Download | `src/datasets/download_tsm.py` — HuggingFace Hub (`cvml-nus/assembly101`) |

---

## 2. Pipeline di Preprocessing — `build_dataset.py`

Lo script `src/datasets/build_dataset.py` trasforma i dati raw in tensori pronti all'uso.

### Flusso per ogni sequenza video:

```
LMDB (feature raw float32)
    ↓  load_sequence_from_lmdb()
    →  dict { frame_no (int) → np.array(2048,) }

CSV annotazioni (coarse actions)
    ↓  build_label_array()
    →  assegna label per ogni frame
       └─ 0 = correct   (default, nessuna coarse action)
       └─ 1 = mistake   (frame nel range [start, end] di una mistake)
       └─ 2 = correction (frame nel range di una correction)
       ⚠️  In caso di sovrapposizione: mistake/correction > correct (priorità per valore max)

Output: torch.save() → data/processed/<sequence_name>.pt
```

### Struttura del `.pt` salvato per sequenza:

```python
{
    'features':      torch.Tensor,   # shape (T, 2048), dtype float32
    'labels':        torch.Tensor,   # shape (T,),      dtype long  → {0, 1, 2}
    'frame_nos':     list[int],      # lista frame number originali (ordinati)
    'sequence_name': str,            # es. "nusar-2021_S04_S05_T001_..."
}
```

- `T` = numero di frame effettivamente presenti nell'LMDB per quella sequenza (variabile tra sequenze).
- Sono esclusi frame con feature di dimensione ≠ 2048.

---

## 3. Dataset PyTorch — `MistakeDetectionDataset`

**File:** `src/datasets/dataloader.py`

### Logica di campionamento

Ogni **sample** non è l'intera sequenza, ma una **storia causale** corrispondente a una singola coarse action annotata nel CSV:

```
Per ogni riga CSV (= una coarse action):
    end_frame_no = row['end']
    end_idx      = indice nel tensore .pt corrispondente a end_frame_no

    sample → features[0 : end_idx + 1]   # tutti i frame dal primo fino al termine dell'azione
             labels  [0 : end_idx + 1]
```

- **Il contesto è causale:** il modello vede la storia completa *fino* alla fine dell'azione, non il futuro.
- Se `end_frame_no` non è presente nell'LMDB, si usa il frame disponibile più vicino ≤ end_frame_no.
- Ogni riga CSV produce **un sample indipendente** → dataset molto più grande del numero di sequenze.

### Output di `__getitem__` (singolo sample, non batchiato):

```python
{
    "features":         torch.Tensor,   # shape (T_storia, 2048), float32
    "labels":           torch.Tensor,   # shape (T_storia,),      long  → {0, 1, 2}
    "sequence_name":    str,
    "original_length":  int,            # = T_storia (lunghezza prima del padding)
}
```

---

## 4. Collate Function & Output del DataLoader

La `generic_collate_fn` applica **padding dinamico** al batch corrente (pad fino alla sequenza più lunga nel batch, non globale).

### Output di ogni batch dal DataLoader:

```python
{
    "features":       torch.Tensor,   # shape [B, T_max, 2048], float32   — padding con 0.0
    "labels":         torch.Tensor,   # shape [B, T_max],       long       — padding con -1
    "attention_mask": torch.Tensor,   # shape [B, T_max],       bool       — True = token reale, False = pad
    "lengths":        torch.Tensor,   # shape [B],              long       — lunghezza vera di ogni sequenza
    "sequence_names": list[str],      # lista di B nomi sequenza
}
```

| Chiave | Shape | Dtype | Note |
|---|---|---|---|
| `features` | `[B, T_max, 2048]` | `float32` | Input per il modello, padding = 0.0 |
| `labels` | `[B, T_max]` | `long` | Ground truth; pad = **-1** (usare `ignore_index=-1` in CrossEntropyLoss) |
| `attention_mask` | `[B, T_max]` | `bool` | `True` = frame reale; usabile come mask per Mamba/xLSTM/Transformer |
| `lengths` | `[B]` | `long` | Utile per `pack_padded_sequence` o per mascherare l'output |
| `sequence_names` | `list[str]` | — | Solo per debug/logging |

---

## 5. Factory — `build_dataloader()`

```python
from src.datasets.dataloader import build_dataloader

loader = build_dataloader(
    processed_dir   = "data/processed",
    annotations_dir = "data/annotations/assembly101-mistake-detection/annots",
    batch_size      = 16,
    shuffle         = True,
    num_workers     = 4,
    pin_memory      = True,   # automaticamente False se no CUDA
)
```

---

## 6. Label Schema — Riepilogo

| Valore | Classe | Descrizione |
|---|---|---|
| `0` | `correct` | Esecuzione corretta / nessuna anomalia |
| `1` | `mistake` | Errore procedurale |
| `2` | `correction` | Correzione di un errore precedente |
| `-1` | `PAD` | Token di padding — **ignorato dalla loss** |

**Classe sbilanciata:** `correct` (0) è dominante. Prevedere strategie di class weighting o focal loss.

---

## 7. Pattern d'Uso nei Modelli

```python
for batch in train_loader:
    x    = batch["features"]        # [B, T_max, 2048]  → input
    y    = batch["labels"]          # [B, T_max]         → target
    mask = batch["attention_mask"]  # [B, T_max]         → maschera padding

    # Forward pass (esempio generico)
    logits = model(x, mask)         # [B, T_max, 3]  — 3 classi

    # Loss: ignora i token di padding
    loss = criterion(
        logits.view(-1, 3),
        y.view(-1),
        # CrossEntropyLoss deve avere ignore_index=-1
    )
```

> ⚠️ **Mamba/xLSTM:** queste architetture elaborano sequenze in ordine temporale.
> Il padding a fine sequenza non causa problemi di data leakage, ma è consigliato
> mascherare l'output prima di calcolare la loss usando `attention_mask`.
