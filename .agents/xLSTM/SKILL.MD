# 🧠 Context Skill: xLSTM per Mistake Detection

---

## 1. Missione

Sei un AI Engineer esperto in reti ricorrenti avanzate.
Implementare l'architettura **xLSTM** per il task di **Mistake Detection frame-by-frame** su Assembly101.
Il modello deve essere confrontabile con Mamba SSM (già implementato) e con la baseline TempAgg.

L'infrastruttura di training, evaluation e dataloader è **già costruita e funzionante**.
Il tuo compito è:
1. Creare il modello `xLSTMMistakeDetector` in `src/models/xlstm_model.py`
2. Creare lo script di training dedicato `src/training/train_xlstm.py`
3. Creare i runner/wrapper SLURM in `scripts/`

---

## 2. Libreria xLSTM

Il pacchetto ufficiale è **`xlstm`**, già installato nella `latest.sif` via `pip install --user xlstm`.

**Architettura scelta: mLSTM pura** — raccomandata dal paper originale per sequenze molto lunghe
(i video Assembly101 arrivano fino a ~25k frame). Nessun blocco sLSTM.

```python
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
)
```

### Configurazione mLSTM pura

```python
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4,
            qkv_proj_blocksize=4,
            num_heads=4,
        )
    ),
    context_length=max_seq_len,   # lunghezza massima sequenza (es. 25000)
    num_blocks=n_layers,          # numero totale di blocchi mLSTM
    embedding_dim=d_model,        # dimensione interna (es. 512)
    slstm_at=[],                  # lista vuota = nessun blocco sLSTM → mLSTM puro
)
backbone = xLSTMBlockStack(cfg)
```

---

## 3. Specifiche del Modello

### Input/Output
- **Input:** `[Batch, T_max, 2048]` — feature TSM estratte da Assembly101
- **Output:** `[Batch, T_max, 3]` — logit per frame (correct=0, mistake=1, correction=2)

### Architettura `xLSTMMistakeDetector(nn.Module)`

```
Input [B, T_max, 2048]
    → InputProjection    (Linear 2048 → d_model, LayerNorm, GELU, Dropout)
    → xLSTM Backbone     (xLSTMBlockStack — n_layers blocchi mLSTM puri)
    → ClassificationHead (LayerNorm → Linear d_model → 3)
    → Output [B, T_max, 3]
```

### Gestione della Padding Mask
Il dataloader restituisce `batch["attention_mask"]` di shape `[B, T_max]` (bool, `False` = padding).

La libreria `xlstm` **non supporta nativamente una attention_mask**: usa la strategia di
**zero-out** dei frame di padding prima di passare le feature al backbone.

```python
# x: [B, T, d_model], mask: [B, T] bool
if attention_mask is not None:
    x = x * attention_mask.unsqueeze(-1).float()
```

### Parametri Default

| Parametro    | Valore | Note                                   |
|--------------|--------|----------------------------------------|
| `input_dim`  | 2048   | Feature TSM                            |
| `d_model`    | 512    | Allineato con Mamba per confronto fair |
| `n_layers`   | 6      | Blocchi mLSTM                          |
| `num_classes`| 3      | correct / mistake / correction         |
| `dropout`    | 0.2    |                                        |
| `max_seq_len`| 25000  | Troncamento sequenze lunghe            |

> ⚠️ Potrebbero emergere problemi di memoria con `d_model=512` e sequenze di 25k frame.
> In caso, ridurre `max_seq_len` o `batch_size` durante il training.

### Firma obbligatoria del modello

```python
class xLSTMMistakeDetector(nn.Module):
    def forward(
        self,
        features: torch.Tensor,                          # [B, T, 2048]
        attention_mask: Optional[torch.Tensor] = None,   # [B, T] bool
    ) -> torch.Tensor:                                   # [B, T, 3]
        ...
```

---

## 4. Dataloader da Usare

**USA ESCLUSIVAMENTE `build_whole_video_dataloaders`** da `src/datasets/dataloader.py`.

```python
from src.datasets.dataloader import build_whole_video_dataloaders

train_loader, val_loader, test_loader = build_whole_video_dataloaders(
    processed_dir = args.processed_dir,   # "data/processed"
    batch_size    = args.batch_size,      # 4 (video interi → batch piccoli)
    val_split     = args.val_split,       # 0.15
    test_split    = args.test_split,      # 0.15
    num_workers   = args.num_workers,     # 4
    pin_memory    = (device_str == "cuda"),
    max_seq_len   = args.max_seq_len,     # 25000
    seed          = args.seed,
)
```

Il batch restituito ha questa struttura:
```python
batch = {
    "features":       # [B, T_max, 2048]  — feature TSM
    "labels":         # [B, T_max]         — int64, -1 = padding
    "attention_mask": # [B, T_max]         — bool, False = padding
}
```

> ⚠️ **NON usare** `build_split_dataloaders` (quello è per la baseline con sequenze di azioni, non video interi).

---

## 5. Integrazione con l'Infrastruttura Esistente

### `MistakeDetectionLightningModule` (già esistente, riutilizzare)

Il LightningModule in `src/training/training_loop.py` è **generico e model-agnostic**.
Accetta qualsiasi `nn.Module` con la firma:
```python
model.forward(features: Tensor, attention_mask: Optional[Tensor]) -> logits: Tensor
```
Gestisce già: loss pesata per classe, metriche per classe (precision/recall/F1), warmup + cosine LR schedule, WandB.

---

## 6. Training Script

### File da creare: `src/training/train_xlstm.py`

Modellato su `src/training/train_mamba.py`, con queste differenze:
- Importa `xLSTMMistakeDetector` invece di `MambaMistakeDetector`
- Usa `build_whole_video_dataloaders`
- Usa il `MistakeDetectionLightningModule` esistente (riutilizzato — **non** duplicarlo)
- **Nessun workaround torchvision** (non necessario in `latest.sif`)

```bash
# Uso tipico (interattivo dentro apptainer-gpu-xl):
python src/training/train_xlstm.py \
    --processed_dir   data/processed \
    --batch_size      4 \
    --num_workers     4 \
    --accumulate_grad_batches 2 \
    --d_model         512 \
    --n_layers        6 \
    --dropout         0.2 \
    --max_seq_len     25000 \
    --epochs          50 \
    --lr              5e-5 \
    --weight_decay    1e-3 \
    --seed            42 \
    --wandb_project   mistake-detection \
    --wandb_run_name  "xlstm-wholevid-run1" \
    --ckpt_dir        experiments/checkpoints
```

---

## 7. Script SLURM (architettura a due livelli)

### Runner: `scripts/run_train_xlstm.sh`
- Contiene la logica di lancio (variabili d'ambiente, chiamata Python, parametri di default)
- **NON** contiene direttive `#SBATCH` né chiamate ad `apptainer`
- Eseguibile direttamente dentro il container (`apptainer-gpu-xl` con `latest.sif`)
- Accetta argomenti extra via `"$@"` per sovrascrivere i default a runtime

### Wrapper SLURM: `scripts/train_xlstm.sh`
- Contiene solo le direttive `#SBATCH` e la chiamata `apptainer exec`
- Usa **`latest.sif`** (NON `mamba_env.sif`) — `xlstm` non richiede `mamba-ssm`
- Delega sempre la logica al runner `run_train_xlstm.sh`

---

## 8. Confronto con Mamba

| Aspetto                 | Mamba (SSM)                          | xLSTM (mLSTM puro)                         |
|-------------------------|--------------------------------------|---------------------------------------------|
| Container SIF           | `mamba_env.sif`                      | `latest.sif`                                |
| Libreria chiave         | `mamba-ssm`                          | `xlstm`                                     |
| Alias interattivo       | `mamba-docker`                       | `apptainer-gpu-xl`                          |
| Input projection        | Linear → LayerNorm → GELU → Dropout  | idem                                        |
| Backbone                | `n` blocchi `MambaBlock`             | `xLSTMBlockStack` (mLSTM puro, `slstm_at=[]`) |
| Mask handling           | zero-out features                    | zero-out features                           |
| Workaround torchvision  | ✓ (necessario)                       | ✗ (non necessario in `latest.sif`)          |
| Script runner           | `run_train_mamba.sh`                 | `run_train_xlstm.sh`                        |
| Script SLURM            | `train_mamba.sh`                     | `train_xlstm.sh`                            |
| Training entry          | `train_mamba.py`         | `train_xlstm.py`                |
