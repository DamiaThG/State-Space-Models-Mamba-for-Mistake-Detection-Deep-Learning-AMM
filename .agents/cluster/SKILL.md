# 🖥️ Context Skill: GCluster — Ambiente, Hardware e Linee Guida

---

## 1. Il Cluster: GCluster (DMI - UniCT)

Il progetto gira interamente sul **GCluster**, il cluster HPC del Dipartimento di Matematica e Informatica dell'Università di Catania (gcluster.dmi.unict.it), basato su scheduler **SLURM**.

- Accesso via SSH: `ssh <codice-fiscale>@gcluster.dmi.unict.it`
- **Username:** `mssdmn01t05c351v`
- **Home sul cluster:** `/home/mssdmn01t05c351v/`
- **Root del progetto:** `/home/mssdmn01t05c351v/assembly-mistake-detection/`
- Il nodo di login è **solo per sottomettere job**: non eseguire nulla di computazionalmente pesante lì.
- I job si eseguono sui **nodi di calcolo** tramite `srun` (interattivo) o `sbatch` (batch).
- ⚠️ Il comando `apptainer` **non è disponibile sul nodo di login**: deve essere eseguito esclusivamente sui nodi di calcolo (via `srun` o dentro un job `sbatch`).

### Alias definiti in `~/.bashrc`

```bash
# Entra in un container interattivo con latest.sif su gnode10 (gpu-xlarge)
alias apptainer-gpu-xl='srun --account=dl-course-q2 --partition=dl-course-q2 --qos=gpu-xlarge --nodelist=gnode10 --gres=gpu:1 --gres=shard:22000 --mem=32G --pty apptainer shell --nv /shared/sifs/latest.sif'

# Entra in un container interattivo con mamba_env.sif su gnode10 (gpu-xlarge)
alias mamba-docker='srun --account=dl-course-q2 --partition=dl-course-q2 --qos=gpu-xlarge --nodelist=gnode10 --gres=gpu:1 --gres=shard:22000 --mem=32G --pty apptainer shell --nv /home/mssdmn01t05c351v/assembly-mistake-detection/mamba_env.sif'
```

---

## 2. Hardware Disponibile

| Nodo    | Categoria | CPU                    | RAM    | GPU                              | bf16 |
|---------|-----------|------------------------|-------:|----------------------------------|:----:|
| gnode1  | low-end   | 8 core                 | 32 GB  | 1× Nvidia K80 (22 GB VRAM)       | ✗    |
| gnode2  | low-end   | 8 core                 | 32 GB  | 1× Nvidia K80 (22 GB VRAM)       | ✗    |
| gnode3  | low-end   | 8 core                 | 32 GB  | 1× Nvidia K80 (22 GB VRAM)       | ✗    |
| gnode4  | low-end   | 8 core                 | 32 GB  | 1× Nvidia K80 (22 GB VRAM)       | ✗    |
| gnode5  | high-end  | 16 core (32 thread)    | 192 GB | 4× Nvidia V100 (16 GiB VRAM cad) | ✗    |
| **gnode10** | **high-end** | **48 core (96 thread)** | **512 GB** | **4× Nvidia L40S (48 GiB VRAM cad)** | **✓** |

> **Nodo di riferimento del progetto: `gnode10`** — L40S con bf16 nativo, usato per **tutti** i training con `--nodelist=gnode10`.

---

## 3. Autorizzazioni, Account e QoS

- **Account/Partition assegnata:** `dl-course-q2`
- Per scoprire le autorizzazioni attive: `sacctmgr show associations user=$USER format=Account,Partition,QOS,DefaultQOS -P`

### Tabella QoS disponibili

| QoS Name       | CPU | RAM       | GPU VRAM  | Tempo Massimo |
|----------------|----:|----------:|----------:|--------------:|
| `gpu-small`    | 1   | 4096 MB   | 2816 MB   | 4 ore         |
| `gpu-medium`   | 2   | 8192 MB   | 5632 MB   | 6 ore         |
| `gpu-large`    | 4   | 16384 MB  | 11264 MB  | 12 ore        |
| `gpu-xlarge`   | 8   | 49152 MB  | 22528 MB  | 12 ore        |
| `gpu-phd-large`| 4   | 40960 MB  | 16384 MB  | 12 ore        |

> **QoS usata di default per tutti i task:** `gpu-xlarge` (8 CPU, 32 GB RAM, 22 GB VRAM).
> Usa QoS più basse per job veloci/debug: ottieni priorità più alta in coda.

---

## 4. Immagini Apptainer (Container SIF)

Il cluster usa **Apptainer** (non Docker) per isolare gli ambienti. Le immagini SIF sono nelle posizioni indicate.

### Immagine 1: `latest.sif` — Uso Generale

- **Path:** `/shared/sifs/latest.sif` (link simbolico all'immagine più recente)
- **Usata per:** baseline, xLSTM, TeSTra, mambapy, feature extraction, e qualsiasi modello **che NON usa mamba-ssm**
- **Librerie extra installate** (via `pip install --user`):
  - `einops`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `tqdm`, `xlstm`, `mambapy`
  - `lightning`, `wandb`

### Immagine 2: `mamba_env.sif` — Mamba SSM

- **Path:** `/home/mssdmn01t05c351v/assembly-mistake-detection/mamba_env.sif`
- **Usata OBBLIGATORIAMENTE per:** qualsiasi codice che importa `mamba_ssm` (la libreria ufficiale Mamba)
- **Librerie chiave:** `mamba-ssm`, `causal-conv1d`, `lightning`, `wandb`
- **Alias interattivo:** `mamba-docker` (lancia `srun + apptainer shell --nv mamba_env.sif` su gnode10)

> ⚠️ **Regola tassativa:** Se il codice fa `from mamba_ssm import ...` → usa **`mamba_env.sif`**.
> Tutti gli altri modelli → usa **`latest.sif`** (immagine di default del cluster).

> ⚠️ **Attenzione alle librerie:** Non tutte le librerie possono essere installate nel cluster.
> Verifica sempre la compatibilità prima di aggiungere nuove dipendenze.

---

## 5. Architettura degli Script (FONDAMENTALE)

Ogni task (training, estrazione feature, valutazione, ecc.) deve avere **due script separati**:

### Script 1 — Runner: `scripts/run_<task>.sh`
- Contiene tutta la logica: `export` delle variabili d'ambiente, chiamata Python, parametri di default
- **NON** contiene direttive `#SBATCH` né chiamate ad `apptainer`
- Eseguibile **direttamente dentro il container** (interattivo via alias o `apptainer shell`)
- Accetta argomenti extra via `"$@"` per sovrascrivere i default a runtime

### Script 2 — Wrapper SLURM: `scripts/<task>.sh`
- Contiene solo le direttive `#SBATCH` e la chiamata `apptainer exec ... bash /workspace/scripts/run_<task>.sh`
- Non duplica la logica del runner — la delega sempre al runner

### Esempi d'uso

```bash
# ── Modalità interattiva (mamba_env.sif) ────────────────────────────
mamba-docker                          # entra nel container su gnode10
cd /home/mssdmn01t05c351v/assembly-mistake-detection
./scripts/run_train_mamba.sh          # lancia il training direttamente
./scripts/run_train_mamba.sh --batch_size 8 --lr 1e-4   # con override

# ── Modalità interattiva (latest.sif) ───────────────────────────────
apptainer-gpu-xl                      # entra nel container su gnode10
cd /home/mssdmn01t05c351v/assembly-mistake-detection
./scripts/run_train_xlstm.sh          # lancia il training xLSTM

# ── Modalità batch (sbatch) ─────────────────────────────────────────
sbatch scripts/train_mamba.sh         # sottomette il job SLURM
squeue --me                           # controlla lo stato del job
tail -f experiments/logs/mamba_<JOB_ID>.out   # segui l'output
```

### Template Wrapper SLURM (mamba_env.sif)
```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --output=experiments/logs/<job_name>_%j.out
#SBATCH --error=experiments/logs/<job_name>_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tua-email>

mkdir -p experiments/logs experiments/checkpoints

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /home/mssdmn01t05c351v/assembly-mistake-detection/mamba_env.sif \
    bash /workspace/scripts/run_<task>.sh
```

### Template Wrapper SLURM (latest.sif)
```bash
#!/bin/bash
#SBATCH --job-name=<job_name>
#SBATCH --output=experiments/logs/<job_name>_%j.out
#SBATCH --error=experiments/logs/<job_name>_%j.err
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --nodelist=gnode10
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:22000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tua-email>

mkdir -p experiments/logs experiments/checkpoints

apptainer exec --nv \
    --bind $(pwd):/workspace \
    --pwd /workspace \
    /shared/sifs/latest.sif \
    bash /workspace/scripts/run_<task>.sh
```

---

## 6. Variabili d'Ambiente nei Runner Script

Da includere sempre nei runner:

```bash
export WANDB_MODE=offline          # wandb non ha accesso a internet dal cluster
export PYTHONUNBUFFERED=1          # stdout non bufferizzato nei log SLURM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # riduce OOM VRAM
```

> ⚠️ **Non usare `PYTHONNOUSERSITE=1`**: blocca Python dal trovare i pacchetti installati
> con `pip install --user` in `~/.local/` (es. `lightning`, `xlstm`, `mambapy`).

---

## 7. Gestione delle Dipendenze

### Librerie disponibili per immagine

| Libreria         | `latest.sif` | `mamba_env.sif` |
|------------------|:------------:|:---------------:|
| PyTorch          | ✓            | ✓               |
| `lightning`      | ✓ (user)     | ✓ (user)        |
| `wandb`          | ✓ (user)     | ✓ (user)        |
| `einops`         | ✓ (user)     | ✓               |
| `scikit-learn`   | ✓ (user)     | ✓               |
| `matplotlib`     | ✓ (user)     | ✓               |
| `pandas`         | ✓ (user)     | ✓               |
| `tqdm`           | ✓ (user)     | ✓               |
| `xlstm`          | ✓ (user)     | —               |
| `mambapy`        | ✓ (user)     | —               |
| `mamba-ssm`      | ✗            | ✓               |
| `causal-conv1d`  | ✗            | ✓               |

> `(user)` = installata via `pip install --user`, disponibile in `~/.local/`

### Installare nuove librerie
```bash
# Dentro il container (interattivo):
pip install --user <pacchetto>
# I pacchetti installati persistono in ~/.local/ tra le sessioni.
# ⚠️ Non tutte le librerie possono essere installate nel cluster.
```

---

## 8. Regole di Codice e Repository

### Paths
- Scripts eseguiti **sempre dalla root del progetto** (`/home/mssdmn01t05c351v/assembly-mistake-detection/`)
- Import assoluti: `from src.models.mamba_model import ...` ✓ — mai relativi `..` ✗
- Dataset/LMDB: `data/`
- Checkpoint: `experiments/checkpoints/`
- Log metriche: `experiments/logs/`

### Hardware & GPU
- Sempre device-agnostic: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- `torch.set_float32_matmul_precision("high")` all'inizio del training (sfrutta bf16 di L40S)
- `torch.cuda.empty_cache()` alla fine della validation per evitare OOM

### Logging in SLURM
- L'output va su file `.out`/`.err` — no progress bar visive
- `tqdm` con `mininterval=2.0` oppure log testuali per le epoche

### Framework
- **PyTorch Lightning** (`import lightning as L`) per strutturare training/validation
- **Weights & Biases** via `WandbLogger` — sempre `WANDB_MODE=offline` sul cluster

### Codice
- Modularità: modelli in `src/models/`, dataloader in `src/datasets/`, training in `src/training/`
- Seed fissi per riproducibilità: `torch`, `numpy`, `random`
- Tensori contigui per Mamba/xLSTM: `.contiguous()` dove richiesto

---

## 9. Comandi SLURM Utili

```bash
# Stato dei job personali
squeue --me

# Dettagli su un job (incluso il campo Reason se PENDING)
scontrol show job <JOB_ID>

# Cancella un job
scancel <JOB_ID>

# Risorse usate da un job in esecuzione
sstat -aPno TresUsageInMax -j <JOB_ID>

# Storico job
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed,Start,End

# Segui l'output in real-time
tail -f experiments/logs/<job_name>_<JOB_ID>.out

# Ispeziona GPU di un job in esecuzione (senza SSH sul nodo)
srun --jobid=<JOB_ID> --overlap nvidia-smi
```
