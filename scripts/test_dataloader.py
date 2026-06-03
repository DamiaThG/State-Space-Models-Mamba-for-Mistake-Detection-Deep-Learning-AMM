import sys
import time
import torch

# =============================================================================
# 1. CONFIGURAZIONE PERCORSI ASSOLUTI (MODIFICA QUI)
# =============================================================================

# Inserisci il path assoluto della cartella che contiene il file dataloader.py
# Esempio: '/home/mssdmn01t05c351v/assembly-mistake-detection/scripts'
DATALOADER_DIR = '/home/mssdmn01t05c351v/assembly-mistake-detection/src/dataloader'

# Percorsi dei dati sul cluster
PROCESSED_DIR = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/processed'
ANNOTATIONS_DIR = '/home/mssdmn01t05c351v/assembly-mistake-detection/data/annotations/assembly101-mistake-detection/annots'

# =============================================================================

# Aggiungiamo la cartella del dataloader ai percorsi di sistema di Python
if DATALOADER_DIR not in sys.path:
    sys.path.append(DATALOADER_DIR)

# Ora Python troverà il file dataloader.py presente in DATALOADER_DIR
from dataloader import build_dataloader


def main():
    print("Inizializzazione del DataLoader in corso...")
    start_time = time.time()
    
    loader = build_dataloader(
        processed_dir=PROCESSED_DIR,
        annotations_dir=ANNOTATIONS_DIR,
        batch_size=4,
        shuffle=True,
        num_workers=2,
    )
    
    print(f"Tempo: {time.time() - start_time:.2f} sec")
    print(f"Dataset size : {len(loader.dataset)} azioni")
    print("-" * 50)

    for i, batch in enumerate(loader):
        print(f"\n[ TEST BATCH {i+1} ]")
        
        features = batch["features"]          
        labels = batch["labels"]              
        mask = batch["attention_mask"]        
        lengths = batch["lengths"]            
        names = batch["sequence_names"]       
        
        B, T_max, D = features.shape
        
        print(f"  > Video            : {names}")
        print(f"  > Lunghezze reali  : {lengths.tolist()}")
        print(f"  > T_max del batch  : {T_max}")
        print(f"  > Shape Features   : {list(features.shape)}")
        
        # --- BLOCCO DEGLI ASSERT (I TEST VERI E PROPRI) ---
        for b in range(B):
            L = lengths[b].item()
            
            # Controllo 1: L'attention mask deve essere True sui frame reali
            assert mask[b, :L].all(), f"Errore: Frame reali nascosti nel sample {b}"
            
            if L < T_max:
                # Controllo 2: L'attention mask deve essere False sul padding
                assert not mask[b, L:].any(), f"Errore: Mask non è False sul padding nel sample {b}"
                # Controllo 3: Le label di padding devono essere -1
                assert (labels[b, L:] == -1).all(), f"Errore: Label non è -1 sul padding nel sample {b}"
                # Controllo 4: Le features di padding devono essere composte da 0.0
                assert (features[b, L:] == 0.0).all(), f"Errore: Features non sono 0.0 sul padding nel sample {b}"

        print("  > [SUCCESS] Mask e Padding corretti! ✅")
        print(f"  > Label uniche     : {labels[labels != -1].unique().tolist()}")

        if i >= 2:
            break

if __name__ == "__main__":
    main()
