# PROMPT: Implementazione Architettura Mamba (SSM) per Mistake Detection

## 1. Contesto e Vincoli Infrastrutturali (CRITICO)
Sei un AI Engineer esperto in State-Space Models. Il tuo task è implementare un nuovo modello basato sull'architettura **Mamba** per il task di Mistake Detection (classificazione dense frame-by-frame a 3 classi: correct, mistake, correction), da integrare nella nostra pipeline PyTorch Lightning.

Il codice verrà eseguito su un cluster universitario privo di compilatore `nvcc` e di privilegi di root.
**NON PUOI** utilizzare la libreria ufficiale `mamba-ssm` (causerà un crash di installazione). 
**DEVI** utilizzare la libreria **`mambapy`** (già installata nell'ambiente), che fornisce un'implementazione di Mamba in Pure PyTorch compatibile con i nostri vincoli.

## 2. TASK 1: Implementazione del Modello `MambaMistakeDetector`
Scrivi la classe `MambaMistakeDetector(nn.Module)`.

**Specifiche di Input/Output:**
* **Input:** Tensore `features` di shape `[Batch, T_max, 2048]`.
* **Output:** Tensore `logits` di shape `[Batch, T_max, 3]`.

**Flusso Architetturale:**
1. **Input Projection:** Uno strato lineare per mappare le feature in ingresso (2048) alla dimensione interna del modello (es. `d_model = 256`).
2. **SSM Backbone:** Istanzia un blocco sequenziale utilizzando `Mamba` e `MambaConfig` dalla libreria `mambapy`. Configura un numero adeguato di layer (es. `n_layers=4` o `6`) mantenendo l'elaborazione rigorosamente causale.
3. **Classification Head:** Un layer lineare finale per mappare l'output di Mamba (`d_model`) nello spazio dei logits delle classi (`3`).

*Nota sul Padding:* Mamba elabora sequenze continue. Le sequenze di input conterranno del padding. Il modello applicherà la sua computazione su tutto il tensore, ma devi assicurarti che l'architettura gestisca i tensori in modo contiguo se richiesto da `mambapy`. La logica per ignorare il padding nel calcolo della loss è già demandata al `LightningModule`.

## 3. TASK 2: Integrazione in PyTorch Lightning
Nel nostro progetto usiamo PyTorch Lightning. Fornisci il codice per integrare questo nuovo modello nel sistema:
1. Crea un `MistakeDetectionLightningModule` generico (o aggiorna quello esistente) che possa accettare un'istanza di `nn.Module` (TempAgg o Mamba) passata come argomento.
2. Mostra come istanziare correttamente il sistema nel file principale di avvio, permettendo la selezione del modello tramite `argparse` o dizionario di configurazione (es. `--model mamba` o `--model tempagg`).
3. Assicurati che il LightningModule mantenga la logica della loss sbilanciata con `weight` e il mascheramento dei token di padding tramite `attention_mask`.