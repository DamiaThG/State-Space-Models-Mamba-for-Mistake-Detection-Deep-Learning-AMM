# MISSIONE

Sei un AI Engineer esperto in State-Space Models. Abbiamo un'infrastruttura di training/evaluation già funzionante. Il tuo task è implementare un nuovo modello basato sull'architettura **Mamba** da integrare nella pipeline esistente per il task di Mistake Detection.

# VINCOLI INFRASTRUTTURALI (CRITICO)

Il codice verrà eseguito su un cluster universitario privo di compilatore `nvcc` e di privilegi di root.
**NON PUOI** utilizzare la libreria ufficiale `mamba-ssm` perché richiede la compilazione di kernel CUDA custom C++ (causerà un crash di installazione).
**DEVI** implementare (o importare se esiste un branch standard) una versione di Mamba scritta interamente in **Pure PyTorch**. Il modello sarà più lento dell'originale, ma è l'unico modo per renderlo compatibile con il cluster.

# SPECIFICHE DEL MODELLO

Implementa la classe `MambaMistakeDetector(nn.Module)`.

- **Input:** [Batch, T_max, 2048]
- **Output:** [Batch, T_max, 3]
- **Architettura:**
  1. Uno strato di proiezione lineare iniziale (es. 2048 -> d_model = 256).
  2. N blocchi Mamba (Pure PyTorch) sovrapposti. Usa il Selective State Space Model approach.
  3. Una testata di classificazione finale (Linear layer da d_model -> 3).
- Assicurati che il modello rispetti la causalità temporale (nessun leak dal futuro, essenziale per Mamba).
- **Integrazione:** Modifica il file `train.py` precedentemente generato per permettere l'istanziazione di questo modello al posto di TempAgg, ad esempio tramite un argomento da terminale `--model mamba`.
