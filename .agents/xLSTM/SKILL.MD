# MISSIONE

Sei un AI Engineer esperto in reti ricorrenti avanzate. Abbiamo già costruito un'infrastruttura di training/evaluation e testato TempAgg e Mamba. Il tuo task è completare il progetto implementando l'architettura **xLSTM**.

# LIBRERIA xLSTM

Il pacchetto ufficiale xLSTM è già installato nell'ambiente tramite `pip install xlstm`.
Dovrai importare e utilizzare i blocchi forniti da questa libreria ufficiale (sLSTM e mLSTM) per costruire il modello.

# SPECIFICHE DEL MODELLO

Implementa la classe `xLSTMMistakeDetector(nn.Module)`.

- **Input:** [Batch, T_max, 2048]
- **Output:** [Batch, T_max, 3]
- **Architettura:**
  1. Uno strato di proiezione lineare iniziale (es. 2048 -> d_model = 256).
  2. Costruisci il core utilizzando i moduli di `xlstm`. Poiché il nostro task richiede una forte memoria esponenziale causale, un mix di mLSTM (per scalare l'attenzione sul lungo termine) e sLSTM è raccomandato, o semplicemente un'architettura mLSTM pura come indicato dal paper originale per task di sequenze molto lunghe.
  3. Testata di classificazione finale (Linear layer -> 3).

# INTEGRAZIONE E COMPARAZIONE

- Il modello deve restituire la shape temporale densa esatta per essere valutato dallo script `evaluate.py` preesistente.
- Aggiungi la possibilità di istanziare l'xLSTM nel file `train.py` (es. `--model xlstm`).
- Assicurati che l'Attention Mask booleana `[B, T_max]` generata dal nostro dataloader venga gestita correttamente per non far esplodere lo stato di memoria dell'xLSTM sui frame di padding.
