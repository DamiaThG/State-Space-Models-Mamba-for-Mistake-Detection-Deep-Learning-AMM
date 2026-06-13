# Struttura Presentazione Finale - Mistake Detection con SSM
**Durata:** 15 Minuti (circa 5 minuti a testa)
**Formato:** 3 Speaker (A, B, C)
**Slide previste:** 12-14 slide max

---

## 🎤 SPEAKER A: Contesto, Dataset e Formulazione del Problema (Minuti 0 - 5)

### Slide 1: Titolo e Introduzione
* Titolo del Progetto, Nomi dei membri del gruppo.
* **Hook (Gancio):** Perché capire quando un essere umano commette un errore procedurale è fondamentale (es. sistemi di Realtà Aumentata per supporto in catene di montaggio o chirurgia).

### Slide 2: Il Task - Mistake Detection Frame-by-Frame
* Spiegazione visiva del task: identificare temporalmente azioni di `correct`, `mistake` e `correction`.
* Sottolineare l'approccio ambizioso: segmentazione densa frame-by-frame sull'intero video (non classificazione di segmenti pre-tagliati).

### Slide 3: Il Dataset Assembly101 e le sue Sfide
* Breve overview: video di assemblaggio giocattoli, viste multiple (noi usiamo single view con feature TSM).
* **La vera sfida:** Lo sbilanciamento estremo. Più del 77% dei frame è corretto, pochissimi frame di errore reale. Lunghezza immensa delle sequenze (fino a 20k+ frame).

### Slide 4: Oltre i Transformer e le RNN Classiche
* Breve transizione alle architetture: spiegare perché non abbiamo usato Transformer (complessità computazionale quadratica $O(N^2)$ proibitiva per 20k frame) o vecchie LSTM (decadimento del contesto).

---

## 🎤 SPEAKER B: Metodologia e Scelte Architetturali (Minuti 5 - 10)

### Slide 5: Architetture a Confronto
* Diagramma ad alto livello dei 3 modelli studiati.
* **C2F (Baseline):** Convoluzioni temporali locali.
* **Mamba:** Spazio di stato continuo, *Selective Scan*, complessità $O(N)$.
* **xLSTM (mLSTM):** Evoluzione delle RNN con memoria a matrice e processing parallelo.

### Slide 6: Sfide Ingegneristiche e Gestione della VRAM
* Come avete fatto materialmente a trainare sequenze di 20.000 frame?
* Troncamento (`max_seq_len`).
* **Gradient Checkpointing *per-block***: il vero "salvavita" per evitare gli Out Of Memory (OOM).
* Masking e ignore_index per il padding.

### Slide 7: Gestione dello Sbilanciamento Estremo
* La scelta della **Focal Loss** multi-classe.
* L'impatto del `class_weight_exp` per penalizzare l'eccessiva sicurezza sulle classi "facili" e forzare il focus sui `mistake`.

---

## 🎤 SPEAKER C: Risultati, Limiti e Conclusioni (Minuti 10 - 15)

### Slide 8: Risultati Quantitativi
* Inserire la `Tabella_Risultati.md` ridotta (Baseline 1 vs Mamba 4 vs xLSTM 2).
* **Takeaway:** Il paradosso per cui la Baseline (con focus locale ad alta frequenza) vince leggermente sui grandi modelli globali sull'F1 del mistake (25% vs 15%).

### Slide 9: Analisi del Paradosso (Mamba diluisce il contesto?)
* Perché Mamba e xLSTM "perdono" sull'F1 del mistake? 
* Suggerire che lo stato nascosto continuo "diluisce" la precisione al singolo frame per colpa del rumore accumulato.

### Slide 10: Analisi Qualitativa e "Oracle Ceiling" (Il confronto col Paper originale)
* Ricordare che anche nel paper originale (Sener et al. CVPR 2022), usando *etichette umane perfette* fornite all'oracolo, la precisione massima sui Mistake è del 48%. 
* Questo dimostra che il basso F1 non è colpa del codice, ma della mostruosa ambiguità visiva intrinseca al video.

### Slide 11: Limiti Riconosciuti (Il vostro punto di forza sul rigore)
Essere espliciti sui difetti dimostra maturità scientifica.
* **Dataset Bias & Ambiguità Visiva:** Rarità estrema e mancanza di differenze visive chiare tra un errore e la sua correzione.
* **Vincoli Computazionali:** Impossibilità (hardware) di scalare i canali dei modelli o i layer di Mamba oltre certi limiti su sequenze di 20k frame, costringendo a trade-off continui.
* **Training Instabile:** Difficoltà di ottimizzazione con loss fortemente fluttuanti (specialmente in configurazione Whole Video), con alto rischio di memorizzare rumore.
* **Baseline Mancanti:** Mancanza di implementazioni open-source equivalenti per il setting denso frame-by-frame, che ci ha costretto a ricostruire noi stessi la TempAgg C2F.

### Slide 12: Conclusione e Sviluppi Futuri
* Sintesi: I modelli SSM/xLSTM abbattono i limiti di VRAM e memoria a lunghissimo termine, ma sui task ad altissima frequenza (frame-level locale) perdono precisione chirurgica.
* Futuro: Architetture ibride (Convoluzioni locali $\rightarrow$ Mamba globale) o uso di feature multimodali (Pose 3D).
* Q&A.
