# Risultati Sperimentali - Mistake Detection (Assembly101)

Di seguito è riportata la tabella riassuntiva di tutti gli 11 addestramenti effettuati sui modelli (Baseline C2F, Mamba, xLSTM):

| ID Esperimento | Modello | Parametri Trainabili | Lunghezza Sequenza | Miglior Val Loss | Test Loss | Test Correct F1 | Test Mistake F1 | Test Correction F1 | Test Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline_001** | BASELINE 1 | 9.06M | 8000 | 0.4863 | 0.7087 | 93.35% | **25.13%** | 2.91% | **40.46%** |
| **baseline_002** | BASELINE 2 | 2.53M | 8000 | 0.3611 | 0.6847 | 94.16% | **25.62%** | 1.61% | **40.46%** |
| **mamba_001** | MAMBA 1 | 11.36M | None | 0.5371 | 1.7684 | 95.62% | 13.99% | **4.38%** | 37.99% |
| **mamba_003** | MAMBA 3 | 11.36M | 12000 | 0.2959 | 0.4673 | 93.75% | 15.18% | 1.84% | 36.92% |
| **mamba_004** | MAMBA 4 | 11.36M | 8000 | **0.2261** | **0.2605** | 93.76% | 10.88% | 1.61% | 35.41% |
| **mamba_006** | MAMBA 6 | **3.19M** | 12000 | 0.3002 | 0.5340 | 95.05% | 10.45% | 2.12% | 35.87% |
| **xlstm_001** | XLSTM 1 | 10.89M | 8000 | 0.3303 | 0.9828 | **95.84%** | 10.85% | 1.25% | 35.98% |
| **xlstm_002** | XLSTM 2 | 4.54M | 8000 | 0.6906 | 0.9915 | 94.28% | 13.69% | 3.07% | 37.01% |
