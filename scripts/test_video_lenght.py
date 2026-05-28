import os
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    folder_path = Path.home() / "assembly-mistake-detection/data/annotations/assembly101-mistake-detection/annots"
    
    if not folder_path.exists():
        print(f"Errore: La cartella {folder_path} non esiste.")
        return

    # Lista per memorizzare la durata della sequenza/azione PIÙ LUNGA per ogni video
    max_chunk_lengths = []

    csv_files = list(folder_path.glob("*.csv"))
    
    for file_path in csv_files:
        try:
            # Leggiamo il csv.
            df = pd.read_csv(file_path, names=['start', 'end', 'verb', 'this', 'that', 'label', 'remark'], header=None)
            
            if df.empty:
                continue
                
            start_frames = pd.to_numeric(df['start'], errors='coerce')
            end_frames = pd.to_numeric(df['end'], errors='coerce')
            
            # Calcoliamo la durata (in frame) di ogni singola riga/azione nel video
            durations = end_frames - start_frames
            
            # Estraiamo la durata dell'azione più lunga in questo specifico video
            max_duration = durations.max()
            
            # Se abbiamo trovato un valore valido, lo salviamo
            if pd.notna(max_duration):
                max_chunk_lengths.append(max_duration)
                
        except Exception as e:
            print(f"Errore durante la lettura di {file_path.name}: {e}")

    # Convertiamo in array NumPy
    max_chunk_lengths = np.array(max_chunk_lengths)

    if len(max_chunk_lengths) > 0:
        print(f"Totale video analizzati:           {len(max_chunk_lengths)}")
        print(f"Media delle sequenze più lunghe:   {max_chunk_lengths.mean():.0f} frame")
        print(f"Mediana delle sequenze più lunghe: {np.median(max_chunk_lengths):.0f} frame")
        print(f"Percentile 90:                     {np.percentile(max_chunk_lengths, 90):.0f} frame")
        print(f"Percentile 95:                     {np.percentile(max_chunk_lengths, 95):.0f} frame")
        print(f"Sequenza (chunk) max in assoluto:  {max_chunk_lengths.max():.0f} frame")
        print(f"Sequenza (chunk) max più corta:    {max_chunk_lengths.min():.0f} frame")
    else:
        print("Nessun dato trovato per calcolare le statistiche.")

if __name__ == "__main__":
    main()
