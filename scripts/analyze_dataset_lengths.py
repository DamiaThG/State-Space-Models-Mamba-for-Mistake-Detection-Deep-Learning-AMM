import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Analizza le lunghezze dei video e la distribuzione delle azioni.")
    parser.add_argument(
        "--annots_dir", 
        type=str, 
        default="/home/mssdmn01t05c351v/assembly-mistake-detection/data/annotations/assembly101-mistake-detection/annots",
        help="Percorso alla cartella contenente i CSV delle annotazioni."
    )
    args = parser.parse_args()
    
    folder_path = Path(args.annots_dir)
    
    if not folder_path.exists():
        print(f"Errore: La cartella {folder_path} non esiste.")
        return

    video_lengths = []
    
    # Per analizzare in che punto (percentuale) del video avvengono le azioni
    action_relative_positions = [] 

    csv_files = list(folder_path.glob("*.csv"))
    
    for file_path in csv_files:
        try:
            # Leggiamo il csv
            df = pd.read_csv(file_path, names=['start', 'end', 'verb', 'this', 'that', 'label', 'remark'], header=None)
            
            if df.empty:
                continue
                
            start_frames = pd.to_numeric(df['start'], errors='coerce')
            end_frames = pd.to_numeric(df['end'], errors='coerce')
            
            # La lunghezza totale del video è approssimabile all'end frame massimo nel CSV
            video_max_len = end_frames.max()
            
            if pd.notna(video_max_len) and video_max_len > 0:
                video_lengths.append(video_max_len)
                
                # Calcola il punto medio dell'azione come percentuale della lunghezza totale
                mid_points = (start_frames + end_frames) / 2.0
                relative_pos = mid_points / video_max_len
                action_relative_positions.extend(relative_pos.dropna().tolist())
                
        except Exception as e:
            print(f"Errore durante la lettura di {file_path.name}: {e}")

    video_lengths = np.array(video_lengths)
    action_relative_positions = np.array(action_relative_positions)

    if len(video_lengths) > 0:
        print("=== STATISTICHE LUNGHEZZA VIDEO INTERI ===")
        print(f"Totale video analizzati:           {len(video_lengths)}")
        print(f"Media frame per video:             {video_lengths.mean():.0f} frame")
        print(f"Mediana frame per video:           {np.median(video_lengths):.0f} frame")
        print(f"Percentile 90:                     {np.percentile(video_lengths, 90):.0f} frame")
        print(f"Percentile 95:                     {np.percentile(video_lengths, 95):.0f} frame")
        print(f"Percentile 99:                     {np.percentile(video_lengths, 99):.0f} frame")
        print(f"Video max in assoluto:             {video_lengths.max():.0f} frame")
        print(f"Video max più corto:               {video_lengths.min():.0f} frame")
        
        print("\n=== DISTRIBUZIONE DELLE AZIONI NEL TEMPO ===")
        # Dividiamo i video in 4 quartili per vedere dove sono concentrate le azioni
        q1 = np.sum(action_relative_positions <= 0.25)
        q2 = np.sum((action_relative_positions > 0.25) & (action_relative_positions <= 0.50))
        q3 = np.sum((action_relative_positions > 0.50) & (action_relative_positions <= 0.75))
        q4 = np.sum(action_relative_positions > 0.75)
        total_actions = len(action_relative_positions)
        
        print(f"Totale azioni analizzate: {total_actions}")
        print(f"Azioni nel Q1 (0-25% del video):   {q1/total_actions*100:.1f}%")
        print(f"Azioni nel Q2 (25-50% del video):  {q2/total_actions*100:.1f}%")
        print(f"Azioni nel Q3 (50-75% del video):  {q3/total_actions*100:.1f}%")
        print(f"Azioni nel Q4 (75-100% del video): {q4/total_actions*100:.1f}%")
    else:
        print("Nessun dato trovato per calcolare le statistiche.")

if __name__ == "__main__":
    main()
