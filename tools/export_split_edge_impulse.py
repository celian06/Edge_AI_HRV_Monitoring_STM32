import os
import numpy as np
import pandas as pd
import mne
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'generated_csv')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# On garde la meme config
FILES_CONFIG = {
    "chb04": { "path": os.path.join("chb04", "chb04_08.edf"), "seizures": [(1440, 1470)] },
    "chb12": { "path": os.path.join("chb12", "chb12_28.edf"), "seizures": [(181, 215)] },
    "chb13": { "path": os.path.join("chb13", "chb13_04.edf"), "seizures": [] }
}

WINDOW_SIZE = 20        
BASELINE_DURATION = 120 

# ==========================================
# 2. EXTRACTION (Inchange)
# ==========================================

def get_ecg_data(filepath):
    full_path = os.path.join(DATA_DIR, filepath)
    if not os.path.exists(full_path): return None, None

    try:
        raw = mne.io.read_raw_edf(full_path, preload=False, verbose=False)
        ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper() or "EKG" in c.upper()), None)
        if not ecg_ch: return None, None

        raw.pick_channels([ecg_ch])
        raw.load_data()
        raw.filter(1.0, 45.0, verbose=False)
        
        sig = raw.get_data()[0] * 1e6
        peaks, _ = find_peaks(sig, distance=int(0.3*raw.info['sfreq']), prominence=np.percentile(sig, 75))
        
        rr_intervals = (np.diff(peaks) / raw.info['sfreq']) * 1000
        peak_times = peaks[1:] / raw.info['sfreq'] 
        
        mask = (rr_intervals > 300) & (rr_intervals < 1500)
        return rr_intervals[mask], peak_times[mask]

    except Exception:
        return None, None

def extract_features(rr_intervals, times, seizures_intervals):
    rows = []
    
    calib = rr_intervals[times < BASELINE_DURATION]
    if len(calib) < WINDOW_SIZE: return []
    
    baseline_vals = [np.sqrt(np.mean(np.diff(calib[k:k+WINDOW_SIZE])**2)) 
                     for k in range(0, len(calib)-WINDOW_SIZE, WINDOW_SIZE)]
    patient_baseline = np.mean(baseline_vals)

    i = 0
    while i < len(rr_intervals) - WINDOW_SIZE:
        window = rr_intervals[i:i+WINDOW_SIZE]
        t = times[i+WINDOW_SIZE]
        
        is_seizure = 0
        for (start, end) in seizures_intervals:
            if (start - 30) <= t <= end:
                is_seizure = 1
                break
        
        diff = np.diff(window)
        rmssd = np.sqrt(np.mean(diff**2))
        
        rows.append({
            "rmssd_ratio": rmssd / patient_baseline,
            "slope": window[-1] - window[0],
            "rmssd_raw": rmssd,
            "mean_rr": np.mean(window),
            "label": is_seizure
        })
        
        if is_seizure: i += 1 # Overlap max pour Crises
        else: i += 20         # Pas d'overlap pour Normal
        
    return rows

def save_csv(df, filename):
    if df.empty: return
    # Simulation timestamp 10ms
    df = df.copy()
    df.insert(0, 'timestamp', range(0, len(df) * 10, 10))
    
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
        
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"   -> {filename} : {len(df)} lignes")

# ==========================================
# 3. MAIN AVEC EQUILIBRAGE
# ==========================================

def main():
    print("--- GENERATION CSV EQUILIBRE (50/50) ---")
    
    all_rows = []
    for name, config in FILES_CONFIG.items():
        print(f"Extraction : {name}")
        rr, times = get_ecg_data(config["path"])
        if rr is None: continue
        rows = extract_features(rr, times, config["seizures"])
        all_rows.extend(rows)

    df_full = pd.DataFrame(all_rows)
    
    # Separation initiale
    df_seizure = df_full[df_full['label'] == 1]
    df_normal_all = df_full[df_full['label'] == 0]
    
    count_seizure = len(df_seizure)
    count_normal = len(df_normal_all)
    
    print(f"\n[AVANT EQUILIBRAGE]")
    print(f"   - Crises : {count_seizure}")
    print(f"   - Normal : {count_normal} (Trop nombreux!)")
    
    # --- EQUILIBRAGE FORCE (UNDERSAMPLING) ---
    # On ne garde que le meme nombre de 'Normal' que de 'Crise'
    # On ajoute un petit facteur x1.2 pour avoir un peu plus de normal que de crise, mais pas trop
    target_normal = int(count_seizure * 1.2) 
    
    if count_normal > target_normal:
        print(f"\n[ACTION] Suppression de l'excedent de donnees normales...")
        df_normal_balanced = df_normal_all.sample(n=target_normal, random_state=42)
    else:
        df_normal_balanced = df_normal_all

    print(f"[APRES EQUILIBRAGE]")
    print(f"   - Crises : {len(df_seizure)}")
    print(f"   - Normal : {len(df_normal_balanced)}")
    
    # Split Train/Test
    train_s, test_s = train_test_split(df_seizure, test_size=0.2, random_state=42)
    train_n, test_n = train_test_split(df_normal_balanced, test_size=0.2, random_state=42)
    
    print("\nEcriture des fichiers CSV...")
    save_csv(train_s, "seizure_training.csv")
    save_csv(test_s,  "seizure_testing.csv")
    save_csv(train_n, "normal_training.csv")
    save_csv(test_n,  "normal_testing.csv")
    
    print(f"\n[SUCCES] Importez ces nouveaux fichiers sur Edge Impulse.")

if __name__ == "__main__":
    main()