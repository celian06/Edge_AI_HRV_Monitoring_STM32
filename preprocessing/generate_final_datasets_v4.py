import pandas as pd
import numpy as np
import os
import sys
import re
import mne
from scipy.signal import find_peaks

# ==========================================
# 1. CONFIGURATION
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data'))
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'datasets_edgeimpulse')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

SUMMARY_FILE = os.path.join(DATA_DIR, "chb04-summary.txt")
EDF_FILES_NORMAL = ["chb04_08.edf", "chb04_28.edf"]

# Paramètres
WINDOW_SIZE = 20    
STEP_SIZE_NORM = 10 
STEP_SIZE_CRISIS = 2 
TEST_RATIO = 0.2    

# --- PARAMÈTRES DU STRESS TEST (NOUVEAU) ---
TEST_AUGMENTATION_FACTOR = 20  # On multiplie la taille du test par 20
TEST_NOISE_LEVEL = 0.15        # 15% de bruit (C'est violent !)

# --- SEUILS V4 ---
THRESH_CRISIS_MAX = 50.0   
THRESH_SPORT_MIN  = 90.0   
THRESH_NORMAL_MIN = 110.0  
THRESH_NORMAL_MAX = 300.0  

# ==========================================
# 2. FONCTIONS
# ==========================================

def check_source_files():
    print(f"Vérification des données dans : {DATA_DIR}")
    missing = []
    if not os.path.exists(SUMMARY_FILE): missing.append("chb04-summary.txt")
    for f in EDF_FILES_NORMAL:
        if not os.path.exists(os.path.join(DATA_DIR, f)): missing.append(f)
    if missing:
        print(f"\nERREUR : Manque {missing}")
        sys.exit(1)
    print("Sources OK.\n")

def calculate_features(rr_series, context='unknown'):
    diff = np.diff(rr_series)
    rmssd = np.sqrt(np.mean(diff**2))
    sdnn = np.std(rr_series)
    if context == 'normal': ratio = np.random.normal(1.5, 0.5)
    elif context == 'crisis': ratio = np.random.uniform(2.0, 8.0)
    else: ratio = 1.0
    return rmssd, sdnn, max(0.1, ratio)

def extract_rr_from_edf(edf_path, start_s=0, end_s=None):
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper() or "EKG" in c.upper()), None)
        if not ecg_ch: return np.array([])
        raw.pick_channels([ecg_ch])
        if end_s:
            max_dur = raw.times[-1]
            raw.crop(tmin=start_s, tmax=min(end_s, max_dur))
        raw.load_data()
        raw.filter(1.0, 45.0, method='iir', verbose=False)
        sig = raw.get_data()[0] * 1e6
        peaks, _ = find_peaks(sig, distance=int(0.3*raw.info['sfreq']), prominence=np.percentile(sig, 75))
        rr_ms = (np.diff(peaks) / raw.info['sfreq']) * 1000
        return rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
    except Exception: return np.array([])

def split_dataframe(df, ratio=0.2):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * ratio)
    if split_idx == 0 and len(df) > 1: split_idx = 1
    return df.iloc[split_idx:], df.iloc[:split_idx]

def reset_timestamps(df):
    if df.empty: return df
    df['timestamp'] = np.arange(len(df)) * 1000
    return df

# --- NOUVELLE FONCTION : TORTURE TEST ---
def generate_stress_test_data(df_pure):
    """Prend les vraies crises et génère des variantes bruitées/difficiles."""
    print(f"   Génération STRESS TEST (x{TEST_AUGMENTATION_FACTOR}, Bruit {TEST_NOISE_LEVEL*100}%)...")
    
    recs = df_pure.to_dict('records')
    stressed_data = []
    
    # On ajoute d'abord les originaux (pour garder la vérité terrain)
    stressed_data.extend(recs)
    
    for _ in range(TEST_AUGMENTATION_FACTOR):
        for row in recs:
            # Bruit aléatoire agressif
            noise_rmssd = np.random.normal(1, TEST_NOISE_LEVEL)
            noise_sdnn = np.random.normal(1, TEST_NOISE_LEVEL)
            
            new_rmssd = row['rmssd'] * noise_rmssd
            new_sdnn = row['sdnn'] * noise_sdnn
            
            # LF/HF complètement chaotique (pour tester si le modèle ignore bien ce paramètre en crise)
            new_ratio = row['lf_hf_ratio'] * np.random.uniform(0.5, 2.5)
            
            # FILTRE DE RÉALISME :
            # Même bruité, cela doit rester une crise pour être un test valide.
            # Si le bruit fait remonter le RMSSD > 50ms, ce n'est plus une crise, 
            # donc on ne peut pas demander à l'IA de dire "Crise". On jette.
            if new_rmssd < THRESH_CRISIS_MAX:
                clone = row.copy()
                clone['rmssd'] = new_rmssd
                clone['sdnn'] = new_sdnn
                clone['lf_hf_ratio'] = new_ratio
                stressed_data.append(clone)
                
    return pd.DataFrame(stressed_data)

# ==========================================
# 3. EXTRACTION
# ==========================================

def get_data_sport():
    print(f"   [1/3] Génération SPORT...")
    SAMPLES = 1500
    rmssd = np.random.uniform(THRESH_SPORT_MIN, 110.0, SAMPLES)
    sdnn = rmssd * np.random.uniform(0.6, 0.9, SAMPLES)
    intensity = (110.0 - rmssd) / (110.0 - THRESH_SPORT_MIN)
    ratio = 2.0 + (intensity * 6.0) + np.random.normal(0, 0.5, SAMPLES)
    ratio = np.clip(ratio, 0.5, 12.0)
    return pd.DataFrame({'timestamp': 0, 'rmssd': rmssd, 'sdnn': sdnn, 'lf_hf_ratio': ratio})

def get_data_normal():
    print("   [2/3] Extraction NORMAL...")
    data = []
    for f in EDF_FILES_NORMAL:
        path = os.path.join(DATA_DIR, f)
        rr = extract_rr_from_edf(path)
        for i in range(0, len(rr) - WINDOW_SIZE, STEP_SIZE_NORM):
            win = rr[i : i+WINDOW_SIZE]
            rmssd, sdnn, ratio = calculate_features(win, context='normal')
            if THRESH_NORMAL_MIN < rmssd < THRESH_NORMAL_MAX:
                data.append({'timestamp': 0, 'rmssd': rmssd, 'sdnn': sdnn, 'lf_hf_ratio': ratio})
    return pd.DataFrame(data)

def get_data_crisis():
    print("   [3/3] Extraction CRISE...")
    seizures = []
    with open(SUMMARY_FILE, 'r') as f:
        cf = None
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"): cf = line.split(":")[1].strip()
            if "Start Time" in line: start = int(re.search(r'\d+', line.split(":")[1]).group())
            if "End Time" in line:
                end = int(re.search(r'\d+', line.split(":")[1]).group())
                seizures.append({'f': cf, 's': start, 'e': end})
    data = []
    for s in seizures:
        path = os.path.join(DATA_DIR, s['f'])
        if not os.path.exists(path): continue
        rr = extract_rr_from_edf(path, start_s=max(0, s['s']-10), end_s=s['e']+10)
        for i in range(0, len(rr) - WINDOW_SIZE, STEP_SIZE_CRISIS):
            win = rr[i : i+WINDOW_SIZE]
            rmssd, sdnn, ratio = calculate_features(win, context='crisis')
            if rmssd < THRESH_CRISIS_MAX:
                data.append({'timestamp': 0, 'rmssd': rmssd, 'sdnn': sdnn, 'lf_hf_ratio': ratio})
    return pd.DataFrame(data)

# ==========================================
# 4. MAIN
# ==========================================

def main():
    print("--- DÉMARRAGE V4 (STRESS TEST MODE) ---")
    check_source_files()
    
    df_s, df_n, df_c = get_data_sport(), get_data_normal(), get_data_crisis()
    
    if df_c.empty: return print("Pas de crise !")

    # Split
    tr_s, te_s = split_dataframe(df_s, TEST_RATIO)
    tr_n, te_n = split_dataframe(df_n, TEST_RATIO)
    tr_c, te_c = split_dataframe(df_c, TEST_RATIO) # te_c contient les vraies crises pures
    
    # --- MODIFICATION ICI : AUGMENTATION DU TEST ---
    te_c_stress = generate_stress_test_data(te_c)
    
    # Save TEST
    reset_timestamps(te_c_stress).to_csv(os.path.join(OUTPUT_DIR, 'TEST_Crisis_V4.csv'), index=False)
    reset_timestamps(te_n).to_csv(os.path.join(OUTPUT_DIR, 'TEST_Normal_V4.csv'), index=False)
    reset_timestamps(te_s).to_csv(os.path.join(OUTPUT_DIR, 'TEST_Sport_V4.csv'), index=False)
    
    print(f"TEST FILES OK (Dont {len(te_c_stress)} crises pour le stress test).")

    # Balance TRAIN
    target = len(tr_n) + len(tr_s)
    needed = target - len(tr_c)
    final_c = [tr_c]
    if needed > 0:
        recs = tr_c.to_dict('records')
        gen = []
        while len(gen) < needed:
            samp = recs[np.random.randint(0, len(recs))]
            scale = np.random.uniform(0.9, 1.1)
            noise = np.random.normal(1, 0.05)
            new_rmssd = samp['rmssd'] * scale * noise
            if new_rmssd < THRESH_CRISIS_MAX:
                clone = samp.copy()
                clone['rmssd'] = new_rmssd
                clone['sdnn'] = samp['sdnn'] * scale * noise
                clone['lf_hf_ratio'] = samp['lf_hf_ratio'] * np.random.normal(1, 0.1)
                gen.append(clone)
        final_c.append(pd.DataFrame(gen))
    
    # Save TRAIN
    reset_timestamps(tr_s).to_csv(os.path.join(OUTPUT_DIR, 'TRAIN_Sport_V4.csv'), index=False)
    reset_timestamps(tr_n).to_csv(os.path.join(OUTPUT_DIR, 'TRAIN_Normal_V4.csv'), index=False)
    reset_timestamps(pd.concat(final_c)).to_csv(os.path.join(OUTPUT_DIR, 'TRAIN_Crisis_V4.csv'), index=False)
    
    print("TERMINÉ.")

if __name__ == "__main__":
    main()