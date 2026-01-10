import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import shutil
import random

# --- CONFIGURATION STRICTE ---
OUTPUT_DIR = 'edge_impulse_filtered_split'
# On garde uniquement les "bons" patients
SELECTED_PATIENTS = ['sz01', 'sz02', 'sz03', 'sz04', 'sz05']
TEST_RATIO = 0.2  # 20% des données iront dans le dossier TEST

WINDOW_DURATION = 60      
STRIDE = 2                
MAX_AUGMENT = 20          

# --- UTILITAIRES (Inchangés) ---
def get_project_root():
    current_script_path = os.path.abspath(__file__)
    root = current_script_path
    while os.path.basename(root) != 'IA_CareLink':
        parent = os.path.dirname(root)
        if parent == root: raise FileNotFoundError("Racine introuvable")
        root = parent
    return root

def parse_time(time_str):
    try: return float(time_str)
    except: pass
    try:
        parts = [float(p) for p in time_str.split(':')]
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2: return parts[0]*60 + parts[1]
    except: pass
    return None

def get_seizure_intervals(patient_id, root_path):
    times_path = os.path.join(root_path, 'data', 'szdb', 'times.seize')
    intervals = []
    if not os.path.exists(times_path): return []
    with open(times_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3 and parts[0] == patient_id:
                start = parse_time(parts[1])
                end = parse_time(parts[2])
                if start and end: intervals.append((start, end))
    return intervals

def calculate_rolling_rmssd_series(ecg_signal, fs, duration_sec):
    series = []
    ecg_centered = ecg_signal - np.mean(ecg_signal)
    if len(ecg_centered) == 0: return []
    threshold = np.max(ecg_centered) * 0.6
    peaks, _ = find_peaks(ecg_centered, height=threshold, distance=fs*0.3)
    if len(peaks) < 5: return [None] * duration_sec
    peak_times = peaks / fs 
    rr_intervals_ms = np.diff(peak_times) * 1000
    rr_times = peak_times[1:]
    for t in range(0, duration_sec):
        mask = (rr_times >= t) & (rr_times < (t + 30))
        window_rrs = rr_intervals_ms[mask]
        val = None
        if len(window_rrs) > 3:
            diff_rr = np.diff(window_rrs)
            diff_rr = diff_rr[np.abs(diff_rr) < 600] 
            if len(diff_rr) > 2:
                val = np.sqrt(np.mean(diff_rr ** 2))
        series.append(val)
    return series

def is_clean_signal(series, is_crisis=False):
    valid = [x for x in series if x is not None]
    if len(valid) < 40: return False
    arr = np.array(valid)
    if np.max(arr) > 300: return False
    if is_crisis:
        if np.std(arr) > 25: return False 
    return True

def generate_csv_file(folder, filename, rmssd_series):
    filepath = os.path.join(folder, filename)
    try:
        with open(filepath, 'w') as f:
            f.write("timestamp,rmssd_val\n")
            for i, val in enumerate(rmssd_series):
                if val is not None:
                    ts = i * 1000 
                    f.write(f"{ts},{val:.2f}\n")
        return True
    except: return False

def process_and_export():
    root = get_project_root()
    base_out = os.path.join(root, 'data', OUTPUT_DIR)
    
    # Création de l'arborescence complète
    if os.path.exists(base_out): shutil.rmtree(base_out)
    
    # On crée train/crise, train/normal, test/crise, test/normal
    for split in ['train', 'test']:
        for label in ['crise', 'normal']:
            os.makedirs(os.path.join(base_out, split, label))
            
    stats = {'train_c': 0, 'train_n': 0, 'test_c': 0, 'test_n': 0}

    print(f"Génération SPLIT MANUAL (Patients {SELECTED_PATIENTS}) en cours...")

    for patient_id in SELECTED_PATIENTS:
        print(f"  Patient {patient_id}...", end=" ")
        rec_path = os.path.join(root, 'data', 'szdb', patient_id, patient_id)
        if not os.path.exists(rec_path + '.dat'): continue

        try:
            record = wfdb.rdrecord(rec_path)
            full_signal = record.p_signal[:, 0]
            fs = record.fs
        except: continue
            
        seizures = get_seizure_intervals(patient_id, root)

        # --- CRISE ---
        c_count = 0
        for i, (start, end) in enumerate(seizures):
            for k in range(MAX_AUGMENT):
                offset = k * STRIDE
                extract_end = start - offset
                extract_start = extract_end - 90
                
                if extract_start < 0 or offset > 300: break 

                chunk = full_signal[int(extract_start*fs) : int(extract_end*fs)]
                full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
                target_series = full_series[-60:]
                
                if is_clean_signal(target_series, is_crisis=True):
                    # DÉCISION DU SPLIT ICI
                    is_test = random.random() < TEST_RATIO
                    split_dir = 'test' if is_test else 'train'
                    
                    filename = f"crise.{patient_id}.s{i+1}.aug{k}.csv"
                    folder = os.path.join(base_out, split_dir, 'crise')
                    
                    if generate_csv_file(folder, filename, target_series):
                        c_count += 1
                        stats[f'{split_dir}_c'] += 1
        
        print(f"-> {c_count} Crises.", end=" ")

        # --- NORMAL ---
        n_count = 0
        attempts = 0
        target_n = max(c_count, 5)
        duration_total = len(full_signal)/fs
        
        while n_count < target_n and attempts < 300:
            attempts += 1
            rand_t = random.uniform(300, duration_total - 100)
            safe = True
            for (s, e) in seizures:
                if (s - 600) < rand_t < (e + 600): safe = False
            
            if safe:
                chunk = full_signal[int((rand_t-30)*fs) : int((rand_t+60)*fs)]
                full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
                target_series = full_series[-60:]
                
                if is_clean_signal(target_series, is_crisis=False):
                    # DÉCISION DU SPLIT ICI AUSSI
                    is_test = random.random() < TEST_RATIO
                    split_dir = 'test' if is_test else 'train'

                    filename = f"normal.{patient_id}.n{n_count+1}.csv"
                    folder = os.path.join(base_out, split_dir, 'normal')
                    
                    if generate_csv_file(folder, filename, target_series):
                        n_count += 1
                        stats[f'{split_dir}_n'] += 1
        
        print(f"-> {n_count} Normaux.")

    print(f"\n--- TERMINÉ ---")
    print(f"TRAIN -> Crise: {stats['train_c']}, Normal: {stats['train_n']}")
    print(f"TEST  -> Crise: {stats['test_c']}, Normal: {stats['test_n']}")
    print(f"Total : {sum(stats.values())}")

if __name__ == "__main__":
    process_and_export()