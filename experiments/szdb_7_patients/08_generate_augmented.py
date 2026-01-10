import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import shutil
import random

# --- CONFIGURATION AUGMENTÉE ---
OUTPUT_DIR = 'edge_impulse_dataset'
TRAIN_PATIENTS = ['sz01', 'sz02', 'sz03', 'sz04', 'sz05']
TEST_PATIENTS = ['sz06', 'sz07']

WINDOW_DURATION = 60      # 60 secondes par fichier
STRIDE = 5                # On décale la fenêtre de 5 secondes pour créer de nouveaux exemples
MAX_AUGMENT_PER_SEIZURE = 10 # On extrait jusqu'à 10 fenêtres par crise

# --- UTILITAIRES ---
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
        # Fenêtre glissante de 30s
        mask = (rr_times >= t) & (rr_times < (t + 30))
        window_rrs = rr_intervals_ms[mask]
        
        val = None
        if len(window_rrs) > 3:
            diff_rr = np.diff(window_rrs)
            diff_rr = diff_rr[np.abs(diff_rr) < 1000] 
            if len(diff_rr) > 2:
                val = np.sqrt(np.mean(diff_rr ** 2))
        series.append(val)
    return series

def generate_csv_file(folder, filename, rmssd_series, baseline):
    filepath = os.path.join(folder, filename)
    try:
        with open(filepath, 'w') as f:
            f.write("timestamp,rmssd_ratio\n")
            count = 0
            for i, val in enumerate(rmssd_series):
                if val is not None and baseline > 0:
                    ratio = val / baseline
                    ts = i * 1000 
                    f.write(f"{ts},{ratio:.4f}\n")
                    count += 1
            if count < 30: # On rejette si moins de 30 points valides sur 60
                f.close()
                os.remove(filepath)
                return False
        return True
    except: return False

def process_and_export():
    root = get_project_root()
    base_out = os.path.join(root, 'data', OUTPUT_DIR)
    
    # On vide le dossier précédent pour repartir sur du propre
    if os.path.exists(base_out): shutil.rmtree(base_out)
    
    for split in ['train', 'test']:
        for label in ['crise', 'normal']:
            os.makedirs(os.path.join(base_out, split, label))
            
    all_patients = TRAIN_PATIENTS + TEST_PATIENTS
    total_crise = 0
    total_normal = 0

    print(f"Génération AUGMENTÉE en cours...")

    for patient_id in all_patients:
        print(f"  Patient {patient_id}...", end=" ")
        rec_path = os.path.join(root, 'data', 'szdb', patient_id, patient_id)
        if not os.path.exists(rec_path + '.dat'): 
            print("Fichier non trouvé.")
            continue

        try:
            record = wfdb.rdrecord(rec_path)
            full_signal = record.p_signal[:, 0]
            fs = record.fs
        except: continue
            
        seizures = get_seizure_intervals(patient_id, root)
        
        # Baseline
        baseline_chunk = full_signal[:300*fs]
        series_base = calculate_rolling_rmssd_series(baseline_chunk, fs, 270)
        valid_base = [x for x in series_base if x is not None]
        baseline_val = np.mean(valid_base) if valid_base else 40.0

        split_dir = 'train' if patient_id in TRAIN_PATIENTS else 'test'

        # --- GÉNÉRATION CRISE (AUGMENTÉE) ---
        patient_crise_count = 0
        for i, (start, end) in enumerate(seizures):
            # On génère plusieurs fenêtres décalées pour la même crise
            # On remonte jusqu'à 2 minutes avant si possible
            for k in range(MAX_AUGMENT_PER_SEIZURE):
                # Shift: k * 5 secondes en arrière
                offset = k * STRIDE
                extract_end = start - offset
                extract_start = extract_end - 90 # Buffer de calcul
                
                if extract_start < 0: break # On ne peut pas remonter avant le début
                
                # Vérif : ne pas aller trop loin du début de crise (max 3 min avant)
                if offset > 180: break 

                chunk = full_signal[int(extract_start*fs) : int(extract_end*fs)]
                full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
                target_series = full_series[-60:] # On garde les 60 dernières secondes du buffer
                
                filename = f"crise.{patient_id}.s{i+1}.aug{k}.csv"
                folder = os.path.join(base_out, split_dir, 'crise')
                if generate_csv_file(folder, filename, target_series, baseline_val):
                    patient_crise_count += 1
                    total_crise += 1
        
        print(f"-> {patient_crise_count} fichiers Crise.", end=" ")

        # --- GÉNÉRATION NORMAL (EQUILIBRÉE) ---
        # On essaie d'avoir autant de Normal que de Crise générés
        nb_normal_target = max(patient_crise_count, 5) # Au moins 5
        patient_normal_count = 0
        attempts = 0
        duration_total = len(full_signal)/fs
        
        while patient_normal_count < nb_normal_target and attempts < 200:
            attempts += 1
            rand_t = random.uniform(300, duration_total - 100)
            safe = True
            for (s, e) in seizures:
                # Zone exclusion large autour des crises
                if (s - 600) < rand_t < (e + 600): safe = False
            
            if safe:
                chunk = full_signal[int((rand_t-30)*fs) : int((rand_t+60)*fs)]
                full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
                target_series = full_series[-60:]
                
                filename = f"normal.{patient_id}.n{patient_normal_count+1}.csv"
                folder = os.path.join(base_out, split_dir, 'normal')
                if generate_csv_file(folder, filename, target_series, baseline_val):
                    patient_normal_count += 1
                    total_normal += 1
        
        print(f"-> {patient_normal_count} fichiers Normal.")

    print(f"\n--- TERMINÉ ---")
    print(f"Total Crise : {total_crise}")
    print(f"Total Normal : {total_normal}")
    print(f"Total Dataset : {total_crise + total_normal}")
    print("Tu peux maintenant ré-uploader ces dossiers sur Edge Impulse (pense à effacer les anciennes données sur le site avant) !")

if __name__ == "__main__":
    process_and_export()