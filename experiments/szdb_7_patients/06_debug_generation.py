import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import shutil
import random

# --- CONFIGURATION ---
WINDOW_DURATION = 60      
OUTPUT_DIR = 'edge_impulse_dataset'
TRAIN_PATIENTS = ['sz01', 'sz02', 'sz03']
TEST_PATIENTS = ['sz04', 'sz05']

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
    """Version robuste pour HH:MM:SS et secondes brutes"""
    try: return float(time_str)
    except: pass
    try:
        parts = [float(p) for p in time_str.split(':')]
        if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2: return parts[0]*60 + parts[1]
    except Exception as e:
        print(f"DEBUG: Erreur parsing temps '{time_str}': {e}")
    return None

def get_seizure_intervals(patient_id, root_path):
    times_path = os.path.join(root_path, 'data', 'szdb', 'times.seize')
    intervals = []
    
    if not os.path.exists(times_path):
        print(f"ERREUR CRITIQUE: Fichier times.seize introuvable ici: {times_path}")
        return []

    print(f"DEBUG: Lecture times.seize pour {patient_id}...")
    with open(times_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            # On cherche l'ID exact
            if len(parts) >= 3 and parts[0] == patient_id:
                start = parse_time(parts[1])
                end = parse_time(parts[2])
                if start is not None and end is not None: 
                    intervals.append((start, end))
                else:
                    print(f"ATTENTION: Échec parsing ligne: {line.strip()}")
    
    print(f"DEBUG: {len(intervals)} crises trouvées pour {patient_id}")
    return intervals

def calculate_rolling_rmssd_series(ecg_signal, fs, duration_sec):
    series = []
    
    # Prétraitement
    ecg_centered = ecg_signal - np.mean(ecg_signal)
    if len(ecg_centered) == 0: return []
    
    threshold = np.max(ecg_centered) * 0.6
    peaks, _ = find_peaks(ecg_centered, height=threshold, distance=fs*0.3)
    
    if len(peaks) < 5:
        # Pas assez de pics dans ce chunk
        return [None] * duration_sec
    
    peak_times = peaks / fs 
    rr_intervals_ms = np.diff(peak_times) * 1000
    rr_times = peak_times[1:]
    
    for t in range(0, duration_sec):
        # On regarde [t, t+30s] pour calculer le point à l'instant t
        mask = (rr_times >= t) & (rr_times < (t + 30))
        window_rrs = rr_intervals_ms[mask]
        
        val = None
        if len(window_rrs) > 3:
            diff_rr = np.diff(window_rrs)
            diff_rr = diff_rr[np.abs(diff_rr) < 1000] # Filtre artefacts
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
        if count > 0:
            pass # print(f"  -> Fichier généré: {filename} ({count} lignes)")
        else:
            print(f"  -> ATTENTION: Fichier vide (pas de données valides) pour {filename}")
            os.remove(filepath) # On supprime le fichier vide
    except Exception as e:
        print(f"ERREUR écriture fichier {filename}: {e}")

def process_and_export():
    root = get_project_root()
    base_out = os.path.join(root, 'data', OUTPUT_DIR)
    
    if os.path.exists(base_out): shutil.rmtree(base_out)
    
    for split in ['train', 'test']:
        for label in ['crise', 'normal']:
            os.makedirs(os.path.join(base_out, split, label))
            
    all_patients = TRAIN_PATIENTS + TEST_PATIENTS
    
    for patient_id in all_patients:
        print(f"\n--- Traitement {patient_id} ---")
        
        rec_path = os.path.join(root, 'data', 'szdb', patient_id, patient_id)
        # VERIFICATION PATH
        if not os.path.exists(rec_path + '.dat'):
            print(f"ERREUR CRITIQUE: Fichier de données introuvable: {rec_path}.dat")
            continue

        try:
            record = wfdb.rdrecord(rec_path)
            full_signal = record.p_signal[:, 0]
            fs = record.fs
            print(f"Signal chargé. Longueur: {len(full_signal)} points, Fs={fs}Hz")
        except Exception as e:
            print(f"ERREUR lecture WFDB: {e}")
            continue
            
        seizures = get_seizure_intervals(patient_id, root)
        if len(seizures) == 0:
            print(f"PAS DE CRISES trouvées pour {patient_id}. Vérifie times.seize !")
            continue
        
        # 1. BASELINE
        baseline_chunk = full_signal[:300*fs]
        series_base = calculate_rolling_rmssd_series(baseline_chunk, fs, 270)
        valid_base = [x for x in series_base if x is not None]
        
        if valid_base:
            baseline_val = np.mean(valid_base)
            print(f"Baseline calculée: {baseline_val:.2f} ms")
        else:
            baseline_val = 40.0
            print("Echec calcul Baseline (signal bruité ?), usage valeur par défaut 40.0")

        # 2. GENERATION 'CRISE'
        files_created = 0
        for i, (start, end) in enumerate(seizures):
            extract_start = start - 90
            extract_end = start
            
            if extract_start < 0: 
                print(f"Crise {i+1} ignorée (trop proche du début).")
                continue
            
            chunk = full_signal[int(extract_start*fs) : int(extract_end*fs)]
            full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
            target_series = full_series[-60:]
            
            if all(v is None for v in target_series):
                 print(f"  -> Echec génération Crise {i+1}: Pas de RMSSD calculable sur ce segment.")
                 continue

            split_dir = 'train' if patient_id in TRAIN_PATIENTS else 'test'
            filename = f"crise.{patient_id}.s{i+1}.csv"
            folder = os.path.join(base_out, split_dir, 'crise')
            generate_csv_file(folder, filename, target_series, baseline_val)
            files_created += 1
            
        print(f"-> {files_created} fichiers 'Crise' générés.")

        # 3. GENERATION 'NORMAL'
        nb_normal = len(seizures) + 2
        duration_total = len(full_signal)/fs
        count = 0
        attempts = 0
        
        while count < nb_normal and attempts < 100:
            attempts += 1
            rand_t = random.uniform(300, duration_total - 100)
            safe = True
            for (s, e) in seizures:
                if (s - 300) < rand_t < (e + 300): safe = False
            
            if safe:
                chunk = full_signal[int((rand_t-30)*fs) : int((rand_t+60)*fs)]
                full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
                target_series = full_series[-60:]
                
                # Check si données valides
                valid_pts = [x for x in target_series if x is not None]
                if len(valid_pts) > 30: # Au moins 30s de signal propre
                    split_dir = 'train' if patient_id in TRAIN_PATIENTS else 'test'
                    filename = f"normal.{patient_id}.n{count+1}.csv"
                    folder = os.path.join(base_out, split_dir, 'normal')
                    generate_csv_file(folder, filename, target_series, baseline_val)
                    count += 1
        
        print(f"-> {count} fichiers 'Normal' générés.")

if __name__ == "__main__":
    process_and_export()