import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import shutil
import random

# --- CONFIGURATION ---
OUTPUT_DIR = 'edge_impulse_dataset'

# Stratégie de séparation (5 Train / 2 Test)
TRAIN_PATIENTS = ['sz01', 'sz02', 'sz03', 'sz04', 'sz05']
TEST_PATIENTS = ['sz06', 'sz07']

# Paramètres de génération
WINDOW_DURATION = 60      # Durée de chaque fichier CSV (secondes)
SAMPLING_INTERVAL = 1.0   # Une valeur de RMSSD par seconde

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
    """Convertit HH:MM:SS ou secondes brutes en float."""
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
    
    if not os.path.exists(times_path):
        print(f"ERREUR: times.seize introuvable dans {times_path}")
        return []

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
    return intervals

def calculate_rolling_rmssd_series(ecg_signal, fs, duration_sec):
    """Calcule une série temporelle de RMSSD seconde par seconde."""
    series = []
    
    # 1. Centrage du signal (suppression offset DC)
    ecg_centered = ecg_signal - np.mean(ecg_signal)
    if len(ecg_centered) == 0: return []
    
    # 2. Seuil Dynamique (60% du max local)
    threshold = np.max(ecg_centered) * 0.6
    
    # 3. Détection des pics
    peaks, _ = find_peaks(ecg_centered, height=threshold, distance=fs*0.3)
    
    if len(peaks) < 5:
        return [None] * duration_sec
    
    # Conversion en temps et intervalles RR
    peak_times = peaks / fs 
    rr_intervals_ms = np.diff(peak_times) * 1000
    rr_times = peak_times[1:]
    
    # 4. Génération de la série (rolling window)
    for t in range(0, duration_sec):
        # On regarde [t, t+30s] pour calculer la valeur à l'instant t
        mask = (rr_times >= t) & (rr_times < (t + 30))
        window_rrs = rr_intervals_ms[mask]
        
        val = None
        if len(window_rrs) > 3:
            diff_rr = np.diff(window_rrs)
            # Filtrage des artefacts physiologiques (>1500ms ou <300ms)
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
                    # CALCUL DU RATIO (Feature Engineering)
                    ratio = val / baseline
                    ts = i * 1000 
                    f.write(f"{ts},{ratio:.4f}\n")
                    count += 1
            
            # Si le fichier est vide ou quasi-vide, on le supprime pour ne pas polluer l'IA
            if count < 10:
                f.close()
                os.remove(filepath)
                return False
        return True
    except Exception as e:
        print(f"Erreur écriture {filename}: {e}")
        return False

def process_and_export():
    root = get_project_root()
    base_out = os.path.join(root, 'data', OUTPUT_DIR)
    
    # Nettoyage préalable
    if os.path.exists(base_out): shutil.rmtree(base_out)
    
    # Création de l'arborescence Edge Impulse
    for split in ['train', 'test']:
        for label in ['crise', 'normal']:
            os.makedirs(os.path.join(base_out, split, label))
            
    all_patients = TRAIN_PATIENTS + TEST_PATIENTS
    
    print(f"Démarrage de la génération pour {len(all_patients)} patients...")
    print(f"Sortie : {base_out}")

    for patient_id in all_patients:
        print(f"\n--- Traitement {patient_id} ---")
        
        rec_path = os.path.join(root, 'data', 'szdb', patient_id, patient_id)
        
        # Vérification présence fichier
        if not os.path.exists(rec_path + '.dat'):
            print(f"Fichiers manquants pour {patient_id}, on passe.")
            continue

        try:
            record = wfdb.rdrecord(rec_path)
            full_signal = record.p_signal[:, 0]
            fs = record.fs
        except Exception as e:
            print(f"Erreur lecture signal: {e}")
            continue
            
        seizures = get_seizure_intervals(patient_id, root)
        if not seizures:
            print(f"Pas de crises définies pour {patient_id} (vérifier times.seize).")
            continue
        
        # 1. CALCUL DE LA BASELINE (Au repos)
        # On prend les 5 premières minutes (0-300s) pour établir la "norme" du patient
        baseline_chunk = full_signal[:300*fs]
        series_base = calculate_rolling_rmssd_series(baseline_chunk, fs, 270)
        valid_base = [x for x in series_base if x is not None]
        
        if valid_base:
            baseline_val = np.mean(valid_base)
            print(f"Baseline (Repos) : {baseline_val:.2f} ms")
        else:
            baseline_val = 40.0
            print("Baseline indéterminée, usage valeur par défaut (40.0 ms).")

        # Déterminer le dossier de destination (Train ou Test)
        split_dir = 'train' if patient_id in TRAIN_PATIENTS else 'test'

        # 2. GÉNÉRATION DES FICHIERS "CRISE"
        files_created_crise = 0
        for i, (start, end) in enumerate(seizures):
            # Fenêtre cible : [Start - 90s] à [Start]
            extract_start = start - 90
            extract_end = start
            
            if extract_start < 0: continue
            
            chunk = full_signal[int(extract_start*fs) : int(extract_end*fs)]
            full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
            
            # On ne garde que les 60 dernières secondes (les plus proches de la crise)
            target_series = full_series[-60:]
            
            filename = f"crise.{patient_id}.s{i+1}.csv"
            folder = os.path.join(base_out, split_dir, 'crise')
            if generate_csv_file(folder, filename, target_series, baseline_val):
                files_created_crise += 1
            
        print(f"-> {files_created_crise} fichiers 'Crise' générés.")

        # 3. GÉNÉRATION DES FICHIERS "NORMAL"
        # On génère un nombre équilibré de fichiers normaux
        nb_normal_target = files_created_crise + 2
        files_created_normal = 0
        attempts = 0
        duration_total = len(full_signal)/fs
        
        while files_created_normal < nb_normal_target and attempts < 200:
            attempts += 1
            # Tirage au sort d'un moment calme (loin des crises)
            rand_t = random.uniform(300, duration_total - 100)
            safe = True
            for (s, e) in seizures:
                if (s - 300) < rand_t < (e + 300): safe = False
            
            if safe:
                chunk = full_signal[int((rand_t-30)*fs) : int((rand_t+60)*fs)]
                full_series = calculate_rolling_rmssd_series(chunk, fs, 90)
                target_series = full_series[-60:]
                
                filename = f"normal.{patient_id}.n{files_created_normal+1}.csv"
                folder = os.path.join(base_out, split_dir, 'normal')
                if generate_csv_file(folder, filename, target_series, baseline_val):
                    files_created_normal += 1
        
        print(f"-> {files_created_normal} fichiers 'Normal' générés.")

if __name__ == "__main__":
    process_and_export()
    print("\n--- TERMINE ---")
    print(f"Les fichiers sont prêts dans : data/{OUTPUT_DIR}")