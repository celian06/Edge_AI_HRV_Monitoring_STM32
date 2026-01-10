import wfdb
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import os
import shutil
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- CONFIGURATION TOTALE ---
# On tente le coup avec TOUT LE MONDE
ALL_PATIENTS = ['sz01', 'sz02', 'sz03', 'sz04', 'sz05', 'sz06', 'sz07']
WINDOW_DURATION = 60      
MAX_AUGMENT = 10  # On reste raisonnable pour ne pas noyer le modèle

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

def calculate_stats_features(ecg_signal, fs):
    # C'est ici qu'on extrait les stats pour le Random Forest
    # 1. Calcul RMSSD Rolling
    ecg_centered = ecg_signal - np.mean(ecg_signal)
    if len(ecg_centered) == 0: return None
    threshold = np.max(ecg_centered) * 0.6
    peaks, _ = find_peaks(ecg_centered, height=threshold, distance=fs*0.3)
    
    if len(peaks) < 5: return None # Signal trop plat ou bruité
    
    peak_times = peaks / fs 
    rr_intervals_ms = np.diff(peak_times) * 1000
    
    # On calcule le RMSSD sur la fenêtre globale
    diff_rr = np.diff(rr_intervals_ms)
    diff_rr = diff_rr[np.abs(diff_rr) < 600] # Filtre artefacts
    
    if len(diff_rr) < 2: return None
    
    # --- LES FEATURES CLÉS POUR LE RANDOM FOREST ---
    # On calcule les stats sur le RMSSD brut (qui varie dans le temps)
    # Mais ici on a une fenêtre de 60s, donc on a une liste de RR.
    
    # RMSSD global de la fenêtre
    rmssd_val = np.sqrt(np.mean(diff_rr ** 2))
    # Ecart type des intervalles RR (SDNN)
    sdnn_val = np.std(rr_intervals_ms)
    # Moyenne des intervalles RR (Rythme cardiaque inverse)
    mean_rr = np.mean(rr_intervals_ms)
    
    return [rmssd_val, sdnn_val, mean_rr]

def prepare_dataset_in_memory():
    root = get_project_root()
    X = [] # Features
    y = [] # Labels (1=Crise, 0=Normal)
    
    print(f"Traitement des 7 patients pour Random Forest...")
    
    for patient_id in ALL_PATIENTS:
        print(f"  Patient {patient_id}...", end=" ")
        rec_path = os.path.join(root, 'data', 'szdb', patient_id, patient_id)
        if not os.path.exists(rec_path + '.dat'): continue

        try:
            record = wfdb.rdrecord(rec_path)
            full_signal = record.p_signal[:, 0]
            fs = record.fs
        except: continue
            
        seizures = get_seizure_intervals(patient_id, root)
        
        # --- CRISE (Label 1) ---
        c_count = 0
        for (start, end) in seizures:
            # On prend plusieurs fenêtres AVANT la crise (Pré-ictal)
            for k in range(MAX_AUGMENT):
                offset = k * 5 # Décalage de 5 secondes
                extract_end = start - offset
                extract_start = extract_end - 60
                
                if extract_start < 0: break
                
                chunk = full_signal[int(extract_start*fs) : int(extract_end*fs)]
                feats = calculate_stats_features(chunk, fs)
                
                if feats:
                    X.append(feats)
                    y.append(1)
                    c_count += 1
        
        # --- NORMAL (Label 0) ---
        # On prend autant de normal que de crise pour ce patient
        n_count = 0
        target_n = max(c_count, 10)
        attempts = 0
        duration_total = len(full_signal)/fs
        
        while n_count < target_n and attempts < 500:
            attempts += 1
            rand_t = random.uniform(300, duration_total - 100)
            safe = True
            for (s, e) in seizures:
                if (s - 600) < rand_t < (e + 600): safe = False # Loin des crises
            
            if safe:
                chunk = full_signal[int((rand_t-30)*fs) : int((rand_t+30)*fs)]
                feats = calculate_stats_features(chunk, fs)
                if feats:
                    X.append(feats)
                    y.append(0)
                    n_count += 1
                    
        print(f"-> {c_count} Crises / {n_count} Normaux.")

    return np.array(X), np.array(y)

if __name__ == "__main__":
    # 1. Préparation
    X, y = prepare_dataset_in_memory()
    print(f"\nDataset Total : {len(X)} echantillons")
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # 3. Entraînement Random Forest
    print("\nEntrainement du Random Forest sur les 7 patients...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # 4. Résultats
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n--- RESULTATS GLOBAUX (7 PATIENTS) ---")
    print(f"Precision (Accuracy) : {acc:.2%}")
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_test, preds))
    print("\nDetails :")
    print(classification_report(y_test, preds, target_names=['Normal', 'Crise']))
    
    if acc > 0.80:
        print("\nSUCCES : Le Random Forest gere bien les 7 patients !")
        print("Tu peux utiliser ces resultats pour ta soutenance sans chercher d'autre base.")
    else:
        print("\nPROBLEME : Même le Random Forest galere avec sz06/sz07.")