import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'experiments', 'results') # Pour sauver l'image

if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

FILES_CONFIG = {
    "chb04": {
        "path": os.path.join("chb04", "chb04_08.edf"),
        "seizures": [(1440, 1470)]
    },
    "chb12": {
        "path": os.path.join("chb12", "chb12_28.edf"),
        "seizures": [(181, 215)]
    },
    "chb13": {
        "path": os.path.join("chb13", "chb13_04.edf"),
        "seizures": []
    }
}

WINDOW_SIZE = 20        
BASELINE_DURATION = 120 

# ==========================================
# 2. EXTRACTION ET AUGMENTATION
# ==========================================

def get_ecg_data(filepath):
    """Charge le fichier EDF et extrait les intervalles RR"""
    full_path = os.path.join(DATA_DIR, filepath)
    if not os.path.exists(full_path):
        print(f"[ERREUR] Fichier introuvable : {full_path}")
        return None, None

    try:
        raw = mne.io.read_raw_edf(full_path, preload=False, verbose=False)
        ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper() or "EKG" in c.upper()), None)
        
        if not ecg_ch:
            return None, None

        raw.pick_channels([ecg_ch])
        raw.load_data()
        raw.filter(1.0, 45.0, verbose=False)
        
        sig = raw.get_data()[0] * 1e6
        peaks, _ = find_peaks(sig, distance=int(0.3*raw.info['sfreq']), prominence=np.percentile(sig, 75))
        
        rr_intervals = (np.diff(peaks) / raw.info['sfreq']) * 1000
        peak_times = peaks[1:] / raw.info['sfreq'] 
        
        mask = (rr_intervals > 300) & (rr_intervals < 1500)
        return rr_intervals[mask], peak_times[mask]

    except Exception as e:
        print(f"[CRASH] {filepath}: {e}")
        return None, None

def extract_features_augmented(rr_intervals, times, seizures_intervals):
    """Extraction avec fenetrage glissant dynamique (Augmentation)"""
    features = []
    labels = []
    
    # 1. Calibration (Baseline)
    calibration_rr = rr_intervals[times < BASELINE_DURATION]
    baseline_rmssds = []
    if len(calibration_rr) > WINDOW_SIZE:
        for i in range(0, len(calibration_rr) - WINDOW_SIZE, WINDOW_SIZE):
            win = calibration_rr[i:i+WINDOW_SIZE]
            val = np.sqrt(np.mean(np.diff(win)**2))
            baseline_rmssds.append(val)
            
    if not baseline_rmssds:
        return pd.DataFrame(), []

    patient_baseline = np.mean(baseline_rmssds)
    print(f"   -> Baseline : {patient_baseline:.2f} ms")

    # 2. Boucle dynamique (While)
    i = 0
    count_seizures = 0
    
    while i < len(rr_intervals) - WINDOW_SIZE:
        window = rr_intervals[i:i+WINDOW_SIZE]
        time_window = times[i+WINDOW_SIZE]
        
        # Etiquetage
        is_seizure = 0
        for (start, end) in seizures_intervals:
            # On inclut 30s avant la crise (Pre-ictal)
            if (start - 30) <= time_window <= end:
                is_seizure = 1
                break
        
        # Calculs
        diff = np.diff(window)
        rmssd_val = np.sqrt(np.mean(diff**2))
        slope = window[-1] - window[0]
        rmssd_ratio = rmssd_val / patient_baseline # Normalisation
        
        features.append({
            "RMSSD_Raw": rmssd_val,
            "RMSSD_Ratio": rmssd_ratio,
            "Mean_RR": np.mean(window),
            "Std_RR": np.std(window),
            "Slope": slope
        })
        labels.append(is_seizure)
        
        # --- LOGIQUE D'AUGMENTATION ---
        if is_seizure == 1:
            i += 1  # Avance de 1 seulement (Chevauchement maximal) -> X20 donnees
            count_seizures += 1
        else:
            i += 20 # Avance de 20 (Pas de chevauchement pour le repos)

    print(f"   -> Echantillons generes : {len(labels)} (dont {count_seizures} crises)")
    return pd.DataFrame(features), labels

# ==========================================
# 3. MAIN
# ==========================================

def main():
    print("--- ENTRAINEMENT V5 : AUGMENTATION & VISUALISATION ---")
    
    all_features = pd.DataFrame()
    all_labels = []
    
    # 1. Chargement
    for name, config in FILES_CONFIG.items():
        print(f"\nTraitement : {name}")
        rr, times = get_ecg_data(config["path"])
        if rr is None: continue
            
        X_p, y_p = extract_features_augmented(rr, times, config["seizures"])
        
        if not X_p.empty:
            all_features = pd.concat([all_features, X_p], ignore_index=True)
            all_labels.extend(y_p)

    # 2. Preparation
    X = all_features
    y = np.array(all_labels)
    
    print(f"\n[TOTAL] : {len(y)} echantillons")
    print(f"   - Repos : {len(y) - np.sum(y)}")
    print(f"   - Crises : {np.sum(y)} (Augmente !)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 3. SMOTE (Equilibrage final)
    print("Application du SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # 4. Entrainement
    print("Entrainement Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_res, y_train_res)
    
    # 5. Evaluation Visuelle
    y_pred = clf.predict(X_test)
    
    print("\n--- RAPPORT TEXTUEL ---")
    print(classification_report(y_test, y_pred))

    # --- GENERATION MATRICE DE CONFUSION VISUELLE ---
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Crise'],
                yticklabels=['Normal', 'Crise'])
    
    plt.title('Matrice de Confusion V5 (Normalise + Augmente)')
    plt.xlabel('Prediction de l\'IA')
    plt.ylabel('Realite Medicale')
    
    output_img = os.path.join(RESULTS_DIR, 'confusion_matrix_v5.png')
    plt.savefig(output_img)
    print(f"\n[IMAGE] Matrice sauvegardee : {output_img}")
    
    # 6. Sauvegarde Modele
    path = os.path.join(MODELS_DIR, 'rf_model_v5_augmented.pkl')
    joblib.dump(clf, path)
    print(f"Modele sauvegarde : {path}")

if __name__ == "__main__":
    main()