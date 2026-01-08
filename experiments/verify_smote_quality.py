import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. CONFIGURATION (Meme que V5)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

FILES_CONFIG = {
    "chb04": {"path": os.path.join("chb04", "chb04_08.edf"), "seizures": [(1440, 1470)]},
    "chb12": {"path": os.path.join("chb12", "chb12_28.edf"), "seizures": [(181, 215)]},
    "chb13": {"path": os.path.join("chb13", "chb13_04.edf"), "seizures": []}
}

WINDOW_SIZE = 20
BASELINE_DURATION = 120

# ==========================================
# 2. LOGIQUE D'EXTRACTION (Identique V5)
# ==========================================

def get_data_and_features():
    print("--- CHARGEMENT DES DONNEES REELLES ---")
    all_features = []
    all_labels = []

    for name, config in FILES_CONFIG.items():
        path = os.path.join(DATA_DIR, config["path"])
        if not os.path.exists(path): continue
        
        # Lecture
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper() or "EKG" in c.upper()), None)
        if not ecg_ch: continue
        
        raw.pick_channels([ecg_ch])
        raw.load_data()
        raw.filter(1.0, 45.0, verbose=False)
        sig = raw.get_data()[0] * 1e6
        peaks, _ = find_peaks(sig, distance=int(0.3*raw.info['sfreq']), prominence=np.percentile(sig, 75))
        rr = (np.diff(peaks) / raw.info['sfreq']) * 1000
        times = peaks[1:] / raw.info['sfreq']
        
        # Calibration
        calib = rr[times < BASELINE_DURATION]
        if len(calib) < WINDOW_SIZE: continue
        baseline = np.mean([np.sqrt(np.mean(np.diff(calib[i:i+WINDOW_SIZE])**2)) for i in range(0, len(calib)-WINDOW_SIZE, WINDOW_SIZE)])
        
        # Extraction Augmentée
        i = 0
        while i < len(rr) - WINDOW_SIZE:
            win = rr[i:i+WINDOW_SIZE]
            t = times[i+WINDOW_SIZE]
            
            is_seizure = 0
            for s, e in config["seizures"]:
                if (s-30) <= t <= e: is_seizure = 1
            
            diff = np.diff(win)
            feats = {
                "RMSSD_Raw": np.sqrt(np.mean(diff**2)),
                "RMSSD_Ratio": np.sqrt(np.mean(diff**2)) / baseline,
                "Mean_RR": np.mean(win),
                "Std_RR": np.std(win),
                "Slope": win[-1] - win[0]
            }
            
            all_features.append(feats)
            all_labels.append(is_seizure)
            
            if is_seizure: i += 1 # Overlap max
            else: i += 20
            
    return pd.DataFrame(all_features), np.array(all_labels)

# ==========================================
# 3. VERIFICATION VISUELLE
# ==========================================

def main():
    # 1. Recuperer les donnees
    X, y = get_data_and_features()
    
    print(f"Donnees Reelles : {len(y)}")
    print(f" - Crises Reelles : {np.sum(y)}")
    print(f" - Normal Reel : {len(y) - np.sum(y)}")

    # 2. Appliquer SMOTE
    print("\nGeneration SMOTE en cours...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # On identifie les index qui ont ete ajoutes
    num_original = len(X)
    X_synthetic = X_res.iloc[num_original:]
    y_synthetic = y_res[num_original:]
    
    print(f"Donnees Synthetiques generees : {len(X_synthetic)}")

    # 3. Projection PCA (2D) pour visualiser
    print("Calcul PCA (Projection 2D)...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_res)
    
    # Creation d'un DataFrame pour le plot
    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    
    # On cree une colonne 'Type'
    types = []
    for i in range(len(y_res)):
        if i < num_original:
            if y_res[i] == 0: types.append("Normal (Reel)")
            else: types.append("Crise (Reelle)")
        else:
            types.append("Crise (SMOTE/Fausse)")
            
    df_plot['Type'] = types

    # 4. Affichage
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot, x='PC1', y='PC2', hue='Type', 
        style='Type', alpha=0.7, palette={'Normal (Reel)': 'blue', 'Crise (Reelle)': 'red', 'Crise (SMOTE/Fausse)': 'lime'}
    )
    
    plt.title("Vérification de la cohérence SMOTE (Projection PCA)")
    plt.xlabel("Composante Principale 1 (Variance max)")
    plt.ylabel("Composante Principale 2")
    
    output = os.path.join(RESULTS_DIR, 'smote_quality_check.png')
    plt.savefig(output)
    print(f"\n[SUCCES] Image de verification sauvegardee : {output}")
    print("Ouvrez l'image : Si les points verts (SMOTE) sont colles aux rouges (Crises), c'est bon !")

if __name__ == "__main__":
    main()