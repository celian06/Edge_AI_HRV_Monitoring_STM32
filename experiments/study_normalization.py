import pandas as pd
import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# On remonte d'un cran pour trouver data
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Liste des fichiers REPOS à comparer
PATIENTS = {
    "chb04 (Enfant)": "chb04_08.edf",
    "chb01 (Adulte)": "chb01_03.edf", 
    "chb10 (Autre)": "chb10_01.edf"
}

def get_rr_intervals(filename, limit_sec=300):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Manquant : {path}")
        return []
    
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper()), None)
        if not ecg_ch: return []
        
        raw.pick_channels([ecg_ch])
        raw.crop(tmax=limit_sec)
        raw.load_data()
        # Filtre passe-bande standard pour ECG
        raw.filter(1.0, 45.0, verbose=False)
        
        sig = raw.get_data()[0] * 1e6
        peaks, _ = find_peaks(sig, distance=int(0.3*raw.info['sfreq']), prominence=np.percentile(sig, 75))
        rr = (np.diff(peaks) / raw.info['sfreq']) * 1000
        # Filtre physiologique large
        return rr[(rr > 300) & (rr < 1500)]
    except Exception as e:
        print(f"Erreur {filename}: {e}")
        return []

def main():
    print("--- ETUDE DE NORMALISATION MULTI-PATIENTS ---")
    data = []
    
    # 1. Collecte des données brutes
    for name, file in PATIENTS.items():
        print(f"Traitement de {name}...")
        rr = get_rr_intervals(file)
        if len(rr) > 0:
            # Calcul RMSSD glissant (fenêtre 20)
            for i in range(0, len(rr)-20, 20):
                win = rr[i:i+20]
                diff = np.diff(win)
                rmssd = np.sqrt(np.mean(diff**2))
                data.append({'Patient': name, 'RMSSD_Brut': rmssd})

    df = pd.DataFrame(data)
    if df.empty:
        print("Erreur: Aucune donnee extraite. Verifiez le dossier data/.")
        return

    # 2. Calcul des Baselines (Moyenne de repos par patient)
    baselines = df.groupby('Patient')['RMSSD_Brut'].mean().to_dict()
    print("\nBASELINES (Moyenne Repos) :")
    for p, val in baselines.items():
        print(f"   - {p} : {val:.1f} ms")

    # 3. Normalisation (La Clé de la Généralisation)
    # Formule : Valeur / Moyenne_Repos_Patient
    df['Ratio_Normalise'] = df.apply(lambda row: row['RMSSD_Brut'] / baselines[row['Patient']], axis=1)

    # 4. Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique A : Valeurs Brutes (Le problème)
    sns.boxplot(data=df, x='Patient', y='RMSSD_Brut', ax=axes[0], hue='Patient')
    axes[0].set_title("A. Valeurs Brutes (Impossible a generaliser)")
    axes[0].set_ylabel("RMSSD (ms)")
    # Seuil de votre modele V4 actuel
    axes[0].axhline(50, color='red', linestyle='--', label='Seuil V4 (50ms)')
    axes[0].legend()

    # Graphique B : Valeurs Normalisees (La solution)
    sns.boxplot(data=df, x='Patient', y='Ratio_Normalise', ax=axes[1], hue='Patient')
    axes[1].set_title("B. Valeurs Normalisees (Ratio / Repos)")
    axes[1].set_ylabel("Ratio (1.0 = Repos)")
    axes[1].axhline(1.0, color='green', linestyle='--', label='Reference Repos (1.0)')
    axes[1].legend()

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'generalisation_proof.png')
    plt.savefig(output_path)
    print(f"\nGraphique sauvegarde dans : {output_path}")
    plt.show()

if __name__ == "__main__":
    main()