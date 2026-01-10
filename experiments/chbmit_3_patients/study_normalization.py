import pandas as pd
import numpy as np
import os
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

# ==========================================
# 1. CONFIGURATION ET CHEMINS
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# LISTE MISE A JOUR : Patients avec ECG verifie manuellement
PATIENTS = {
    "chb04 (Ref)":       os.path.join("chb04", "chb04_08.edf"),
    "chb12 (Patient A)": os.path.join("chb12", "chb12_28.edf"), 
    "chb13 (Patient B)": os.path.join("chb13", "chb13_04.edf")
}

# ==========================================
# 2. EXTRACTION DU SIGNAL
# ==========================================

def get_rr_intervals(filename, limit_sec=300):
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        print(f"[ERREUR] Fichier manquant : {path}")
        return []
    
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        
        # Tentative 1 : Recherche large (ECG, EKG, V1)
        ecg_ch = next((c for c in raw.ch_names if "ECG" in c.upper() or "EKG" in c.upper()), None)
        
        # Tentative 2 : Verification si echec
        if not ecg_ch:
            # Affichage pour debug si echec
            print(f"[INFO] Pas d'ECG evident dans {filename}.") 
            print(f"       Canaux disponibles : {raw.ch_names[:5]} ...") # On affiche les 5 premiers
            return []
        
        # Chargement
        raw.pick_channels([ecg_ch])
        raw.crop(tmax=limit_sec)
        raw.load_data()
        raw.filter(1.0, 45.0, verbose=False)
        
        sig = raw.get_data()[0] * 1e6
        peaks, _ = find_peaks(sig, distance=int(0.3*raw.info['sfreq']), prominence=np.percentile(sig, 75))
        
        rr = (np.diff(peaks) / raw.info['sfreq']) * 1000
        
        # Filtre physiologique large (Enfant/Adulte)
        return rr[(rr > 300) & (rr < 1500)]

    except Exception as e:
        print(f"[ERREUR] Lecture de {filename}: {e}")
        return []

# ==========================================
# 3. MAIN
# ==========================================

def main():
    print("--- ETUDE DE NORMALISATION MULTI-PATIENTS ---")
    
    data = []
    
    for name, relative_path in PATIENTS.items():
        print(f"Traitement de {name}...")
        rr = get_rr_intervals(relative_path)
        
        if len(rr) > 0:
            print(f"   -> {len(rr)} battements detectes.")
            for i in range(0, len(rr)-20, 20):
                win = rr[i:i+20]
                diff = np.diff(win)
                rmssd = np.sqrt(np.mean(diff**2))
                data.append({'Patient': name, 'RMSSD_Brut': rmssd})
        else:
            print(f"   -> ECHEC pour {name}")

    df = pd.DataFrame(data)
    if df.empty:
        print("\n[STOP] Aucune donnee valide. Arret.")
        return

    # Calcul des Baselines
    baselines = df.groupby('Patient')['RMSSD_Brut'].mean().to_dict()
    print("\n--- MOYENNES DE REPOS (BaseLine) ---")
    for p, val in baselines.items():
        print(f"   - {p} : {val:.1f} ms")

    # Normalisation
    df['Ratio_Normalise'] = df.apply(lambda row: row['RMSSD_Brut'] / baselines[row['Patient']], axis=1)

    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Graphique A
    sns.boxplot(data=df, x='Patient', y='RMSSD_Brut', ax=axes[0], hue='Patient', palette="Set2")
    axes[0].set_title("A. Valeurs Brutes (Heterogenes)")
    axes[0].set_ylabel("RMSSD (ms)")
    axes[0].axhline(50, color='red', linestyle='--', label='Seuil Fixe 50ms')
    axes[0].legend()

    # Graphique B
    sns.boxplot(data=df, x='Patient', y='Ratio_Normalise', ax=axes[1], hue='Patient', palette="Set2")
    axes[1].set_title("B. Valeurs Normalisees (Homogenes)")
    axes[1].set_ylabel("Ratio (1.0 = Repos)")
    axes[1].axhline(1.0, color='green', linestyle='--', label='Repos (1.0)')
    axes[1].legend()

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, 'generalisation_proof.png')
    plt.savefig(output_path)
    print(f"\n[SUCCES] Resultat sauvegarde : {output_path}")
    plt.show()

if __name__ == "__main__":
    main()