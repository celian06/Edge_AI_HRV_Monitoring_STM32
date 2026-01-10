import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# --- FONCTIONS UTILITAIRES ---

def get_project_root():
    """
    Remonte l'arborescence depuis l'emplacement de ce script
    pour trouver la racine du projet 'IA_CareLink'.
    """
    current_script_path = os.path.abspath(__file__)
    root = current_script_path
    
    # On remonte tant qu'on ne trouve pas le dossier racine
    while os.path.basename(root) != 'IA_CareLink':
        parent = os.path.dirname(root)
        if parent == root:
            raise FileNotFoundError("Impossible de trouver la racine du projet IA_CareLink.")
        root = parent
    return root

def calculate_rmssd(rr_intervals_ms):
    """Calcule le RMSSD a partir des intervalles RR (en ms)."""
    if len(rr_intervals_ms) < 2:
        return 0.0
    diff_rr = np.diff(rr_intervals_ms)
    rmssd = np.sqrt(np.mean(diff_rr ** 2))
    return rmssd

def inspect_patient_data(patient_id):
    # 1. Construction du chemin robuste
    try:
        project_root = get_project_root()
        # Chemin: data/szdb/{patient_id}/{patient_id} (sans extension)
        record_path = os.path.join(project_root, 'data', 'szdb', patient_id, patient_id)
        
        print(f"--- Analyse du patient : {patient_id} ---")
        
        if not os.path.exists(record_path + '.hea'):
            print(f"ERREUR CRITIQUE : Fichier non trouve : {record_path}.hea")
            return

        # 2. Lecture des fichiers PhysioNet
        # On lit 20 secondes (sampto=4000 a 200Hz) pour valider
        record = wfdb.rdrecord(record_path)
        signals, fields = wfdb.rdsamp(record_path, sampto=4000)

    except Exception as e:
        print(f"Erreur lors de la lecture : {e}")
        return

    fs = fields['fs']
    print(f"Frequence (Fs) : {fs} Hz")
    
    # 3. Selection du canal ECG
    ecg_signal_raw = signals[:, 0]
    
    # Creation du vecteur temps pour l'affichage (en secondes)
    time_vector = np.arange(len(ecg_signal_raw)) / fs

    # =================================================================
    # 4. Detection des pics R (AMELIOREE : Centrage + Seuil Dynamique)
    # =================================================================
    
    # ETAPE CRITIQUE : Centrage du signal (suppression de l'offset DC)
    # Le signal brut est souvent decale (ex: centre a 1.0mV).
    # En soustrayant la moyenne, on le ramene autour de 0.
    ecg_centered = ecg_signal_raw - np.mean(ecg_signal_raw)
    
    # On calcule l'amplitude max sur ce signal propre
    max_amplitude = np.max(ecg_centered)
    
    # Seuil dynamique a 60% du max.
    # Puisque le signal est centré, l'onde T (plus petite) restera sous ce seuil.
    dynamic_threshold = max_amplitude * 0.6
    
    print(f"Max amplitude (centree) : {max_amplitude:.2f} mV")
    print(f"Seuil de detection : {dynamic_threshold:.2f} mV")

    # Detection des pics sur le signal CENTRE
    # distance=fs*0.3 -> On interdit 2 pics en moins de 300ms (max 200 bpm)
    peaks, _ = find_peaks(ecg_centered, height=dynamic_threshold, distance=fs*0.3)
    # =================================================================
    
    # 5. Calcul RMSSD
    rmssd_val = 0
    if len(peaks) > 1:
        # Conversion indices -> temps (ms)
        rr_intervals = np.diff(peaks) / fs * 1000
        rmssd_val = calculate_rmssd(rr_intervals)
        print(f"Nombre de pics R detectes : {len(peaks)}")
        print(f"RMSSD estime (sur 20s) : {rmssd_val:.2f} ms")
    else:
        print("ATTENTION : Pas assez de pics detectes.")

    # 6. Visualisation
    plt.figure(figsize=(12, 6))
    
    # On affiche le signal CENTRE pour que le seuil visuel soit correct
    plt.plot(time_vector, ecg_centered, label='Signal ECG Centre', color='blue', alpha=0.7)
    
    # Affichage des pics detectes
    plt.plot(time_vector[peaks], ecg_centered[peaks], "x", label='Pics R (Valides)', color='red', markersize=10, markeredgewidth=2)
    
    # Affichage de la ligne de seuil
    plt.axhline(y=dynamic_threshold, color='green', linestyle='--', alpha=0.8, label=f'Seuil ({dynamic_threshold:.2f}mV)')
    
    plt.title(f"Validation Patient {patient_id} (Adultes) - RMSSD={rmssd_val:.1f}ms")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Lance l'inspection sur le patient sz01
    inspect_patient_data('sz01')