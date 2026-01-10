import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# --- CONFIGURATION ---
PATIENTS = ['sz01', 'sz02', 'sz03', 'sz04', 'sz05', 'sz06', 'sz07']
MINUTES_BEFORE = 10  # On regarde 10 minutes avant pour voir la descente

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

def get_first_seizure(patient_id, root_path):
    times_path = os.path.join(root_path, 'data', 'szdb', 'times.seize')
    if not os.path.exists(times_path): return None
    
    # On lit le fichier et on prend la PREMIÈRE crise trouvée pour ce patient
    with open(times_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3 and parts[0] == patient_id:
                return parse_time(parts[1]) # Retourne le start_time
    return None

def calculate_rolling_rmssd(ecg_signal, fs, start_time_sec, duration_sec):
    # 1. Prétraitement (Centrage + Seuil dynamique)
    ecg_centered = ecg_signal - np.mean(ecg_signal)
    
    # On calibre le seuil sur la première minute du signal (censée être calme)
    calib_len = int(60 * fs)
    threshold = np.max(ecg_centered[:calib_len]) * 0.6 if len(ecg_centered) > calib_len else np.max(ecg_centered)*0.6
    
    peaks, _ = find_peaks(ecg_centered, height=threshold, distance=fs*0.3)
    
    peak_times = peaks / fs 
    rr_intervals_ms = np.diff(peak_times) * 1000
    rr_times = peak_times[1:] # Temps associé à l'intervalle
    
    t_vals = []
    rmssd_vals = []
    
    # Calcul glissant seconde par seconde
    for t in range(0, duration_sec):
        # Fenêtre de 30s se terminant à t
        mask = (rr_times >= (t - 30)) & (rr_times < t)
        window_rrs = rr_intervals_ms[mask]
        
        if len(window_rrs) > 5:
            diff_rr = np.diff(window_rrs)
            diff_rr = diff_rr[np.abs(diff_rr) < 1000] # Filtre artefacts
            if len(diff_rr) > 2:
                val = np.sqrt(np.mean(diff_rr ** 2))
                t_vals.append(start_time_sec + t)
                rmssd_vals.append(val)
                
    return np.array(t_vals), np.array(rmssd_vals)

def generate_proof_plot(patient_id):
    root = get_project_root()
    record_path = os.path.join(root, 'data', 'szdb', patient_id, patient_id)
    
    # Création dossier de sortie
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    print(f"Traitement {patient_id}...")
    
    seizure_start = get_first_seizure(patient_id, root)
    if seizure_start is None:
        print(f"  -> Pas de crise trouvée pour {patient_id}")
        return

    # Fenêtre : [Début - 10min] à [Début + 30s]
    view_start = max(0, seizure_start - (MINUTES_BEFORE * 60))
    view_end = seizure_start + 30
    duration = view_end - view_start
    
    # Lecture
    samp_from = int(view_start * 200)
    samp_to = int(view_end * 200)
    
    try:
        signals, fields = wfdb.rdsamp(record_path, sampfrom=samp_from, sampto=samp_to)
        fs = fields['fs']
        ecg = signals[:, 0]
    except Exception as e:
        print(f"  -> Erreur lecture: {e}")
        return

    # Calcul RMSSD
    t_axis, rmssd_vals = calculate_rolling_rmssd(ecg, fs, view_start, int(duration))
    
    if len(rmssd_vals) == 0:
        print("  -> Signal trop bruité, impossible de calculer RMSSD.")
        return

    # Calcul de la Baseline (sur les 3 premières minutes de la vue)
    baseline_val = np.mean(rmssd_vals[:180]) if len(rmssd_vals) > 180 else np.mean(rmssd_vals)

    # GRAPHIQUE
    plt.figure(figsize=(12, 6))
    
    plt.plot(t_axis, rmssd_vals, color='blue', linewidth=1.5, label='RMSSD (VFC)')
    
    # Ligne de Baseline (Vert)
    plt.axhline(y=baseline_val, color='green', linestyle='--', label=f'Baseline ({baseline_val:.1f}ms)')
    
    # Ligne d'alerte théorique (-50%)
    plt.axhline(y=baseline_val*0.5, color='orange', linestyle=':', label='Seuil Alerte (-50%)')
    
    # Zone Crise
    plt.axvline(x=seizure_start, color='red', linewidth=2, label='DÉBUT CRISE')
    
    # Zone Pré-Ictal (La cible de l\'IA)
    plt.axvspan(seizure_start - 60, seizure_start, color='yellow', alpha=0.3, label='Zone Pré-Ictale (1 min)')
    
    plt.title(f"Preuve visuelle - Patient {patient_id} - Chute VFC pré-crise")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("RMSSD (ms)")
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Sauvegarde
    filename = f"proof_{patient_id}.png"
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path)
    plt.close() # Ferme la figure pour libérer la mémoire
    
    print(f"  -> Graphique sauvegardé : {save_path}")

if __name__ == "__main__":
    for p in PATIENTS:
        generate_proof_plot(p)
    
    print("\nTerminé ! Va voir dans le dossier experiments/v2.../results/")