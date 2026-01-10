import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# --- CONFIGURATION ---
MINUTES_TO_VIEW = 10  # On regarde 10 minutes avant la crise pour voir l'état "Normal"

# --- UTILITAIRES (Mêmes que précédemment) ---
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

def get_seizure_time(patient_id, root_path):
    times_path = os.path.join(root_path, 'data', 'szdb', 'times.seize')
    if not os.path.exists(times_path): return None, None
    with open(times_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3 and parts[0] == patient_id:
                return parse_time(parts[1]), parse_time(parts[2])
    return None, None

def calculate_rolling_rmssd(rr_intervals_ms, rr_times, window_size_sec=30):
    t_vals = []
    rmssd_vals = []
    if len(rr_times) < 2: return np.array([]), np.array([])
    
    start_time = rr_times[0]
    end_time = rr_times[-1]
    current_t = start_time + window_size_sec
    
    while current_t < end_time:
        mask = (rr_times >= (current_t - window_size_sec)) & (rr_times < current_t)
        window_rrs = rr_intervals_ms[mask]
        
        # Filtrage strict pour le graphique : on veut une courbe propre
        if len(window_rrs) > 5:
            diff_rr = np.diff(window_rrs)
            # On ignore les sauts absurdes (artefacts de mouvement)
            diff_rr = diff_rr[np.abs(diff_rr) < 600] 
            
            if len(diff_rr) > 2:
                val = np.sqrt(np.mean(diff_rr ** 2))
                t_vals.append(current_t)
                rmssd_vals.append(val)
        current_t += 1.0 
    return np.array(t_vals), np.array(rmssd_vals)

def verify_compression(patient_id):
    project_root = get_project_root()
    record_path = os.path.join(project_root, 'data', 'szdb', patient_id, patient_id)
    
    start_seizure, end_seizure = get_seizure_time(patient_id, project_root)
    if start_seizure is None: return

    print(f"--- Vérification Compression {patient_id} ---")
    
    # 1. On regarde loin en arrière (MINUTES_TO_VIEW avant la crise)
    view_start = max(0, start_seizure - (MINUTES_TO_VIEW * 60))
    view_end = start_seizure + 30 # On s'arrête juste après le début de crise
    
    # Lecture
    samp_from = int(view_start * 200)
    samp_to = int(view_end * 200)
    
    try:
        signals, fields = wfdb.rdsamp(record_path, sampfrom=samp_from, sampto=samp_to)
    except:
        print("Erreur lecture fichier.")
        return

    fs = fields['fs']
    ecg = signals[:, 0]
    time_vec = np.linspace(view_start, view_end, len(ecg))
    
    # 2. Traitement Signal (Centré)
    ecg_centered = ecg - np.mean(ecg)
    
    # Seuil intelligent : calculé sur le début du graphique (censé être calme)
    # On prend les 60 premières secondes de la vue pour calibrer
    calibration_window = ecg_centered[:200*60] 
    threshold = np.max(calibration_window) * 0.6
    
    peaks, _ = find_peaks(ecg_centered, height=threshold, distance=fs*0.3)
    
    peak_times = time_vec[peaks]
    rr_ms = np.diff(peak_times) * 1000 
    rr_times_for_calc = peak_times[1:] 
    
    # 3. RMSSD
    rmssd_t, rmssd_v = calculate_rolling_rmssd(rr_ms, rr_times_for_calc)
    
    # 4. Calcul de la Baseline (Moyenne sur les 3 premières minutes de la vue)
    # On suppose que 10 min avant la crise, le patient va bien.
    baseline_mask = rmssd_t < (view_start + 180)
    if np.any(baseline_mask):
        baseline_val = np.mean(rmssd_v[baseline_mask])
    else:
        baseline_val = np.mean(rmssd_v)

    # 5. GRAPHIQUE
    plt.figure(figsize=(14, 6))
    
    # Trace la courbe RMSSD
    plt.plot(rmssd_t, rmssd_v, color='blue', linewidth=2, label='RMSSD (VFC)')
    
    # Trace la Baseline (Pointillés Verts)
    plt.axhline(y=baseline_val, color='green', linestyle='--', linewidth=2, label=f'Baseline Repos (~{baseline_val:.1f}ms)')
    
    # Trace la zone de Danger (50% de la baseline)
    plt.axhline(y=baseline_val * 0.5, color='orange', linestyle=':', label='Seuil Alerte (-50%)')

    # Zone Crise
    plt.axvline(x=start_seizure, color='red', linewidth=3, label='DÉBUT CRISE')
    plt.axvspan(start_seizure, view_end, color='red', alpha=0.2)
    
    # Zone d'intérêt (Pré-Ictal)
    plt.axvspan(start_seizure - 60, start_seizure, color='yellow', alpha=0.3, label='Pré-Ictal (1 min avant)')

    plt.title(f"Évolution VFC Patient {patient_id} - {MINUTES_TO_VIEW} min avant la crise")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("RMSSD (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Test sur le patient sz01 (Adulte)
    verify_compression('sz01')
    # Tu peux changer pour 'sz02' ou 'sz04' pour comparer