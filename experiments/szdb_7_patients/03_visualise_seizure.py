import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

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
    try:
        return float(time_str)
    except ValueError:
        pass 
    try:
        parts = time_str.split(':')
        parts = [float(p) for p in parts]
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2: return parts[0] * 60 + parts[1]
    except Exception as e:
        return None
    return None

def get_seizure_time(patient_id, root_path):
    """Lit times.seize et renvoie (debut, fin) en secondes."""
    times_path = os.path.join(root_path, 'data', 'szdb', 'times.seize')
    
    if not os.path.exists(times_path):
        print(f"ERREUR : Fichier introuvable : {times_path}")
        return None, None

    with open(times_path, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 3 and parts[0] == patient_id:
                start_seconds = parse_time(parts[1])
                end_seconds = parse_time(parts[2])
                return start_seconds, end_seconds
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
        
        # On demande au moins 5 battements pour calculer un RMSSD fiable
        if len(window_rrs) > 5:
            diff_rr = np.diff(window_rrs)
            # Filtrage des valeurs absurdes (artefacts) > 2000ms ou < 300ms
            diff_rr = diff_rr[np.abs(diff_rr) < 1000] 
            
            if len(diff_rr) > 2:
                val = np.sqrt(np.mean(diff_rr ** 2))
                t_vals.append(current_t)
                rmssd_vals.append(val)
        
        current_t += 1.0 
        
    return np.array(t_vals), np.array(rmssd_vals)

def visualize_seizure_event(patient_id):
    project_root = get_project_root()
    record_path = os.path.join(project_root, 'data', 'szdb', patient_id, patient_id)
    
    start_seizure, end_seizure = get_seizure_time(patient_id, project_root)
    if start_seizure is None: return

    print(f"--- Analyse {patient_id} ---")
    
    # Fenêtre : 3 min avant -> 1 min après
    view_start = max(0, start_seizure - 180) 
    view_end = end_seizure + 60
    
    samp_from = int(view_start * 200) 
    samp_to = int(view_end * 200)
    
    try:
        signals, fields = wfdb.rdsamp(record_path, sampfrom=samp_from, sampto=samp_to)
    except ValueError:
        print("Erreur fenetre.")
        return

    fs = fields['fs']
    ecg = signals[:, 0]
    time_vec = np.linspace(view_start, view_end, len(ecg))
    
    # --- CORRECTION DU SEUILLAGE ---
    ecg_centered = ecg - np.mean(ecg)
    
    # On détermine l'index où commence la crise DANS notre fenêtre locale
    idx_crisis_start = int((start_seizure - view_start) * fs)
    
    # On calcule l'amplitude max UNIQUEMENT sur la partie calme (avant la crise)
    # Si la fenêtre commence direct par la crise, on prend tout (fallback)
    if idx_crisis_start > 200: 
        safe_zone = ecg_centered[:idx_crisis_start]
        ref_amplitude = np.max(safe_zone)
    else:
        ref_amplitude = np.max(ecg_centered)

    threshold = ref_amplitude * 0.6
    print(f"Seuil calibré sur zone calme : {threshold:.2f} mV")
    
    # Détection
    peaks, _ = find_peaks(ecg_centered, height=threshold, distance=fs*0.3)
    
    print(f"Pics détectés : {len(peaks)}")
    
    peak_times = time_vec[peaks]
    rr_ms = np.diff(peak_times) * 1000 
    rr_times_for_calc = peak_times[1:] 
    
    rmssd_t, rmssd_v = calculate_rolling_rmssd(rr_ms, rr_times_for_calc)
    
    # GRAPHIQUE
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(time_vec, ecg_centered, color='black', alpha=0.6, label='ECG')
    ax1.axhline(y=threshold, color='green', linestyle='--', label='Seuil Détection')
    ax1.set_ylabel('Amplitude (mV)')
    ax1.set_title(f"ECG Patient {patient_id}")
    ax1.grid(True)
    
    ax2.plot(rmssd_t, rmssd_v, color='blue', linewidth=2, label='RMSSD (30s)')
    ax2.set_ylabel('RMSSD (ms)')
    ax2.set_xlabel('Temps (secondes)')
    ax2.grid(True)
    
    for ax in [ax1, ax2]:
        ax.axvspan(start_seizure, end_seizure, color='red', alpha=0.3, label='CRISE')
        ax.axvspan(start_seizure - 30, start_seizure, color='orange', alpha=0.3, label='Pre-Ictal')
        ax.axvline(start_seizure, color='red', linestyle='--')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_seizure_event('sz01')