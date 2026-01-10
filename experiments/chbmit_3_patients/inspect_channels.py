import os
import mne

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

FILES_TO_INSPECT = {
    "chb01": os.path.join("chb01", "chb01_03.edf"),
    "chb10": os.path.join("chb10", "chb10_01.edf")
}

def inspect():
    print("--- INSPECTION DES CANAUX ---")
    for name, relative_path in FILES_TO_INSPECT.items():
        path = os.path.join(DATA_DIR, relative_path)
        print(f"\nFICHIER : {name} ({relative_path})")
        
        if not os.path.exists(path):
            print(" -> Fichier introuvable.")
            continue
            
        try:
            # Lecture rapide des en-tetes
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            print(f" -> Nombre de canaux : {len(raw.ch_names)}")
            print(f" -> Liste des canaux : {raw.ch_names}")
        except Exception as e:
            print(f" -> Erreur : {e}")

if __name__ == "__main__":
    inspect()