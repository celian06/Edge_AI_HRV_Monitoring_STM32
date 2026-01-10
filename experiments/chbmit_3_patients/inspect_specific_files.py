import os
import mne

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'data'))

# Les fichiers qui posent probleme
FILES_TO_INSPECT = {
    "chb05": os.path.join("chb05", "chb05_13.edf"),
    "chb08": os.path.join("chb08", "chb08_02.edf")
}

def inspect():
    print("--- INSPECTION DES NOMS DE CANAUX ---")
    print(f"Dossier Data : {DATA_DIR}\n")
    
    for name, relative_path in FILES_TO_INSPECT.items():
        path = os.path.join(DATA_DIR, relative_path)
        print(f"FICHIER : {name}")
        
        if not os.path.exists(path):
            print(f" -> [ERREUR] Fichier introuvable : {path}")
            continue
            
        try:
            # Lecture des en-tetes
            raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
            
            # On cherche tout ce qui pourrait ressembler a un coeur
            suspects = [ch for ch in raw.ch_names if "ECG" in ch.upper() or "EKG" in ch.upper() or "V1" in ch.upper()]
            
            print(f" -> Tous les canaux ({len(raw.ch_names)}) :")
            print(raw.ch_names)
            
            if suspects:
                print(f" -> Candidats probables pour le coeur : {suspects}")
            else:
                print(" -> AUCUN canal ne contient 'ECG' ou 'EKG'.")
                
        except Exception as e:
            print(f" -> Erreur : {e}")
        print("-" * 30)

if __name__ == "__main__":
    inspect()