import wfdb
import os

def get_project_root():
    current_script_path = os.path.abspath(__file__)
    root = current_script_path
    while os.path.basename(root) != 'IA_CareLink':
        parent = os.path.dirname(root)
        if parent == root: raise FileNotFoundError("Racine introuvable")
        root = parent
    return root

def investigate_patient(patient_id):
    project_root = get_project_root()
    folder_path = os.path.join(project_root, 'data', 'szdb', patient_id)
    record_path_no_ext = os.path.join(folder_path, patient_id)
    
    print(f"--- Enquête sur le patient : {patient_id} ---")

    # 1. ANALYSE DU FICHIER D'ANNOTATIONS (.ari)
    print(f"\n[1] Inspection du fichier .ari ...")
    try:
        # On force la lecture de l'extension 'ari'
        ann = wfdb.rdann(record_path_no_ext, extension='ari')
        
        # On regarde quels sont les symboles présents (ex: 'N' pour Normal, '+' pour Rythme...)
        unique_labels = set(ann.symbol)
        print(f"   -> Nombre d'annotations trouvées : {len(ann.sample)}")
        print(f"   -> Types de symboles trouvés : {unique_labels}")
        
        # On affiche les notes auxiliaires (souvent du texte comme "Seizure start")
        aux_notes = [note for note in ann.aux_note if note.strip() != '']
        if len(aux_notes) > 0:
            print(f"   -> Notes trouvées (Indices précieux !) :")
            # On affiche les 10 premières notes uniques
            print(f"      {list(set(aux_notes))[:10]}")
        else:
            print("   -> Aucune note textuelle dans le fichier .ari.")
            
    except Exception as e:
        print(f"   -> Impossible de lire .ari : {e}")

    # 2. ANALYSE DU HEADER (.hea)
    print(f"\n[2] Lecture du fichier .hea (Texte brut) ...")
    hea_file = os.path.join(folder_path, f"{patient_id}.hea")
    try:
        with open(hea_file, 'r') as f:
            content = f.readlines()
            print("   -> Contenu du Header :")
            for line in content:
                # On affiche surtout les lignes de commentaires (commençant par #)
                if line.startswith('#') or 'seizure' in line.lower():
                    print(f"      [INFO] {line.strip()}")
                else:
                    # On affiche quand même les premières lignes techniques pour info
                    pass 
    except Exception as e:
        print(f"   -> Impossible de lire .hea : {e}")

if __name__ == "__main__":
    investigate_patient('sz01')