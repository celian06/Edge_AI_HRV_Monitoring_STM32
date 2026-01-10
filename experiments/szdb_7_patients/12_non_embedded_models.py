import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---
# Pointez vers le dossier généré par le script 11 (le split manuel)
BASE_DIR = 'data/edge_impulse_filtered_split' 

def load_data_from_folder(folder_path, label_value):
    data = []
    if not os.path.exists(folder_path): return []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            try:
                # On lit le CSV et on calcule les stats (Même logique qu'Edge Impulse)
                df = pd.read_csv(filepath)
                vals = df['rmssd_val'].values
                # Features statistiques simples
                features = [
                    np.mean(vals),
                    np.std(vals),
                    np.min(vals),
                    np.max(vals),
                    np.percentile(vals, 25), # Q1
                    np.percentile(vals, 75)  # Q3
                ]
                data.append(features + [label_value])
            except: pass
    return data

# 1. Chargement des données (Train et Test séparés comme avant)
print("Chargement des données...")
X_train, y_train = [], []
X_test, y_test = [], []

# Charger TRAIN
train_crise = load_data_from_folder(os.path.join(BASE_DIR, 'train', 'crise'), 1)
train_normal = load_data_from_folder(os.path.join(BASE_DIR, 'train', 'normal'), 0)
train_data = np.array(train_crise + train_normal)
X_train = train_data[:, :-1]
y_train = train_data[:, -1]

# Charger TEST
test_crise = load_data_from_folder(os.path.join(BASE_DIR, 'test', 'crise'), 1)
test_normal = load_data_from_folder(os.path.join(BASE_DIR, 'test', 'normal'), 0)
test_data = np.array(test_crise + test_normal)
X_test = test_data[:, :-1]
y_test = test_data[:, -1]

print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# 2. Random Forest (Le Challenger)
print("\n--- RANDOM FOREST ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, preds_rf):.2%}")
print(classification_report(y_test, preds_rf, target_names=['Normal', 'Crise']))

# 3. SVM (L'alternative)
print("\n--- SVM (Support Vector Machine) ---")
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)
preds_svm = svm.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, preds_svm):.2%}")
print(classification_report(y_test, preds_svm, target_names=['Normal', 'Crise']))