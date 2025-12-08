# 🏥 Carelink - Système Embarqué de Détection d'Épilepsie (STM32)

> **Projet I-NOVGAMES / Bureau d'Études M2**
> *Détection précoce des crises par analyse multi-modale (HRV + Mouvement) sur microcontrôleur STM32 WB55.*

---

## 🎯 Objectif du Projet

Carelink est un dispositif portable (bracelet) conçu pour détecter les crises d'épilepsie tonico-cloniques et focales en temps réel.
Il repose sur une architecture **Multi-Capteurs** fusionnant deux modèles d'IA :
1.  **Analyse Cardiaque (PPG/ECG) :** Détecte l'effondrement du tonus vagal (signe précurseur).
2.  **Analyse Mouvement (IMU) :** Détecte les convulsions rythmiques.

**Cible actuelle :** Modèle personnalisé pour le patient `chb04` (Pédiatrique, Base CHB-MIT).

---

## 🧠 Architecture IA (Modèle V4 Lite)

Le cœur du système est le modèle **HRV V4 Lite**, optimisé pour l'embarqué.

### 1. La Stratégie "Grand Fossé" (Safety Gap)
Pour garantir 0% de faux positifs (notamment durant le sport), nous avons défini des zones physiologiques strictes basées sur le RMSSD :
* 🔴 **CRISE (< 50 ms) :** Effondrement vagal majeur. Déclenchement alerte.
* ⚫ **ZONE TAMPON (50 ms - 90 ms) :** Zone d'incertitude. Le modèle est entraîné pour ignorer cette zone.
* 🟢 **NORMAL / SPORT (> 90 ms) :** Zone de sécurité. Même avec un effort intense, le patient reste au-dessus de 90ms.

### 2. Optimisation Embarquée (Flatten Average)
* **Pré-traitement :** Aucun DSP complexe (FFT/Spectral) sur le microcontrôleur.
* **Entrée IA :** 3 valeurs flottantes (Moyenne glissante sur 2s).
* **Modèle :** Réseau de Neurones Dense (Float32).

---

## 📂 Structure du Dépôt

```text
Carelink-STM32/
│
├── data/                      # Données brutes (Non incluses, voir Installation)
│   └── chb04-summary.txt      # Annotations des crises
│
├── preprocessing/             # Pipeline Data Science (Python)
│   ├── generate_dataset_v4_lite.py  # Script maître de génération des datasets
│   └── datasets_edgeimpulse/        # Fichiers CSV prêts pour l'entraînement
│
├── edge_impulse_lib/          # Librairie C++ exportée (Le Cerveau)
│   ├── edge-impulse-sdk/      # Moteur d'inférence TensorFlow Lite Micro
│   ├── model-parameters/      # Configuration du modèle V4
│   └── tflite-model/          # Poids du réseau de neurones
│
├── stm32_firmware/            # Code source de l'application (C/C++)
│
└── README.md
