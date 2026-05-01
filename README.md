# 🔐 Système Biométrique d'Accès

Authentification multi-facteur : **Liveness + Visage + Voix**

## Installation

### 1. Prérequis système (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install -y python3-pip cmake libboost-all-dev libopenblas-dev liblapack-dev portaudio19-dev
```

### 2. Environnement Python
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> ⚠️ `face-recognition` nécessite `dlib` qui peut prendre 5-10 min à compiler.

---

## Utilisation

### Enrôler un utilisateur
```bash
python main.py --enroll --name Alice
```
→ Capture le visage (webcam) puis la voix (micro)

### Vérifier l'accès
```bash
python main.py --verify
```

**Flow :**
1. Clignez des yeux 2 fois (anti-spoofing)
2. Regardez la caméra (reconnaissance faciale)
3. Parlez votre phrase (vérification vocale)

---

## Structure
```
biometric/
├── main.py                  ← Point d'entrée
├── requirements.txt
├── modules/
│   ├── liveness.py          ← Détection clignement (EAR)
│   ├── face_module.py       ← face_recognition (dlib)
│   └── voice_module.py      ← SpeechBrain ECAPA-TDNN
└── data/
    └── users/
        └── Alice/
            ├── face.npy     ← Embedding visage
            └── voice.npy    ← Embedding voix
```

## Seuils (modifiables)
| Paramètre | Fichier | Valeur par défaut |
|---|---|---|
| Clignements requis | liveness.py | 2 |
| Distance visage max | face_module.py | 0.5 |
| Similarité voix min | voice_module.py | 0.75 |
