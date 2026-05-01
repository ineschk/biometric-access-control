"""
Module: face_recognition_module.py
Enregistre et vérifie les visages via face_recognition (dlib).
Stockage : fichiers .npy dans data/users/<name>/face.npy
"""

import os
import cv2
import numpy as np
import face_recognition

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "users")
FACE_THRESHOLD = 0.5   # distance max (plus bas = plus strict)
CAPTURE_SAMPLES = 5    # photos prises lors de l'enrôlement


def _user_dir(name: str) -> str:
    path = os.path.join(DATA_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def enroll_face(name: str, camera_index: int = 0) -> bool:
    """Capture plusieurs frames et sauvegarde l'embedding moyen."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[FACE] Impossible d'ouvrir la caméra.")
        return False

    encodings = []
    print(f"[FACE] Enrôlement de {name} — regardez la caméra ({CAPTURE_SAMPLES} captures)...")

    while len(encodings) < CAPTURE_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        found = face_recognition.face_encodings(rgb)

        if found:
            encodings.append(found[0])
            print(f"[FACE] Capture {len(encodings)}/{CAPTURE_SAMPLES}")

        cv2.putText(frame, f"Captures: {len(encodings)}/{CAPTURE_SAMPLES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.imshow("Enrôlement visage", frame)
        cv2.waitKey(300)

    cap.release()
    cv2.destroyAllWindows()

    if not encodings:
        print("[FACE] ❌ Aucun visage détecté.")
        return False

    mean_encoding = np.mean(encodings, axis=0)
    save_path = os.path.join(_user_dir(name), "face.npy")
    np.save(save_path, mean_encoding)
    print(f"[FACE] ✅ Visage enregistré pour {name}.")
    return True


def verify_face(camera_index: int = 0) -> tuple[bool, str, float]:
    """
    Capture un frame et compare avec tous les utilisateurs enregistrés.
    Retourne (success, name, confidence_score).
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False, "", 0.0

    print("[FACE] Vérification du visage...")
    best_match = None
    best_distance = 1.0

    # Charger tous les embeddings enregistrés
    known = {}
    if os.path.exists(DATA_DIR):
        for user in os.listdir(DATA_DIR):
            fpath = os.path.join(DATA_DIR, user, "face.npy")
            if os.path.exists(fpath):
                known[user] = np.load(fpath)

    if not known:
        print("[FACE] ❌ Aucun utilisateur enregistré.")
        cap.release()
        return False, "", 0.0

    # Capture jusqu'à trouver un visage
    for _ in range(60):
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if encodings:
            enc = encodings[0]
            for name, ref in known.items():
                dist = face_recognition.face_distance([ref], enc)[0]
                if dist < best_distance:
                    best_distance = dist
                    best_match = name

            cv2.imshow("Vérification visage", frame)
            cv2.waitKey(100)
            break

        cv2.imshow("Vérification visage", frame)
        cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()

    score = 1.0 - best_distance  # convertir en score de confiance
    if best_match and score >= (1.0 - FACE_THRESHOLD):
        print(f"[FACE] ✅ Visage reconnu : {best_match} (score={score:.2f})")
        return True, best_match, score
    else:
        print(f"[FACE] ❌ Visage non reconnu (meilleur score={score:.2f})")
        return False, "", score
