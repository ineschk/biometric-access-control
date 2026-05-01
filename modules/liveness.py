"""
Module: liveness.py
Détecte si la personne devant la caméra est réelle (pas une photo/vidéo).
Méthode : détection de clignement des yeux via Eye Aspect Ratio (EAR)
"""

import cv2
import face_recognition
import numpy as np
from collections import deque


EAR_THRESHOLD = 0.25      # en dessous = œil fermé
BLINK_FRAMES   = 2         # frames consécutives pour valider un clignement
REQUIRED_BLINKS = 2        # clignements requis pour valider "vivant"
MAX_FRAMES     = 200       # timeout


def _eye_aspect_ratio(eye_points):
    """Calcule le EAR pour un œil donné."""
    # distances verticales
    A = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    B = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    # distance horizontale
    C = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (A + B) / (2.0 * C) if C > 0 else 0


def check_liveness(camera_index=0) -> bool:
    """
    Ouvre la webcam et demande à l'utilisateur de cligner des yeux.
    Retourne True si liveness validée, False sinon.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[LIVENESS] Impossible d'ouvrir la caméra.")
        return False

    blink_count = 0
    frames_below = 0
    frame_total = 0
    in_blink = False

    print("[LIVENESS] Clignez des yeux 2 fois pour prouver que vous êtes réel...")

    while frame_total < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_total += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks_list = face_recognition.face_landmarks(rgb)

        ear = 1.0  # valeur par défaut (œil ouvert)

        if landmarks_list:
            lm = landmarks_list[0]
            left_ear  = _eye_aspect_ratio(lm["left_eye"])
            right_ear = _eye_aspect_ratio(lm["right_eye"])
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                frames_below += 1
                in_blink = True
            else:
                if in_blink and frames_below >= BLINK_FRAMES:
                    blink_count += 1
                    print(f"[LIVENESS] Clignement détecté ({blink_count}/{REQUIRED_BLINKS})")
                frames_below = 0
                in_blink = False

        # Affichage
        status = f"Clignements: {blink_count}/{REQUIRED_BLINKS} | EAR: {ear:.2f}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if blink_count >= REQUIRED_BLINKS:
            print("[LIVENESS] ✅ Liveness validée !")
            cap.release()
            cv2.destroyAllWindows()
            return True

    cap.release()
    cv2.destroyAllWindows()
    print("[LIVENESS] ❌ Liveness échouée (timeout ou pas assez de clignements).")
    return False
