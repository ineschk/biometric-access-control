"""
main.py — Système biométrique d'accès
Usage:
  python main.py --enroll --name <prénom>    → Enrôler un utilisateur
  python main.py --verify                    → Vérifier l'accès
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from modules.liveness import check_liveness
from modules.face_module import enroll_face, verify_face
from modules.voice_module import enroll_voice, verify_voice


def enroll(name: str):
    print(f"\n{'='*50}")
    print(f"  ENRÔLEMENT : {name.upper()}")
    print(f"{'='*50}\n")

    ok_face = enroll_face(name)
    if not ok_face:
        print("❌ Enrôlement visage échoué. Abandon.")
        return

    ok_voice = enroll_voice(name)
    if not ok_voice:
        print("❌ Enrôlement vocal échoué. Abandon.")
        return

    print(f"\n✅ {name} enrôlé avec succès !")


def verify():
    print(f"\n{'='*50}")
    print(f"  VÉRIFICATION D'ACCÈS")
    print(f"{'='*50}\n")

    # Étape 1 : Liveness
    print("--- Étape 1/3 : Détection de vivacité ---")
    if not check_liveness():
        print("\n🚫 ACCÈS REFUSÉ — Liveness échouée.")
        return

    # Étape 2 : Reconnaissance faciale
    print("\n--- Étape 2/3 : Reconnaissance faciale ---")
    face_ok, name, face_score = verify_face()
    if not face_ok:
        print("\n🚫 ACCÈS REFUSÉ — Visage non reconnu.")
        return

    # Étape 3 : Vérification vocale
    print(f"\n--- Étape 3/3 : Vérification vocale ({name}) ---")
    voice_ok, voice_score = verify_voice(name)

    # Décision finale
    print(f"\n{'='*50}")
    if face_ok and voice_ok:
        print(f"  ✅ ACCÈS ACCORDÉ — Bienvenue, {name.upper()} !")
        print(f"  Visage: {face_score:.2f} | Voix: {voice_score:.2f}")
    else:
        print(f"  🚫 ACCÈS REFUSÉ")
        print(f"  Visage: {face_score:.2f} | Voix: {voice_score:.2f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Système biométrique")
    parser.add_argument("--enroll", action="store_true", help="Mode enrôlement")
    parser.add_argument("--verify", action="store_true", help="Mode vérification")
    parser.add_argument("--name", type=str, help="Nom de l'utilisateur (requis pour --enroll)")
    args = parser.parse_args()

    if args.enroll:
        if not args.name:
            print("❌ --name requis pour l'enrôlement. Ex: python main.py --enroll --name Alice")
            sys.exit(1)
        enroll(args.name)
    elif args.verify:
        verify()
    else:
        parser.print_help()
