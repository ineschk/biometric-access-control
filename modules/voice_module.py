"""
Module: voice_module.py
Enregistre et vérifie la voix via SpeechBrain ECAPA-TDNN (speaker embedding).
Stockage : data/users/<name>/voice.npy
"""

import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import torch
import wave

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "users")
SAMPLE_RATE = 16000
RECORD_SECONDS = 8
VOICE_THRESHOLD = 0.50

_model = None


def _get_model():
    global _model
    if _model is None:
        print("[VOICE] Chargement du modèle SpeechBrain...")
        from speechbrain.inference import EncoderClassifier
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa",
            run_opts={"device": "cpu"}
        )
        print("[VOICE] Modèle chargé.")
    return _model


def _record(seconds: int = RECORD_SECONDS) -> str:
    print(f"[VOICE] Parlez maintenant ({seconds} secondes)...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("[VOICE] Enregistrement terminé.")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    return tmp.name


def _get_embedding(wav_path: str) -> np.ndarray:
    model = _get_model()
    
    # Charger WAV manuellement (évite torchaudio et k2)
    with wave.open(wav_path, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_bytes = wf.readframes(n_frames)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        fs = wf.getframerate()
        n_channels = wf.getnchannels()
    
    # Mono
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    
    # Resample à 16kHz si besoin
    if fs != SAMPLE_RATE:
        ratio = SAMPLE_RATE / fs
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    
    # Tensor pour SpeechBrain
    signal = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    # Encoder avec le modèle ECAPA (embedding pur)
    embeddings = model.encode_batch(signal)
    return embeddings.squeeze().detach().numpy()


def _user_dir(name: str) -> str:
    path = os.path.join(DATA_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def enroll_voice(name: str) -> bool:
    print(f"[VOICE] Enrôlement vocal de {name}.")
    print("[VOICE] Dites votre phrase d'accès...")

    wav_path = _record_debug(seconds=10, debug_name=name)
    embedding = _get_embedding(wav_path)
    os.unlink(wav_path)

    save_path = os.path.join(_user_dir(name), "voice.npy")
    np.save(save_path, embedding)
    print(f"[VOICE] ✅ Voix enregistrée pour {name}.")
    return True


def verify_voice(expected_name: str) -> tuple[bool, float]:
    ref_path = os.path.join(DATA_DIR, expected_name, "voice.npy")
    if not os.path.exists(ref_path):
        print(f"[VOICE] ❌ Aucune voix enregistrée pour {expected_name}.")
        return False, 0.0

    ref_embedding = np.load(ref_path)

    print(f"[VOICE] Vérification vocale pour {expected_name}...")
    print("[VOICE] Dites votre phrase d'accès...")

    wav_path = _record(seconds=8)
    test_embedding = _get_embedding(wav_path)
    os.unlink(wav_path)

    # Similarité cosinus
    sim = float(np.dot(ref_embedding, test_embedding) /
                (np.linalg.norm(ref_embedding) * np.linalg.norm(test_embedding)))

    print(f"[VOICE] Similarité vocale : {sim:.3f} (seuil={VOICE_THRESHOLD})")

    if sim >= VOICE_THRESHOLD:
        print(f"[VOICE] ✅ Voix vérifiée pour {expected_name}.")
        return True, sim
    else:
        print(f"[VOICE] ❌ Voix non reconnue.")
        return False, sim

def _record_debug(seconds: int = RECORD_SECONDS, debug_name: str = "debug") -> str:
    """Enregistre et sauvegarde le fichier WAV pour vérification."""
    print(f"[VOICE] Parlez maintenant ({seconds} secondes)...")
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype='float32')
    sd.wait()
    print("[VOICE] Enregistrement terminé.")

    # Sauvegarde dans le dossier data pour vérification
    debug_path = os.path.join(DATA_DIR, f"{debug_name}_test.wav")
    wav.write(debug_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"[VOICE] 🔊 Fichier sauvegardé : {debug_path}")
    return debug_path