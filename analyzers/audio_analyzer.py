"""Audio-Emotionserkennung via Mikrofon.

Nutzt librosa fuer Feature-Extraktion und ein vortrainiertes SpeechBrain-Modell
fuer Speech-Emotion-Recognition. Laeuft in einem eigenen Thread.
"""

import logging
import threading
import time

import numpy as np

from analyzers.audio_quality import compute_snr_proxy_db, quality_from_snr_db
from config import (
    AUDIO_CHUNK_SECONDS,
    AUDIO_DEVICE_INDEX,
    AUDIO_EMA_ALPHA,
    AUDIO_INFERENCE_COOLDOWN,
    AUDIO_SAMPLE_RATE,
    AUDIO_SNR_DB_CEIL,
    AUDIO_SNR_DB_FLOOR,
    EMOTION_MAP,
)
from core.ema_utils import update_ema_vector_inplace

log = logging.getLogger("emotion-light.audio")

# Mapping von SpeechBrain/IEMOCAP Labels auf unsere 7 Emotionen
_LABEL_MAP = {
    "hap": "happy",
    "happiness": "happy",
    "happy": "happy",
    "sad": "sad",
    "sadness": "sad",
    "ang": "angry",
    "anger": "angry",
    "angry": "angry",
    "fea": "fear",
    "fear": "fear",
    "sur": "surprise",
    "surprise": "surprise",
    "dis": "disgust",
    "disgust": "disgust",
    "neu": "neutral",
    "neutral": "neutral",
    "excited": "happy",
    "frustrated": "angry",
    "other": "neutral",
}

_EMOTIONS = list(EMOTION_MAP.keys())


class AudioEmotionAnalyzer:
    """Erfasst Audio vom Mikrofon und erkennt Emotionen in einem Hintergrund-Thread."""

    def __init__(self):
        import sounddevice as sd  # noqa: F401 — Verfuegbarkeit pruefen

        self._lock = threading.Lock()
        self._running = False
        n = len(_EMOTIONS)
        self._ema = {e: 1.0 / n for e in _EMOTIONS}
        self._result = {
            "emotion": "neutral",
            "confidence": 0.0,
            "quality": 0.0,
            "snr_proxy_db": -40.0,
            "rms": 0.0,
            "ema_vector": self._ema.copy(),
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "speech_rate": 0.0,
            "energy_variance": 0.0,
        }
        self._classifier = None
        self._noise_floor_rms = 0.002

    def start(self):
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _load_model(self):
        """Laedt das SpeechBrain Emotion-Recognition-Modell (lazy)."""
        if self._classifier is not None:
            return
        try:
            # Kompatibilitaets-Shim fuer torchaudio >= 2.0
            import torchaudio

            if not hasattr(torchaudio, "list_audio_backends"):
                torchaudio.list_audio_backends = lambda: ["soundfile"]

            # SpeechBrain/HF meldet beim Laden haeufig nicht-kritische Hinweise als WARNING.
            logging.getLogger("speechbrain").setLevel(logging.ERROR)
            logging.getLogger(
                "speechbrain.lobes.models.huggingface_transformers.huggingface"
            ).setLevel(logging.ERROR)

            # Windows-Fix: os.symlink durch Kopierfallback ersetzen.
            # Ohne Developer Mode oder Admin-Rechte scheitert os.symlink auf Windows
            # mit WinError 1314. SpeechBrain und huggingface_hub erstellen intern Symlinks.
            import functools
            import os
            import shutil

            _orig_symlink = os.symlink

            def _symlink_or_copy(src, dst, *args, **kwargs):
                try:
                    _orig_symlink(src, dst, *args, **kwargs)
                except (OSError, NotImplementedError):
                    # Fallback: Datei kopieren statt Symlink erstellen
                    if os.path.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)

            os.symlink = _symlink_or_copy

            # Kompatibilitaets-Shim fuer huggingface_hub >= 0.23
            import huggingface_hub

            patched_hf_functions = []
            for _fn_name in ("hf_hub_download", "model_info", "list_repo_files"):
                _fn = getattr(huggingface_hub, _fn_name, None)
                if _fn is not None:

                    @functools.wraps(_fn)
                    def _patched(*args, _orig=_fn, **kwargs):
                        if "use_auth_token" in kwargs:
                            kwargs["token"] = kwargs.pop("use_auth_token")
                        return _orig(*args, **kwargs)

                    setattr(huggingface_hub, _fn_name, _patched)
                    patched_hf_functions.append((_fn_name, _fn))

            try:
                import torch
                from speechbrain.inference.classifiers import EncoderClassifier

                # Begrenze CPU-Parallelitaet fuer Audio-Inferenz, damit Video-FPS stabil bleiben.
                torch.set_num_threads(1)
                try:
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                _MODEL_ID = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
                try:
                    local_path = huggingface_hub.snapshot_download(
                        _MODEL_ID,
                        local_files_only=True,
                        local_dir_use_symlinks=False,
                    )
                    source = local_path
                    savedir = local_path
                except Exception:
                    source = _MODEL_ID
                    savedir = "pretrained_models/emotion-recognition"

                self._classifier = EncoderClassifier.from_hparams(
                    source=source,
                    savedir=savedir,
                    run_opts={"device": "cpu"},
                )
                log.info("SpeechBrain Emotion-Modell geladen.")
            finally:
                # Patches immer rueckgaengig machen, um globale Seiteneffekte zu vermeiden.
                os.symlink = _orig_symlink
                for fn_name, original in patched_hf_functions:
                    setattr(huggingface_hub, fn_name, original)
        except Exception as exc:
            log.error("SpeechBrain Modell konnte nicht geladen werden: %s", exc)
            self._running = False

    def _loop(self):
        import sounddevice as sd

        self._load_model()
        if self._classifier is None:
            return

        chunk_samples = int(AUDIO_SAMPLE_RATE * AUDIO_CHUNK_SECONDS)

        while self._running:
            try:
                # Audio-Chunk aufnehmen
                audio = sd.rec(
                    chunk_samples,
                    samplerate=AUDIO_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=AUDIO_DEVICE_INDEX,
                )
                sd.wait()
                audio = audio.flatten()

                # Stille erkennen: wenn RMS zu niedrig → ueberspringen
                rms = np.sqrt(np.mean(audio**2))

                # Adaptive floor follows quiet/background levels and enables a stable SNR proxy.
                percentile_floor = float(np.percentile(np.abs(audio), 25))
                target_floor = max(1e-6, percentile_floor)
                floor_alpha = 0.20 if target_floor < self._noise_floor_rms else 0.04
                self._noise_floor_rms = (
                    1.0 - floor_alpha
                ) * self._noise_floor_rms + floor_alpha * target_floor

                snr_proxy_db = compute_snr_proxy_db(audio, self._noise_floor_rms)
                quality = quality_from_snr_db(
                    snr_proxy_db,
                    snr_floor_db=AUDIO_SNR_DB_FLOOR,
                    snr_ceil_db=AUDIO_SNR_DB_CEIL,
                )

                if rms < 0.005:
                    continue

                # Prosodische Merkmale extrahieren
                prosodic = self._extract_prosodic_features(audio)

                # Klassifizierung
                import torch

                signal = torch.tensor(audio).unsqueeze(0)
                prediction = self._classifier.classify_batch(signal)

                # prediction: (out_prob, score, index, text_lab)
                scores = prediction[0].squeeze().numpy()
                labels = prediction[3]

                # Scores auf unsere 7 Emotionen mappen
                emotion_scores = {e: 0.0 for e in _EMOTIONS}
                for i, label in enumerate(labels):
                    label_lower = label.lower().strip()
                    mapped = _LABEL_MAP.get(label_lower, "neutral")
                    if i < len(scores):
                        emotion_scores[mapped] = max(
                            emotion_scores[mapped], float(scores[i]) * 100.0
                        )

                # EMA aktualisieren
                alpha = AUDIO_EMA_ALPHA
                update_ema_vector_inplace(self._ema, emotion_scores, alpha, _EMOTIONS)

                best = max(_EMOTIONS, key=lambda e: self._ema[e])
                with self._lock:
                    self._result = {
                        "emotion": best,
                        "confidence": self._ema[best],
                        "quality": quality,
                        "snr_proxy_db": float(snr_proxy_db),
                        "rms": float(rms),
                        "ema_vector": self._ema.copy(),
                        "pitch_mean_hz": prosodic.get("pitch_mean_hz", 0.0),
                        "pitch_std_hz": prosodic.get("pitch_std_hz", 0.0),
                        "speech_rate": prosodic.get("speech_rate", 0.0),
                        "energy_variance": prosodic.get("energy_variance", 0.0),
                    }

                if AUDIO_INFERENCE_COOLDOWN > 0:
                    time.sleep(AUDIO_INFERENCE_COOLDOWN)

            except Exception as exc:
                log.debug("Audio-Analyse fehlgeschlagen: %s", exc)
                time.sleep(1.0)

    def _extract_prosodic_features(self, audio: np.ndarray) -> dict:
        """Extrahiert prosodische Merkmale aus einem Audio-Chunk.

        Returns dict mit:
          pitch_mean_hz  – mittlere Grundfrequenz (F0) in Hz
          pitch_std_hz   – Standardabweichung der F0 (Intonationsvarianz)
          speech_rate    – geschaetzte Sprechrate (Energie-Onsets/Sekunde)
          energy_variance – Varianz der Frame-Energie (Dynamik)
        """
        result = {
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "speech_rate": 0.0,
            "energy_variance": 0.0,
        }
        try:
            import librosa

            # F0 via pyin (robust pitch tracking)
            f0, voiced_flag, _ = librosa.pyin(
                audio,
                fmin=60.0,
                fmax=500.0,
                sr=AUDIO_SAMPLE_RATE,
                frame_length=2048,
            )
            voiced_f0 = (
                f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
            )
            if len(voiced_f0) > 2:
                result["pitch_mean_hz"] = float(np.mean(voiced_f0))
                result["pitch_std_hz"] = float(np.std(voiced_f0))

            # Energie pro Frame
            hop_length = 512
            energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            if len(energy) > 1:
                result["energy_variance"] = float(np.var(energy))

            # Sprechrate: Onset-Erkennung als Proxy fuer Silben
            onsets = librosa.onset.onset_detect(
                y=audio,
                sr=AUDIO_SAMPLE_RATE,
                hop_length=hop_length,
                backtrack=False,
            )
            duration_s = len(audio) / AUDIO_SAMPLE_RATE
            if duration_s > 0:
                result["speech_rate"] = float(len(onsets) / duration_s)

        except Exception as exc:
            log.debug("Prosodische Analyse fehlgeschlagen: %s", exc)
        return result

    def get(self) -> dict:
        with self._lock:
            return self._result.copy()

    def stop(self):
        self._running = False
