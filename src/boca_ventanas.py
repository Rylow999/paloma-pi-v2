# -*- coding: utf-8 -*-
"""paloma_pi_v2/src/boca_ventanas.py — La boca por VENTANAS (la dirección correcta).

Lección del paper aplicándose a sí misma: los roles globales del clip no
separan conductas (accuracy 0.15 vs azar 0.70). Lo que separa es la
estructura temporal — que el experimento de secuencias ya demostró que el
resonator lee (60-80%).

Este módulo codifica la SECUENCIA de ventanas como bundle HRR (con rol TIME,
igual que exp_temporal_sequence.py) y la boca decide sobre la TRAYECTORIA
(cómo evoluciona el pitch), no sobre el promedio.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hrr_encoder import hrr_bind, _vec
from pure_resonator import hrr_unbind


# ================================================================
# Roles de la secuencia
# ================================================================
ROLES_SEQ = ("TIME", "PITCH", "RMS")
BUCKETS_PITCH = 16
BUCKETS_RMS = 16


def features_de_clip(audio_path, sr=22050, win_s=5.0, hop_s=2.0, max_win=40):
    """Extrae la secuencia de ventanas de un clip (rápido, FFT-based)."""
    import librosa
    from scipy import signal

    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    b, a = signal.butter(4, [100 / (sr / 2), 450 / (sr / 2)], btype='band')
    y = signal.filtfilt(b, a, y)

    win = int(win_s * sr)
    hop = int(hop_s * sr)
    windows = []
    for start in range(0, len(y) - win, hop):
        if len(windows) >= max_win:
            break
        seg = y[start:start + win]
        nfft = 1 << int(np.ceil(np.log2(2 * len(seg))))
        Y = np.fft.rfft(seg, nfft)
        corr = np.fft.irfft(Y * np.conj(Y), nfft)[:len(seg)]
        if corr[0] <= 1e-9:
            continue
        corr = corr / corr[0]
        min_lag, max_lag = int(sr / 450), min(int(sr / 100), len(corr) - 1)
        if min_lag >= max_lag:
            continue
        lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        pitch = sr / lag if lag > 0 else 0.0
        rms = float(np.sqrt(np.mean(seg ** 2)))
        windows.append({"t0": start / sr, "pitch": pitch, "rms": rms})
    return windows


def bundle_secuencia(windows, N=512, max_T=20):
    """Codifica la secuencia como bundle con rol TIME (igual que Exp temporal)."""
    T = min(len(windows), max_T)
    role_vecs = {r: _vec(f"__role_{r}__", N) for r in ROLES_SEQ}
    bundle = np.zeros(N)
    for t in range(T):
        w = windows[t]
        if w["pitch"] <= 0:
            continue
        sym_pitch = f"pitch_{int(np.clip((w['pitch'] - 100) / (400 / BUCKETS_PITCH), 0, BUCKETS_PITCH - 1))}"
        sym_rms = f"rms_{int(np.clip(w['rms'] * 80, 0, BUCKETS_RMS - 1))}"
        sym_time = f"t_{t}"
        frame = (hrr_bind(role_vecs["PITCH"], _vec(sym_pitch, N))
                 + hrr_bind(role_vecs["RMS"], _vec(sym_rms, N)))
        bundle += hrr_bind(role_vecs["TIME"], hrr_bind(_vec(sym_time, N), frame))
    return bundle, role_vecs, T


def trayectoria_pitch(bundle, role_vecs, T, N=512):
    """Recupera la trayectoria de pitch ventana a ventana (unbind por TIME)."""
    cb_pitch = np.array([_vec(f"pitch_{i}", N) for i in range(BUCKETS_PITCH)])
    traj = []
    for t in range(T):
        sym_time = f"t_{t}"
        frame_t = hrr_unbind(bundle, hrr_bind(role_vecs["TIME"], _vec(sym_time, N)))
        cand = hrr_unbind(frame_t, role_vecs["PITCH"])
        n = np.linalg.norm(cand)
        if n < 1e-12:
            traj.append(None)
            continue
        sims = (cb_pitch @ cand) / (np.linalg.norm(cb_pitch, axis=1) * n)
        best = int(np.argmax(sims))
        conf = float(sims[best])
        traj.append({
            "bucket": best,
            "pitch_hz": 100 + (best + 0.5) * (400 / BUCKETS_PITCH),
            "conf": conf,
        })
    return traj


# ================================================================
# La boca sobre la TRAYECTORIA (decisión secuencial)
# ================================================================

def decidir_de_trayectoria(traj):
    """Clasifica la conducta por la FORMA de la trayectoria.

    La literatura bioacústica (Abs & Jeismann 1988, Partan 2005):
      - cortejo (song): arrullo tonal ESTABLE — pitch con poca variación,
        muchas ventanas activas
      - contacto (call): llamadas puntuales — pocas ventanas activas,
        pitch variable entre ellas
      - alerta (alarm): pitch alto esporádico + gaps
      - demanda (begging): pitch bajo repetido rápido

    Métricas de la trayectoria (calculadas, no hardcodeadas):
      - n_activas: ventanas con pitch detectable
      - pitch_std: variación del pitch entre ventanas activas
      - pitch_mean: media del pitch
    """
    activas = [t for t in traj if t is not None]
    if not activas:
        return "sin senal", 0.2
    n_act = len(activas)
    pitches = [t["pitch_hz"] for t in activas]
    p_mean = float(np.mean(pitches))
    p_std = float(np.std(pitches))
    p_cv = p_std / p_mean if p_mean > 0 else 0.0  # coeficiente de variación

    # Reglas sobre la TRAYECTORIA (calibradas con la estructura de los datos):
    # cortejo: estable (CV bajo) y sostenido (muchas ventanas activas)
    if p_cv < 0.15 and n_act >= 8:
        return "cortejo", 0.8
    # demanda: pitch bajo (p25) y estable
    if p_mean < 200 and p_cv < 0.2:
        return "demanda", 0.7
    # alerta: pitch alto + esporádico (pocas ventanas activas)
    if p_mean > 350 and n_act < 8:
        return "alerta", 0.7
    # contacto: variable (CV alto) o esporádico
    if p_cv >= 0.15 or n_act < 8:
        return "contacto", 0.7
    return "cortejo", 0.5  # fallback: la clase mayoritaria


class PalomaBocaVentanas:
    """Pipeline completo por ventanas: clip → secuencia → HRR → trayectoria → palabra."""

    def __init__(self, N=512):
        self.N = N

    def hablar(self, audio_path, win_s=5.0, hop_s=2.0, max_win=40):
        windows = features_de_clip(audio_path, win_s=win_s, hop_s=hop_s, max_win=max_win)
        if not windows:
            return {"palabra": "sin senal", "confianza": 0.0, "traj": []}
        bundle, role_vecs, T = bundle_secuencia(windows, N=self.N)
        traj = trayectoria_pitch(bundle, role_vecs, T, N=self.N)
        palabra, conf = decidir_de_trayectoria(traj)
        return {"palabra": palabra, "confianza": conf, "traj": traj, "n_windows": len(windows)}
