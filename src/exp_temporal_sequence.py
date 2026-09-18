#!/usr/bin/env python3
"""
Paloma-π v2 — Secuencias temporales (ventanas deslizantes).

El arrullo no es un evento instantáneo: es una SECUENCIA. Pregunta: ¿el
resonator sigue la trayectoria temporal del estado (pitch, actividad)
ventana a ventana, o solo colapsa al promedio?

Protocolo:
  1. Cada arrullo largo se corta en ventanas de 5s con hop de 2s
  2. Cada ventana → features propias (pitch, energía, sílabas locales)
  3. Bundle HRR secuencial: cada ventana es un estado en el tiempo
  4. Test A: recuperación por ventana (accuracy por ventana)
  5. Test B (el fuerte): secuencia completa — ¿el orden de las ventanas
     queda recuperable? (bind con rol TIME)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extractor_audio import CooExtractor
from hrr_encoder import hrr_bind
from pure_resonator import _vec, hrr_unbind

DATA = Path(__file__).parent.parent / "data"


def _seg_features(seg, sr):
    from scipy import signal
    b, a = signal.butter(4, [100 / (sr / 2), 450 / (sr / 2)], btype='band')
    y = signal.filtfilt(b, a, seg)
    rms = float(np.sqrt(np.mean(y ** 2)))
    zcr = float(np.mean(np.abs(np.diff(np.sign(y)))) / 2)
    # Pitch por autocorrelación vía FFT (rápido, no lazo de Python)
    n = len(y)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    Y = np.fft.rfft(y, nfft)
    corr = np.fft.irfft(Y * np.conj(Y), nfft)[:n]
    if corr[0] <= 1e-9:
        return {"pitch": 0.0, "rms": rms, "zcr": zcr}
    corr = corr / corr[0]
    min_lag = int(sr / 450)   # 450 Hz máximo
    max_lag = int(sr / 100)   # 100 Hz mínimo
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
    if min_lag >= max_lag:
        return {"pitch": 0.0, "rms": rms, "zcr": zcr}
    lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
    pitch = sr / lag if lag > 0 else 0.0
    return {"pitch": float(pitch), "rms": rms, "zcr": zcr}


def window_features(y, sr, win_s=5.0, hop_s=2.0):
    """Extrae features por ventana (fast)."""
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    out = []
    for start in range(0, len(y) - win, hop):
        seg = y[start:start + win]
        if len(seg) < win:
            break
        out.append(_seg_features(seg, sr))
    return out


def main():
    print("=" * 70)
    print("Paloma-π v2 — Secuencias temporales (ventanas deslizantes)")
    print("=" * 70)

    # Tomamos los 3 arrullos más largos
    import pandas as pd
    df = pd.read_csv(DATA / "features_large.csv")
    df = df[df.valid]
    longest = df.nlargest(3, "duration_s")
    print(f"Clips seleccionados (más largos):")
    for _, r in longest.iterrows():
        print(f"  {Path(r['file']).name}: {r['duration_s']:.0f}s")

    results = []
    N = 512
    for _, row in longest.iterrows():
        audio_file = row["file"]
        print(f"\n=== {Path(audio_file).name} ===")
        import librosa
        y, sr = librosa.load(audio_file, sr=22050, mono=True)

        # 1. Ventanas
        win_s, hop_s = 5.0, 2.0
        win, hop = int(win_s * sr), int(hop_s * sr)
        windows = []
        for start in range(0, len(y) - win, hop):
            seg = y[start:start + win]
            if len(seg) < win:
                continue
            f = _seg_features(seg, sr)
            windows.append({
                "t0": start / sr,
                "pitch": f["pitch"],
                "rms": f["rms"],
                "zcr": f["zcr"],
            })
        print(f"  Ventanas: {len(windows)} (de {win_s}s cada {hop_s}s)")
        pitches = [w["pitch"] for w in windows if w["pitch"] > 0]
        if not pitches:
            print("  (sin pitch detectable — ruido puro)")
            continue
        print(f"  Pitch mediano: {np.median(pitches):.0f} Hz  "
              f"(rango {min(pitches):.0f}-{max(pitches):.0f})")
        print(f"  RMS medio: {np.mean([w['rms'] for w in windows]):.4f}")

        # 2. Bundle por ventana + rol TIME (el orden cuenta)
        #    state_t = bind(TIME_t, bind(PITCH, pitch_t) + bind(ENERGY, rms_t))
        T = min(len(windows), 20)  # cap de 20 ventanas para el test
        role_vecs = {r: _vec(f"__role_{r}__", N) for r in ("TIME", "PITCH", "RMS")}
        id_vec = _vec("paloma_seq", N)

        # Codificamos la secuencia completa como superposición con rol TIME
        bundle_seq = np.zeros(N)
        for t in range(T):
            w = windows[t]
            sym_pitch = f"pitch_{int(np.clip((w['pitch'] - 100) / 400 * 15, 0, 15))}"
            sym_rms = f"rms_{int(np.clip(w['rms'] * 80, 0, 15))}"  # rms ~ 0-0.2
            sym_time = f"t_{t}"
            frame = (hrr_bind(role_vecs["PITCH"], _vec(sym_pitch, N))
                     + hrr_bind(role_vecs["RMS"], _vec(sym_rms, N)))
            bundle_seq += hrr_bind(role_vecs["TIME"],
                                   hrr_bind(_vec(sym_time, N), frame))

        # 3. Decodificación: ¿podemos recuperar la trayectoria de pitch?
        #    Para cada t, preguntamos: unbind(bundle, TIME_t) ≈ frame_t
        pitches_dec = []
        for t in range(T):
            sym_time = f"t_{t}"
            frame_t = hrr_unbind(bundle_seq,
                                 hrr_bind(role_vecs["TIME"], _vec(sym_time, N)))
            # Ahora frame_t ≈ bind(PITCH, pitch_t) + bind(RMS, rms_t)
            cand_pitch = hrr_unbind(frame_t, role_vecs["PITCH"])
            # cleanup sobre codebook de pitch
            cb_pitch = np.array([_vec(f"pitch_{i}", N) for i in range(16)])
            n = np.linalg.norm(cand_pitch)
            if n > 0:
                sims = (cb_pitch @ cand_pitch) / (np.linalg.norm(cb_pitch, axis=1) * n)
                best = int(np.argmax(sims))
                pitch_hz_dec = 100 + (best + 0.5) * (400 / 16)
            else:
                pitch_hz_dec = 0.0
            pitches_dec.append({
                "t": t, "true_pitch": windows[t]["pitch"],
                "dec_pitch": float(pitch_hz_dec),
            })

        ok = sum(1 for p in pitches_dec
                 if abs(p["true_pitch"] - p["dec_pitch"]) < 40 and p["true_pitch"] > 0)
        total = sum(1 for p in pitches_dec if p["true_pitch"] > 0)
        acc = ok / total if total else 0.0
        print(f"  Recuperación de trayectoria pitch: {ok}/{total} "
              f"({acc:.2f} accuracy ±40Hz)")
        results.append({
            "file": Path(audio_file).name,
            "n_windows": len(windows),
            "pitch_recovered": ok,
            "pitch_total": total,
            "pitch_acc": acc,
        })

    out = DATA / "temporal_seq_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
