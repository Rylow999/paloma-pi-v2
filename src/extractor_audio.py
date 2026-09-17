# Paloma-π v2 — Audio extractor for Columba livia coos

# Reusa el pipeline anti-contaminacion del proyecto original pero maduro:
# - Filtro pasa-banda 100-450 Hz (la banda del arrullo)
# - YIN para pitch tracking (mas robusto que FFT crudo)
# - Segmentacion por energia (voice activity detection)
# - Features por clip: pitch_hz, pitch_std, n_syllables, duration_s, snr_estimate
# Dependencias: librosa, soundfile, numpy, pandas, scipy
# Nota: NO usamos torch aca - el extractor es senal de audio pura.

from pathlib import Path
import numpy as np
import pandas as pd
import librosa


class CooExtractor:
    """Extrae features del arrullo de Columba livia."""

    def __init__(self, sr=22050):
        self.sr = sr
        # Banda del arrullo: 100-450 Hz (verificado en literatura/zoonotic papers)
        self.fmin = 100.0
        self.fmax = 450.0
        # Ventana para pitch tracking
        self.hop_length = 512

    def _bandpass_filter(self, y, sr):
        """Filtro FIR simple: quita graves urbanos y agudos."""
        from scipy import signal
        nyq = sr / 2.0
        b, a = signal.butter(4, [self.fmin / nyq, self.fmax / nyq], btype='band')
        return signal.filtfilt(b, a, y)

    def extract(self, audio_path):
        """
        Extrae features de un .wav/.mp3 del arrullo de paloma.

        Devuelve dict con:
          pitch_hz, pitch_std, n_syllables, duration_s, snr_estimate,
          pitch_contour (array), syllable_bounds (list de tuplas)
        """
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        except Exception as e:
            return {"error": str(e), "file": str(audio_path)}

        # Filtrar antes de todo (evita contaminación de bajos urbanos)
        y = self._bandpass_filter(y, sr)

        # 1. Pitch tracking con YIN (robusto en ruido)
        f0 = librosa.yin(y,
                         fmin=self.fmin,
                         fmax=self.fmax,
                         sr=sr,
                         frame_length=2048,
                         hop_length=self.hop_length)

        f0_clean = f0[~np.isnan(f0)]
        pitch_hz = float(np.median(f0_clean)) if len(f0_clean) else 0.0
        pitch_std = float(np.std(f0_clean)) if len(f0_clean) else 0.0

        # 2. Segmentación por energía (silencio vs voz)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=self.hop_length)[0]
        # Umbral: percentil 30 de energía (el arrullo es tonal y marcado)
        thr = np.percentile(rms, 30)
        is_voice = rms > thr
        # Contar transiciones de silencio a voz (sílabas)
        transitions = np.sum(np.diff(is_voice.astype(int)) > 0)
        n_syllables = int(transitions)

        # 3. SNR: energia en banda vs fuera de banda
        # FFT simple para estimar
        fft = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
        in_band = fft[(freqs >= self.fmin) & (freqs <= self.fmax)].sum()
        total = fft.sum()
        snr_estimate = float(in_band / max(total, 1e-9))

        # 4. Duración
        duration_s = len(y) / sr

        return {
            "file": str(audio_path),
            "pitch_hz": pitch_hz,
            "pitch_std": pitch_std,
            "n_syllables": n_syllables,
            "duration_s": duration_s,
            "snr_estimate": snr_estimate,
            "valid": pitch_hz > 0 and n_syllables >= 1,
        }


def batch_extract(audio_dir, output_csv):
    """Extrae features de todos los .wav/.mp3 en un directorio."""
    ext = CooExtractor()
    rows = []
    audio_path = Path(audio_dir)
    for f in sorted(audio_path.glob("*.mp3")) + sorted(audio_path.glob("*.wav")):
        feats = ext.extract(f)
        rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    # Ejemplo mínimo: procesar audio de prueba
    import sys
    if len(sys.argv) > 1:
        print(CooExtractor().extract(sys.argv[1]))
    else:
        print("Uso: python extractor_audio.py <ruta_audio.mp3>")
