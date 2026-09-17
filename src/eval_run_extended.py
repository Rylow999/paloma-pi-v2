# Evaluacion extendida: 6 roles (pitch_media, pitch_std separados)
# Precondicion: dataset real, al menos 10 muestras.

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from hrr_encoder import (PalomaHRREncoder, make_codebooks,
                         bucketize_pitch, bucketize_syll, bucketize_dur, bucketize_snr)
from pure_resonator import PalomaPureResonator

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

df = pd.read_csv(DATA / "features_real.csv")
df = df[df.valid == True].reset_index(drop=True)

# Separar pitch en dos variables (media y std), como pide la literatura
# Los machos con mas experiencia tienen pitch_std bajo (estables)
# Los jovenes tienen pitch_std alto (variable)

# Normalizar: pitch_hz está en [100,500], pitch_std en [0,120]
def bucket_pitch_med(h):
    return int(np.clip((h - 100) / 400 * 15, 0, 15))

def bucket_pitch_std(s):
    return int(np.clip((s - 0) / 120 * 15, 0, 15))

# Roles expandidos
ROLES_EXT = ["PITCH_MED", "PITCH_STD", "SYLL", "DUR", "ID", "SNR"]

def make_codebooks_ext(recordists):
    return {
        "PITCH_MED": [f"pm_{i}" for i in range(16)],
        "PITCH_STD": [f"ps_{i}" for i in range(16)],
        "SYLL": [f"syll_{i}" for i in range(16)],
        "DUR": [f"dur_{i}" for i in range(16)],
        "ID": [f"rec_{r}" for r in recordists],
        "SNR": [f"snr_{i}" for i in range(16)],
    }

class PalomaResonator6:
    """Resonator puro con 6 roles. Sin Gram."""
    def __init__(self, N=512, seed=7):
        self.N = N
        self.rng = np.random.RandomState(seed)
        self.role_vecs = {r: self._vec(f"__role_{r}__") for r in ROLES_EXT}
        self.codebooks = make_codebooks_ext([])
        self.sym_vecs = {s: self._vec(s) for cbs in self.codebooks.values() for s in cbs}
        self.cb_arr = {r: np.array([self.sym_vecs[s] for s in self.codebooks[r]]) for r in ROLES_EXT}
        self.cb_names = {r: self.codebooks[r] for r in ROLES_EXT}

    def _vec(self, name):
        seed = int(hashlib.sha1(str(name).encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(self.N) / np.linalg.norm(rng.randn(self.N))

    def cleanup(self, v, rol):
        CB = self.cb_arr[rol]
        n = np.linalg.norm(v)
        if n == 0: return self.cb_names[rol][0]
        sims = (CB @ v) / (np.linalg.norm(CB, axis=1) * n)
        return self.cb_names[rol][int(np.argmax(sims))]

    def decode(self, bundle, T=100):
        roles = ROLES_EXT
        est = []
        for _ in roles:
            v = self.rng.randn(self.N)
            est.append(v / np.linalg.norm(v))
        for _ in range(T):
            new = []
            for i, rol in enumerate(roles):
                others = bundle.copy()
                for j in range(len(roles)):
                    if j != i:
                        others -= np.fft.ifft(np.fft.fft(self.role_vecs[roles[j]]) * np.fft.fft(est[j])).real
                cand = np.fft.ifft(np.fft.fft(others) * np.conj(np.fft.fft(self.role_vecs[rol]))).real
                new.append(self.sym_vecs[self.cleanup(cand, rol)])
            est = new
        return {rol: self.cleanup(est[i], rol) for i, rol in enumerate(roles)}

import hashlib

class PalomaEncoder6:
    def __init__(self, N=512):
        self.N = N
        self.role_vecs = {r: self._vec(f"__role_{r}__") for r in ROLES_EXT}

    def _vec(self, name):
        seed = int(hashlib.sha1(str(name).encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return rng.randn(self.N) / np.linalg.norm(rng.randn(self.N))

    def encode(self, row):
        truth = {}
        bundle = np.zeros(self.N)
        # pitch media y std separados
        pm = bucket_pitch_med(row["pitch_hz"])
        ps = bucket_pitch_std(row["pitch_std"])
        truth["PITCH_MED"] = f"pm_{pm}"
        truth["PITCH_STD"] = f"ps_{ps}"
        truth["SYLL"] = f"syll_{bucketize_syll(row['n_syllables'])}"
        truth["DUR"] = f"dur_{bucketize_dur(row['duration_s'])}"
        truth["ID"] = f"rec_{row['recordist']}"
        truth["SNR"] = f"snr_{bucketize_snr(row['snr_estimate'])}"
        for rol, sym in truth.items():
            bundle += np.fft.ifft(np.fft.fft(self.role_vecs[rol]) * np.fft.fft(self._vec(sym))).real
        return bundle, truth

# Ejecutar evaluación
results = []
for idx, row in df.iterrows():
    encoder = PalomaEncoder6(N=512)
    bundle, truth = encoder.encode(row)
    # El codebook incluye TODOS los recs como posibles candidatos
    all_rec = df.recordist.unique().tolist()
    codebooks = {
        "PITCH_MED": [f"pm_{i}" for i in range(16)],
        "PITCH_STD": [f"ps_{i}" for i in range(16)],
        "SYLL": [f"syll_{i}" for i in range(16)],
        "DUR": [f"dur_{i}" for i in range(16)],
        "ID": [f"rec_{r}" for r in all_rec],
        "SNR": [f"snr_{i}" for i in range(16)],
    }
    resonator = PalomaResonator6(N=512)
    dec = resonator.decode(bundle, T=150)
    accs = {r: float(dec[r] == truth[r]) for r in ROLES_EXT}
    accs["joint"] = float(all(v == 1.0 for v in accs.values()))
    accs["pitch_med_dec"] = abs(int(dec["PITCH_MED"].split("_")[1]) - int(truth["PITCH_MED"].split("_")[1]))
    results.append(accs)
    print(f"Grabación {idx}: joint={accs['joint']}, pitch_err={accs['pitch_med_err']}")

# Resumen
print(f"\n=== Resultados 6 roles ===")
for r in ROLES_EXT:
    acc_r = np.mean([res[r] for res in results])
    print(f"  {r}: {acc_r:.3f}")
print(f"\nJoint accuracy (6/6): {np.mean([res['joint'] for res in results]):.3f}")
