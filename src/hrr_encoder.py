# Paloma-π v2: HRR encoder para arrullos de Columba livia
#
# Mapea los features extraidos por extractor_audio.py a vectores HRR con roles:
#   PITCH = pitch_hz bucketizado (16 buckets log: 100-500 Hz)
#   SYLL  = n_syllables bucketizado (16 buckets exponencial)
#   DUR   = duration_s bucketizada (16 buckets log)
#   ID    = hash del recordista (un vector fijo por grabador, 16 slots)
#   SNR   = snr_estimate bucketizado (16 buckets lineal)
#
# Bundle = sum( bind(role_R, symbol_R) ) para los 5 roles.
# Devuelve tambien el dict de "ground truth" para evaluar el decoder.

import hashlib
import numpy as np


def hrr_bind(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real


def _vec(name, N):
    """Vector gaussiano determinista por nombre (misma convencion que state_encoder)."""
    seed = int(hashlib.sha1(str(name).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    v = rng.randn(N)
    return v / np.linalg.norm(v)


ROLES = ["PITCH", "SYLL", "DUR", "ID", "SNR"]
BUCKETS = 16


def bucketize_pitch(pitch_hz):
    # Buckets log entre 100 y 500 Hz
    if pitch_hz <= 0:
        return 0
    b = int(np.log2(pitch_hz / 100.0) / np.log2(500.0 / 100.0) * BUCKETS)
    return max(0, min(BUCKETS - 1, b))


def bucketize_syll(n):
    # Exponencial: 1,2,4,8,16,32,64,128,256,...
    if n <= 0:
        return 0
    b = int(np.log2(max(n, 1)))
    return max(0, min(BUCKETS - 1, b))


def bucketize_dur(d):
    # Log entre 1 y 300 s
    if d <= 0:
        return 0
    b = int(np.log(max(d, 1.0)) / np.log(300.0) * BUCKETS)
    return max(0, min(BUCKETS - 1, b))


def bucketize_snr(s):
    return max(0, min(BUCKETS - 1, int(s * BUCKETS)))


class PalomaHRREncoder:
    """Encoder HRR de features de arrullo a vector bundle."""

    def __init__(self, N=512, seed=7):
        self.N = N
        self.rng = np.random.RandomState(seed)
        self.role_vecs = {r: _vec(f"__role_{r}__", N) for r in ROLES}

    def encode_fact(self, feats, recorder_id):
        """feats: dict de extractor_audio; recorder_id: string/nombre.

        Devuelve (bundle, truth) donde truth es el dict rol -> simbolo
        para evaluar el decoder despues.
        """
        # Simbolos discretos por rol
        symbols = {
            "PITCH": f"pitch_{bucketize_pitch(feats['pitch_hz'])}",
            "SYLL": f"syll_{bucketize_syll(feats['n_syllables'])}",
            "DUR": f"dur_{bucketize_dur(feats['duration_s'])}",
            "ID": f"rec_{hashlib.sha1(str(recorder_id).encode()).hexdigest()[:6]}",
            "SNR": f"snr_{bucketize_snr(feats['snr_estimate'])}",
        }

        bundle = np.zeros(self.N)
        for role in ROLES:
            bundle += hrr_bind(self.role_vecs[role], _vec(symbols[role], self.N))
        return bundle, symbols


# Codebooks por rol (para el resonator)
def make_codebooks(recorder_ids):
    """Devuelve dict rol -> lista de simbolos posibles."""
    cbs = {
        "PITCH": [f"pitch_{i}" for i in range(BUCKETS)],
        "SYLL": [f"syll_{i}" for i in range(BUCKETS)],
        "DUR": [f"dur_{i}" for i in range(BUCKETS)],
        "ID": [f"rec_{hashlib.sha1(str(r).encode()).hexdigest()[:6]}" for r in recorder_ids],
        "SNR": [f"snr_{i}" for i in range(BUCKETS)],
    }
    return cbs
