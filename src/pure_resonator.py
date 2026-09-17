# Paloma-π v2: pure resonator decoder para roles de arrullo.
# Adaptacion de pandora/transducer/pure_resonator.py (mismo motor, sin Gram).

import hashlib
import numpy as np


def hrr_bind(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def hrr_unbind(c, r):
    return np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(r))).real


def _vec(name, N):
    seed = int(hashlib.sha1(str(name).encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    v = rng.randn(N)
    return v / np.linalg.norm(v)


class PalomaPureResonator:
    def __init__(self, codebooks, N=512, seed=7):
        self.codebooks = codebooks          # dict rol -> [simbolos]
        self.N = N
        self.rng = np.random.RandomState(seed)
        self.role_vecs = {r: _vec(f"__role_{r}__", N) for r in codebooks}
        self.sym_vecs = {s: _vec(s, N) for syms in codebooks.values() for s in syms}
        self.cb_arr = {
            r: np.array([self.sym_vecs[s] for s in syms])
            for r, syms in codebooks.items()
        }
        self.cb_names = codebooks

    def cleanup(self, v, rol):
        CB = self.cb_arr[rol]
        n = np.linalg.norm(v)
        if n == 0:
            return self.cb_names[rol][0]
        sims = (CB @ v) / (np.linalg.norm(CB, axis=1) * n)
        return self.cb_names[rol][int(np.argmax(sims))]

    def decode(self, bundle, T=100):
        roles = list(self.codebooks.keys())
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
                        others = others - hrr_bind(self.role_vecs[roles[j]], est[j])
                cand = hrr_unbind(others, self.role_vecs[rol])
                new.append(self.sym_vecs[self.cleanup(cand, rol)])
            est = new
        return {rol: self.cleanup(est[i], rol) for i, rol in enumerate(roles)}
