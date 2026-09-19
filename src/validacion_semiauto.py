#!/usr/bin/env python3
"""
Validacion semi-automatica (sin BORIS): el bundle HRR agrupa conductas?

Test:
  1. Por cada clip, bundle HRR de 5 roles (pitch, syll, dur, id, snr)
  2. Conducta etiquetada por el recordista (campo 'type' de xeno-canto,
     mapeado a categorias normalizadas)
  3. Metrica: silhouette score y 1-NN accuracy (leave-one-out) sobre
     la matriz de similitud coseno entre bundles
  4. Null model: 1000 permutaciones de etiquetas, p-value empirico

Si 1-NN accuracy > percentil 95 del null -> hay estructura real.
"""
import json
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hrr_encoder import PalomaHRREncoder, make_codebooks

BASE = Path(__file__).parent.parent
DATA = BASE / "data"


def map_type(t):
    t = (t or "").lower()
    if "courtship" in t or "song" in t:
        return "cortejo"
    if "alarm" in t or "flight" in t:
        return "alerta"
    if "begging" in t or "nest" in t:
        return "demanda"
    if "wing" in t:
        return "mecanica"
    if "call" in t:
        return "contacto"
    return "otro"


def main():
    df = pd.read_csv(DATA / "features_large.csv")
    df = df[df.valid].reset_index(drop=True)
    meta = json.loads((DATA / "xeno_canto_metadata_large.json").read_text())
    rec_map = {str(r["id"]): r for r in meta}
    df["recordist"] = df["file"].map(
        lambda f: rec_map.get(Path(f).stem, {}).get("rec", "unknown"))
    df["type_raw"] = df["file"].map(
        lambda f: rec_map.get(Path(f).stem, {}).get("type", ""))
    df["conducta"] = df["type_raw"].map(map_type)

    # Solo clases con >=4 ejemplos (estimable)
    counts = df.conducta.value_counts()
    keep = counts[counts >= 4].index.tolist()
    df = df[df.conducta.isin(keep)].reset_index(drop=True)
    print(f"Clases usadas: {keep}  (n={len(df)})")
    print(df.conducta.value_counts().to_string())

    # Bundles
    N = 512
    encoder = PalomaHRREncoder(N=N)
    uniq_rec = list(dict.fromkeys(df["recordist"]))
    bundles = []
    for _, row in df.iterrows():
        b, _ = encoder.encode_fact(row.to_dict(), row["recordist"])
        bundles.append(b)
    B = np.array(bundles)
    Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
    S = Bn @ Bn.T  # matriz de similitud coseno

    y = df["conducta"].values
    n = len(y)

    def knn_acc(S, y):
        np.fill_diagonal(S_use := S.copy(), -np.inf)
        nn = S_use.argmax(axis=1)
        return float((y[nn] == y).mean())

    # Silhouette simple sobre similitud (a - b)/max(a,b)
    def silhouette(S, y):
        vals = []
        for i in range(len(y)):
            same = S[i, y == y[i]]
            diff = S[i, y != y[i]]
            a = same[same < 0.9999].mean() if (same < 0.9999).any() and (y == y[i]).sum() > 1 else same.mean()
            a = np.mean([s for j, s in enumerate(S[i]) if y[j] == y[i] and j != i])
            b = diff.mean() if len(diff) else 0.0
            if max(a, b) > 0:
                vals.append((a - b) / max(abs(a), abs(b)))
        return float(np.mean(vals))

    acc_real = knn_acc(S, y)
    sil_real = silhouette(S, y)
    print(f"\nReal:  1-NN acc = {acc_real:.3f}   silhouette = {sil_real:.3f}")

    # Null model: permutar etiquetas
    rng = np.random.RandomState(0)
    null_acc, null_sil = [], []
    for it in range(500):
        yp = rng.permutation(y)
        null_acc.append(knn_acc(S, yp))
        null_sil.append(silhouette(S, yp))
    null_acc = np.array(null_acc); null_sil = np.array(null_sil)

    p_acc = (np.sum(null_acc >= acc_real) + 1) / (len(null_acc) + 1)
    p_sil = (np.sum(null_sil >= sil_real) + 1) / (len(null_sil) + 1)

    print(f"\nNull model (500 permutaciones):")
    print(f"  1-NN acc null: {null_acc.mean():.3f} ± {null_acc.std():.3f}")
    print(f"  silhouette null: {null_sil.mean():.3f} ± {null_sil.std():.3f}")
    print(f"\n  p-value accuracy:  {p_acc:.4f}")
    print(f"  p-value silhouette: {p_sil:.4f}")
    sig = "SI" if p_acc < 0.05 or p_sil < 0.05 else "NO"
    print(f"\n  Hay estructura conductual significativa: {sig}")

    out = {
        "n_clips": int(n),
        "classes": keep,
        "acc_1nn_real": acc_real,
        "silhouette_real": sil_real,
        "null_acc_mean": float(null_acc.mean()),
        "null_acc_std": float(null_acc.std()),
        "null_sil_mean": float(null_sil.mean()),
        "null_sil_std": float(null_sil.std()),
        "p_acc": float(p_acc),
        "p_sil": float(p_sil),
    }
    (DATA / "validacion_semiauto.json").write_text(json.dumps(out, indent=2))
    print(f"\nGuardado: {DATA/'validacion_semiauto.json'}")


if __name__ == "__main__":
    main()
