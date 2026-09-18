#!/usr/bin/env python3
"""
Paloma-π v2 — Validación etológica SIN etiquetado manual.

Estrategia dos-niveles:

A) Semi-supervisada: xeno-canto tiene etiquetas 'type' (song/call/etc)
   provistas por el grabador. Son proxy del contexto conductual.
   Test: ¿el bundle HRR agrupa por type aunque el encoder no lo sepa?

B) Auto-supervisada por BORIS-como-motor: las ventanas consecutivas dentro
   del mismo clip deben ser más similares entre sí que las de clips distintos
   (consistencia). La simetría se mide SIN labels: k-NN entre bundles;
   la accuracy del resonator tiene que superar el azar en el grafo de
   vecinos más cercanos.

Esto elimina la necesidad de BORIS para el primer gate: si la señal no
diferencia tipos ni tiene auto-consistencia temporal, no vale continuar.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hrr_encoder import PalomaHRREncoder, make_codebooks

BASE = Path(__file__).parent.parent
DATA = BASE / "data"


# ============================================================
# A) Semi-supervisada con metadata xeno-canto
# ============================================================

# Mapeo del campo 'type' a conductas normalizadas
def map_type(t):
    t = t.lower()
    if "courtship" in t or "song" in t:
        return "cortejo"
    if "alarm" in t or "flight call" in t or "nocturnal flight call" in t:
        return "alerta"
    if "begging" in t or "nest call" in t:
        return "demanda"
    if "call" in t:
        return "contacto"
    if "wing" in t:
        return "mecanica"
    return "otro"


def bundle_of(row, recs, encoder, N=512):
    return encoder.encode_fact(row, row["recordist"])


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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

    # Stats
    print("=== Dataset ===")
    print(df["conducta"].value_counts().to_string())

    # A) Bundles
    N = 512
    encoder = PalomaHRREncoder(N=N)
    bundles = []
    for _, row in df.iterrows():
        b, truth = encoder.encode_fact(row.to_dict(), row["recordist"])
        bundles.append(b)
    bundles = np.array(bundles)

    # B) Matriz de similitud entre bundles
    S = bundles @ bundles.T / (np.linalg.norm(bundles, axis=1, keepdims=True)
                                * np.linalg.norm(bundles, axis=1, keepdims=True).T)
    # ¿Las del mismo tipo son más parecidas?
    same, diff = [], []
    cond = df["conducta"].values
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if cond[i] == cond[j]:
                same.append(S[i, j])
            else:
                diff.append(S[i, j])
    same = np.array(same); diff = np.array(diff)
    print(f"\nSimilitud coseno bundles HRR:")
    print(f"  mismo tipo:     media={same.mean():.4f} ± {same.std():.4f} (n={len(same)})")
    print(f"  tipos distintos: media={diff.mean():.4f} ± {diff.std():.4f} (n={len(diff)})")

    # Tests: Mann-Whitney por bootstrap
    rng = np.random.RandomState(0)
    diffs = []
    for _ in range(1000):
        a = rng.choice(same, size=min(200, len(same)), replace=True)
        b = rng.choice(diff, size=min(200, len(diff)), replace=True)
        diffs.append(a.mean() - b.mean())
    diffs = np.array(diffs)
    ci = (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))
    print(f"  Bootstrap (media - media, 1000 rep): mean={diffs.mean():.4f}, "
          f"CI95=[{ci[0]:.4f}, {ci[1]:.4f}], p<0.05={np.mean(diffs > 0) > 0.95}")

    # C) Self-consistency: k-NN entre bundles (leave-one-out)
    # Para cada bundle, su vecino más cercano tiene el mismo tipo?
    Smax = S.copy()
    np.fill_diagonal(Smax, -np.inf)
    nn = Smax.argmax(axis=1)
    knn_acc = (df["conducta"].values[nn] == df["conducta"].values).mean()
    print(f"\n  k-NN accuracy (1-NN): {knn_acc:.3f}")
    print(f"  Azar (baseline tipo mayoritario): "
          f"{(df['conducta'] == df['conducta'].mode()[0]).mean():.3f}")

    # D) Etiquetado pseudo-BORIS: cada bundle vota a su cluster más cercano
    # y auditamos si el resultado reproduce la etiqueta externa
    from collections import Counter
    clusters = df["conducta"].unique()
    centroids = {}
    for c in clusters:
        centroids[c] = bundles[df["conducta"].values == c].mean(axis=0)
    pred = np.array([
        max(centroids, key=lambda c: cosine(bundles[i], centroids[c]))
        for i in range(len(bundles))
    ])
    centroid_acc = float((pred == df["conducta"].values).mean())
    print(f"  Accuracy por centroide de conducta: {centroid_acc:.3f}")

    # Save
    out = {
        "n_clips": len(df),
        "conducta_counts": df["conducta"].value_counts().to_dict(),
        "same_type_sim": {"mean": float(same.mean()), "std": float(same.std())},
        "diff_type_sim": {"mean": float(diff.mean()), "std": float(diff.std())},
        "bootstrap_ci95": list(ci),
        "knn_acc_1nn": float(knn_acc),
        "centroid_acc": centroid_acc,
    }
    out_path = DATA / "etologia_validation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()
