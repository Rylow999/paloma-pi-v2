# Paloma-π v2: eval run sobre dataset xeno-canto real.
#
# Protocolo observer-relativity:
#   - Mismos bundles (encoder HRR sobre features reales)
#   - Mismo resonator PURO (sin Gram) — decoder candidato robusto
#   - Medimos accuracy de recuperacion por rol (PITCH, SYLL, DUR, ID, SNR)
#   - Agregacion: accuracy por rol + joint accuracy (todos los roles bien)
#
# Esto es el analogo a Exp 7 del FHRR paper, pero con AUDIO REAL de paloma
# (no sintetico) y una configuracion de 5 roles en vez de 6.

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hrr_encoder import PalomaHRREncoder, make_codebooks, bucketize_pitch
from pure_resonator import PalomaPureResonator


BASE = Path(__file__).parent.parent
DATA = BASE / "data"


def main():
    # Cargamos features extraidos por extractor_audio
    feats = pd.read_csv(DATA / "features_real.csv")
    metadata = json.loads((DATA / "xeno_canto_metadata/recordings.json").read_text())
    # recordist -> id (la metadata usa 'rec' como recordista)
    rec_ids = [r.get("rec", "unknown") for r in metadata[:len(feats)]]
    feats["recordist"] = rec_ids[:len(feats)]

    # Filtramos validos
    feats = feats[feats["valid"] == True].reset_index(drop=True)
    print(f"Grabaciones validas: {len(feats)} / {len(pd.read_csv(DATA/'features_real.csv'))}")

    # Codebooks: lista de recordists (para ID) + buckets fijos
    unique_recs = list(dict.fromkeys(feats["recordist"]))  # preservar orden
    codebooks = make_codebooks(unique_recs)

    # Encoder + resonator puro
    N = 512
    encoder = PalomaHRREncoder(N=N)
    resonator = PalomaPureResonator(codebooks, N=N)

    # Evaluacion
    roles = list(codebooks.keys())
    per_role_acc = {r: [] for r in roles}
    joint_accs = []

    for i, row in feats.iterrows():
        fdict = row.to_dict()
        bundle, truth = encoder.encode_fact(fdict, fdict["recordist"])
        dec = resonator.decode(bundle, T=150)
        accs = [dec[r] == truth[r] for r in roles]
        joint = float(all(accs))
        joint_accs.append(joint)
        for r in roles:
            per_role_acc[r].append(float(dec[r] == truth[r]))

    # Reporte
    print(f"\n=== Paloma-π v2: resonator puro sobre xeno-canto (n={len(feats)}) ===")
    for r in roles:
        a = per_role_acc[r]
        print(f"  rol {r:>6}: acc mean = {np.mean(a):.3f}")

    print(f"\n  JOINT accuracy (5/5 roles correctos): {np.mean(joint_accs):.3f}")
    print(f"  N = {N}, codebook sizes: { {r: len(c) for r, c in codebooks.items()} }")

    # Save results
    out = {
        "n_samples": len(feats),
        "roles": roles,
        "per_role_acc": {r: float(np.mean(per_role_acc[r])) for r in roles},
        "joint_acc": float(np.mean(joint_accs)),
        "N": N,
        "codebook_sizes": {r: len(c) for r, c in codebooks.items()},
    }
    (DATA / "v2_eval_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nResultados guardados en {DATA/'v2_eval_results.json'}")


if __name__ == "__main__":
    main()
