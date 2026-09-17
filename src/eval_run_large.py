"""Paloma-π v2 — evaluación en dataset grande (83 clips)."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from hrr_encoder import (PalomaHRREncoder, make_codebooks,
                         bucketize_pitch, bucketize_syll, bucketize_dur, bucketize_snr)
from pure_resonator import PalomaPureResonator

BASE = Path(__file__).parent.parent
DATA = BASE / 'data'

df = pd.read_csv(DATA / 'features_large.csv')
df = df[df.valid == True].reset_index(drop=True)
print(f'Validas: {len(df)} clips')

meta = json.loads((DATA / 'xeno_canto_metadata_large.json').read_text())
# Mapeo por filename -> recordista
name_map = {}
for r in meta:
    rid = r['id']
    rec = r.get('rec', 'unknown')
    name_map[f'{rid}.mp3'] = rec
df['recordist'] = df['file'].map(lambda f: name_map.get(Path(f).name, 'unknown'))

unique_recs = list(dict.fromkeys(df['recordist']))
print(f'Recordistas únicos: {len(unique_recs)}')

codebooks = make_codebooks(unique_recs)
print(f'Codebook ID: {len(codebooks["ID"])} candidatos')

N = 1024  # más dim para acomodar los >50 IDs
encoder = PalomaHRREncoder(N=N)
resonator = PalomaPureResonator(codebooks, N=N)

roles = list(codebooks.keys())
per_role = {r: [] for r in roles}
joints = []

for i, row in df.iterrows():
    r = row.to_dict()
    bundle, truth = encoder.encode_fact(r, r['recordist'])
    dec = resonator.decode(bundle, T=200)
    accs = [dec[k] == truth[k] for k in roles]
    joints.append(float(all(accs)))
    for k in roles:
        per_role[k].append(float(dec[k] == truth[k]))
    if i % 10 == 0:
        print(f'  [{i+1}/{len(df)}] joint={joints[-1]:.0f}')

res = {
    'n': len(df),
    'unique_recorders': len(unique_recs),
    'per_role_acc': {k: float(np.mean(v)) for k, v in per_role.items()},
    'joint_acc': float(np.mean(joints)),
    'codebook_sizes': {k: len(v) for k, v in codebooks.items()},
}
(DATA / 'v2_eval_large.json').write_text(json.dumps(res, indent=2))
print('\n=== RESULTADOS GRANDE ===')
for k, v in res['per_role_acc'].items():
    print(f'  {k}: {v:.3f}')
print(f'  JOINT: {res["joint_acc"]:.3f}')
