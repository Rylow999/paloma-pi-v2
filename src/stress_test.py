# Paloma-π v2: stress test — accuracy de ID en función de n_samples y ruído
import json, hashlib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from hrr_encoder import bucketize_pitch, bucketize_syll, bucketize_dur, bucketize_snr
from pure_resonator import PalomaPureResonator, _vec, hrr_bind


def bucket_pm(h): return int(np.clip((h - 100) / 400 * 15, 0, 15))
def bucket_ps(s): return int(np.clip(s / 120 * 15, 0, 15))

ROLES6 = ['PITCH_MED','PITCH_STD','SYLL','DUR','ID','SNR']

def encode6(row, N=512):
    truth = {
        'PITCH_MED': f'pm_{bucket_pm(row["pitch_hz"])}',
        'PITCH_STD': f'ps_{bucket_ps(row["pitch_std"])}',
        'SYLL': f'syll_{bucketize_syll(row["n_syllables"])}',
        'DUR': f'dur_{bucketize_dur(row["duration_s"])}',
        'ID': f'rec_{hashlib.sha1(str(row["recordist"]).encode()).hexdigest()[:6]}',
        'SNR': f'snr_{bucketize_snr(row["snr_estimate"])}',
    }
    b = np.zeros(N)
    for r in ROLES6:
        b += hrr_bind(_vec(f'__role_{r}__', N), _vec(truth[r], N))
    return b, truth


def run_trial(bundle, truth, codebooks, N=512, noise_sigma=0.0, seed=42):
    rng = np.random.RandomState(seed)
    if noise_sigma > 0:
        bundle = bundle + noise_sigma * rng.randn(len(bundle))
    res = PalomaPureResonator(codebooks, N=N, seed=seed)
    t0 = time.time()
    dec = res.decode(bundle, T=150)
    dt = time.time() - t0
    accs = {r: float(dec[r] == truth[r]) for r in truth}
    accs['joint'] = float(all(accs[r] for r in truth))
    return accs, dt


BASE = Path(__file__).parent.parent
DATA = BASE / 'data'
df = pd.read_csv(DATA/'features_real.csv')
df = df[df.valid == True].reset_index(drop=True)
meta = pd.read_json(DATA/'xeno_canto_metadata/recordings.json')
rec_map = {str(r['id']): r.get('rec','u') for r in meta.to_dict('records')}
df['recordist'] = df['file'].str.replace('.mp3','',regex=False).map(
    lambda x: rec_map.get(x.split('/')[-1], 'unknown'))

import time
results = []
noise_lvls = [0.0, 0.05, 0.1, 0.2]
n_ids = [1,2,3,5,10]

for n_id in n_ids:
  for noise in noise_lvls:
    accs = {r:[] for r in ROLES6}
    joints = []
    dts = []
    for trial in range(10):
      sub = df.sample(n=min(n_id,len(df)), random_state=trial)
      bundle, truth = encode6(sub.iloc[0])
      cbs = {
        'PITCH_MED': [f'pm_{i}' for i in range(16)],
        'PITCH_STD': [f'ps_{i}' for i in range(16)],
        'SYLL': [f'syll_{i}' for i in range(16)],
        'DUR': [f'dur_{i}' for i in range(16)],
        'ID': [f'rec_{hashlib.sha1(r.encode()).hexdigest()[:6]}' for r in sub.recordist.unique()],
        'SNR': [f'snr_{i}' for i in range(16)],
      }
      a, dt = run_trial(bundle, truth, cbs, noise_sigma=noise, seed=trial)
      for r in ROLES6: accs[r].append(a[r])
      joints.append(a['joint'])
      dts.append(dt)
    row = dict(n_id=n_id, noise=noise,
               ID=np.mean(accs['ID']),
               joint=np.mean(joints),
               time_ms=np.mean(dts)*1000)
    results.append(row)
    print(f'n_id={n_id:2d} noise={noise:.2f}  ID={row["ID"]:.3f} joint={row["joint"]:.3f}')

out = DATA/'stress_test_v4.json'
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {out}")
