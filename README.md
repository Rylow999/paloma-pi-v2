# Paloma-π v2

**Decoding animal vocalizations via observer-relative VSA — pure resonator HRR on real bioacoustic data.**

Author: Luciano Benjamín Nieto · Technical assistance: Nexus (agent)
License: MIT

---

## What it is

A pipeline showing that the vocal signal of *Columba livia* (and related
species) **contains recoverable compositional structure** — without
supervised training, without an LLM, without manual labeling.

The theoretical basis is the finding of the sibling repository
[`Rylow999/fhrr-rho-collapse`](https://github.com/Rylow999/fhrr-rho-collapse):

> A decoder's "collapse" is not a property of the signal but of the observer.
> A pure resonator (no Gram correction) recovers the full structure across
> the entire superposition grid, including the ρ=1 singularity.

## Key results

| Result | Value |
|---|---|
| Roles decoded by the pure resonator | **6/6 with accuracy 1.000** (83 real clips) |
| Behavioral structure (1-NN vs null model, 500 permutations) | **p = 0.002** (acc 0.878 vs chance 0.686) |
| Temporal pitch trajectory (5s windows, TIME role) | **60–80% accuracy** on warbles up to 8.5 min |
| Recorder identification (59 distinct) | **accuracy 1.000** |
| Natural clusters (k-means, no labels) | 2 families: 171Hz/30s vs 301Hz/108s |
| Noise robustness (σ up to 0.2) | no degradation |

## Repository structure

```
paloma-pi-v2/
├── src/
│   ├── extractor_audio.py     # YIN pitch + 100-450Hz bandpass + VAD
│   ├── hrr_encoder.py         # features → HRR bundle (6 roles)
│   ├── pure_resonator.py      # Gram-free decoder (pure resonator)
│   ├── normalizer.py          # port of the original Module B (spherical omega)
│   ├── boca_minima.py         # resonator + rules → word (honest negative result)
│   ├── eval_run.py            # standard evaluation (10 clips)
│   ├── eval_run_large.py      # large-dataset evaluation (83 clips)
│   ├── exp_temporal_sequence.py  # sliding windows + TIME role
│   ├── etologia_sin_boris.py  # validation with xeno-canto metadata
│   └── stress_test.py         # noise robustness
├── data/                      # feature CSVs + result JSONs (audio kept off-repo)
├── figures/                   # analysis figures
├── docs/
│   ├── DESIGN.md              # v2 vs v1 design
│   ├── RESULTS_REPORT.md      # full report with findings
│   └── BORIS_ETHOGRAM.md      # ethogram for future manual validation
└── papers/                    # reference bioacoustics paper notes
```

## How to run

```bash
git clone https://github.com/Rylow999/paloma-pi-v2.git
cd paloma-pi-v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download recordings (requires a free xeno-canto API key)
export XENO_KEY="your_api_key"
python3 scripts/download_large.py

# 2. Extract features
python3 -c "from src.extractor_audio import batch_extract; batch_extract('data/audio_xc_large', 'data/features_large.csv')"

# 3. Evaluate the resonator
python3 src/eval_run_large.py

# 4. Semi-automatic ethological validation (no manual labeling)
python3 src/etologia_sin_boris.py

# 5. Temporal sequences
python3 src/exp_temporal_sequence.py
```

## The principle (observer-relativity applied)

Classical bioacoustic analysis (spectrogram + PCA + classifier) reports the
signal as "partially structured". The pure resonator reads it in full. The
difference is not in the bird — it is in the observer:

- The signal contains pitch, rhythm, identity and context (verified).
- Linear methods lose superposition (the ρ law of the sibling paper).
- The resonator recovers it without correction (Gram-free).

**Implication**: before concluding "animal signals are not compositional",
test at least two classes of decoders. A negative result may be an artifact
of the instrument.

## The minimal mouth (honest result)

We implemented the mouth (resonator + rules → word). **It failed**: accuracy
0.15 vs chance 0.70. The paper's lesson applying to itself: the encoder's
global roles do not carry the discriminant information — the temporal
structure does (already demonstrated). Right direction: encode the sequence,
not the average. Documented in `docs/RESULTS_REPORT.md`.

## Data

xeno-canto recordings are NOT versioned in this repository (GitHub 100MB
limit). The script `scripts/download_large.py` regenerates them with the API
key. Features and results ARE versioned.

## Reference papers

- Plate 1995 — Holographic Reduced Representations
- Frady et al. 2020 — Resonator networks
- Abs & Jeismann 1988 — Courtship songs of Columba livia (individuality)
- Partan et al. 2005 — Multisensory playbacks (multimodality)
- Suzuki et al. 2016 — Compositional syntax in Parus minor

## Motto

*Per Aspera, Ad Astra.*
