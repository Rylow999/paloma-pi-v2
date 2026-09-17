# Paloma-π v2 — Decoding multimodal animal signals with observer-relative VSA

**Autor:** Luciano Benjamín Nieto + Nexus (asistencia técnica)
**Base:** exp_SGM_0020 del pseudopaloma original + hallazgos FHRR (repo público)
**Especie target:** Columba livia (rock dove / paloma doméstica)
**Lema:** el problema no es si la señal "tiene composo" — es si existe un
decoder para el cual la estructura es visible. La ley de ρ del repo FHRR nos
dice dónde mirar.

---

## 1. Qué cambió respecto al diseño viejo (agosto 2026)

| Aspecto | Versión vieja (idea futura) | Versión v2 (ahora) |
|---|---|---|
| Decoder | similarity-NN (lineal, FAILED v0.25 top1=0.020) | **resonator puro** (9/10 accuracy inclusive en ρ=1, sin Gram) |
| Composición | un vector ω único ∈ R^4 | bundle HRR por roles (audio × pose) |
| Validación | similitud decide (subjetivo) | sweep de decoders: gram/pure/pinv/gradient/MLP |
| Ground truth | inexistente (nota crítica) | BORIS + xeno-canto real |
| Conducta | simulada | playback con registro etológico |

Repos públicos que alimentan esto:
- `Rylow999/fhrr-rho-collapse` — la ley de ρ, el observer-relativity result
- `pandora/transducer/state_encoder.py` (en SGM) — encoder state→HRR already working

---

## 2. Anatomía de la señal de Columba livia (paper-based)

Datos que tenemos:
- **Pitch del arrullo:** 100–400 Hz (estable, tonal)
- **Estructura silábica:** disyllabic, pattern `coo-COO-coo` con strophe 1.2–1.8 seg
- **Individuos distinguibles** (Abs & Jeismann 1988, Bioacoustics): 6 temporal
  relations differ across males; courtship songs individually stable
- **Multimodalidad video+audio necesaria:** Partan et al. 2005 (Animal Behaviour)
  showed females respond stronger to combined cues than audio alone

→ Esto **ya es una estructura composicional**: cada arrullo es [estructura
silábica (2 shots) × contenido del individuo (su signature)]. Se puede codificar
como HRR con roles `STRUCT`, `INDIVIDUAL`, `CONTEXT`.

## 3. El pipeline (maduro, reusa componentes reales)

```
AUDIO (xeno-canto .mp3 → PCM 22050Hz)
  │
  ▼
extract_features
  - pitch_hz (YIN / parselmouth / librosa yin)
  - spectral_centroid
  - syllable duration sequence (voice activity detection)
  │
  ▼
PGM HRR ENCODER  (reusa state_encoder.py patterns)
  - role STRUCT   = hash de la duración relativa sílaba 1 vs 2
  - role PITCH    = bucketized pitch_hz (16 buckets)
  - role IDENTITY = hash del individual si hay metadata
  - role CONTEXT  = señal externa si la hay (cortejo, agonismo, alarma)
  │
  ▼
PURE RESONATOR DECODER (resonator puro de pure_resonator.py)
  → recupera los 4 roles
  │
  ▼
VALIDACIÓN ETOLOGICA
  - matching self-consistency (bootstrap sobre la fecha del recordista)
  - cross-individual: dos individuos conocidos deben dar IDs distintos
```

Total: 3 componentes, cada uno probado individualmente en el repo FHRR.

## 4. Diferencias críticas con v1

1. **NO adaptativa todavía** — primero ver si el resonator puro puede decodificar
   la estructura fija (STRUCT/PITCH/IDENTITY). Si sí, agregar aprendizaje
   después. No mezclar los dos problemas.
2. **Métrica es accuracy de recuperación del rol**, igual que Exp 7 del FHRR.
   No "similaridad entre vectores" (la métrica vieja que fallaba).
3. **Multi-decoder sweep** como protocolo: gram y pinv sirven de negative
   controls; el MLP (si el resonator falla) mide qué tanto hay en la señal;
   gradient mide la riqueza disponible.

## 5. Datos reales — xeno-canto

La API pública de xeno-canto lista ~120 grabaciones etiquetadas
`Columba livia` con calidad A-B. Especies del género:
- Son todas `song` (courtship) o `call` (alarm/contact)
- Algunas tienen grabador/fecha/lugar — eso da self-supervised ID

Plan: descargar 30 grabaciones con calidad A, generar features con librosa,
armar un CSV `pitch_hz, syllable_t [t0,t1,t2], duration, recordist_id`, y
correr el pipeline.

Fallback: v1 mod convolucional sintético (como `exp_SGM_0020` original) si
la API no da ancho de banda.

## 6. Estado actual (16 sep 2026)

```bash
PALOMA_PI_V2/
├── docs/                ← este doc
├── data/                ← xeno-canto descargas + features.csv
├── src/
│   ├── extractor_audio.py      (pitch, syllables via librosa)
│   ├── hrr_encoder.py         (roles → bundle HRR: reusa patterns de Pandora)
│   ├── pure_resonator.py       (exactamente el de SGM, solo cambia codebook)
│   └── eval_run.py            (multi-decoder sweep sobre dataset real)
└── papers/
    ├── Abs_Jeismann_1988.md   (notas del paper bioacústico)
    ├── Partan_2005.md         (multimodalidad cortejo)
    └── Suzuki_2016.md         (compositional syntax in Parus minor)
```

## 7. Recordatorio para después

El objetivo **no es reemplazar el trabajo etológico**, es probar la tesis
observer-relative: si un solo decoder encuentra estructura consistente,
la señal tiene composicionalidad *para esa clase de observadores*, aunque el
uso más común (similitud coseno) no la vea.

Si no encontramos estructura, también es un resultado: las señales de paloma
en la base son holísticas (cada arrullo es una unidad), no composicionales.
Eso lo decía la literatura clásica. V2 le pone el número.

*Per aspera.*
