# Paloma-π v2

**Decoding animal vocalizations via observer-relative VSA — pure resonator HRR on real bioacoustic data.**

Autor: Luciano Benjamín Nieto · Asistencia técnica: Nexus (agente)
Licencia: MIT

---

## Qué es

Un pipeline que demuestra que la señal vocal de *Columba livia* (y especies
cercanas) **contiene estructura composicional recuperable** — sin
entrenamiento supervisado, sin LLM, sin etiquetado manual.

La base teórica es el hallazgo del repo hermano
[`Rylow999/fhrr-rho-collapse`](https://github.com/Rylow999/fhrr-rho-collapse):

> El "colapso" de un decoder no es propiedad de la señal sino del observador.
> Un resonator puro (sin corrección de Gram) recupera la estructura completa
> en todo el grid de superposición, incluida la singularidad ρ=1.

## Resultados principales

| Resultado | Valor |
|---|---|
| Roles decodificados por el resonator puro | **6/6 con accuracy 1.000** (83 clips reales) |
| Estructura conductual (1-NN vs null model, 500 permutaciones) | **p = 0.002** (acc 0.878 vs azar 0.686) |
| Trayectoria temporal del pitch (ventanas de 5s, rol TIME) | **60–80% accuracy** en arrullos de hasta 8.5 min |
| Identificación de grabador (59 distintos) | **accuracy 1.000** |
| Clusters naturales (k-means, sin etiquetas) | 2 familias: 171Hz/30s vs 301Hz/108s |
| Robustez al ruido (σ hasta 0.2) | sin degradación |

## Estructura

```
paloma-pi-v2/
├── src/
│   ├── extractor_audio.py     # YIN pitch + bandpass 100-450Hz + VAD
│   ├── hrr_encoder.py         # features → bundle HRR (6 roles)
│   ├── pure_resonator.py      # decoder sin Gram (resonator puro)
│   ├── normalizer.py          # port del Módulo B original (omega esférico)
│   ├── boca_minima.py         # resonator + reglas → palabra (resultado honesto)
│   ├── eval_run.py            # evaluación estándar (10 clips)
│   ├── eval_run_large.py      # evaluación dataset grande (83 clips)
│   ├── exp_temporal_sequence.py  # ventanas deslizantes + rol TIME
│   ├── etologia_sin_boris.py  # validación con metadata xeno-canto
│   └── stress_test.py         # robustez al ruido
├── data/                      # features CSV + resultados JSON (audio en disco local)
├── figures/                   # figuras del análisis
├── docs/
│   ├── DESIGN.md              # diseño v2 vs v1
│   ├── RESULTS_REPORT.md      # informe completo con hallazgos
│   └── BORIS_ETHOGRAM.md      # ethogram para validación manual futura
└── papers/                    # notas de papers bioacústicos de referencia
```

## Cómo correr

```bash
git clone https://github.com/Rylow999/paloma-pi-v2.git
cd paloma-pi-v2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Descargar grabaciones (requiere API key de xeno-canto, gratis)
export XENO_KEY="tu_api_key"
python3 scripts/download_large.py

# 2. Extraer features
python3 -c "from src.extractor_audio import batch_extract; batch_extract('data/audio_xc_large', 'data/features_large.csv')"

# 3. Evaluar el resonator
python3 src/eval_run_large.py

# 4. Validación etológica semi-automática (sin etiquetado manual)
python3 src/etologia_sin_boris.py

# 5. Secuencias temporales
python3 src/exp_temporal_sequence.py
```

## El principio (observer-relativity aplicado)

El análisis clásico de bioacústica (espectrograma + PCA + clasificador)
reporta la señal como "parcialmente estructurada". El resonator puro la lee
completa. La diferencia no está en el ave — está en el observador:

- La señal contiene pitch, ritmo, identidad y contexto (verificado).
- Los métodos lineales pierden la superposición (la ley ρ del paper hermano).
- El resonator la recupera sin corrección (sin Gram).

**Implicación**: antes de concluir "la señal animal no es composicional",
hay que probar al menos dos clases de decoders. El resultado negativo puede
ser un artefacto del instrumento.

## La boca mínima (resultado honesto)

Implementamos la boca (resonator + reglas → palabra). **Falló**: accuracy
0.15 vs azar 0.70. La lección del paper aplicándose a sí misma: los roles
globales del encoder no contienen la información discriminante — la
estructura temporal sí (ya demostrada). Dirección correcta: codificar la
secuencia, no el promedio. Documentado en `docs/RESULTS_REPORT.md`.

## Datos

Las grabaciones de xeno-canto NO se versionan en el repo (límite de GitHub
100MB). El script `scripts/download_large.py` las regenera con la API key.
Los features y resultados SÍ están versionados.

## Papers de referencia

- Plate 1995 — Holographic Reduced Representations
- Frady et al. 2020 — Resonator networks
- Abs & Jeismann 1988 — Courtship songs of Columba livia (individualidad)
- Partan et al. 2005 — Multisensory playbacks (multimodalidad)
- Suzuki et al. 2016 — Compositional syntax in Parus minor

*Per aspera.*
