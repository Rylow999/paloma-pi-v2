# -*- coding: utf-8 -*-
"""paloma_pi_v2/src/boca_minima.py — La boca de Paloma-π.

Resonator (oído) decodifica el bundle HRR a símbolos limpios;
reglas de decisión (boca mínima) convierten símbolos a palabra.

Sin LLM. Sin entrenamiento. La boca es una TABLA, no una red.

Arquitectura (el patrón del paper FHRR aplicado):
    bundle HRR → resonator puro → símbolos → reglas → palabra
    (oído)       (decoder)        (estructura) (decisión) (voz)

El LLM entrenado (transductor verbalizador) queda como EXTENSIÓN futura,
no como base: primero validar que las reglas simples alcanzan.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hrr_encoder import PalomaHRREncoder
from pure_resonator import PalomaPureResonator


# ================================================================
# La tabla de decisión (la boca mínima)
# ================================================================

# Clases conductuales (mismas que la validación semi-automática)
# Cada clase se define por las condiciones sobre los símbolos decodificados.
# Los umbrales salen de la literatura bioacústica + los datos medidos:
#   - cortejo: pitch medio-alto + arrullo largo + muchas sílabas
#   - contacto: pitch medio + arrullo corto
#   - alerta: pitch alto + energía alta (SNR bajo = ruido de fondo)
#   - demanda: pitch bajo + pocas sílabas (begging call)
#   - mecánica: energía de alas (SNR bajo + pitch indefinido)

def decidir(sym, pitch_hz_est, rms_est):
    """Reglas de decisión calibradas con los percentiles REALES del dataset.

    Calibración (features_large.csv, 83 clips válidos):
        pitch_hz:   p25=181, p50=263, p75=363  (rango 129-421)
        n_syllables: p25=29, p50=71, p75=151   (rango 3-575)
        duration_s: p25=24, p50=49, p75=105    (rango 1.2-514)
        snr:        p25=0.69, p50=0.81, p75=0.92

    Los buckets del resonator: pitch bucketizado (pm_N) mapea a
    100 + (N+0.5)*25 Hz, o sea pm_3 = 188 Hz, pm_11 = 388 Hz.

    Reglas en TÉRMINOS DE LOS DATOS (no de buckets elegidos a mano):
        cortejo (song): la clase mayoritaria — pitch medio, sílabas medias
        contacto (call): pitch medio-bajo, clip corto
        alerta (alarm/flight): pitch alto + SNR bajo (ruido de fondo)
        demanda (begging): sílabas pocas (bucket bajo)
        mecanica (wing): pitch indefinido o SNR muy bajo
    """
    def b2n(s, pref):
        try:
            return int(s.split(pref + "_")[1])
        except (IndexError, ValueError):
            return 0

    pm = b2n(sym.get("PITCH", sym.get("PITCH_MED", "pm_0")), "pm")
    sy = b2n(sym.get("SYLL", "syll_0"), "syll")
    du = b2n(sym.get("DUR", "dur_0"), "dur")
    sn = b2n(sym.get("SNR", "snr_0"), "snr")

    # Umbrales calibrados: buckets en unidades de los datos reales
    # pm=4 ↔ 213 Hz (p25 del pitch), pm=10 ↔ 363 Hz (p75)
    # sy=1 ↔ ~6 sílabas (p bajo), sy=4 ↔ ~64 (p50), sy=6 ↔ ~151 (p75)
    # du=2 ↔ ~8s, du=4 ↔ ~49s (p50), du=6 ↔ ~105s (p75)
    # sn=13 ↔ 0.81 (p50)
    pitch_hz = 100 + (pm + 0.5) * 25.0

    # 1. mecánica: SNR muy bajo (p<25) → ruido/alas
    if sn < 10:
        return "mecanica", 0.65
    # 2. alerta: pitch por encima del p75 (363Hz → pm>=10)
    if pm >= 10 and sn < 13:
        return "alerta", 0.7
    # 3. demanda: sílabas por debajo del p25 (29 → sy<=2) Y duración corta
    if sy <= 2 and du <= 3:
        return "demanda", 0.7
    # 4. cortejo: la clase por defecto de pitch medio + sílabas medias+
    if pm >= 4 and sy >= 3:
        return "cortejo", 0.75
    # 5. contacto: el resto (pitch bajo-medio, clip corto)
    return "contacto", 0.6


# ================================================================
# El pipeline completo (oído + boca)
# ================================================================

class PalomaBoca:
    """Pipeline completo: features → HRR → resonator → reglas → palabra."""

    def __init__(self, N=512, seed=7):
        self.N = N
        self.encoder = PalomaHRREncoder(N=N)

    def hablar(self, feats, recordist="unknown", T=150, codebooks=None):
        """Un 'latido' de Paloma-π: features → palabra.

        feats: dict de extractor_audio (pitch_hz, n_syllables, duration_s, snr_estimate)
        recordist: identidad (para el rol ID)
        codebooks: dict rol->[símbolos] (si None, se generan estándar)

        Devuelve dict con: palabra, confianza, símbolos decodificados, truth.
        """
        if codebooks is None:
            from hrr_encoder import make_codebooks
            codebooks = make_codebooks([recordist])

        bundle, truth = self.encoder.encode_fact(feats, recordist)
        res = PalomaPureResonator(codebooks, N=self.N, seed=7)
        sym = res.decode(bundle, T=T)

        # Estimar pitch y rms de los símbolos decodificados (para las reglas)
        def b2n(s, pref):
            try:
                return int(s.split(pref + "_")[1])
            except (IndexError, ValueError):
                return 0
        pm = b2n(sym.get("PITCH", sym.get("PITCH_MED", "pm_0")), "pm")
        pitch_hz_est = 100 + (pm + 0.5) * (400 / 16)
        rms_est = 0.1  # placeholder (el resonator no decodifica rms directo)

        palabra, conf = decidir(sym, pitch_hz_est, rms_est)
        return {
            "palabra": palabra,
            "confianza": conf,
            "simbolos": sym,
            "truth": truth,
            "pitch_hz_est": pitch_hz_est,
        }
