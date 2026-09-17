# ============================================
# MODULO B: NORMALIZADOR TOPOLOGICO
# (El Puente hacia SGM)
# ============================================
#
# Este modulo toma las 4 metricas crudas extraidas por el Modulo A
# y las transforma en un vector omega (∈ R^4) compatible con el
# motor cognitivo SGM del Modulo C.
#
# Flujo del pipeline:
#   1. Recepcion de metricas crudas: [pitch, head_bob, throat, tension]
#   2. Heuristicas de normalizacion a rango [-1.0, 1.0]
#   3. Normalizacion L2 a la esfera unitaria (||ω|| = 1)
#   4. Output: vector omega listo para SGM
#
# Ecuaciones SGM aplicadas:
#   - Restriccion matematica: ω_sphere = ω_raw / ||ω_raw||_2
#   - Si ω_raw = 0 → ω_sphere = 0 (caso degenerado)
#
# Referencia: SGM v1.0 §3 — El vector semantico omega es la
# representacion pura del conocimiento, sin texto asociado.
# Su magnitud debe ser unitaria para que la similitud coseno
# (Eq. 2) funcione correctamente en el grafo.

import numpy as np


class TopologicalNormalizer:
    """
    Puente entre el codificador sensorial L1 y el motor SGM.
    
    Convierte features crudos del dominio fisico (Hz, ratios)
    en vectores semanticos normalizados del dominio topologico.
    
    La normalizacion es CRITICA para el correcto funcionamiento
    del SGM porque:
    
    1. La similitud coseno (Eq. 2) requiere vectores unitarios:
       cos(θ) = (ω_a · ω_b) / (||ω_a|| · ||ω_b||)
       Si ||ω|| ≠ 1, la similitud no representa puramente
       el angulo entre conceptos.
    
    2. El movimiento por afinidad (Eq. 2) usa la norma:
       P(m|n) ∝ exp(-α · ||ω_m - ω_n||)
       Vectores no normalizados distorsionan las distancias.
    
    3. La interferencia de ondas (Eq. 7) depende de ||ω||:
       I_i(t) = ||ω_i|| · cos(φ_i - φ_root)
       Solo con ||ω|| = 1, la interferencia depende puramente
       de la fase.
    """
    
    def __init__(self):
        """
        Inicializa el normalizador con parametros biologicos
        de referencia para Columba livia.
        
        Rangos biologicos observados:
        - pitch: 100-400 Hz (arrullo tipico)
        - head_bob: 0-8 Hz (frecuencia maxima observable)
        - throat_inflation: 0.0-1.0 (ya normalizado)
        - flight_tension: 0.0-1.0 (ya normalizado)
        """
        # Rangos biologicos para normalizacion z-score adaptada
        self.pitch_mean = 250.0   # Hz (centro del rango)
        self.pitch_std = 75.0     # Hz (cubre 100-400)
        
        self.head_bob_max = 6.0   # Hz (maximo biologico razonable)
        
        # Factores de escala para equilibrar las dimensiones
        # Cada dimension tiene un peso diferente en el comportamiento
        self.dimension_weights = np.array([1.0, 1.0, 1.0, 1.0])
    
    def normalize_pitch(self, pitch_hz):
        """
        Normaliza la frecuencia fundamental del arrullo.
        
        Metodo: z-score truncado a [-1, 1]
        - pitch = 250 Hz → 0.0 (valor central)
        - pitch = 100 Hz → -1.0 (bajo)
        - pitch = 400 Hz → +1.0 (alto)
        
        Interpretacion semantica:
        - Valores negativos: arrullo grave, lento → estado calmado
        - Valores positivos: arrullo agudo, rapido → estado agitado
        - Cercano a 0: arrullo tipico, neutral
        
        Conexion SGM:
            La frecuencia del arrullo es el indicador acustico mas
            directo del estado emocional. En el grafo, se alinea
            con nodos de tipo "calma" (ω_dim0 ≈ -1) o "alerta"
            (ω_dim0 ≈ +1).
        
        Args:
            pitch_hz: Frecuencia fundamental en Hz (de Modulo A)
        
        Returns:
            float en rango [-1.0, 1.0]
        """
        if pitch_hz <= 0:
            return 0.0
        
        # z-score
        z = (pitch_hz - self.pitch_mean) / self.pitch_std
        
        # Truncar a [-1, 1] (clip)
        return float(np.clip(z, -1.0, 1.0))
    
    def normalize_head_bob(self, head_bob_freq):
        """
        Normaliza la frecuencia de tics de cabeza.
        
        Metodo: normalizacion por maximo
        - 0 Hz → -1.0 (sin movimiento)
        - 3 Hz → 0.0 (promedio)
        - 6+ Hz → +1.0 (maximo)
        
        Interpretacion semantica:
        - Valores negativos: quieto, atento sin moverse
        - Valores positivos: exploracion activa, cortejo intenso
        - Cercano a 0: comportamiento social moderado
        
        Conexion SGM:
            Los head-bobs son gestos comunicativos en Columba livia.
            La frecuencia determina el tipo de mensaje: cortejo
            (ritmico), exploracion (erratico), o alerta (ausente).
        
        Args:
            head_bob_freq: Frecuencia en Hz (de Modulo A)
        
        Returns:
            float en rango [-1.0, 1.0]
        """
        # Normalizar: 0→-1, max/2→0, max→+1
        normalized = 2.0 * (head_bob_freq / self.head_bob_max) - 1.0
        return float(np.clip(normalized, -1.0, 1.0))
    
    def normalize_throat(self, throat_inflation):
        """
        Normaliza el ratio de expansion del cuello.
        
        Metodo: mapeo lineal a [-1, 1]
        - 0.0 (sin expansion) → -1.0
        - 0.5 (expansion media) → 0.0
        - 1.0 (maxima expansion) → +1.0
        
        Interpretacion semantica:
        - Valores negativos: cuello relajado, estado neutral
        - Valores positivos: cuello inflado, exhibicion de cortejo
        - Valores cercanos a 0: inflacion moderada, vocalizacion
        
        Conexion SGM:
            La inflacion del cuello en machos de Columba livia es
            un display sexual. En el grafo, valores altos se
            asocian con el nodo "cortejo".
        
        Args:
            throat_inflation: Ratio 0.0-1.0 (de Modulo A)
        
        Returns:
            float en rango [-1.0, 1.0]
        """
        # Mapeo: [0, 1] → [-1, 1]
        return float(2.0 * throat_inflation - 1.0)
    
    def normalize_tension(self, flight_tension):
        """
        Normaliza el nivel de tension postural.
        
        Metodo: mapeo lineal a [-1, 1]
        - 0.0 (relajado) → -1.0
        - 0.5 (moderado) → 0.0
        - 1.0 (alerta maxima) → +1.0
        
        Interpretacion semantica:
        - Valores negativos: postura relajada, descanso
        - Valores positivos: rigida, alerta, listo para volar
        - Cercano a 0: estado de calma vigilante
        
        Conexion SGM:
            La tension postural es el indicador mas directo del
            estado de alerta. En el grafo, valores altos activan
            el nodo "alerta" y pueden disparar phase-hijacking
            si la valencia supera θ_emerg = 0.30.
        
        Args:
            flight_tension: Ratio 0.0-1.0 (de Modulo A)
        
        Returns:
            float en rango [-1.0, 1.0]
        """
        # Mapeo: [0, 1] → [-1, 1]
        return float(2.0 * flight_tension - 1.0)
    
    def normalize_to_sphere(self, omega_raw):
        """
        Normaliza un vector a la esfera unitaria (L2 norm = 1).
        
        Este es el paso CRITICO del pipeline. Garantiza que:
        ||ω||_2 = sqrt(ω_0² + ω_1² + ω_2² + ω_3²) = 1.0
        
        Si el vector es cero (todas las metricas son cero o no
        se pudieron medir), se retorna el vector cero.
        
        Relevancia para SGM:
            - Eq. 2 (afinidad): P(m|n) ∝ exp(-α · ||ω_m - ω_n||)
              Con vectores unitarios, ||ω_m - ω_n|| representa
              puramente la distancia angular entre conceptos.
            
            - Eq. 7 (interferencia): I_i = ||ω_i|| · cos(φ_i - φ_root)
              Con ||ω_i|| = 1, la interferencia solo depende de
              la fase, separando claramente semantica (ω) de
              temporalidad (φ).
        
        Args:
            omega_raw: Array numpy de 4 dimensiones en rango [-1, 1]
        
        Returns:
            omega_sphere: Vector unitario (∈ R^4, ||ω|| = 1) o vector 0
        """
        omega_raw = np.array(omega_raw, dtype=np.float64)
        
        # Caso degenerado: vector cero
        norm = np.linalg.norm(omega_raw)
        if norm < 1e-10:
            return np.zeros(4, dtype=np.float64)
        
        # Normalizacion L2 a la esfera unitaria
        omega_sphere = omega_raw / norm
        
        # Verificacion de invariante (debug)
        final_norm = np.linalg.norm(omega_sphere)
        assert abs(final_norm - 1.0) < 1e-6, f"Norma L2 fallida: {final_norm}"
        
        return omega_sphere
    
    def process(self, features):
        """
        Pipeline completo de normalizacion.
        
        Toma las metricas crudas del Modulo A y produce el vector
        omega normalizado para el Modulo C (SGM).
        
        Flujo:
        1. Normalizar cada dimension individualmente a [-1, 1]
        2. Aplicar pesos dimensionales
        3. Proyectar a la esfera unitaria (L2 norm = 1)
        
        Args:
            features: dict con metricas del Modulo A:
                {
                    'pitch_hz': float,
                    'head_bob_freq': float,
                    'throat_inflation': float,
                    'flight_tension': float
                }
        
        Returns:
            dict con:
                {
                    'omega_raw': array [-1, 1]^4 (antes de L2),
                    'omega_sphere': array unitario (∈ esfera S^3),
                    'omega_norm_l2': float (debe ser 1.0),
                    'normalization_steps': dict (pasos intermedios)
                }
        """
        # --- Paso 1: Normalizar cada dimension a [-1, 1] ---
        pitch_norm = self.normalize_pitch(features.get('pitch_hz', 0))
        head_bob_norm = self.normalize_head_bob(features.get('head_bob_freq', 0))
        throat_norm = self.normalize_throat(features.get('throat_inflation', 0))
        tension_norm = self.normalize_tension(features.get('flight_tension', 0))
        
        omega_raw = np.array([
            pitch_norm,
            head_bob_norm,
            throat_norm,
            tension_norm
        ], dtype=np.float64)
        
        # --- Paso 2: Aplicar pesos dimensionales ---
        omega_weighted = omega_raw * self.dimension_weights
        
        # --- Paso 3: Proyectar a esfera unitaria ---
        omega_sphere = self.normalize_to_sphere(omega_weighted)
        
        result = {
            'omega_raw': omega_raw,
            'omega_sphere': omega_sphere,
            'omega_norm_l2': float(np.linalg.norm(omega_sphere)),
            'normalization_steps': {
                'pitch_normalized': round(pitch_norm, 4),
                'head_bob_normalized': round(head_bob_norm, 4),
                'throat_normalized': round(throat_norm, 4),
                'tension_normalized': round(tension_norm, 4),
            },
            'semantic_reading': self._interpret_semantic_reading(omega_sphere)
        }
        
        return result
    
    def _interpret_semantic_reading(self, omega_sphere):
        """
        Proporciona una lectura semantica del vector omega.
        
        Esto es util para debugging y para la interfaz de usuario.
        No es usado por el motor SGM (que opera puramente con
        vectores numericos).
        
        Args:
            omega_sphere: Vector unitario en R^4
        
        Returns:
            dict con interpretacion de cada dimension
        """
        dims = ['pitch', 'head_bob', 'throat', 'tension']
        reading = {}
        
        for i, dim in enumerate(dims):
            val = omega_sphere[i]
            if val > 0.3:
                level = "alto"
            elif val > -0.3:
                level = "neutral"
            else:
                level = "bajo"
            reading[dim] = {
                'value': round(val, 4),
                'level': level
            }
        
        return reading
    
    def batch_normalize(self, features_list):
        """
        Normaliza una lista de vectores de features.
        
        Util para procesar multiples ventanas de tiempo o
        multiples videos.
        
        Args:
            features_list: Lista de dicts con metricas
        
        Returns:
            Lista de dicts con vectores omega normalizados
        """
        return [self.process(f) for f in features_list]
