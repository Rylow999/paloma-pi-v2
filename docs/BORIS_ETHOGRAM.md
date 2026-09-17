# Ethogram para validación Paloma-π v2 en BORIS

5 conductas base (cubren lo que el audio puede separar):

| Code | Conducta | Categoría |
|---|---|---|
| CTX | Cortejo (arrullo + display) | Evento |
| ALT | Alerta (tensión postural alta) | Estado |
| AGG | Agonismo (persecución/ataque) | Evento |
| FEED | Alimentación (picoteo) | Evento |
| REST | Reposo (plumaje relajado) | Estado |

**Procedimiento sugerido:**
1. `cd ~/paloma-pi-v2 && .venv-boris/bin/python -m boris` (abre BORIS GUI)
2. File > New project > Add media (elegir `data/audio_samples/*.mp3`)
3. Preferences > Ethogram > Importar tabla de arriba (o crear manualmente)
4. Etiquetar por tiempo los clips (5-10 min por clip)
5. Export > Events table (CSV) → me pasás el CSV
6. Yo mapeo: conducta etiquetada ↔ resonator predictions → accuracy etológica

**Criterio de éxito (delta mínimo):**
- El resonator asigna CTX a clips que vos etiquetaste como cortejo: >70% = pasa.
- El resonator separa ALT de CTX: accuracy de discriminación >80% = pasa.

Si falla, ajustamos el codebook (símbolos más finos) o el threshold de confianza.
