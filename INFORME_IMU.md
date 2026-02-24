# Informe de desarrollo IMU (Android -> PC)

## Estado actual
- Proyecto: captura de IMU desde Android por HTTP en red local y procesado offline/live.
- Resultado actual: pipeline funcionando de extremo a extremo, incluyendo visualización en vivo con retardo controlado.
- Ajuste de procesado que mejor encaja hasta ahora: configuración sensible con `g_fc=0.03`, `fmin=0.02`, `fmax=1.20`, `height_min_change=0.00005`, `height_step_min=0.000001`, `confirm_samples=6`.

## Objetivo
- Capturar datos de IMU (al menos `t` y `az`, idealmente `ax, ay, az, gx, gy, gz`) desde Android.
- Guardar sesiones de ~30 s.
- Procesar señal para estimar dinámica vertical y altura relativa.
- Reducir ruido usando umbrales y confirmación temporal de movimiento.

## Bitácora por fechas

### 2026-02-11
- Se creó receptor inicial en `imu_server.py` (UDP, puerto 8000, procesado básico en vivo).
- Se separó el flujo en dos scripts:
  - `capture_http_imu.py`: captura por `POST /data`.
  - `process_imu_session.py`: procesado offline con métricas y gráficas.
- Se resolvió problema de conectividad WSL <-> móvil usando alternativas:
  - `portproxy` en Windows.
  - ejecución directa del script en Windows cuando conviene.
- Se confirmaron capturas válidas con archivos `session_..._30s.csv`.

### 2026-02-18
- Se mejoró `process_imu_session.py`:
  - soporte `.csv` y `.npz`.
  - modo sin gráfica (`--no-plot`) para entornos sin `matplotlib`.
  - `--height-min-change` (deadband de altura).
- Se creó `process_imu_step_by_step.py` para visualizar etapas:
  - `step1`: señal cruda.
  - `step2`: eliminación de gravedad.
  - `step3`: band-pass y periodo dominante.
  - `step4`: altura integrada.
  - `all`: vista completa.
- Se añadió confirmación temporal de cambios de altura:
  - `--height-step-min` (paso mínimo por muestra).
  - `--confirm-samples` (N muestras consecutivas para confirmar subida/bajada).
- Se amplió `capture_http_imu.py` para guardar columnas completas cuando llegan:
  - `t, ax, ay, az, gx, gy, gz` (si faltan campos, quedan como `NaN`).

### 2026-02-18 (actualización del día)
- Se preparó y rellenó la memoria del TFG en la plantilla oficial de la FIC:
  - personalización de `memoria_tfg.tex` (datos, dedicatoria, agradecimientos),
  - redacción de `introducion.tex`, `demo.tex`, `conclusions.tex`,
  - redacción de resumen y palabras clave en `portada/resumo.tex` y `portada/palabras_chave.tex`,
  - inclusión de gráficas de validación y compilación correcta con `xelatex`.
- Se añadió bibliografía técnica base y citas en el texto:
  - `TittertonWeston2004`, `Farrell2008`, `Oppenheim2010`, `AndroidSensors`.
- Se creó `live_imu_http.py` para procesado en vivo por `POST /data`:
  - modo live con gráfica (si hay `matplotlib`) o consola,
  - cálculo en vivo de `fs`, `Tp`, `std(h)`, `Hs`,
  - opción de guardado de sesión al salir (`--save-on-exit`).
- Se añadió procesado por bloques con retardo en live:
  - `--chunk-seconds` y `--fft-min-samples`.
- Se añadió modo de ventana deslizante en live:
  - `--window-seconds` + `--update-seconds` (ejemplo recomendado: ventana 30 s, update 1 s).
- Se unificaron parámetros offline/live para usar el mismo ajuste:
  - `--g-fc`, `--fmin`, `--fmax`,
  - `--height-min-change`, `--height-step-min`, `--confirm-samples`.
- Se implementó panel de control interactivo en live:
  - sliders para ajuste rápido,
  - cajas de texto + botón `Aplicar` para introducir valores exactos,
  - ajuste de layout para evitar solapamiento de textos.
- Se amplió `process_imu_step_by_step.py`:
  - procesado por bloques con `--chunk-seconds`,
  - mínimo FFT configurable con `--fft-min-samples`,
  - control de unidades y escala de altura en gráfica (`--height-unit`, `--height-ylim`).
- Se creó `auto_tune_imu.py`:
  - búsqueda iterativa de parámetros en varias rondas,
  - score de calidad por bloques,
  - persistencia del mejor set en `auto_tune_state.json`,
  - generación de comando recomendado para reutilizar el mejor ajuste.
- Se añadieron scripts de soporte WSL para puerto 8001:
  - `setup_wsl_portproxy_8001.bat`,
  - `remove_wsl_portproxy_8001.bat`.

## Flujo actual (recomendado)
1. Captura:
   - `python3 capture_http_imu.py --host 0.0.0.0 --port 8001 --seconds 30`
2. Procesado rápido (métricas):
   - `python3 process_imu_session.py session_YYYYMMDD_HHMMSS_30s.csv --no-plot`
3. Procesado paso a paso (gráfico):
   - `python3 process_imu_step_by_step.py session_YYYYMMDD_HHMMSS_30s.csv --mode all`
4. Procesado en vivo (ventana deslizante):
   - `python3 live_imu_http.py --host 0.0.0.0 --port 8001 --window-seconds 30 --update-seconds 1`
5. Autoajuste de parámetros:
   - `python3 auto_tune_imu.py session_YYYYMMDD_HHMMSS_30s.csv --chunk-seconds 5 --trials 120 --rounds 4`

## Configuraciones probadas de detección de cambio

### Serie estricta (sin confirmaciones)
- `height_min_change` en rango `0.001` a `0.010` m con `confirm_samples=20`.
- Resultado en `session_20260211_165829_30s.csv`: 0 subidas / 0 bajadas confirmadas.

### Serie sensible (ajustada)
Probadas con:
- `--g-fc 0.03 --fmin 0.02 --fmax 1.20`
- umbrales más pequeños y menos muestras de confirmación.

Comandos:
- `python3 process_imu_step_by_step.py session_20260211_165829_30s.csv --mode step4 --g-fc 0.03 --fmin 0.02 --fmax 1.20 --height-min-change 0.00010 --height-step-min 0.000002 --confirm-samples 8`
- `python3 process_imu_step_by_step.py session_20260211_165829_30s.csv --mode step4 --g-fc 0.03 --fmin 0.02 --fmax 1.20 --height-min-change 0.00005 --height-step-min 0.000001 --confirm-samples 6`  <-- **casi perfecta**
- `python3 process_imu_step_by_step.py session_20260211_165829_30s.csv --mode step4 --g-fc 0.03 --fmin 0.02 --fmax 1.20 --height-min-change 0.00002 --height-step-min 0.0000005 --confirm-samples 5`

## Interpretación técnica actual
- El pipeline funciona, pero es sensible a:
  - orientación del móvil,
  - frecuencia de movimiento real,
  - banda de filtros,
  - relación señal/ruido de aceleración vertical.
- Con movimientos suaves a mano, mucha energía cae en muy baja frecuencia y se puede perder si los filtros son estrictos.
- La confirmación por persistencia (N muestras) ayuda a evitar falsos positivos por ruido instantáneo.

## Próximos pasos sugeridos
- Hacer una sesión de calibración controlada:
  - 10 s quieto + 20 s movimiento vertical claro y periódico.
- Guardar automáticamente un reporte por sesión (JSON/MD) con parámetros y métricas.
- Opcional avanzado: usar orientación (`rotation vector`) en Android para proyectar aceleración al eje vertical global.

## Archivos clave
- `capture_http_imu.py`
- `process_imu_session.py`
- `process_imu_step_by_step.py`
- `live_imu_http.py`
- `auto_tune_imu.py`
- `setup_wsl_portproxy_8000.bat`
- `remove_wsl_portproxy_8000.bat`
- `setup_wsl_portproxy_8001.bat`
- `remove_wsl_portproxy_8001.bat`
- `auto_tune_state.json`
- `modelo-tfg-fic-v1.6_2223xun/memoria_tfg.tex`
- `modelo-tfg-fic-v1.6_2223xun/contido/introducion.tex`
- `modelo-tfg-fic-v1.6_2223xun/contido/demo.tex`
- `modelo-tfg-fic-v1.6_2223xun/contido/conclusions.tex`

---
Última actualización: 2026-02-18
