# Referencias sobre IMU para caracterización de olas (replicables)

## 1) OpenMetBuoy-v2021 (Geosciences, 2022)
- Paper: https://www.mdpi.com/2076-3263/12/3/110
- Código/hardware/procesado: https://github.com/jerabaul29/OpenMetBuoy-v2021a
- Aporte para el TFG:
  - Pipeline completo de boya con IMU (acelerómetro, giroscopio, magnetómetro).
  - Procesado de orientación y estimación de espectro/estado de mar.
  - Buen candidato para replicar porque es abierto y documentado.

## 2) Smartphone vs buoy: metodología multi-instrumento (The Cryosphere, 2025)
- Paper: https://tc.copernicus.org/articles/19/6927/2025/
- Código: https://github.com/Turbotice/icewave
- Dataset (DOI citado por el paper): https://doi.org/10.57745/OUWL0Z
- Aporte para el TFG:
  - Comparación explícita entre teléfono y boya para medir movimiento/ondas.
  - Scripts y estructura de análisis reutilizables (sincronización, FFT, procesamiento).
  - Muy alineado con comparar Android vs plataforma alternativa.

## 3) SKIB - Surface Kinematics Buoy (Ocean Science, 2018)
- Paper: https://os.copernicus.org/articles/14/1449/2018/
- Dataset del artículo: ftp://ftp.ifremer.fr/ifremer/ww3/COM/PAPERS/2018_OS_GUIMARAES_ETAL/dataset
- Aporte para el TFG:
  - Método de cálculo de parámetros de oleaje a partir de cinemática de superficie.
  - Validación frente a referencia operacional.
  - Útil como base metodológica para métricas y validación.

## 4) Miniaturized strapdown inertial wave sensor (Frontiers, 2022)
- Paper: https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2022.991996/full
- Aporte para el TFG:
  - Descripción técnica de algoritmo strapdown inercial para olas.
  - Incluye marco de coordenadas, integración y análisis espectral/direccional.
  - Útil para justificar decisiones de procesado en memoria.

## 5) Methods and Errors of Wave Measurements Using Conventional IMUs
- Artículo: https://journals.rcsi.science/1573-160X/article/view/284255
- Repo citado (fusión sensorial): https://github.com/memsindustrygroup/Open-Source-Sensor-Fusion
- Aporte para el TFG:
  - Discusión de errores y límites de IMUs convencionales.
  - Referencias de técnicas de fusión de sensores (Kalman/TRIAD y variantes).
  - Útil para sección de limitaciones y mejora futura.

## Nota de selección para réplica
- Prioridad alta para réplica directa: referencias 1 y 2 (por disponer de código y enfoque aplicable).
- Prioridad media para soporte metodológico: referencias 3, 4 y 5.

## Decision metodologica adoptada (2026-02-28)
- Se adopta como metodo principal de procesado de olas un enfoque tipo OpenMetBuoy (Welch + momentos espectrales):
  - PSD de aceleracion vertical.
  - Conversion a espectro de elevacion dividiendo por `(2*pi*f)^4`.
  - Calculo de `Hs`, `Tz`, `Tc` a partir de `m0`, `m2`, `m4`.
- Criterio de uso:
  - Ventanas largas (objetivo 10-20 min) para estimacion robusta con modo OMB puro.
  - Ventanas cortas (p.ej. 30 s) se mantienen como modo de monitorizacion rapida, no como estimacion principal.

## Flujo funcional propuesto del TFG

### Bloque Captura (Android)
- Leer sensores `accelerometer` + `gyroscope` (ideal tambien `rotation vector`).
- Normalizar timestamps y tasa de muestreo real.
- Guardar sesion local en CSV/JSON (`t,ax,ay,az,gx,gy,gz` + metadatos de movil).

### Bloque Procesado on-device (Android)
- Ventana larga configurable (objetivo 10-20 min).
- Pipeline OMB:
  - PSD (Welch) de aceleracion vertical.
  - Conversion a elevacion con `(2*pi*f)^4`.
  - Calculo de `Hs`, `Tz`, `Tc`.
- Modo rapido opcional (30 s) solo diagnostico.

### Bloque Visualizacion
- Grafica de olas en tiempo (`h(t)`).
- Panel con metricas actuales (`Hs`, `Tz`, `Tc`, `fs` real, calidad).
- Indicador de si la ventana ya es valida para resultado robusto.

### Bloque Exportacion
- Exportar resultados por sesion (CSV + resumen JSON/MD).
- Incluir parametros usados (filtros, banda, tamano de ventana, version app).

### Bloque ESP32 + MPU6050 (fase 2)
- Mismo formato de salida que Android.
- Mismo algoritmo y mismas metricas para comparacion justa.

### Bloque Comparativa final
- Misma prueba fisica para ambos.
- Tabla de comparacion: `Hs/Tz/Tc`, estabilidad, ruido, consumo, latencia.
- Conclusiones de viabilidad y limites de cada plataforma.
