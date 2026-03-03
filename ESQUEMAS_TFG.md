# Esquemas Visuales del TFG

## 1) Arquitectura general
```mermaid
flowchart LR
    A[Android - Modo Autonomo] --> A1[Captura IMU]
    A1 --> A2[Procesado OMB on-device]
    A2 --> A3[Visualizacion en app]
    A2 --> A4[Exportacion CSV/JSON]

    B[ESP32 + MPU6050] --> B1[Captura IMU]
    B1 --> B2[Procesado OMB en ESP32]
    B2 --> B3[Envio de resultados]

    B3 --> C[Android - Modo Display]
    C --> C1[Recepcion de metricas]
    C1 --> C2[Visualizacion en app]
    C1 --> C3[Exportacion CSV/JSON]

    A4 --> D[Repositorio de sesiones]
    C3 --> D
    D --> E[Comparativa final Android vs ESP32]
```

## 2) Flujo Android (captura -> procesado -> visualizacion)
```mermaid
flowchart TD
    S[Inicio sesion] --> C[Captura sensores<br/>acc + gyro (+ rotation vector)]
    C --> T[Normalizar timestamps<br/>y fs real]
    T --> W{Ventana suficiente?}
    W -- No --> Q[Modo rapido diagnostico<br/>30s]
    W -- Si --> P[Procesado OMB<br/>Welch + (2*pi*f)^4 + Hs/Tz/Tc]
    Q --> V[Visualizacion parcial]
    P --> V[Visualizacion robusta]
    V --> X[Exportar sesion<br/>CSV + resumen JSON/MD]
    X --> F[Fin]
```

## 3) Comparativa Android vs ESP32
```mermaid
flowchart TD
    P0[Definir prueba fisica unica] --> P1[Ejecutar Android]
    P0 --> P2[Ejecutar ESP32]

    P1 --> M1[Obtener Hs/Tz/Tc + fs + ruido + latencia + consumo]
    P2 --> M2[Obtener Hs/Tz/Tc + fs + ruido + latencia + consumo]

    M1 --> C1[Unificar formato de resultados]
    M2 --> C1

    C1 --> T[Tabla comparativa]
    T --> G[Graficas comparativas]
    G --> R[Conclusiones:<br/>viabilidad, precision, limites]
```

