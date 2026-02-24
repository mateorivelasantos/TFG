# Contexto compartido entre chats (2 ordenadores)

Este archivo sirve para transferir contexto entre chats y equipos sin perder continuidad.

## Regla de uso (siempre)
1. Al empezar un chat nuevo, pedir: "Lee `CHAT_CONTEXT.md` y continuamos desde ahí".
2. Antes de cerrar sesión, actualizar este archivo.
3. Hacer `git add`, `git commit` y `git push`.
4. En el otro equipo: `git pull` antes de seguir.

## Formato de actualización
Copiar y pegar este bloque en cada actualización:

```md
## Handoff YYYY-MM-DD HH:MM (TZ) - Equipo X
- Autor chat/equipo:
- Commit actual (`git rev-parse --short HEAD`):
- Rama:
- Objetivo activo:
- Cambios realizados:
  - ...
- Decisiones tomadas:
  - ...
- Estado/resultado:
  - ...
- Próximos pasos (ordenados):
  1. ...
  2. ...
- Bloqueos o dudas:
  - ...
- Comandos clave ejecutados:
  - `...`
```

## Handoff 2026-02-24 18:40 (UTC+01:00) - Equipo actual
- Autor chat/equipo: Chat + mateo (equipo actual)
- Commit actual (`git rev-parse --short HEAD`): `7ab2a8c`
- Rama: `main`
- Objetivo activo: mantener continuidad entre chats/equipos y no perder contexto operativo del TFG.
- Cambios realizados:
  - Se inicializó Git en `/home/mateo/TFG`.
  - Se creó `.gitignore` básico para Python (`.venv`, `__pycache__`, `*.pyc`).
  - Se hizo commit inicial.
  - Se configuró remoto `origin` a `https://github.com/mateorivelasantos/TFG.git`.
  - Se hizo `push` exitoso de `main` a GitHub.
- Decisiones tomadas:
  - Usar este archivo como fuente única de traspaso entre chats.
  - Mantener handoffs breves y accionables (qué se hizo, qué falta, bloqueos).
- Estado/resultado:
  - Repo sincronizado en GitHub y listo para trabajo en ambos ordenadores.
  - Pipeline IMU funcional (captura, procesado offline/live, autoajuste) según `INFORME_IMU.md`.
- Próximos pasos (ordenados):
  1. En el otro ordenador, clonar o hacer `git pull`.
  2. Iniciar chat nuevo pidiendo lectura de `CHAT_CONTEXT.md`.
  3. Continuar tarea técnica y actualizar este archivo al cerrar.
- Bloqueos o dudas:
  - Ninguno a nivel Git actualmente.
- Comandos clave ejecutados:
  - `git remote add origin https://github.com/mateorivelasantos/TFG.git`
  - `git push -u origin main`

## Resumen técnico vivo
- Estado técnico detallado del TFG: ver `INFORME_IMU.md`.
- Scripts clave:
  - `capture_http_imu.py`
  - `process_imu_session.py`
  - `process_imu_step_by_step.py`
  - `live_imu_http.py`
  - `auto_tune_imu.py`
