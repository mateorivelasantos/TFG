#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDF_EXPORT="${IDF_EXPORT:-$HOME/esp/esp-idf/export.sh}"
DEFAULT_TARGET="esp32s3"

# Herramientas instaladas por usuario (p.ej. cmake/ninja con pip --user).
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -f "$IDF_EXPORT" ]]; then
  echo "No encuentro export.sh en: $IDF_EXPORT"
  echo "Define IDF_EXPORT o instala ESP-IDF en ~/esp/esp-idf"
  exit 1
fi

source "$IDF_EXPORT" >/dev/null

pick_port() {
  local p
  for p in /dev/ttyUSB* /dev/ttyACM*; do
    [[ -e "$p" ]] || continue
    echo "$p"
    return 0
  done
  return 1
}

require_port() {
  local port="${1:-}"
  if [[ -z "$port" ]]; then
    port="$(pick_port || true)"
  fi
  if [[ -z "$port" ]]; then
    echo "No se detecto puerto serie (/dev/ttyUSB* o /dev/ttyACM*)." >&2
    echo "Adjunta el dispositivo a WSL y vuelve a intentar." >&2
    exit 1
  fi
  if [[ ! -r "$port" || ! -w "$port" ]]; then
    echo "Puerto sin permisos: $port" >&2
    echo "Prueba: sudo chmod 666 $port" >&2
    return 1
  fi
  echo "$port"
}

ensure_target() {
  local current_target=""
  if [[ -f sdkconfig ]]; then
    current_target="$(awk -F'"' '/^CONFIG_IDF_TARGET="/ { print $2; exit }' sdkconfig || true)"
  fi
  if [[ "$current_target" != "$DEFAULT_TARGET" ]]; then
    echo "Configurando target a ${DEFAULT_TARGET}..."
    idf.py set-target "$DEFAULT_TARGET"
  fi
}

cmd="${1:-help}"
arg="${2:-}"

cd "$SCRIPT_DIR"

case "$cmd" in
  help|-h|--help)
    cat <<USAGE
Uso: ./esp.sh <comando> [puerto]

Comandos:
  target          Configura target a ${DEFAULT_TARGET}
  build           Compila
  diag-build      Compila firmware de diagnostico IMU
  flash [PORT]    Flashea (autodetecta puerto si no se pasa)
  monitor [PORT]  Abre monitor serie
  fm [PORT]       Flash + monitor
  diag-fm [PORT]  Flash + monitor del firmware de diagnostico IMU
  clean           idf.py fullclean
  port            Muestra puerto detectado
USAGE
    ;;

  target)
    idf.py set-target "$DEFAULT_TARGET"
    ;;

  build)
    ensure_target
    idf.py build
    ;;

  diag-build)
    ensure_target
    BOYA_DIAG_IMU=1 idf.py build
    ;;

  flash)
    ensure_target
    port="$(require_port "$arg")"
    idf.py -p "$port" -b 115200 flash
    ;;

  monitor)
    port="$(require_port "$arg")"
    idf.py -p "$port" monitor
    ;;

  fm)
    ensure_target
    port="$(require_port "$arg")"
    idf.py -p "$port" -b 115200 flash monitor
    ;;

  diag-fm)
    ensure_target
    port="$(require_port "$arg")"
    BOYA_DIAG_IMU=1 idf.py -p "$port" -b 115200 flash monitor
    ;;

  clean)
    idf.py fullclean
    ;;

  port)
    p="$(pick_port || true)"
    if [[ -z "$p" ]]; then
      echo "(sin puerto detectado)"
    else
      echo "$p"
    fi
    ;;

  *)
    echo "Comando no reconocido: $cmd"
    echo "Ejecuta: ./esp.sh help"
    exit 1
    ;;
esac
