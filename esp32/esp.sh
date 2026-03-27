#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDF_EXPORT="${IDF_EXPORT:-$HOME/esp/esp-idf/export.sh}"
DEFAULT_TARGET="esp32s3"

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
    echo "No se detecto puerto serie (/dev/ttyUSB* o /dev/ttyACM*)."
    echo "Adjunta el dispositivo a WSL y vuelve a intentar."
    exit 1
  fi
  if [[ ! -r "$port" || ! -w "$port" ]]; then
    echo "Puerto sin permisos: $port"
    echo "Prueba: sudo chmod 666 $port"
    exit 1
  fi
  echo "$port"
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
    idf.py build
    ;;

  diag-build)
    BOYA_DIAG_IMU=1 idf.py build
    ;;

  flash)
    port="$(require_port "$arg")"
    idf.py -p "$port" -b 115200 flash
    ;;

  monitor)
    port="$(require_port "$arg")"
    idf.py -p "$port" monitor
    ;;

  fm)
    port="$(require_port "$arg")"
    idf.py -p "$port" -b 115200 flash monitor
    ;;

  diag-fm)
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
