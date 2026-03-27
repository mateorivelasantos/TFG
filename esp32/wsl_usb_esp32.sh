#!/usr/bin/env bash
set -euo pipefail

PORT_GLOB_1="/dev/ttyUSB*"
PORT_GLOB_2="/dev/ttyACM*"

pick_port() {
  local p
  for p in $PORT_GLOB_1 $PORT_GLOB_2; do
    [[ -e "$p" ]] || continue
    echo "$p"
    return 0
  done
  return 1
}

echo "[1/4] Cargando modulos USB serie en WSL..."
sudo modprobe usbserial
sudo modprobe cp210x

echo "[2/4] Comprobando puerto serie..."
port="$(pick_port || true)"
if [[ -z "${port:-}" ]]; then
  echo "No hay puerto serie en WSL."
  echo "Haz el attach desde Windows y vuelve a ejecutar este script."
  echo "Comando habitual en PowerShell (Admin):"
  echo "  usbipd attach --wsl --busid 1-3"
  exit 1
fi

echo "[3/4] Ajustando permisos para $port ..."
sudo chmod 666 "$port"

echo "[4/4] Estado listo."
ls -l "$port"
echo
echo "Puerto listo: $port"
echo "Siguiente paso:"
echo "  cd /home/mrivela/TFG/esp32"
echo "  ./esp.sh fm $port"
