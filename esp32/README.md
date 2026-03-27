# ESP32-S3 - BOYA IMU (ESP-IDF)

Proyecto base en C++ para captura IMU y envio a Android.

## Flujo rapido (recomendado)
Desde `esp32/`:

```bash
./esp.sh target   # solo la primera vez o si cambias de chip
./esp.sh build
./esp.sh fm       # flash + monitor (autodetecta puerto)
```

## Flujo rapido en WSL + Windows
Si usas `usbipd` y no quieres repetir todos los pasos a mano:

En PowerShell como administrador:
```powershell
cd C:\Users\Mateo\AndroidStudioProjects\boya
powershell -ExecutionPolicy Bypass -File \\wsl$\kali-linux\home\mrivela\TFG\esp32\attach_esp32_wsl.ps1
```

En WSL:
```bash
cd /home/mrivela/TFG/esp32
./wsl_usb_esp32.sh
./esp.sh fm /dev/ttyUSB0
```

## Comandos utiles
```bash
./esp.sh help
./esp.sh port
./esp.sh flash /dev/ttyUSB0
./esp.sh monitor /dev/ttyUSB0
./esp.sh clean
```

## Si falla por permisos del puerto
```bash
sudo chmod 666 /dev/ttyUSB0
```

## Requisitos
- ESP-IDF instalado (`~/esp/esp-idf`)
- Entorno Linux/WSL con acceso al puerto serie

## Estructura
- `main/main.cpp`: punto de entrada (`app_main`)
- `esp.sh`: wrapper rapido para compilar/flashear
- `wsl_usb_esp32.sh`: prepara modulos USB y permisos en WSL
- `attach_esp32_wsl.ps1`: helper para `usbipd attach` desde Windows con deteccion automatica del CP210x
- `CMakeLists.txt`, `main/CMakeLists.txt`: configuracion ESP-IDF
- `sdkconfig.defaults`: defaults iniciales

## Feedback en pantalla OLED
- Soporte añadido para OLED SSD1306 por I2C (128x64).
- Deteccion automatica en direccion `0x3C` (fallback `0x3D`).
- Muestra estado en tiempo real:
  - SSID/AP activo
  - IMU detectada (`WHO_AM_I`)
  - Duracion y nombre de captura configurada
  - Ultima accion HTTP (`CAP START`, `CAP STOP`, `CAP DOWN`, etc.)
  - Heartbeat
- Si la OLED no esta conectada o no responde, el firmware sigue funcionando sin bloquearse.

Pines I2C actuales en [main.cpp](/home/mrivela/TFG/esp32/main/main.cpp):
- OLED Heltec: `SDA = GPIO17`, `SCL = GPIO18`
- MCU externo: `SDA = GPIO41`, `SCL = GPIO42`
