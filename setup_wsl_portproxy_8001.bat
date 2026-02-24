@echo off
setlocal enabledelayedexpansion

REM Ejecutar como Administrador.
REM Redirige 0.0.0.0:8001 (Windows) -> <IP WSL>:8001

for /f "tokens=1" %%i in ('wsl hostname -I') do set WSL_IP=%%i

if "%WSL_IP%"=="" (
  echo [ERROR] No se pudo obtener la IP de WSL.
  echo Inicia WSL y vuelve a ejecutar.
  exit /b 1
)

echo [INFO] IP WSL detectada: %WSL_IP%

echo [INFO] Eliminando reglas previas de portproxy para 8001...
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8001 >nul 2>&1
netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=8001 >nul 2>&1

echo [INFO] Creando portproxy 0.0.0.0:8001 -> %WSL_IP%:8001
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8001 connectaddress=%WSL_IP% connectport=8001
if errorlevel 1 (
  echo [ERROR] No se pudo crear portproxy. Ejecuta este archivo como Administrador.
  exit /b 1
)

echo [INFO] Configurando firewall para TCP/8001...
netsh advfirewall firewall show rule name="WSL_8001" >nul 2>&1
if errorlevel 1 (
  netsh advfirewall firewall add rule name="WSL_8001" dir=in action=allow protocol=TCP localport=8001
) else (
  netsh advfirewall firewall set rule name="WSL_8001" new enable=Yes
)

echo.
echo [OK] Configuracion completada.
echo [OK] Envia desde el movil a: http://IP_WINDOWS:8001/data
echo.
echo [INFO] Ver estado:
echo   netsh interface portproxy show v4tov4

echo.
echo [TIP] En WSL ejecuta:
echo   python3 live_imu_http.py --host 0.0.0.0 --port 8001 --window-seconds 30 --update-seconds 1

endlocal
