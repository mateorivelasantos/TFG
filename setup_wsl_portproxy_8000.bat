@echo off
setlocal enabledelayedexpansion

REM Ejecutar como Administrador.
REM Este script redirige 0.0.0.0:8000 (Windows) -> <IP WSL>:8000

for /f "tokens=1" %%i in ('wsl hostname -I') do set WSL_IP=%%i

if "%WSL_IP%"=="" (
  echo [ERROR] No se pudo obtener la IP de WSL.
  echo Asegurate de que WSL este iniciado: wsl -d Ubuntu -e true
  exit /b 1
)

echo [INFO] IP WSL detectada: %WSL_IP%

echo [INFO] Limpiando portproxy anterior (si existe)...
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8000 >nul 2>&1
netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=8000 >nul 2>&1

echo [INFO] Creando portproxy 0.0.0.0:8000 -> %WSL_IP%:8000
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=8000 connectaddress=%WSL_IP% connectport=8000
if errorlevel 1 (
  echo [ERROR] Fallo al crear portproxy. Ejecuta este .bat como Administrador.
  exit /b 1
)

echo [INFO] Configurando regla de firewall TCP/8000...
netsh advfirewall firewall show rule name="WSL_8000" >nul 2>&1
if errorlevel 1 (
  netsh advfirewall firewall add rule name="WSL_8000" dir=in action=allow protocol=TCP localport=8000
) else (
  netsh advfirewall firewall set rule name="WSL_8000" new enable=Yes
)

echo.
echo [OK] Portproxy configurado.
echo [OK] Envia desde el movil a: http://IP_DE_WINDOWS_WIFI:8000/data
echo.
echo [INFO] Ver estado:
echo   netsh interface portproxy show v4tov4
echo.

endlocal
