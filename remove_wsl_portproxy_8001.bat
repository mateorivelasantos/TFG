@echo off
setlocal

REM Ejecutar como Administrador.

echo [INFO] Eliminando portproxy en 8001...
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8001 >nul 2>&1
netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=8001 >nul 2>&1

echo [INFO] Eliminando regla firewall WSL_8001...
netsh advfirewall firewall delete rule name="WSL_8001" >nul 2>&1

echo [OK] Limpieza completada.
endlocal
