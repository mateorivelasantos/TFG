param(
    [string]$BusId = "",
    [string]$WslDistro = "",
    [string]$VidPid = "10c4:ea60",
    [string]$DeviceMatch = "CP210x"
)

$ErrorActionPreference = "Stop"

function Resolve-WslDistro {
    param([string]$PreferredDistro)

    if ($PreferredDistro) {
        return $PreferredDistro
    }

    $listV = wsl -l -v 2>$null
    if ($listV) {
        foreach ($line in ($listV -split "`r?`n")) {
            $cleanLine = $line -replace "`0", ""
            if ($cleanLine -match '^\s*\*\s+([^\s]+)') {
                return $Matches[1]
            }
        }
    }

    $listQ = wsl -l -q 2>$null
    if ($listQ) {
        foreach ($line in ($listQ -split "`r?`n")) {
            $name = ($line -replace "`0", "").Trim()
            if ($name) {
                return $name
            }
        }
    }

    throw "No se pudo detectar automaticamente la distro WSL. Pasa -WslDistro manualmente."
}

function Resolve-BusId {
    param(
        [string]$PreferredBusId,
        [string]$VidPid,
        [string]$DeviceMatch
    )

    if ($PreferredBusId) {
        return $PreferredBusId
    }

    $listOutput = usbipd list
    $lines = $listOutput -split "`r?`n"

    foreach ($line in $lines) {
        if (($line -match [regex]::Escape($VidPid)) -or ($line -match [regex]::Escape($DeviceMatch))) {
            if ($line -match '^\s*([0-9]+-[0-9]+)\s+') {
                return $Matches[1]
            }
        }
    }

    throw "No se pudo encontrar automaticamente el BUSID del dispositivo USB ($VidPid / $DeviceMatch)."
}

$ResolvedBusId = Resolve-BusId -PreferredBusId $BusId -VidPid $VidPid -DeviceMatch $DeviceMatch
$ResolvedWslDistro = Resolve-WslDistro -PreferredDistro $WslDistro

Write-Host "[1/5] Reiniciando WSL..." -ForegroundColor Cyan
wsl --shutdown
Start-Sleep -Seconds 2

Write-Host "[2/5] Reiniciando usbipd..." -ForegroundColor Cyan
try {
    Restart-Service usbipd -ErrorAction Stop
}
catch {
    Write-Host "No se pudo reiniciar usbipd automaticamente. Sigue el flujo igualmente." -ForegroundColor Yellow
}

Write-Host "[3/5] Preparando modulos USB/IP dentro de WSL..." -ForegroundColor Cyan
try {
    wsl -d $ResolvedWslDistro -u root -- sh -lc "modprobe usbip_core && modprobe vhci_hcd"
    if ($LASTEXITCODE -ne 0) {
        throw "modprobe devolvio codigo $LASTEXITCODE"
    }
}
catch {
    throw "No se pudieron cargar los modulos USB/IP en WSL. Revisa el nombre de la distro con -WslDistro. Detalle: $($_.Exception.Message)"
}

Write-Host "[4/5] Compartiendo dispositivo USB ($ResolvedBusId)..." -ForegroundColor Cyan
usbipd bind --force --busid $ResolvedBusId | Out-Host

Write-Host "[5/5] Adjuntando a WSL..." -ForegroundColor Cyan
usbipd attach --wsl $ResolvedWslDistro --busid $ResolvedBusId | Out-Host

Write-Host ""
Write-Host "BUSID detectado: $ResolvedBusId" -ForegroundColor Green
Write-Host "Distro WSL: $ResolvedWslDistro" -ForegroundColor Green
Write-Host "Si el attach fue bien, en WSL ejecuta:" -ForegroundColor Green
Write-Host "  cd ~/TFG/esp32"
Write-Host "  ./wsl_usb_esp32.sh"
