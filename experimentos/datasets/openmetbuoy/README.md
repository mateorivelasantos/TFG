# Datasets OpenMetBuoy (validacion local)

## Que hay en esta carpeta

Se copiaron 3 datasets publicos para inspeccion local:

- `2022_CAGE.nc` (release 2024)
- `2022_AWI_UTOKYO.nc` (release 2024)
- `data_drift_waves_Greenland_2022_seals_cruise.nc` (release 2022)

Origen:

- https://github.com/jerabaul29/2024_OpenMetBuoy_data_release_MarginalIceZone_SeaIce_OpenOcean
- https://github.com/jerabaul29/data_release_sea_ice_drift_waves_in_ice_marginal_ice_zone_2022

## Conclusion rapida para nuestro TFG

Estos datasets abiertos de OpenMetBuoy **no** incluyen serie temporal IMU cruda de 6 ejes
(`ax, ay, az, gx, gy, gz` por muestra).

Incluyen principalmente:

- posicion/tiempo (`time`, `lat`, `lon`)
- espectros (`accel_energy_spectrum`, `elevation_energy_spectrum`, `wave_spectrum`)
- parametros ya procesados (`Hs0`, `T02`, `T24`, `hs`, `tp`, `tz0`, etc.)

Por tanto:

- sirven para validar productos de oleaje/salida final,
- no sirven para testear directamente un pipeline que necesite la serie IMU cruda punto a punto.

## Inspeccion de variables

Usa:

```bash
python3 scan_openmet_netcdf.py 2022_CAGE.nc
python3 scan_openmet_netcdf.py 2022_AWI_UTOKYO.nc
python3 scan_openmet_netcdf.py data_drift_waves_Greenland_2022_seals_cruise.nc
```

Nota: requiere `netCDF4`:

```bash
pip install netCDF4
```
