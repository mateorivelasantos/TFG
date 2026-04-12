# Validacion algoritmo (2 datasets)

Contenido:

- `validate_algorithm_two_datasets.py`: script de validacion.
- `export_dunex_to_android_csv.py`: convierte `mission_*.nc` (DUNEX/microSWIFT) al CSV exacto de Android (`t_ms,ax,ay,az,gx,gy,gz`).
- `datasets/`: NetCDF descargados.
- `resultados/validation_two_datasets.json`: ultimo informe generado.

Ejecucion:

```bash
/tmp/tfg_venv/bin/python experimentos/validacion_algoritmo/validate_algorithm_two_datasets.py --max-samples-per-dataset 2
```

Si ya tienes los datasets descargados y no quieres volver a bajarlos:

```bash
/tmp/tfg_venv/bin/python experimentos/validacion_algoritmo/validate_algorithm_two_datasets.py --max-samples-per-dataset 2 --skip-download
```

## Exportar a formato Android

Ejemplo con `dunex_mission_2.nc`:

```bash
python3 experimentos/validacion_algoritmo/export_dunex_to_android_csv.py \
  --input experimentos/validacion_algoritmo/datasets/dunex_mission_2.nc \
  --output-dir experimentos/validacion_algoritmo/resultados/android_ready
```

Salida generada (una por trayectoria):

- `resultados/android_ready/dunex_mission_2_traj39_android.csv`
- `resultados/android_ready/dunex_mission_2_traj10_android.csv`
- `resultados/android_ready/dunex_mission_2_traj6_android.csv`
