#!/usr/bin/env python3
"""
Convierte un NetCDF de DUNEX microSWIFT (mission_*.nc) al formato CSV que
consume la app Android:

    t_ms,ax,ay,az,gx,gy,gz

Uso rapido:
  python3 export_dunex_to_android_csv.py \
      --input datasets/dunex_mission_2.nc \
      --output-dir resultados/android_ready
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError as exc:
    raise SystemExit(
        "Falta netCDF4. Instala dependencias con:\n"
        "  python3 -m venv .venv\n"
        "  . .venv/bin/activate\n"
        "  pip install netCDF4 numpy"
    ) from exc


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def _build_rows(
    t_sec: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
) -> Iterable[tuple[int, float, float, float, float, float, float]]:
    mask = _finite_mask(t_sec, ax, ay, az, gx, gy, gz)
    if not np.any(mask):
        return []

    t = t_sec[mask].astype(float)
    axv = ax[mask].astype(float)
    ayv = ay[mask].astype(float)
    azv = az[mask].astype(float)
    gxv = gx[mask].astype(float)
    gyv = gy[mask].astype(float)
    gzv = gz[mask].astype(float)

    order = np.argsort(t)
    t = t[order]
    axv = axv[order]
    ayv = ayv[order]
    azv = azv[order]
    gxv = gxv[order]
    gyv = gyv[order]
    gzv = gzv[order]

    keep = np.ones_like(t, dtype=bool)
    keep[1:] = np.diff(t) > 0.0
    t = t[keep]
    axv = axv[keep]
    ayv = ayv[keep]
    azv = azv[keep]
    gxv = gxv[keep]
    gyv = gyv[keep]
    gzv = gzv[keep]

    if t.size < 2:
        return []

    t_ms = np.round((t - t[0]) * 1000.0).astype(np.int64)
    return zip(t_ms.tolist(), axv.tolist(), ayv.tolist(), azv.tolist(), gxv.tolist(), gyv.tolist(), gzv.tolist())


def export_file(input_nc: Path, output_dir: Path, trajectory_index: int | None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []

    with Dataset(input_nc, "r") as ds:
        t_sec = np.array(ds.variables["time"][:], dtype=float)
        traj_values = np.array(ds.variables["trajectory"][:]).astype(int)

        ax_all = np.array(ds.variables["acceleration_x_body"][:], dtype=float)
        ay_all = np.array(ds.variables["acceleration_y_body"][:], dtype=float)
        az_all = np.array(ds.variables["acceleration_z_body"][:], dtype=float)
        gx_all = np.array(ds.variables["rotation_rate_x"][:], dtype=float)
        gy_all = np.array(ds.variables["rotation_rate_y"][:], dtype=float)
        gz_all = np.array(ds.variables["rotation_rate_z"][:], dtype=float)

        indices = [trajectory_index] if trajectory_index is not None else list(range(ax_all.shape[0]))

        for i in indices:
            if i < 0 or i >= ax_all.shape[0]:
                raise ValueError(f"trajectory_index fuera de rango: {i}")

            rows = list(
                _build_rows(
                    t_sec=t_sec,
                    ax=ax_all[i],
                    ay=ay_all[i],
                    az=az_all[i],
                    gx=gx_all[i],
                    gy=gy_all[i],
                    gz=gz_all[i],
                )
            )
            if not rows:
                continue

            traj_id = int(traj_values[i]) if i < traj_values.size else i
            out_name = f"{input_nc.stem}_traj{traj_id}_android.csv"
            out_path = output_dir / out_name

            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t_ms", "ax", "ay", "az", "gx", "gy", "gz"])
                for row in rows:
                    t_ms, ax, ay, az, gx, gy, gz = row
                    writer.writerow(
                        [
                            int(t_ms),
                            f"{ax:.6f}",
                            f"{ay:.6f}",
                            f"{az:.6f}",
                            f"{gx:.6f}",
                            f"{gy:.6f}",
                            f"{gz:.6f}",
                        ]
                    )

            out_paths.append(out_path)

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta DUNEX mission_*.nc a CSV compatible con Android app.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Ruta al .nc (ej: datasets/dunex_mission_2.nc)",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("resultados/android_ready"),
        type=Path,
        help="Directorio de salida para CSVs Android",
    )
    parser.add_argument(
        "--trajectory-index",
        type=int,
        default=None,
        help="Indice de trayectoria (0..N-1). Si no se indica, exporta todas.",
    )
    args = parser.parse_args()

    out_paths = export_file(args.input, args.output_dir, args.trajectory_index)
    if not out_paths:
        raise SystemExit("No se generaron CSVs (sin muestras validas).")

    print("CSV(s) generados:")
    for p in out_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
