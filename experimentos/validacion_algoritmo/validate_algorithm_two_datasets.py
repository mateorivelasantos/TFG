#!/usr/bin/env python3
"""Valida el algoritmo OMB usando dos datasets publicos.

Dataset 1: PANGAEA 958689 (OpenMetBuoy 2022)
Dataset 2: Zenodo 17087019 (KVS buoy 2025)

Estrategia:
1) Carga espectros de elevacion S_eta(f) de ambos datasets.
2) Selecciona ventanas validas.
3) Sintetiza az(t) ideal a partir de S_eta(f) (banda OMB).
4) Ejecuta el pipeline OMB existente sobre az(t).
5) Compara Hs/Tz/Tp del algoritmo frente a referencia de momentos espectrales.

Nota: esta validacion comprueba la coherencia del pipeline espectral.
No sustituye una validacion con IMU cruda de campo.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import netCDF4 as nc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import process_imu_openmetbuoy as omb


PANGAEA_URL = (
    "https://download.pangaea.de/dataset/958689/files/data_packed_as_netcdf.nc"
)
ZENODO_2025_URL = (
    "https://zenodo.org/records/17087019/files/"
    "2025_KVS_buoy17_deployment_nonQCdata_v01.nc"
)


def as_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.ma.asarray(x)
    return np.ma.filled(arr, np.nan).astype(np.float64, copy=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validacion dual del algoritmo OMB sobre datasets publicos"
    )
    p.add_argument(
        "--datasets-dir",
        default=str(SCRIPT_DIR / "datasets"),
        help="Directorio donde guardar/cargar los datasets",
    )
    p.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=3,
        help="Maximo de muestras espectrales por dataset",
    )
    p.add_argument(
        "--duration-sec",
        type=float,
        default=20.0 * 60.0,
        help="Duracion de la serie sintetica (s)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para fases aleatorias de sintesis",
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="No descargar; usar solo archivos ya presentes",
    )
    p.add_argument(
        "--out-json",
        default=str(SCRIPT_DIR / "resultados" / "validation_two_datasets.json"),
        help="Ruta del informe JSON de salida",
    )
    return p.parse_args()


def ensure_file(url: str, path: Path, skip_download: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    if skip_download:
        raise FileNotFoundError(f"No existe {path} y --skip-download esta activo")
    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    path.write_bytes(data)
    print(f"[ok] guardado en {path} ({path.stat().st_size} bytes)")


def omb_band_limits(cfg: omb.OMBConfig) -> tuple[float, float]:
    df = cfg.fs_target_hz / cfg.fft_length
    fmin = cfg.welch_bin_min * df
    fmax = (cfg.welch_bin_max - 1) * df
    return float(fmin), float(fmax)


def _valid_spectrum_indices(
    freq_hz: np.ndarray,
    spectra: np.ndarray,
    fmin: float,
    fmax: float,
    limit: int,
) -> list[int]:
    idx: list[int] = []
    freq_mask = (freq_hz >= fmin) & (freq_hz <= fmax) & np.isfinite(freq_hz)
    if not np.any(freq_mask):
        return idx

    for i in range(spectra.shape[0]):
        s = as_float_array(spectra[i])
        m = freq_mask & np.isfinite(s) & (s > 0.0)
        if np.count_nonzero(m) >= 8:
            idx.append(i)
        if len(idx) >= limit:
            break
    return idx


def moments_metrics(freq_hz: np.ndarray, seta: np.ndarray) -> dict:
    freq = np.asarray(freq_hz, dtype=np.float64)
    s = np.asarray(seta, dtype=np.float64)
    if freq.size < 2:
        raise ValueError("frecuencia insuficiente")
    df = np.gradient(freq)

    m0 = float(np.sum(s * df))
    m2 = float(np.sum((freq**2) * s * df))

    hs = 4.0 * math.sqrt(max(m0, 0.0))
    tz = math.sqrt(m0 / m2) if m0 > 0 and m2 > 0 else float("nan")

    kpk = int(np.argmax(s))
    fp = float(freq[kpk])
    tp = 1.0 / fp if fp > 0 else float("nan")

    return {
        "m0": m0,
        "m2": m2,
        "hs_m": hs,
        "tz_s": tz,
        "tp_s": tp,
    }


def synthesize_az_from_spectrum(
    freq_hz: np.ndarray,
    seta: np.ndarray,
    duration_sec: float,
    fs_hz: float,
    gravity_mps2: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(duration_sec * fs_hz))
    if n < 100:
        raise ValueError("duracion/fs demasiado cortos para sintesis")
    t = np.arange(n, dtype=np.float64) / fs_hz

    f = np.asarray(freq_hz, dtype=np.float64)
    s = np.asarray(seta, dtype=np.float64)
    df = np.gradient(f)

    # eta(t) = sum A_i cos(2*pi*f_i*t + phi_i), con A_i = sqrt(2 S_i df_i)
    amp = np.sqrt(np.maximum(0.0, 2.0 * s * df))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=f.shape[0])

    omega = 2.0 * np.pi * f
    cos_term = np.cos(np.outer(omega, t) + phi[:, None])
    az_dyn = -np.sum((omega[:, None] ** 2) * (amp[:, None] * cos_term), axis=0)
    az = gravity_mps2 + az_dyn
    return t, az


def run_omb_on_signal(t_sec: np.ndarray, az: np.ndarray, cfg: omb.OMBConfig) -> dict:
    t_uniform, az_uniform = omb.resample_to_uniform_fs(t_sec, az, cfg.fs_target_hz)
    return omb.compute_omb_metrics(az_uniform, cfg, window_position="tail")


def extract_pangaea_samples(
    ds: nc.Dataset,
    fmin: float,
    fmax: float,
    limit: int,
) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    freq = as_float_array(ds.variables["frequency"][:])
    spectra = as_float_array(ds.variables["wave_spectrum"][:])  # [traj, obs, f]
    yielded = 0
    # Reducimos a una trayectoria por simplicidad, recorriendo observaciones.
    for traj in range(spectra.shape[0]):
        obs_idx = _valid_spectrum_indices(freq, spectra[traj], fmin, fmax, limit)
        for oi in obs_idx:
            s_full = as_float_array(spectra[traj, oi])
            m = (freq >= fmin) & (freq <= fmax) & np.isfinite(s_full) & (s_full > 0.0)
            yield (f"pangaea:traj{traj}:obs{oi}", freq[m], s_full[m])
            yielded += 1
            if yielded >= limit:
                return


def extract_zenodo_2025_samples(
    ds: nc.Dataset,
    fmin: float,
    fmax: float,
    limit: int,
) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    freq = as_float_array(ds.variables["frequencies_waves_imu"][:])
    if "processed_elevation_energy_spectrum" in ds.variables:
        spectra = as_float_array(ds.variables["processed_elevation_energy_spectrum"][:])
    else:
        spectra = as_float_array(ds.variables["elevation_energy_spectrum"][:])
    # [traj, obs, f]
    yielded = 0
    for traj in range(spectra.shape[0]):
        obs_idx = _valid_spectrum_indices(freq, spectra[traj], fmin, fmax, limit)
        for oi in obs_idx:
            s_full = as_float_array(spectra[traj, oi])
            m = (freq >= fmin) & (freq <= fmax) & np.isfinite(s_full) & (s_full > 0.0)
            yield (f"zenodo2025:traj{traj}:obs{oi}", freq[m], s_full[m])
            yielded += 1
            if yielded >= limit:
                return


def pct_err(est: float, ref: float) -> float:
    if not np.isfinite(est) or not np.isfinite(ref) or ref == 0.0:
        return float("nan")
    return float(100.0 * (est - ref) / ref)


def main() -> None:
    args = parse_args()
    cfg = omb.OMBConfig()
    fmin, fmax = omb_band_limits(cfg)
    rng = np.random.default_rng(args.seed)

    base_dir = Path(args.datasets_dir)
    pangaea_nc = base_dir / "pangaea_958689_data_packed_as_netcdf.nc"
    zenodo_nc = base_dir / "zenodo_17087019_2025_KVS_buoy17.nc"
    ensure_file(PANGAEA_URL, pangaea_nc, args.skip_download)
    ensure_file(ZENODO_2025_URL, zenodo_nc, args.skip_download)

    results: list[dict] = []

    with nc.Dataset(pangaea_nc) as ds_p:
        for name, f, s in extract_pangaea_samples(
            ds_p, fmin, fmax, args.max_samples_per_dataset
        ):
            ref = moments_metrics(f, s)
            t, az = synthesize_az_from_spectrum(
                f, s, args.duration_sec, cfg.fs_target_hz, cfg.gravity_mps2, rng
            )
            est = run_omb_on_signal(t, az, cfg)
            results.append(
                {
                    "sample": name,
                    "reference": ref,
                    "estimate": {
                        "hs_m": est["hs_m"],
                        "tz_s": est["tz_s"],
                        "tp_s": est["tp_s"],
                    },
                    "errors_pct": {
                        "hs_pct": pct_err(est["hs_m"], ref["hs_m"]),
                        "tz_pct": pct_err(est["tz_s"], ref["tz_s"]),
                        "tp_pct": pct_err(est["tp_s"], ref["tp_s"]),
                    },
                }
            )

    with nc.Dataset(zenodo_nc) as ds_z:
        for name, f, s in extract_zenodo_2025_samples(
            ds_z, fmin, fmax, args.max_samples_per_dataset
        ):
            ref = moments_metrics(f, s)
            t, az = synthesize_az_from_spectrum(
                f, s, args.duration_sec, cfg.fs_target_hz, cfg.gravity_mps2, rng
            )
            est = run_omb_on_signal(t, az, cfg)
            results.append(
                {
                    "sample": name,
                    "reference": ref,
                    "estimate": {
                        "hs_m": est["hs_m"],
                        "tz_s": est["tz_s"],
                        "tp_s": est["tp_s"],
                    },
                    "errors_pct": {
                        "hs_pct": pct_err(est["hs_m"], ref["hs_m"]),
                        "tz_pct": pct_err(est["tz_s"], ref["tz_s"]),
                        "tp_pct": pct_err(est["tp_s"], ref["tp_s"]),
                    },
                }
            )

    if not results:
        raise SystemExit("No se encontraron muestras validas en los datasets.")

    hs_err = np.array([r["errors_pct"]["hs_pct"] for r in results], dtype=np.float64)
    tz_err = np.array([r["errors_pct"]["tz_pct"] for r in results], dtype=np.float64)
    tp_err = np.array([r["errors_pct"]["tp_pct"] for r in results], dtype=np.float64)

    def abs_mean(x: np.ndarray) -> float:
        return float(np.nanmean(np.abs(x)))

    summary = {
        "n_samples": len(results),
        "omb_band_hz": [fmin, fmax],
        "mean_abs_error_pct": {
            "hs_pct": abs_mean(hs_err),
            "tz_pct": abs_mean(tz_err),
            "tp_pct": abs_mean(tp_err),
        },
    }

    print("=== VALIDACION DOS DATASETS ===")
    print(f"Muestras totales: {summary['n_samples']}")
    print(
        "MAE% | "
        f"Hs={summary['mean_abs_error_pct']['hs_pct']:.3f} | "
        f"Tz={summary['mean_abs_error_pct']['tz_pct']:.3f} | "
        f"Tp={summary['mean_abs_error_pct']['tp_pct']:.3f}"
    )
    for r in results:
        e = r["errors_pct"]
        print(
            f"- {r['sample']}: "
            f"Hs err={e['hs_pct']:.3f}% | "
            f"Tz err={e['tz_pct']:.3f}% | "
            f"Tp err={e['tp_pct']:.3f}%"
        )

    out = {
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] informe guardado en {out_path}")


if __name__ == "__main__":
    main()
