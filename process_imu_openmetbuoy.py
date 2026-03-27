#!/usr/bin/env python3
"""Procesado IMU estilo OpenMetBuoy (Welch + momentos espectrales Hs/Tz/Tc).

Replica la logica del firmware de OpenMetBuoy-v2021a:
- fs objetivo 10 Hz
- FFT len 2048
- solape 75% (step 512)
- bins [9, 64) para el espectro enviado
- ventana Hann implementada como sin(pi*n/N)^2 con factor energetico 1.63
- conversion de S_aa(f) a S_eta(f) dividiendo por (2*pi*f)^4
- calculo de momentos m0, m2, m4 por suma discreta (* df)

Notas:
- El firmware transmite Tz/Tc como frecuencia (sqrt(m2/m0), sqrt(m4/m2));
  el decoder oficial invierte esos valores para reportar periodos en segundos.
  Este script reporta ambos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OMBConfig:
    fs_target_hz: float = 10.0
    gravity_mps2: float = 9.81
    fft_length: int = 2048
    fft_overlap: int = 512
    number_welch_segments_target: int = 21
    start_margin_samples: int = 50
    welch_bin_min: int = 9
    welch_bin_max: int = 64  # exclusivo
    hanning_energy_scaling: float = 1.63


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Procesa una sesion IMU con metodo OpenMetBuoy")
    p.add_argument("file", help="Ruta del archivo de entrada (.csv o .npz)")
    p.add_argument(
        "--out-json",
        default="",
        help="Ruta JSON de salida con metricas y espectro (opcional)",
    )
    p.add_argument(
        "--window-position",
        choices=("head", "tail"),
        default="tail",
        help="Si sobran muestras, usar ventana al principio o al final (default: tail)",
    )
    return p.parse_args()


def load_session(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ext = path.suffix.lower()

    if ext == ".npz":
        data = np.load(path)
        if "t" not in data or "az" not in data:
            raise ValueError("El NPZ debe contener claves 't' y 'az'.")
        t = np.asarray(data["t"], dtype=np.float64)
        az = np.asarray(data["az"], dtype=np.float64)
    elif ext == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV sin cabecera")

            names = [n.strip().lower() for n in reader.fieldnames]
            t_vals: list[float] = []
            az_vals: list[float] = []

            for row in reader:
                row_n = {k.strip().lower(): v for k, v in row.items() if k is not None}

                t_raw = None
                for k in ("t", "t_ms", "time", "timestamp"):
                    if k in row_n:
                        t_raw = row_n[k]
                        break

                az_raw = None
                for k in ("az", "a_z", "acc_z"):
                    if k in row_n:
                        az_raw = row_n[k]
                        break

                if t_raw is None or az_raw is None:
                    continue

                try:
                    t_vals.append(float(t_raw))
                    az_vals.append(float(az_raw))
                except ValueError:
                    continue

        if len(t_vals) < 100:
            raise ValueError("CSV con muy pocas muestras validas para t/az")

        t = np.asarray(t_vals, dtype=np.float64)
        az = np.asarray(az_vals, dtype=np.float64)

        # Heuristica: si viene en ms (header t_ms o dt tipica > 1), convertir a s.
        if "t_ms" in names:
            t = t / 1000.0
        else:
            dt = np.diff(t)
            dt = dt[dt > 0]
            if dt.size > 10 and float(np.median(dt)) > 1.0:
                t = t / 1000.0
    else:
        raise ValueError("Formato no soportado. Usa .csv o .npz")

    mask = np.isfinite(t) & np.isfinite(az)
    t = t[mask]
    az = az[mask]
    if t.size < 100:
        raise ValueError("No hay suficientes muestras validas tras limpiar NaN/Inf")

    order = np.argsort(t)
    t = t[order]
    az = az[order]

    dt = np.diff(t)
    keep = np.ones_like(t, dtype=bool)
    keep[1:] = dt > 0
    t = t[keep]
    az = az[keep]

    if t.size < 100:
        raise ValueError("No hay suficientes muestras con timestamp estrictamente creciente")

    # Rebase temporal.
    t = t - t[0]
    return t, az


def fs_estimate(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size < 10:
        return 0.0
    return float(1.0 / np.median(dt))


def resample_to_uniform_fs(t: np.ndarray, az: np.ndarray, fs_target: float) -> tuple[np.ndarray, np.ndarray]:
    if fs_target <= 0:
        raise ValueError("fs_target debe ser > 0")

    dt = 1.0 / fs_target
    t_end = float(t[-1])
    # Incluye ultimo instante dentro de error numerico.
    t_uniform = np.arange(0.0, t_end + 0.5 * dt, dt, dtype=np.float64)
    if t_uniform.size < 100:
        raise ValueError("Serie remuestreada demasiado corta")

    az_uniform = np.interp(t_uniform, t, az)
    return t_uniform, az_uniform


def omb_hanning_window(n: int, energy_scaling: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    s = np.sin(np.pi * idx / float(n))
    return energy_scaling * (s * s)


def compute_omb_metrics(
    az_uniform: np.ndarray,
    cfg: OMBConfig,
    window_position: str,
) -> dict:
    n = az_uniform.size
    max_segments = int((n - cfg.start_margin_samples - cfg.fft_length) // cfg.fft_overlap + 1)
    if max_segments < 1:
        needed = cfg.start_margin_samples + cfg.fft_length
        raise ValueError(
            f"Muestras insuficientes tras remuestreo: N={n}, minimo={needed} para 1 segmento"
        )

    n_segments = min(cfg.number_welch_segments_target, max_segments)

    used_len = cfg.start_margin_samples + (n_segments - 1) * cfg.fft_overlap + cfg.fft_length
    if window_position == "head":
        base = 0
    else:
        base = n - used_len

    if base < 0:
        raise ValueError("Ventana de analisis invalida; revisa longitud de serie")

    x = az_uniform[base : base + used_len]

    df = cfg.fs_target_hz / cfg.fft_length
    bins = np.arange(cfg.welch_bin_min, cfg.welch_bin_max, dtype=np.int64)
    freqs = bins * df

    welch = np.zeros(freqs.shape[0], dtype=np.float64)
    window = omb_hanning_window(cfg.fft_length, cfg.hanning_energy_scaling)

    for seg in range(n_segments):
        start = cfg.start_margin_samples + seg * cfg.fft_overlap
        end = start + cfg.fft_length

        segment = x[start:end].astype(np.float64, copy=True)
        segment = segment - cfg.gravity_mps2
        segment *= window

        fft_vals = np.fft.rfft(segment, n=cfg.fft_length)

        for i, k in enumerate(bins):
            energy = float((fft_vals[k].real * fft_vals[k].real) + (fft_vals[k].imag * fft_vals[k].imag))
            energy /= float(n_segments)
            welch[i] += 2.0 * energy / cfg.fft_length / cfg.fft_length / df

    omega4 = (2.0 * np.pi * freqs) ** 4
    seta = welch / omega4

    # Replica firmware: integracion por suma discreta (no trapezoidal).
    m0 = float(np.sum(seta) * df)
    m2 = float(np.sum((freqs**2) * seta) * df)
    m4 = float(np.sum((freqs**4) * seta) * df)

    sqrt_m0 = math.sqrt(max(m0, 0.0))
    sqrt_m2 = math.sqrt(max(m2, 0.0))
    sqrt_m4 = math.sqrt(max(m4, 0.0))

    hs = 4.0 * sqrt_m0

    # Valores "raw" que guarda firmware (realmente son frecuencia media en Hz).
    tz_raw = sqrt_m2 / sqrt_m0 if sqrt_m0 > 0.0 else float("nan")
    tc_raw = sqrt_m4 / sqrt_m2 if sqrt_m2 > 0.0 else float("nan")

    # Valores en periodo (s) como usa el decoder oficial al leer paquete.
    tz_s = 1.0 / tz_raw if np.isfinite(tz_raw) and tz_raw > 0.0 else float("nan")
    tc_s = 1.0 / tc_raw if np.isfinite(tc_raw) and tc_raw > 0.0 else float("nan")

    kmax = int(np.argmax(seta))
    fp = float(freqs[kmax])
    tp_s = 1.0 / fp if fp > 0.0 else float("nan")

    max_welch = float(np.max(welch)) if welch.size else 0.0
    if max_welch > 0:
        welch_q16 = np.clip(np.round(welch / max_welch * 65000.0), 0, 65000).astype(np.uint16)
    else:
        welch_q16 = np.zeros_like(welch, dtype=np.uint16)

    return {
        "samples_uniform_total": int(n),
        "samples_used": int(used_len),
        "analysis_base_index": int(base),
        "segments_used": int(n_segments),
        "df_hz": float(df),
        "frequency_min_hz": float(freqs[0]),
        "frequency_max_hz": float(freqs[-1]),
        "m0": m0,
        "m2": m2,
        "m4": m4,
        "hs_m": float(hs),
        "tz_raw_hz": float(tz_raw),
        "tc_raw_hz": float(tc_raw),
        "tz_s": float(tz_s),
        "tc_s": float(tc_s),
        "tp_s": float(tp_s),
        "welch_accel_psd": welch.tolist(),
        "welch_accel_psd_q16": welch_q16.tolist(),
        "welch_accel_psd_max": float(max_welch),
        "frequencies_hz": freqs.tolist(),
    }


def main() -> None:
    args = parse_args()
    cfg = OMBConfig()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"No existe el archivo: {path}")

    t, az = load_session(path)
    fs_in = fs_estimate(t)
    if fs_in <= 0:
        raise SystemExit("No se pudo estimar frecuencia de muestreo de entrada")

    t_u, az_u = resample_to_uniform_fs(t, az, cfg.fs_target_hz)

    metrics = compute_omb_metrics(az_u, cfg, args.window_position)

    print(f"FILE={path}")
    print(f"input_fs≈{fs_in:.4f}Hz | input_N={len(t)} | input_duracion≈{t[-1]:.2f}s")
    print(
        f"OMB fs={cfg.fs_target_hz:.2f}Hz | uniform_N={metrics['samples_uniform_total']} "
        f"| used_N={metrics['samples_used']} | segments={metrics['segments_used']}"
    )
    print(
        f"OMB bins=[{cfg.welch_bin_min},{cfg.welch_bin_max}) "
        f"f=[{metrics['frequency_min_hz']:.5f},{metrics['frequency_max_hz']:.5f}]Hz "
        f"df={metrics['df_hz']:.6f}Hz"
    )
    print(
        "RESULTADOS OMB: "
        f"Hs={metrics['hs_m']:.5f}m "
        f"Tz={metrics['tz_s']:.5f}s "
        f"Tc={metrics['tc_s']:.5f}s "
        f"Tp={metrics['tp_s']:.5f}s"
    )

    if args.out_json:
        out_path = Path(args.out_json)
        payload = {
            "input_file": str(path),
            "config": {
                "fs_target_hz": cfg.fs_target_hz,
                "gravity_mps2": cfg.gravity_mps2,
                "fft_length": cfg.fft_length,
                "fft_overlap": cfg.fft_overlap,
                "number_welch_segments_target": cfg.number_welch_segments_target,
                "start_margin_samples": cfg.start_margin_samples,
                "welch_bin_min": cfg.welch_bin_min,
                "welch_bin_max": cfg.welch_bin_max,
                "hanning_energy_scaling": cfg.hanning_energy_scaling,
            },
            "metrics": metrics,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] JSON guardado en {out_path}")


if __name__ == "__main__":
    main()
