#!/usr/bin/env python3
"""Genera un CSV IMU sintetico (sin envio HTTP).

Salida compatible con el pipeline actual:
cabecera -> t,ax,ay,az,gx,gy,gz
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from dataclasses import dataclass


@dataclass
class WaveComponent:
    amplitude_m: float
    period_s: float
    phase_rad: float


class RunningStats:
    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        d2 = x - self.mean
        self.m2 += d * d2

    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.n - 1))


def parse_components(text: str) -> list[WaveComponent]:
    out: list[WaveComponent] = []
    raw = [p.strip() for p in text.split(";") if p.strip()]
    for part in raw:
        toks = [x.strip() for x in part.split(",")]
        if len(toks) != 3:
            raise ValueError(f"Componente invalida: {part} (esperado A,T,phase)")
        a = float(toks[0])
        t = float(toks[1])
        p = float(toks[2])
        if t <= 0:
            raise ValueError("Cada periodo debe ser > 0")
        out.append(WaveComponent(a, t, p))
    if not out:
        raise ValueError("Debes indicar al menos una componente")
    return out


def elevation_and_vertical_acc(t_s: float, comps: list[WaveComponent]) -> tuple[float, float]:
    eta = 0.0
    az_wave = 0.0
    for c in comps:
        omega = 2.0 * math.pi / c.period_s
        s = math.sin(omega * t_s + c.phase_rad)
        eta += c.amplitude_m * s
        az_wave += -c.amplitude_m * (omega**2) * s
    return eta, az_wave


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generador CSV sintetico de IMU")
    p.add_argument("--seconds", type=float, default=1200.0, help="Duracion simulada (s)")
    p.add_argument("--fs", type=float, default=50.0, help="Frecuencia de muestreo (Hz)")
    p.add_argument(
        "--components",
        default="0.08,8.0,0.0;0.03,4.5,1.2;0.015,12.0,0.5",
        help="Componentes de elevacion: 'A,T,phase;A,T,phase;...'",
    )
    p.add_argument("--noise-std", type=float, default=0.05, help="Ruido gaussiano en az (m/s2)")
    p.add_argument("--xy-noise-std", type=float, default=0.02, help="Ruido ax/ay (m/s2)")
    p.add_argument("--gyro-noise-std", type=float, default=0.01, help="Ruido gx/gy/gz (rad/s)")
    p.add_argument("--drift-az", type=float, default=0.0, help="Deriva lineal en az (m/s3)")
    p.add_argument("--gravity", type=float, default=9.81, help="Gravedad base (m/s2)")
    p.add_argument("--seed", type=int, default=123, help="Semilla aleatoria")
    p.add_argument("--out-prefix", default="simulated_session", help="Prefijo del CSV")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.seconds <= 0:
        raise ValueError("--seconds debe ser > 0")
    if args.fs <= 0:
        raise ValueError("--fs debe ser > 0")

    comps = parse_components(args.components)
    random.seed(args.seed)

    dt = 1.0 / args.fs
    n_samples = int(math.floor(args.seconds * args.fs)) + 1

    eta_stats = RunningStats()
    az_wave_stats = RunningStats()
    az_total_stats = RunningStats()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"{args.out_prefix}_{ts}_{int(args.seconds)}s.csv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])

        for i in range(n_samples):
            t_s = i * dt
            eta, az_wave = elevation_and_vertical_acc(t_s, comps)
            az = args.gravity + az_wave + args.drift_az * t_s + random.gauss(0.0, args.noise_std)
            ax = random.gauss(0.0, args.xy_noise_std)
            ay = random.gauss(0.0, args.xy_noise_std)
            gx = random.gauss(0.0, args.gyro_noise_std)
            gy = random.gauss(0.0, args.gyro_noise_std)
            gz = random.gauss(0.0, args.gyro_noise_std)

            w.writerow(
                [
                    f"{t_s:.6f}",
                    f"{ax:.9f}",
                    f"{ay:.9f}",
                    f"{az:.9f}",
                    f"{gx:.9f}",
                    f"{gy:.9f}",
                    f"{gz:.9f}",
                ]
            )

            eta_stats.add(eta)
            az_wave_stats.add(az_wave)
            az_total_stats.add(az)

    hs_theory = 4.0 * eta_stats.std()
    print(f"[OK] CSV generado: {out_file}")
    print(
        f"[INFO] N={n_samples} fs={args.fs:.2f}Hz duracion≈{(n_samples - 1) * dt:.2f}s "
        f"| eta_std={eta_stats.std():.5f}m Hs_teorica≈{hs_theory:.5f}m"
    )
    print(
        f"[INFO] az_wave_std={az_wave_stats.std():.5f}m/s2 "
        f"| az_total_std={az_total_stats.std():.5f}m/s2"
    )


if __name__ == "__main__":
    main()
