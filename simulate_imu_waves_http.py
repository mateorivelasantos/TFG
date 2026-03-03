#!/usr/bin/env python3
"""Simula datos IMU de olas y los envia por HTTP (POST /data).

Genera aceleracion vertical con:
- gravedad
- una o varias componentes sinusoidales de ola
- ruido gaussiano
- deriva lenta opcional

Formato enviado (compatible con capture_http_imu.py):
{"t": <ms>, "ax": ..., "ay": ..., "az": ..., "gx": ..., "gy": ..., "gz": ...}
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse


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
    comps: list[WaveComponent] = []
    chunks = [c.strip() for c in text.split(";") if c.strip()]
    for chunk in chunks:
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Componente invalida: '{chunk}'. Usa A,T,phase")
        amp = float(parts[0])
        period = float(parts[1])
        phase = float(parts[2])
        if period <= 0:
            raise ValueError("El periodo de cada componente debe ser > 0")
        comps.append(WaveComponent(amp, period, phase))
    if not comps:
        raise ValueError("Debes indicar al menos una componente de ola")
    return comps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulador IMU de olas por HTTP")
    p.add_argument("--url", default="http://127.0.0.1:8001/data", help="URL destino POST")
    p.add_argument("--duration-seconds", type=float, default=1200.0, help="Duracion simulada (s)")
    p.add_argument("--fs", type=float, default=50.0, help="Frecuencia de muestreo (Hz)")
    p.add_argument(
        "--components",
        default="0.08,8.0,0.0;0.03,4.5,1.2",
        help="Componentes de elevacion: 'A,T,phase;A,T,phase;...'",
    )
    p.add_argument("--noise-std", type=float, default=0.05, help="Ruido gaussiano en az (m/s2)")
    p.add_argument("--xy-noise-std", type=float, default=0.02, help="Ruido ax/ay (m/s2)")
    p.add_argument("--gyro-noise-std", type=float, default=0.01, help="Ruido gx/gy/gz (rad/s)")
    p.add_argument("--drift-az", type=float, default=0.0, help="Deriva lineal en az (m/s3)")
    p.add_argument("--gravity", type=float, default=9.81, help="Gravedad base (m/s2)")
    p.add_argument("--seed", type=int, default=42, help="Semilla pseudoaleatoria")
    p.add_argument("--report-every", type=float, default=5.0, help="Progreso cada N segundos simulados")
    p.add_argument(
        "--realtime",
        action="store_true",
        help="Si se activa, respeta el tiempo real entre muestras",
    )
    return p


def elevation_and_vertical_acc(
    t_s: float,
    components: Iterable[WaveComponent],
) -> tuple[float, float]:
    eta = 0.0
    az_wave = 0.0
    for comp in components:
        omega = 2.0 * math.pi / comp.period_s
        s = math.sin(omega * t_s + comp.phase_rad)
        eta += comp.amplitude_m * s
        az_wave += -comp.amplitude_m * (omega**2) * s
    return eta, az_wave


def open_connection(parsed_url) -> tuple[http.client.HTTPConnection, str, str]:
    scheme = parsed_url.scheme.lower()
    host = parsed_url.hostname
    if host is None:
        raise ValueError("URL sin host valido")
    port = parsed_url.port
    path = parsed_url.path or "/"
    if parsed_url.query:
        path = f"{path}?{parsed_url.query}"

    if scheme == "http":
        conn = http.client.HTTPConnection(host, port or 80, timeout=10)
    elif scheme == "https":
        conn = http.client.HTTPSConnection(host, port or 443, timeout=10)
    else:
        raise ValueError("Solo se soporta http:// o https://")
    return conn, path, host


def send_one(conn: http.client.HTTPConnection, path: str, payload: dict) -> int:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    status = resp.status
    resp.read()
    return status


def main() -> None:
    args = build_parser().parse_args()
    if args.fs <= 0:
        raise ValueError("--fs debe ser > 0")
    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds debe ser > 0")

    components = parse_components(args.components)
    random.seed(args.seed)

    parsed = urlparse(args.url)
    conn, path, host = open_connection(parsed)
    dt = 1.0 / args.fs
    # Incluimos la muestra final para alcanzar exactamente "duration_seconds".
    n_samples = int(math.floor(args.duration_seconds * args.fs)) + 1
    t0_ms = int(time.time() * 1000)

    eta_stats = RunningStats()
    az_wave_stats = RunningStats()
    az_total_stats = RunningStats()

    sent_ok = 0
    sent_err = 0
    last_report_t = -1.0

    print(
        f"[SIM] Destino={args.url} host={host} | duracion={args.duration_seconds:.1f}s "
        f"fs={args.fs:.2f}Hz N={n_samples}"
    )
    print(f"[SIM] Componentes={args.components}")

    wall_start = time.perf_counter()
    try:
        for i in range(n_samples):
            t_s = i * dt
            eta, az_wave = elevation_and_vertical_acc(t_s, components)
            az = (
                args.gravity
                + az_wave
                + args.drift_az * t_s
                + random.gauss(0.0, args.noise_std)
            )
            ax = random.gauss(0.0, args.xy_noise_std)
            ay = random.gauss(0.0, args.xy_noise_std)
            gx = random.gauss(0.0, args.gyro_noise_std)
            gy = random.gauss(0.0, args.gyro_noise_std)
            gz = random.gauss(0.0, args.gyro_noise_std)

            payload = {
                "t": t0_ms + int(round(t_s * 1000.0)),
                "ax": ax,
                "ay": ay,
                "az": az,
                "gx": gx,
                "gy": gy,
                "gz": gz,
            }

            try:
                status = send_one(conn, path, payload)
            except Exception:
                sent_err += 1
                try:
                    conn.close()
                except Exception:
                    pass
                conn, path, host = open_connection(parsed)
                continue

            if status == 200:
                sent_ok += 1
            else:
                sent_err += 1

            eta_stats.add(eta)
            az_wave_stats.add(az_wave)
            az_total_stats.add(az)

            if args.report_every > 0:
                crrt_slot = math.floor(t_s / args.report_every)
                if crrt_slot != last_report_t:
                    last_report_t = crrt_slot
                    print(
                        f"[SIM] t={t_s:7.2f}s | sent_ok={sent_ok} sent_err={sent_err} "
                        f"| az={az:8.4f}"
                    )

            if args.realtime:
                target = wall_start + t_s
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    wall_elapsed = time.perf_counter() - wall_start
    eta_std = eta_stats.std()
    hs_theory = 4.0 * eta_std

    print("\n[SIM] Finalizado")
    print(
        f"[SIM] enviados_ok={sent_ok} errores={sent_err} "
        f"| wall_time={wall_elapsed:.2f}s | throughput={sent_ok / max(wall_elapsed, 1e-9):.1f} msg/s"
    )
    print(
        f"[SIM] eta_std={eta_std:.5f}m => Hs_teorica≈{hs_theory:.5f}m "
        f"| az_wave_std={az_wave_stats.std():.5f}m/s2 | az_total_std={az_total_stats.std():.5f}m/s2"
    )


if __name__ == "__main__":
    main()
