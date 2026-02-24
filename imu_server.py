#!/usr/bin/env python3
"""Servidor IMU por red local.

Escucha en UDP puerto 8000, recibe datos IMU de un Android y los procesa.
Acepta mensajes en JSON o CSV.

Formatos soportados:
1) JSON:
   {"ax":0.1,"ay":0.2,"az":9.8,"gx":0.01,"gy":0.02,"gz":0.03,"ts":1700000000}
2) CSV:
   ax,ay,az,gx,gy,gz,ts
"""

from __future__ import annotations

import argparse
import json
import math
import socket
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class ImuSample:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    ts: Optional[float] = None


class ImuProcessor:
    def __init__(self, window_size: int = 25, motion_threshold: float = 1.2) -> None:
        self.window: Deque[ImuSample] = deque(maxlen=window_size)
        self.motion_threshold = motion_threshold

    def process(self, sample: ImuSample) -> Dict[str, float | bool]:
        self.window.append(sample)

        acc_mag = math.sqrt(sample.ax**2 + sample.ay**2 + sample.az**2)
        gyro_mag = math.sqrt(sample.gx**2 + sample.gy**2 + sample.gz**2)

        # Promedio de magnitud de aceleración en ventana para suavizar ruido.
        avg_acc = sum(math.sqrt(s.ax**2 + s.ay**2 + s.az**2) for s in self.window) / len(self.window)

        is_moving = abs(acc_mag - 9.81) > self.motion_threshold or gyro_mag > 0.3

        return {
            "acc_magnitude": acc_mag,
            "gyro_magnitude": gyro_mag,
            "avg_acc_magnitude": avg_acc,
            "moving": is_moving,
        }


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_payload(payload: str) -> Optional[ImuSample]:
    payload = payload.strip()
    if not payload:
        return None

    # Intento JSON primero.
    if payload.startswith("{") and payload.endswith("}"):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None

        return ImuSample(
            ax=_to_float(data.get("ax")),
            ay=_to_float(data.get("ay")),
            az=_to_float(data.get("az")),
            gx=_to_float(data.get("gx")),
            gy=_to_float(data.get("gy")),
            gz=_to_float(data.get("gz")),
            ts=_to_float(data.get("ts"), default=0.0) or None,
        )

    # Fallback CSV: ax,ay,az,gx,gy,gz,ts(optional)
    parts = [p.strip() for p in payload.split(",")]
    if len(parts) < 6:
        return None

    ts = _to_float(parts[6], default=0.0) if len(parts) >= 7 else None
    return ImuSample(
        ax=_to_float(parts[0]),
        ay=_to_float(parts[1]),
        az=_to_float(parts[2]),
        gx=_to_float(parts[3]),
        gy=_to_float(parts[4]),
        gz=_to_float(parts[5]),
        ts=ts or None,
    )


def run_server(host: str, port: int, window: int, motion_threshold: float) -> None:
    processor = ImuProcessor(window_size=window, motion_threshold=motion_threshold)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))

    print(f"[IMU] Escuchando UDP en {host}:{port}")
    print("[IMU] Esperando datos... (Ctrl+C para salir)")

    while True:
        data, addr = sock.recvfrom(8192)
        text = data.decode("utf-8", errors="ignore")

        sample = parse_payload(text)
        if sample is None:
            print(f"[WARN] Payload invalido desde {addr}: {text[:120]!r}")
            continue

        result = processor.process(sample)

        print(
            "[OK] "
            f"from={addr[0]}:{addr[1]} "
            f"acc={result['acc_magnitude']:.3f}m/s2 "
            f"gyro={result['gyro_magnitude']:.3f}rad/s "
            f"avg_acc={result['avg_acc_magnitude']:.3f} "
            f"moving={result['moving']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receptor IMU UDP en puerto 8000")
    parser.add_argument("--host", default="0.0.0.0", help="Host de escucha (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Puerto de escucha (default: 8000)")
    parser.add_argument("--window", type=int, default=25, help="Ventana de promedio (default: 25)")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=1.2,
        help="Umbral de movimiento para aceleracion (default: 1.2)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_server(args.host, args.port, args.window, args.motion_threshold)
    except KeyboardInterrupt:
        print("\n[IMU] Servidor detenido.")


if __name__ == "__main__":
    main()
