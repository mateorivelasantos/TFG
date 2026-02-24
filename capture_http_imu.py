#!/usr/bin/env python3
"""Captura IMU por HTTP y guarda sesion de 30s en NPZ/CSV.

Espera POST en /data con JSON:
  minimo: {"t": <ms>, "az": <m/s2>}
  completo: {"t": <ms>, "ax": ..., "ay": ..., "az": ..., "gx": ..., "gy": ..., "gz": ...}
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Deque, List, Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Captura IMU por HTTP y guarda NPZ/CSV")
    p.add_argument("--host", default="0.0.0.0", help="Host de escucha (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Puerto HTTP (default: 8000)")
    p.add_argument("--seconds", type=float, default=30.0, help="Duracion de captura en segundos")
    p.add_argument("--max-samples", type=int, default=200000, help="Maximo de muestras en buffer")
    p.add_argument("--out-prefix", default="session", help="Prefijo del archivo de salida")
    return p


class CaptureState:
    def __init__(self, max_samples: int, capture_seconds: float) -> None:
        self.t_buf: Deque[float] = deque(maxlen=max_samples)
        self.ax_buf: Deque[float] = deque(maxlen=max_samples)
        self.ay_buf: Deque[float] = deque(maxlen=max_samples)
        self.az_buf: Deque[float] = deque(maxlen=max_samples)
        self.gx_buf: Deque[float] = deque(maxlen=max_samples)
        self.gy_buf: Deque[float] = deque(maxlen=max_samples)
        self.gz_buf: Deque[float] = deque(maxlen=max_samples)
        self.lock = threading.Lock()
        self.t0: Optional[float] = None
        self.capture_start_tr: Optional[float] = None
        self.capture_seconds = capture_seconds
        self.done = False

    def add_sample(
        self,
        t_s: float,
        az: float,
        ax: Optional[float] = None,
        ay: Optional[float] = None,
        gx: Optional[float] = None,
        gy: Optional[float] = None,
        gz: Optional[float] = None,
    ) -> None:
        with self.lock:
            if self.t0 is None:
                self.t0 = t_s

            tr = t_s - self.t0

            # Asegura monotonia para evitar dt <= 0.
            if len(self.t_buf) and tr <= self.t_buf[-1]:
                tr = self.t_buf[-1] + 0.001

            if self.capture_start_tr is None:
                self.capture_start_tr = tr
                print(f"[CAPTURA] Iniciada en t={self.capture_start_tr:.3f}s")

            if not self.done:
                self.t_buf.append(tr)
                self.ax_buf.append(float("nan") if ax is None else ax)
                self.ay_buf.append(float("nan") if ay is None else ay)
                self.az_buf.append(az)
                self.gx_buf.append(float("nan") if gx is None else gx)
                self.gy_buf.append(float("nan") if gy is None else gy)
                self.gz_buf.append(float("nan") if gz is None else gz)
                if (tr - self.capture_start_tr) >= self.capture_seconds:
                    self.done = True
                    print(f"[CAPTURA] {self.capture_seconds:.1f}s completos.")

    def is_done(self) -> bool:
        with self.lock:
            return self.done

    def export_lists(self) -> Tuple[
        List[float], List[float], List[float], List[float], List[float], List[float], List[float]
    ]:
        with self.lock:
            t = list(self.t_buf)
            ax = list(self.ax_buf)
            ay = list(self.ay_buf)
            az = list(self.az_buf)
            gx = list(self.gx_buf)
            gy = list(self.gy_buf)
            gz = list(self.gz_buf)

        if len(t) == 0:
            return t, ax, ay, az, gx, gy, gz

        t0c = t[0]
        t_out: List[float] = []
        ax_out: List[float] = []
        ay_out: List[float] = []
        az_out: List[float] = []
        gx_out: List[float] = []
        gy_out: List[float] = []
        gz_out: List[float] = []

        for ti, axi, ayi, azi, gxi, gyi, gzi in zip(t, ax, ay, az, gx, gy, gz):
            tr = ti - t0c
            if tr <= self.capture_seconds:
                t_out.append(tr)
                ax_out.append(axi)
                ay_out.append(ayi)
                az_out.append(azi)
                gx_out.append(gxi)
                gy_out.append(gyi)
                gz_out.append(gzi)
        return t_out, ax_out, ay_out, az_out, gx_out, gy_out, gz_out


def make_handler(state: CaptureState):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/data":
                self.send_response(404)
                self.end_headers()
                return

            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            try:
                d = json.loads(body.decode())
                t_s = float(d["t"]) / 1000.0
                az = float(d["az"])
                ax = float(d["ax"]) if "ax" in d else None
                ay = float(d["ay"]) if "ay" in d else None
                gx = float(d["gx"]) if "gx" in d else None
                gy = float(d["gy"]) if "gy" in d else None
                gz = float(d["gz"]) if "gz" in d else None
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"bad payload")
                return

            state.add_sample(t_s=t_s, az=az, ax=ax, ay=ay, gx=gx, gy=gy, gz=gz)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler


def run_capture(host: str, port: int, state: CaptureState, out_prefix: str) -> str:
    handler = make_handler(state)
    server = HTTPServer((host, port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Servidor HTTP escuchando en {host}:{port} (POST /data)")
    print("Esperando muestras...")

    try:
        while not state.is_done():
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[CAPTURA] Interrumpida por usuario, guardando lo recibido.")
    finally:
        server.shutdown()
        server.server_close()

    t, ax, ay, az, gx, gy, gz = state.export_lists()
    if len(t) == 0:
        raise RuntimeError("No se recibieron muestras antes de finalizar.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    if np is not None:
        fname = f"{out_prefix}_{ts}_{int(state.capture_seconds)}s.npz"
        np.savez(
            fname,
            t=np.array(t, dtype=np.float64),
            ax=np.array(ax, dtype=np.float64),
            ay=np.array(ay, dtype=np.float64),
            az=np.array(az, dtype=np.float64),
            gx=np.array(gx, dtype=np.float64),
            gy=np.array(gy, dtype=np.float64),
            gz=np.array(gz, dtype=np.float64),
        )
    else:
        fname = f"{out_prefix}_{ts}_{int(state.capture_seconds)}s.csv"
        with open(fname, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz"])
            for ti, axi, ayi, azi, gxi, gyi, gzi in zip(t, ax, ay, az, gx, gy, gz):
                w.writerow(
                    [
                        f"{ti:.6f}",
                        f"{axi:.9f}",
                        f"{ayi:.9f}",
                        f"{azi:.9f}",
                        f"{gxi:.9f}",
                        f"{gyi:.9f}",
                        f"{gzi:.9f}",
                    ]
                )
    return fname


def main() -> None:
    args = build_parser().parse_args()
    state = CaptureState(max_samples=args.max_samples, capture_seconds=args.seconds)
    if np is None:
        print("[WARN] numpy no esta instalado. Se guardara CSV en lugar de NPZ.")

    try:
        fname = run_capture(args.host, args.port, state, args.out_prefix)
        t, *_ = state.export_lists()
        print(f"[OK] Guardado: {fname} | N={len(t)} muestras | duracion≈{t[-1]:.2f}s")
    except Exception as exc:
        print(f"[ERROR] {exc}")


if __name__ == "__main__":
    main()
