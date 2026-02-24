#!/usr/bin/env python3
"""Monitor IMU en vivo por HTTP.

Recibe JSON por POST /data desde Android:
  {"t": <ms>, "az": <m/s2>, ...}

Procesa en ventana deslizante y muestra métricas en tiempo real.
Si matplotlib está disponible, abre visualización live.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Deque, Optional

import numpy as np

from process_imu_session import (
    apply_height_deadband,
    apply_sustained_height_confirmation,
    bandpass_1pole_offline,
    dominant_period_fft,
    fs_est,
    height_from_accel_fft,
    lowpass_1pole_offline,
)


class LiveState:
    def __init__(self, max_samples: int) -> None:
        self.t: Deque[float] = deque(maxlen=max_samples)
        self.az: Deque[float] = deque(maxlen=max_samples)
        self.lock = threading.Lock()
        self.t0: Optional[float] = None
        self.count = 0

    def add(self, t_s: float, az: float) -> None:
        with self.lock:
            if self.t0 is None:
                self.t0 = t_s
            tr = t_s - self.t0
            if self.t and tr <= self.t[-1]:
                tr = self.t[-1] + 0.001
            self.t.append(tr)
            self.az.append(az)
            self.count += 1

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, int]:
        with self.lock:
            t = np.array(self.t, dtype=np.float64)
            az = np.array(self.az, dtype=np.float64)
            c = self.count
        return t, az, c



def make_handler(state: LiveState):
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
            except Exception:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"bad payload")
                return

            state.add(t_s=t_s, az=az)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, fmt: str, *args) -> None:
            return

    return Handler



def compute_metrics(
    t: np.ndarray,
    az: np.ndarray,
    g_fc: float,
    fmin: float,
    fmax: float,
    height_min_change: float,
    height_step_min: float,
    confirm_samples: int,
):
    if len(t) < 20:
        return None

    fs = fs_est(t)
    if fs <= 0:
        return None

    g_est = lowpass_1pole_offline(az, fs, g_fc)
    az_dyn = az - g_est
    az_bp = bandpass_1pole_offline(az_dyn, fs, fmin, fmax)
    tp = dominant_period_fft(
        az_bp,
        fs,
        fmin=max(0.03, fmin),
        fmax=min(1.5, fmax),
        min_samples=128,
    )
    h = height_from_accel_fft(
        az_dyn,
        fs,
        fmin=fmin,
        fmax=fmax,
        min_samples=128,
    )

    h = apply_height_deadband(h, height_min_change)
    h, _ = apply_sustained_height_confirmation(h, height_step_min, confirm_samples)

    if h is None:
        hs = np.nan
        hrms = np.nan
    else:
        hrms = float(np.std(h))
        hs = 4.0 * hrms

    return {
        "fs": float(fs),
        "tp": float(tp),
        "hs": float(hs),
        "hrms": float(hrms),
        "az_dyn": az_dyn,
        "az_bp": az_bp,
        "h": h,
    }


def compute_chunked_height(
    t: np.ndarray,
    az: np.ndarray,
    chunk_seconds: float,
    g_fc: float,
    fmin: float,
    fmax: float,
    fft_min_samples: int,
    height_min_change: float,
    height_step_min: float,
    confirm_samples: int,
):
    if chunk_seconds <= 0 or len(t) < 20:
        return None
    t0 = float(t[0])
    n_complete = int((float(t[-1]) - t0) // chunk_seconds)
    if n_complete <= 0:
        return None

    t_out_parts = []
    h_out_parts = []
    last_metrics = None
    prev_end = 0.0

    for i in range(n_complete):
        start = t0 + i * chunk_seconds
        end = start + chunk_seconds
        m = (t >= start) & (t < end)
        if np.sum(m) < max(20, fft_min_samples):
            continue

        tb = t[m]
        azb = az[m]
        tb_rel = tb - tb[0]

        fs = fs_est(tb_rel)
        if fs <= 0:
            continue

        g_est = lowpass_1pole_offline(azb, fs, g_fc)
        az_dyn = azb - g_est
        az_bp = bandpass_1pole_offline(az_dyn, fs, fmin, fmax)
        tp = dominant_period_fft(
            az_bp,
            fs,
            fmin=max(0.03, fmin),
            fmax=min(1.5, fmax),
            min_samples=fft_min_samples,
        )
        hb = height_from_accel_fft(
            az_dyn,
            fs,
            fmin=fmin,
            fmax=fmax,
            min_samples=fft_min_samples,
        )
        if hb is None:
            continue

        hb = apply_height_deadband(hb, height_min_change)
        hb, _ = apply_sustained_height_confirmation(hb, height_step_min, confirm_samples)
        if hb is None:
            continue

        hb = hb - hb[0] + prev_end
        prev_end = float(hb[-1])

        t_out_parts.append(tb)
        h_out_parts.append(hb)
        last_metrics = {
            "fs": float(fs),
            "tp": float(tp),
            "hrms": float(np.std(hb)),
            "hs": float(4.0 * np.std(hb)),
            "chunk_index": i + 1,
            "n_complete": n_complete,
        }

    if not t_out_parts:
        return None

    t_out = np.concatenate(t_out_parts)
    h_out = np.concatenate(h_out_parts)
    return t_out, h_out, last_metrics


def compute_window_height(
    t: np.ndarray,
    az: np.ndarray,
    window_seconds: float,
    g_fc: float,
    fmin: float,
    fmax: float,
    height_min_change: float,
    height_step_min: float,
    confirm_samples: int,
):
    if window_seconds <= 0 or len(t) < 20:
        return None
    if float(t[-1]) - float(t[0]) < window_seconds:
        return None

    start = float(t[-1]) - window_seconds
    m = t >= start
    if np.sum(m) < 20:
        return None

    tw = t[m]
    azw = az[m]
    tw_rel = tw - tw[0]
    mm = compute_metrics(
        tw_rel,
        azw,
        g_fc,
        fmin,
        fmax,
        height_min_change,
        height_step_min,
        confirm_samples,
    )
    if mm is None or mm["h"] is None:
        return None
    return tw, mm["h"], mm



def run_text_mode(state: LiveState, args: argparse.Namespace) -> None:
    print("[LIVE] Modo texto activo (sin matplotlib). Ctrl+C para salir.")
    last_count = -1
    last_complete = -1
    last_update_idx = -1
    while True:
        time.sleep(args.refresh)
        t, az, count = state.snapshot()
        if count == last_count:
            continue
        last_count = count

        if args.window_seconds > 0:
            if len(t) <= 1:
                continue
            idx = int((float(t[-1]) - float(t[0])) // args.update_seconds) if args.update_seconds > 0 else count
            if idx == last_update_idx:
                continue
            last_update_idx = idx
            out = compute_window_height(
                t,
                az,
                args.window_seconds,
                args.g_fc,
                args.fmin,
                args.fmax,
                args.height_min_change,
                args.height_step_min,
                args.confirm_samples,
            )
            if out is None:
                print(f"[LIVE] esperando ventana de {args.window_seconds:.1f}s")
                continue
            _, _, mm = out
            print(
                f"[LIVE][window {args.window_seconds:.1f}s] "
                f"fs={mm['fs']:.2f}Hz Tp={mm['tp']:.2f}s "
                f"Hs={mm['hs']:.4f}m std(h)={mm['hrms']:.4f} "
                f"(update={args.update_seconds:.1f}s)"
            )
            continue

        if args.chunk_seconds > 0:
            c = int((float(t[-1]) - float(t[0])) // args.chunk_seconds) if len(t) > 1 else 0
            if c == last_complete:
                continue
            last_complete = c
            out = compute_chunked_height(
                t,
                az,
                args.chunk_seconds,
                args.g_fc,
                args.fmin,
                args.fmax,
                args.fft_min_samples,
                args.height_min_change,
                args.height_step_min,
                args.confirm_samples,
            )
            if out is None:
                print(f"[LIVE] muestras={count} (esperando primer bloque de {args.chunk_seconds:.1f}s)")
                continue
            _, _, lm = out
            print(
                f"[LIVE][chunk {lm['chunk_index']}/{lm['n_complete']}] "
                f"fs={lm['fs']:.2f}Hz Tp={lm['tp']:.2f}s "
                f"Hs={lm['hs']:.4f}m std(h)={lm['hrms']:.4f} "
                f"(retardo≈{args.chunk_seconds:.1f}s)"
            )
            continue

        m = compute_metrics(
            t,
            az,
            args.g_fc,
            args.fmin,
            args.fmax,
            args.height_min_change,
            args.height_step_min,
            args.confirm_samples,
        )
        if m is None:
            print(f"[LIVE] muestras={count} (aun sin datos suficientes)")
            continue
        print(
            f"[LIVE] N={count} fs={m['fs']:.2f}Hz "
            f"Tp={m['tp']:.2f}s Hs={m['hs']:.4f}m std(h)={m['hrms']:.4f}"
        )



def run_plot_mode(state: LiveState, args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, TextBox

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.subplots_adjust(bottom=0.40)

    line_az, = ax1.plot([], [], label="az cruda", alpha=0.30)
    line_dyn, = ax1.plot([], [], label="az sin g", alpha=0.8)
    line_bp, = ax1.plot([], [], label="az band-pass", lw=1.5)
    ax1.set_ylabel("m/s2")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    line_h, = ax2.plot([], [], label="h (freq)", lw=1.6)
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("m")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    txt = ax1.text(0.01, 0.98, "", transform=ax1.transAxes, va="top")

    # Panel de control en vivo (mismos parametros que offline).
    ax_gfc = fig.add_axes([0.08, 0.24, 0.26, 0.03])
    ax_fmin = fig.add_axes([0.38, 0.24, 0.26, 0.03])
    ax_fmax = fig.add_axes([0.68, 0.24, 0.26, 0.03])
    ax_hmc = fig.add_axes([0.08, 0.18, 0.26, 0.03])
    ax_hstep = fig.add_axes([0.38, 0.18, 0.26, 0.03])
    ax_conf = fig.add_axes([0.68, 0.18, 0.26, 0.03])

    s_gfc = Slider(ax_gfc, "g_fc", 0.01, 0.40, valinit=args.g_fc, valstep=0.005)
    s_fmin = Slider(ax_fmin, "fmin", 0.005, 0.50, valinit=args.fmin, valstep=0.005)
    s_fmax = Slider(ax_fmax, "fmax", 0.10, 3.00, valinit=args.fmax, valstep=0.01)
    s_hmc = Slider(
        ax_hmc,
        "h_min",
        0.0,
        0.0100,
        valinit=args.height_min_change,
        valstep=0.00005,
    )
    s_hstep = Slider(
        ax_hstep,
        "h_step",
        0.0,
        0.000050,
        valinit=args.height_step_min,
        valstep=0.000001,
    )
    s_conf = Slider(ax_conf, "confirmN", 1, 30, valinit=args.confirm_samples, valstep=1)

    # Evita solape de textos: el valor exacto ya se muestra en las cajas inferiores.
    for s in (s_gfc, s_fmin, s_fmax, s_hmc, s_hstep, s_conf):
        s.valtext.set_visible(False)
        s.label.set_fontsize(10)

    # Entrada numerica directa + boton aplicar.
    tx_gfc_ax = fig.add_axes([0.08, 0.10, 0.12, 0.04])
    tx_fmin_ax = fig.add_axes([0.22, 0.10, 0.12, 0.04])
    tx_fmax_ax = fig.add_axes([0.36, 0.10, 0.12, 0.04])
    tx_hmc_ax = fig.add_axes([0.50, 0.10, 0.12, 0.04])
    tx_hstep_ax = fig.add_axes([0.64, 0.10, 0.12, 0.04])
    tx_conf_ax = fig.add_axes([0.78, 0.10, 0.08, 0.04])
    btn_apply_ax = fig.add_axes([0.88, 0.10, 0.08, 0.04])

    t_gfc = TextBox(tx_gfc_ax, "g_fc", initial=f"{args.g_fc:.4f}")
    t_fmin = TextBox(tx_fmin_ax, "fmin", initial=f"{args.fmin:.4f}")
    t_fmax = TextBox(tx_fmax_ax, "fmax", initial=f"{args.fmax:.4f}")
    t_hmc = TextBox(tx_hmc_ax, "h_min", initial=f"{args.height_min_change:.6f}")
    t_hstep = TextBox(tx_hstep_ax, "h_step", initial=f"{args.height_step_min:.6f}")
    t_conf = TextBox(tx_conf_ax, "N", initial=f"{int(args.confirm_samples)}")
    b_apply = Button(btn_apply_ax, "Aplicar")

    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    def apply_from_text(_event=None) -> None:
        try:
            g = _clamp(float(t_gfc.text), 0.01, 0.40)
            nmin = _clamp(float(t_fmin.text), 0.005, 0.50)
            nmax = _clamp(float(t_fmax.text), 0.10, 3.00)
            hmin = _clamp(float(t_hmc.text), 0.0, 0.0100)
            hstep = _clamp(float(t_hstep.text), 0.0, 0.000050)
            nconf = int(round(_clamp(float(t_conf.text), 1, 30)))
        except ValueError:
            return

        if nmax <= nmin:
            nmax = nmin + 0.01

        s_gfc.set_val(g)
        s_fmin.set_val(nmin)
        s_fmax.set_val(nmax)
        s_hmc.set_val(hmin)
        s_hstep.set_val(hstep)
        s_conf.set_val(nconf)

        t_gfc.set_val(f"{g:.4f}")
        t_fmin.set_val(f"{nmin:.4f}")
        t_fmax.set_val(f"{nmax:.4f}")
        t_hmc.set_val(f"{hmin:.6f}")
        t_hstep.set_val(f"{hstep:.6f}")
        t_conf.set_val(f"{nconf}")

    b_apply.on_clicked(apply_from_text)
    t_gfc.on_submit(apply_from_text)
    t_fmin.on_submit(apply_from_text)
    t_fmax.on_submit(apply_from_text)
    t_hmc.on_submit(apply_from_text)
    t_hstep.on_submit(apply_from_text)
    t_conf.on_submit(apply_from_text)

    last_update_idx = -1

    while plt.fignum_exists(fig.number):
        plt.pause(args.refresh)
        t, az, count = state.snapshot()
        if len(t) < 20:
            continue

        g_fc = float(s_gfc.val)
        fmin = float(s_fmin.val)
        fmax = float(s_fmax.val)
        if fmax <= fmin:
            fmax = fmin + 0.01
        height_min_change = float(s_hmc.val)
        height_step_min = float(s_hstep.val)
        confirm_samples = int(round(s_conf.val))

        if args.window_seconds > 0:
            idx = int((float(t[-1]) - float(t[0])) // args.update_seconds) if args.update_seconds > 0 else count
            if idx == last_update_idx:
                continue
            last_update_idx = idx

            line_az.set_data(t, az)
            line_dyn.set_data([], [])
            line_bp.set_data([], [])
            ax1.set_xlim(float(t[0]), float(t[-1]))
            ymin, ymax = float(np.min(az)), float(np.max(az))
            pad = 0.2 * (ymax - ymin + 1e-6) + 0.1
            ax1.set_ylim(ymin - pad, ymax + pad)

            outw = compute_window_height(
                t,
                az,
                args.window_seconds,
                g_fc,
                fmin,
                fmax,
                height_min_change,
                height_step_min,
                confirm_samples,
            )
            if outw is None:
                line_h.set_data([], [])
                txt.set_text(
                    f"N={count} esperando ventana {args.window_seconds:.1f}s "
                    f"(update={args.update_seconds:.1f}s)"
                )
                fig.canvas.draw_idle()
                continue

            tw, hw, mm = outw
            line_h.set_data(tw, hw)
            ax2.set_xlim(float(t[0]), float(t[-1]))
            hymin, hymax = float(np.min(hw)), float(np.max(hw))
            hpad = 0.15 * (hymax - hymin + 1e-6) + 0.001
            ax2.set_ylim(hymin - hpad, hymax + hpad)
            txt.set_text(
                f"N={count} window={args.window_seconds:.1f}s update={args.update_seconds:.1f}s "
                f"Tp={mm['tp']:.2f}s std(h)={mm['hrms']:.4f}m Hs={mm['hs']:.4f}m "
                f"g={g_fc:.3f} [{fmin:.3f},{fmax:.2f}] "
                f"hmin={height_min_change:.5f} step={height_step_min:.6f} N={confirm_samples}"
            )
            fig.canvas.draw_idle()
            continue

        if args.chunk_seconds > 0:
            line_az.set_data(t, az)
            line_dyn.set_data([], [])
            line_bp.set_data([], [])
            ax1.set_xlim(float(t[0]), float(t[-1]))
            ymin, ymax = float(np.min(az)), float(np.max(az))
            pad = 0.2 * (ymax - ymin + 1e-6) + 0.1
            ax1.set_ylim(ymin - pad, ymax + pad)

            out = compute_chunked_height(
                t,
                az,
                args.chunk_seconds,
                g_fc,
                fmin,
                fmax,
                args.fft_min_samples,
                height_min_change,
                height_step_min,
                confirm_samples,
            )
            if out is None:
                line_h.set_data([], [])
                txt.set_text(
                    f"N={count} esperando bloque {args.chunk_seconds:.1f}s "
                    f"(retardo≈{args.chunk_seconds:.1f}s)"
                )
                fig.canvas.draw_idle()
                continue

            th, hh, lm = out
            line_h.set_data(th, hh)
            ax2.set_xlim(float(t[0]), float(t[-1]))
            hymin, hymax = float(np.min(hh)), float(np.max(hh))
            hpad = 0.15 * (hymax - hymin + 1e-6) + 0.001
            ax2.set_ylim(hymin - hpad, hymax + hpad)
            txt.set_text(
                f"N={count} chunk={args.chunk_seconds:.1f}s "
                f"Tp={lm['tp']:.2f}s std(h)={lm['hrms']:.4f}m Hs={lm['hs']:.4f}m "
                f"(retardo≈{args.chunk_seconds:.1f}s) "
                f"g={g_fc:.3f} [{fmin:.3f},{fmax:.2f}] "
                f"hmin={height_min_change:.5f} step={height_step_min:.6f} N={confirm_samples}"
            )
            fig.canvas.draw_idle()
            continue

        m = compute_metrics(
            t,
            az,
            g_fc,
            fmin,
            fmax,
            height_min_change,
            height_step_min,
            confirm_samples,
        )
        if m is None:
            continue

        az_dyn = m["az_dyn"]
        az_bp = m["az_bp"]
        h = m["h"]

        line_az.set_data(t, az)
        line_dyn.set_data(t, az_dyn)
        line_bp.set_data(t, az_bp)

        ax1.set_xlim(float(t[0]), float(t[-1]))
        y_all = np.concatenate([az, az_dyn, az_bp])
        ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
        pad = 0.2 * (ymax - ymin + 1e-6) + 0.1
        ax1.set_ylim(ymin - pad, ymax + pad)

        if h is None:
            line_h.set_data(t, np.zeros_like(t))
            txt.set_text(
                f"N={count} fs={m['fs']:.2f}Hz Tp={m['tp']:.2f}s Hs=None"
            )
        else:
            line_h.set_data(t, h)
            hymin, hymax = float(np.min(h)), float(np.max(h))
            hpad = 0.15 * (hymax - hymin + 1e-6) + 0.001
            ax2.set_ylim(hymin - hpad, hymax + hpad)
            txt.set_text(
                f"N={count} fs={m['fs']:.2f}Hz Tp={m['tp']:.2f}s "
                f"std(h)={m['hrms']:.4f}m Hs={m['hs']:.4f}m "
                f"g={g_fc:.3f} [{fmin:.3f},{fmax:.2f}] "
                f"hmin={height_min_change:.5f} step={height_step_min:.6f} N={confirm_samples}"
            )

        fig.canvas.draw_idle()



def save_csv(state: LiveState, out_file: str) -> None:
    t, az, _ = state.snapshot()
    if len(t) == 0:
        print("[LIVE] No hay datos para guardar.")
        return
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "az"])
        for ti, azi in zip(t, az):
            w.writerow([f"{ti:.6f}", f"{azi:.9f}"])
    print(f"[LIVE] Guardado CSV: {out_file}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monitor IMU live por HTTP")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--refresh", type=float, default=0.15, help="Refresco UI/metrics en segundos")
    p.add_argument("--g-fc", type=float, default=0.03)
    p.add_argument("--fmin", type=float, default=0.02)
    p.add_argument("--fmax", type=float, default=1.20)
    p.add_argument("--height-min-change", type=float, default=0.0)
    p.add_argument("--height-step-min", type=float, default=0.0)
    p.add_argument("--confirm-samples", type=int, default=1)
    p.add_argument("--window-seconds", type=float, default=0.0, help="Ventana deslizante de procesado en segundos")
    p.add_argument("--update-seconds", type=float, default=1.0, help="Paso de actualizacion para ventana deslizante")
    p.add_argument("--chunk-seconds", type=float, default=0.0, help="Procesado por bloques (retardo) en segundos")
    p.add_argument("--fft-min-samples", type=int, default=128, help="Minimo de muestras para FFT por bloque")
    p.add_argument("--save-on-exit", default="", help="Ruta CSV para guardar al salir")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    state = LiveState(max_samples=args.max_samples)

    server = HTTPServer((args.host, args.port), make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"[LIVE] HTTP escuchando en {args.host}:{args.port} (POST /data)")
    print("[LIVE] Payload esperado: {\"t\": <ms>, \"az\": <m/s2>}")

    try:
        try:
            import matplotlib.pyplot as _plt  # noqa: F401
            run_plot_mode(state, args)
        except ModuleNotFoundError:
            run_text_mode(state, args)
    except KeyboardInterrupt:
        print("\n[LIVE] Detenido por usuario.")
    finally:
        server.shutdown()
        server.server_close()
        if args.save_on_exit:
            save_csv(state, args.save_on_exit)


if __name__ == "__main__":
    main()
