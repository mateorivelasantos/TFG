#!/usr/bin/env python3
"""Procesa una sesion NPZ de IMU (t, az) y muestra graficas interactivas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# ---------- DSP helpers ----------
def fs_est(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[dt > 0]
    return 1.0 / np.median(dt) if len(dt) > 5 else 0.0


def lowpass_1pole_offline(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    if fs <= 0 or fc <= 0:
        return x.copy()
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * fc)
    a = dt / (rc + dt)
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def bandpass_1pole_offline(x: np.ndarray, fs: float, fmin: float, fmax: float) -> np.ndarray:
    if fs <= 0:
        return x.copy()
    dt = 1.0 / fs

    # HP (1er orden)
    rc = 1.0 / (2 * np.pi * max(fmin, 1e-6))
    a = rc / (rc + dt)
    y_hp = np.empty_like(x, dtype=np.float64)
    y_hp[0] = 0.0
    x_prev = x[0]
    for i in range(1, len(x)):
        y_hp[i] = a * (y_hp[i - 1] + x[i] - x_prev)
        x_prev = x[i]

    # LP (1er orden)
    rc2 = 1.0 / (2 * np.pi * max(fmax, 1e-6))
    a2 = dt / (rc2 + dt)
    y_lp = np.empty_like(x, dtype=np.float64)
    y_lp[0] = y_hp[0]
    for i in range(1, len(x)):
        y_lp[i] = y_lp[i - 1] + a2 * (y_hp[i] - y_lp[i - 1])

    return y_lp


def dominant_period_fft(
    y: np.ndarray, fs: float, fmin: float = 0.04, fmax: float = 1.0, min_samples: int = 256
) -> float:
    n = len(y)
    if n < min_samples or fs <= 0:
        return np.nan
    y0 = y - np.mean(y)
    w = np.hanning(n)
    Y = np.abs(np.fft.rfft(y0 * w))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return np.nan
    fpk = freqs[mask][np.argmax(Y[mask])]
    return float(1.0 / fpk) if fpk > 0 else np.nan


def height_from_accel_fft(
    a: np.ndarray, fs: float, fmin: float, fmax: float, min_samples: int = 256
) -> np.ndarray | None:
    """Altura por doble integracion en frecuencia.

    H(f) = A(f) / (2*pi*f)^2, usando mascara [fmin,fmax] para evitar DC.
    """
    n = len(a)
    if n < min_samples or fs <= 0:
        return None

    a0 = a - np.mean(a)
    w = np.hanning(n)
    A = np.fft.rfft(a0 * w)
    freqs = np.fft.rfftfreq(n, d=1 / fs)

    mask = (freqs >= max(fmin, 1e-6)) & (freqs <= fmax)

    H = np.zeros_like(A, dtype=np.complex128)
    denom = (2 * np.pi * freqs[mask]) ** 2
    H[mask] = A[mask] / denom

    h = np.fft.irfft(H, n=n)
    h = h - np.mean(h)
    return h


def apply_height_deadband(h: np.ndarray | None, min_change_m: float) -> np.ndarray | None:
    """Filtra microcambios de altura con deadband y retencion de ultimo valor."""
    if h is None or min_change_m <= 0:
        return h
    out = np.empty_like(h, dtype=np.float64)
    out[0] = h[0]
    for i in range(1, len(h)):
        if abs(h[i] - out[i - 1]) >= min_change_m:
            out[i] = h[i]
        else:
            out[i] = out[i - 1]
    return out


def apply_sustained_height_confirmation(
    h: np.ndarray | None, min_step_m: float, min_samples: int
) -> tuple[np.ndarray | None, np.ndarray]:
    """Confirma cambios de altura solo si persisten N muestras con el mismo signo.

    Retorna:
      - h_confirmada: altura con retencion mientras no haya confirmacion
      - state: -1 bajando confirmado, +1 subiendo confirmado, 0 sin confirmar
    """
    if h is None:
        return None, np.array([], dtype=np.int8)

    n = len(h)
    state = np.zeros(n, dtype=np.int8)
    if min_step_m <= 0 or min_samples <= 1:
        return h.copy(), state

    out = np.empty_like(h, dtype=np.float64)
    out[0] = h[0]

    run_sign = 0
    run_len = 0

    for i in range(1, n):
        step = h[i] - h[i - 1]
        sign = 1 if step >= min_step_m else (-1 if step <= -min_step_m else 0)

        if sign == 0:
            run_sign = 0
            run_len = 0
            out[i] = out[i - 1]
            continue

        if sign == run_sign:
            run_len += 1
        else:
            run_sign = sign
            run_len = 1

        if run_len >= min_samples:
            state[i] = np.int8(sign)
            out[i] = h[i]
        else:
            out[i] = out[i - 1]

    return out, state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Procesa una sesion IMU (.npz o .csv)")
    p.add_argument("file", help="Ruta al archivo .npz o .csv")
    p.add_argument("--no-plot", action="store_true", help="Solo calcula metricas, sin graficas")
    p.add_argument(
        "--height-min-change",
        type=float,
        default=0.0,
        help="Cambio minimo de altura (m) para considerar que cambia (deadband)",
    )
    p.add_argument(
        "--height-step-min",
        type=float,
        default=0.0,
        help="Cambio minimo por muestra en altura (m) para confirmar direccion",
    )
    p.add_argument(
        "--confirm-samples",
        type=int,
        default=1,
        help="Numero de muestras consecutivas para confirmar subida/bajada",
    )
    return p.parse_args()


def load_session(path: str) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".npz":
        data = np.load(path)
        if "t" not in data or "az" not in data:
            raise ValueError("El NPZ debe tener claves 't' y 'az'.")
        t = data["t"].astype(np.float64)
        az = data["az"].astype(np.float64)
        return t, az

    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSV sin cabecera.")

            names = [n.strip().lower() for n in reader.fieldnames]
            t_list: list[float] = []
            az_list: list[float] = []

            for row in reader:
                row_norm = {k.strip().lower(): v for k, v in row.items() if k is not None}
                t_raw = None
                az_raw = None
                for t_name in ("t", "t_ms", "time", "timestamp"):
                    if t_name in row_norm:
                        t_raw = row_norm[t_name]
                        break
                for az_name in ("az", "a_z", "acc_z"):
                    if az_name in row_norm:
                        az_raw = row_norm[az_name]
                        break

                if t_raw is None or az_raw is None:
                    continue

                try:
                    t_list.append(float(t_raw))
                    az_list.append(float(az_raw))
                except ValueError:
                    continue

        if len(t_list) < 10:
            raise ValueError("CSV sin suficientes muestras validas para t/az.")

        t = np.array(t_list, dtype=np.float64)
        az = np.array(az_list, dtype=np.float64)

        # Si viene en ms (cabecera t_ms o valores grandes), pasa a segundos.
        if "t_ms" in names or np.nanmedian(np.diff(t[t.size // 4 :])) > 1.0:
            t = t / 1000.0

        # Rebase temporal a t=0.
        t = t - t[0]
        return t, az

    raise ValueError("Formato no soportado. Usa .npz o .csv")


def main() -> None:
    args = parse_args()

    t, az = load_session(args.file)

    fs = fs_est(t)
    print(f"FILE={args.file}")
    print(f"fs≈{fs:.2f} Hz | N={len(t)} | duracion≈{t[-1]:.2f}s")

    # Calculo base para modo texto.
    g_fc = 0.08
    fmin = 0.08
    fmax = 0.50
    g_est = lowpass_1pole_offline(az, fs, g_fc)
    az_dyn = az - g_est
    az_bp = bandpass_1pole_offline(az_dyn, fs, fmin, fmax)
    Tp = dominant_period_fft(az_bp, fs, fmin=max(0.03, fmin), fmax=min(1.5, fmax))
    h0 = height_from_accel_fft(az_dyn, fs, fmin=fmin, fmax=fmax)
    h0 = apply_height_deadband(h0, args.height_min_change)
    h0, state0 = apply_sustained_height_confirmation(h0, args.height_step_min, args.confirm_samples)

    if h0 is None:
        print(f"Tp≈{Tp:.2f}s | h=None (insuficientes muestras o fs invalida)")
    else:
        hrms = float(np.std(h0))
        hs_approx = 4.0 * hrms
        print(
            f"Tp≈{Tp:.2f}s | std(h)≈{hrms:.4f} m | Hs≈{hs_approx:.4f} m "
            f"| deadband={args.height_min_change:.4f} m "
            f"| step_min={args.height_step_min:.6f} m | N={args.confirm_samples}"
        )
        if state0.size:
            up = int(np.sum(state0 > 0))
            down = int(np.sum(state0 < 0))
            print(f"muestras confirmadas -> subida={up} bajada={down}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider
    except ModuleNotFoundError:
        print("matplotlib no esta instalado. Ejecuta con --no-plot o instala python3-matplotlib.")
        return

    plt.ion()
    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_axes([0.08, 0.42, 0.86, 0.53])
    ax2 = fig.add_axes([0.08, 0.12, 0.86, 0.23])

    line_az, = ax1.plot(t, az, label="az cruda", alpha=0.25)
    line_az_dyn, = ax1.plot(t, az * 0, label="az sin g", alpha=0.8)
    line_az_bp, = ax1.plot(t, az * 0, label="az band-pass", lw=2)

    line_h, = ax2.plot(t, az * 0, label="h (freq)", lw=2)

    txt1 = ax1.text(0.01, 0.98, "", transform=ax1.transAxes, va="top")

    ax1.set_title("Aceleracion (offline)")
    ax1.set_xlabel("Tiempo (s)")
    ax1.set_ylabel("m/s2")
    ax1.grid(True)
    ax1.legend(loc="upper right")

    ax2.set_title("Altura por integracion espectral (offline)")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_ylabel("m")
    ax2.grid(True)
    ax2.legend(loc="upper right")

    ax_gfc = fig.add_axes([0.10, 0.04, 0.28, 0.03])
    ax_fmin = fig.add_axes([0.42, 0.04, 0.22, 0.03])
    ax_fmax = fig.add_axes([0.68, 0.04, 0.22, 0.03])

    s_gfc = Slider(ax_gfc, "g_fc (Hz)", 0.02, 0.30, valinit=0.08, valstep=0.01)
    s_fmin = Slider(ax_fmin, "fmin (Hz)", 0.01, 0.30, valinit=0.08, valstep=0.01)
    s_fmax = Slider(ax_fmax, "fmax (Hz)", 0.20, 3.00, valinit=0.50, valstep=0.05)

    ax_radio = fig.add_axes([0.10, 0.12, 0.20, 0.10])
    radio = RadioButtons(ax_radio, ("h desde az_dyn", "h desde az_bp"), active=0)

    ax_btn = fig.add_axes([0.34, 0.12, 0.12, 0.06])
    btn = Button(ax_btn, "Aplicar")

    def recompute(_=None) -> None:
        g_fc = float(s_gfc.val)
        fmin = float(s_fmin.val)
        fmax = float(s_fmax.val)
        if fmax <= fmin:
            fmax = fmin + 0.05

        g_est = lowpass_1pole_offline(az, fs, g_fc)
        az_dyn = az - g_est
        az_bp = bandpass_1pole_offline(az_dyn, fs, fmin, fmax)

        Tp = dominant_period_fft(az_bp, fs, fmin=max(0.03, fmin), fmax=min(1.5, fmax))

        src = radio.value_selected
        if src == "h desde az_dyn":
            h = height_from_accel_fft(az_dyn, fs, fmin=fmin, fmax=fmax)
        else:
            h = height_from_accel_fft(az_bp, fs, fmin=fmin, fmax=fmax)
        h = apply_height_deadband(h, args.height_min_change)
        h, state = apply_sustained_height_confirmation(h, args.height_step_min, args.confirm_samples)

        line_az.set_ydata(az)
        line_az_dyn.set_ydata(az_dyn)
        line_az_bp.set_ydata(az_bp)
        line_az_bp.set_label(f"az band-pass [{fmin:.2f},{fmax:.2f}] Hz")
        ax1.legend(loc="upper right")

        y_all = np.concatenate([az_dyn, az_bp])
        yspan = float(np.max(y_all) - np.min(y_all))
        ypad = 0.5 + 0.2 * (yspan + 1e-6)
        ax1.set_ylim(float(np.min(y_all) - ypad), float(np.max(y_all) + ypad))

        if h is None:
            line_h.set_ydata(np.zeros_like(t))
            txt1.set_text(
                f"fs≈{fs:.1f} Hz | g_fc={g_fc:.2f} | band=[{fmin:.2f},{fmax:.2f}] | "
                f"Tp≈{Tp:.2f}s | h=None | db={args.height_min_change:.3f}m | "
                f"step={args.height_step_min:.4f} | N={args.confirm_samples}"
            )
        else:
            line_h.set_ydata(h)
            hrms = float(np.std(h))
            hs_approx = 4.0 * hrms
            up = int(np.sum(state > 0))
            down = int(np.sum(state < 0))
            ax2.set_ylim(float(np.min(h) - 0.05), float(np.max(h) + 0.05))
            txt1.set_text(
                f"fs≈{fs:.1f} Hz | g_fc={g_fc:.2f} | band=[{fmin:.2f},{fmax:.2f}] | "
                f"Tp≈{Tp:.2f}s | std(h)≈{hrms:.4f} m | Hs≈{hs_approx:.4f} m | "
                f"db={args.height_min_change:.3f}m | step={args.height_step_min:.4f} | "
                f"N={args.confirm_samples} | up={up} down={down}"
            )

        fig.canvas.draw_idle()

    btn.on_clicked(recompute)
    recompute()
    plt.show(block=True)


if __name__ == "__main__":
    main()
