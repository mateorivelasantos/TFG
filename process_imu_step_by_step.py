#!/usr/bin/env python3
"""Procesado IMU paso a paso para visualizar el efecto de cada etapa.

Etapas:
  step1: az cruda
  step2: estimacion de gravedad + az dinamica
  step3: band-pass + periodo dominante
  step4: altura por doble integracion en frecuencia
  all:   panel completo con todas las etapas
"""

from __future__ import annotations

import argparse

import numpy as np

from process_imu_session import (
    apply_height_deadband,
    apply_sustained_height_confirmation,
    bandpass_1pole_offline,
    dominant_period_fft,
    fs_est,
    height_from_accel_fft,
    load_session,
    lowpass_1pole_offline,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualiza el procesado IMU por etapas")
    p.add_argument("file", help="Ruta al .csv o .npz")
    p.add_argument(
        "--mode",
        choices=["step1", "step2", "step3", "step4", "all"],
        default="all",
        help="Etapa a visualizar (default: all)",
    )
    p.add_argument("--g-fc", type=float, default=0.08, help="Corte LP para gravedad")
    p.add_argument("--fmin", type=float, default=0.08, help="Banda minima")
    p.add_argument("--fmax", type=float, default=0.50, help="Banda maxima")
    p.add_argument(
        "--height-source",
        choices=["dyn", "bp"],
        default="dyn",
        help="Fuente para altura: az dinamica o band-pass",
    )
    p.add_argument(
        "--height-min-change",
        type=float,
        default=0.0,
        help="Cambio minimo de altura (m) para considerar cambio real",
    )
    p.add_argument(
        "--height-step-min",
        type=float,
        default=0.0,
        help="Cambio minimo por muestra en altura para confirmar direccion",
    )
    p.add_argument(
        "--confirm-samples",
        type=int,
        default=1,
        help="Muestras consecutivas para confirmar subida/bajada",
    )
    p.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.0,
        help="Si >0, procesa por bloques no solapados de N segundos",
    )
    p.add_argument(
        "--fft-min-samples",
        type=int,
        default=128,
        help="Minimo de muestras para activar FFT/integracion espectral",
    )
    p.add_argument(
        "--height-unit",
        choices=["m", "cm", "mm"],
        default="m",
        help="Unidad para visualizar altura en grafica (default: m)",
    )
    p.add_argument(
        "--height-ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=None,
        help="Limites fijos del eje Y de altura en la unidad seleccionada",
    )
    p.add_argument("--save", default="", help="Ruta PNG para guardar figura en vez de mostrar")
    return p.parse_args()


def compute_pipeline(
    t: np.ndarray,
    az: np.ndarray,
    fs: float,
    g_fc: float,
    fmin: float,
    fmax: float,
    fft_min_samples: int,
):
    g_est = lowpass_1pole_offline(az, fs, g_fc)
    az_dyn = az - g_est
    az_bp = bandpass_1pole_offline(az_dyn, fs, fmin, fmax)
    Tp = dominant_period_fft(
        az_bp,
        fs,
        fmin=max(0.03, fmin),
        fmax=min(1.5, fmax),
        min_samples=fft_min_samples,
    )
    h_dyn = height_from_accel_fft(
        az_dyn,
        fs,
        fmin=fmin,
        fmax=fmax,
        min_samples=fft_min_samples,
    )
    h_bp = height_from_accel_fft(
        az_bp,
        fs,
        fmin=fmin,
        fmax=fmax,
        min_samples=fft_min_samples,
    )
    return g_est, az_dyn, az_bp, Tp, h_dyn, h_bp


def print_summary(fs: float, t: np.ndarray, az: np.ndarray, az_dyn: np.ndarray, az_bp: np.ndarray, Tp: float, h: np.ndarray | None) -> None:
    print(f"fs≈{fs:.2f} Hz | N={len(t)} | duracion≈{t[-1]:.2f}s")
    print(f"std(az)={np.std(az):.4f} | std(az_dyn)={np.std(az_dyn):.4f} | std(az_bp)={np.std(az_bp):.4f}")
    if h is None:
        print(f"Tp≈{Tp:.2f}s | h=None")
    else:
        hrms = float(np.std(h))
        hs = 4.0 * hrms
        print(f"Tp≈{Tp:.2f}s | std(h)≈{hrms:.4f} m | Hs≈{hs:.4f} m")


def height_scale(unit: str) -> float:
    if unit == "cm":
        return 100.0
    if unit == "mm":
        return 1000.0
    return 1.0


def split_in_chunks(t: np.ndarray, x: np.ndarray, chunk_seconds: float) -> list[tuple[np.ndarray, np.ndarray]]:
    if chunk_seconds <= 0:
        return [(t, x)]
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    t0 = float(t[0])
    t_end = float(t[-1])
    start = t0
    while start < t_end + 1e-9:
        end = start + chunk_seconds
        m = (t >= start) & (t < end)
        if np.sum(m) >= 20:
            tc = t[m] - t[m][0]
            xc = x[m]
            chunks.append((tc, xc))
        start = end
    return chunks


def process_one(args: argparse.Namespace, t: np.ndarray, az: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray | None, np.ndarray]:
    fs = fs_est(t)
    g_est, az_dyn, az_bp, Tp, h_dyn, h_bp = compute_pipeline(
        t=t,
        az=az,
        fs=fs,
        g_fc=args.g_fc,
        fmin=args.fmin,
        fmax=args.fmax,
        fft_min_samples=args.fft_min_samples,
    )
    h = h_dyn if args.height_source == "dyn" else h_bp
    h = apply_height_deadband(h, args.height_min_change)
    h, state = apply_sustained_height_confirmation(h, args.height_step_min, args.confirm_samples)
    return fs, g_est, az_dyn, az_bp, Tp, h, state


def main() -> None:
    args = parse_args()
    if args.fmax <= args.fmin:
        raise ValueError("fmax debe ser mayor que fmin")

    t, az = load_session(args.file)
    chunks = split_in_chunks(t, az, args.chunk_seconds)
    if not chunks:
        raise ValueError("No hay suficientes datos para los bloques solicitados.")

    fs, g_est, az_dyn, az_bp, Tp, h, state = process_one(args, chunks[0][0], chunks[0][1])

    print(f"FILE={args.file}")
    print(f"height_min_change={args.height_min_change:.4f} m")
    print(f"height_step_min={args.height_step_min:.6f} m | confirm_samples={args.confirm_samples}")
    print(f"height_unit={args.height_unit}")
    if args.chunk_seconds > 0:
        print(f"chunk_seconds={args.chunk_seconds:.2f} | bloques={len(chunks)}")

    for i, (tc, azc) in enumerate(chunks, start=1):
        fs_i, _, az_dyn_i, az_bp_i, tp_i, h_i, st_i = process_one(args, tc, azc)
        print(f"--- Bloque {i} ({tc[0]:.2f}s..{tc[-1]:.2f}s relativo) ---")
        print_summary(fs_i, tc, azc, az_dyn_i, az_bp_i, tp_i, h_i)
        if st_i.size:
            print(f"muestras confirmadas -> subida={int(np.sum(st_i > 0))} bajada={int(np.sum(st_i < 0))}")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib no esta instalado. Instala python3-matplotlib o usa process_imu_session.py --no-plot")
        return

    hscale = height_scale(args.height_unit)
    hylabel = args.height_unit

    if args.chunk_seconds > 0 and args.mode == "step4":
        n = len(chunks)
        ncols = 2 if n > 3 else 1
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])
        axs = axs.flatten()

        for i, (tc, azc) in enumerate(chunks):
            fs_i, _, _, _, tp_i, h_i, _ = process_one(args, tc, azc)
            ax = axs[i]
            if h_i is None:
                ax.plot(tc, np.zeros_like(tc), label="h=None")
                ax.set_title(f"Bloque {i+1} | insuficiente")
            else:
                hrms_i = float(np.std(h_i))
                hs_i = 4.0 * hrms_i
                h_plot = h_i * hscale
                ax.plot(tc, h_plot, label=f"h ({args.height_source})")
                ax.set_title(f"Bloque {i+1} | Tp={tp_i:.2f}s | Hs={hs_i:.4f}m")
                if args.height_ylim is not None:
                    ax.set_ylim(args.height_ylim[0], args.height_ylim[1])
            ax.set_xlabel("Tiempo bloque (s)")
            ax.set_ylabel(hylabel)
            ax.grid(True)
            ax.legend(loc="upper right")

        for j in range(len(chunks), len(axs)):
            axs[j].axis("off")
        fig.suptitle(f"Step4 por bloques de {args.chunk_seconds:.1f}s", fontsize=13)
        fig.tight_layout()

    elif args.mode == "all":
        fig, axs = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
        fig.suptitle("Procesado IMU paso a paso", fontsize=14)

        ax = axs[0, 0]
        ax.plot(t, az, label="az cruda", alpha=0.9)
        ax.set_title("Step 1: Senal cruda")
        ax.set_ylabel("m/s2")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax = axs[0, 1]
        ax.plot(t, az, label="az cruda", alpha=0.35)
        ax.plot(t, g_est, label=f"g_est LP ({args.g_fc:.2f} Hz)", lw=2)
        ax.plot(t, az_dyn, label="az dinamica", alpha=0.9)
        ax.set_title("Step 2: Remocion de gravedad")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax = axs[1, 0]
        ax.plot(t, az_dyn, label="az dinamica", alpha=0.7)
        ax.plot(t, az_bp, label=f"az band-pass [{args.fmin:.2f},{args.fmax:.2f}] Hz", lw=2)
        ax.set_title(f"Step 3: Filtrado en banda | Tp≈{Tp:.2f}s")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("m/s2")
        ax.grid(True)
        ax.legend(loc="upper right")

        ax = axs[1, 1]
        if h is None:
            ax.plot(t, np.zeros_like(t), label="h=None")
            ax.set_title("Step 4: Altura (insuficiente)")
        else:
            hrms = float(np.std(h))
            hs = 4.0 * hrms
            h_plot = h * hscale
            ax.plot(t, h_plot, label=f"h desde az_{args.height_source}")
            ax.set_title(
                f"Step 4: Altura | std(h)={hrms:.4f} m | Hs≈{hs:.4f} m | "
                f"db={args.height_min_change:.3f}m | step={args.height_step_min:.4f} | N={args.confirm_samples}"
            )
            if args.height_ylim is not None:
                ax.set_ylim(args.height_ylim[0], args.height_ylim[1])
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel(hylabel)
        ax.grid(True)
        ax.legend(loc="upper right")

        fig.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=(12, 5))

        if args.mode == "step1":
            ax.plot(t, az, label="az cruda")
            ax.set_title("Step 1: Senal cruda")
            ax.set_ylabel("m/s2")

        elif args.mode == "step2":
            ax.plot(t, az, label="az cruda", alpha=0.35)
            ax.plot(t, g_est, label=f"g_est LP ({args.g_fc:.2f} Hz)", lw=2)
            ax.plot(t, az_dyn, label="az dinamica")
            ax.set_title("Step 2: Remocion de gravedad")
            ax.set_ylabel("m/s2")

        elif args.mode == "step3":
            ax.plot(t, az_dyn, label="az dinamica", alpha=0.7)
            ax.plot(t, az_bp, label=f"az band-pass [{args.fmin:.2f},{args.fmax:.2f}] Hz", lw=2)
            ax.set_title(f"Step 3: Filtrado en banda | Tp≈{Tp:.2f}s")
            ax.set_ylabel("m/s2")

        elif args.mode == "step4":
            if h is None:
                ax.plot(t, np.zeros_like(t), label="h=None")
                ax.set_title("Step 4: Altura (insuficiente)")
            else:
                hrms = float(np.std(h))
                hs = 4.0 * hrms
                h_plot = h * hscale
                ax.plot(t, h_plot, label=f"h desde az_{args.height_source}")
                ax.set_title(
                    f"Step 4: Altura | std(h)={hrms:.4f} m | Hs≈{hs:.4f} m | "
                    f"db={args.height_min_change:.3f}m | step={args.height_step_min:.4f} | N={args.confirm_samples}"
                )
                if args.height_ylim is not None:
                    ax.set_ylim(args.height_ylim[0], args.height_ylim[1])
            ax.set_ylabel(hylabel)

        ax.set_xlabel("Tiempo (s)")
        ax.grid(True)
        ax.legend(loc="upper right")
        fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=160)
        print(f"[OK] Figura guardada en {args.save}")
    else:
        plt.show(block=True)


if __name__ == "__main__":
    main()
