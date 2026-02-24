#!/usr/bin/env python3
"""Autoajuste de parametros IMU por busqueda iterativa.

Prueba muchas combinaciones de:
- g_fc, fmin, fmax
- height_min_change, height_step_min, confirm_samples

Evalua por bloques temporales y optimiza una funcion de calidad.
Guarda el mejor resultado para reutilizarlo como punto de partida.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass
class Params:
    g_fc: float
    fmin: float
    fmax: float
    height_min_change: float
    height_step_min: float
    confirm_samples: int


@dataclass
class EvalMetrics:
    score: float
    coverage: float
    mean_hs: float
    std_hs: float
    mean_tp: float
    std_tp: float
    mean_confirm_ratio: float
    valid_chunks: int
    total_chunks: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-tuning de parametros para procesado IMU")
    p.add_argument("file", help="CSV o NPZ de sesion")
    p.add_argument("--chunk-seconds", type=float, default=5.0, help="Tamano de bloque para evaluacion")
    p.add_argument("--fft-min-samples", type=int, default=128, help="Minimo de muestras por bloque")
    p.add_argument("--trials", type=int, default=180, help="Combinaciones por ronda")
    p.add_argument("--rounds", type=int, default=4, help="Rondas de refinamiento")
    p.add_argument("--topk", type=int, default=8, help="Top resultados a mostrar")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    p.add_argument("--target-hs", type=float, default=-1.0, help="Si >0, intenta acercarse a este Hs (m)")
    p.add_argument(
        "--state-file",
        default="auto_tune_state.json",
        help="Archivo para persistir el mejor set entre ejecuciones",
    )
    return p.parse_args()


def split_chunks(t: np.ndarray, x: np.ndarray, chunk_seconds: float, fft_min_samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    chunks: list[tuple[np.ndarray, np.ndarray]] = []
    t0 = float(t[0])
    t_end = float(t[-1])
    start = t0
    while start < t_end + 1e-9:
        end = start + chunk_seconds
        m = (t >= start) & (t < end)
        if np.sum(m) >= max(20, fft_min_samples):
            tc = t[m] - t[m][0]
            xc = x[m]
            chunks.append((tc, xc))
        start = end
    return chunks


def eval_one_chunk(tc: np.ndarray, azc: np.ndarray, p: Params, fft_min_samples: int) -> tuple[float, float, float] | None:
    fs = fs_est(tc)
    if fs <= 0:
        return None

    g_est = lowpass_1pole_offline(azc, fs, p.g_fc)
    az_dyn = azc - g_est
    az_bp = bandpass_1pole_offline(az_dyn, fs, p.fmin, p.fmax)

    tp = dominant_period_fft(
        az_bp,
        fs,
        fmin=max(0.03, p.fmin),
        fmax=min(1.5, p.fmax),
        min_samples=fft_min_samples,
    )

    h = height_from_accel_fft(
        az_dyn,
        fs,
        fmin=p.fmin,
        fmax=p.fmax,
        min_samples=fft_min_samples,
    )
    if h is None:
        return None

    h = apply_height_deadband(h, p.height_min_change)
    h, state = apply_sustained_height_confirmation(h, p.height_step_min, p.confirm_samples)
    if h is None or len(h) == 0:
        return None

    hs = 4.0 * float(np.std(h))
    confirm_ratio = float(np.sum(state != 0)) / max(1, len(state))
    tp = float(tp) if np.isfinite(tp) else np.nan
    return hs, tp, confirm_ratio


def score_params(
    chunks: list[tuple[np.ndarray, np.ndarray]],
    p: Params,
    fft_min_samples: int,
    target_hs: float,
) -> EvalMetrics:
    hs_list: list[float] = []
    tp_list: list[float] = []
    cr_list: list[float] = []

    for tc, azc in chunks:
        out = eval_one_chunk(tc, azc, p, fft_min_samples)
        if out is None:
            continue
        hs, tp, cr = out
        hs_list.append(hs)
        if np.isfinite(tp):
            tp_list.append(tp)
        cr_list.append(cr)

    valid = len(hs_list)
    total = len(chunks)
    if valid == 0:
        return EvalMetrics(
            score=-1e9,
            coverage=0.0,
            mean_hs=np.nan,
            std_hs=np.nan,
            mean_tp=np.nan,
            std_tp=np.nan,
            mean_confirm_ratio=0.0,
            valid_chunks=0,
            total_chunks=total,
        )

    mean_hs = float(np.mean(hs_list))
    std_hs = float(np.std(hs_list))
    mean_tp = float(np.mean(tp_list)) if tp_list else np.nan
    std_tp = float(np.std(tp_list)) if tp_list else np.nan
    mean_cr = float(np.mean(cr_list)) if cr_list else 0.0
    coverage = float(valid) / max(1, total)

    # Score base sin supervision.
    score = 2.2 * coverage + 1.5 * mean_cr + 0.9 * math.tanh(mean_hs / 0.05) - 2.0 * std_hs
    if np.isfinite(std_tp):
        score -= 0.15 * std_tp

    # Penaliza valores de Hs no realistas para esta fase prototipo.
    if mean_hs > 1.5:
        score -= (mean_hs - 1.5) * 1.5

    # Si hay objetivo de Hs, fuerza ajuste a objetivo.
    if target_hs > 0:
        score -= 2.0 * abs(mean_hs - target_hs)

    return EvalMetrics(
        score=score,
        coverage=coverage,
        mean_hs=mean_hs,
        std_hs=std_hs,
        mean_tp=mean_tp,
        std_tp=std_tp,
        mean_confirm_ratio=mean_cr,
        valid_chunks=valid,
        total_chunks=total,
    )


def sample_params(rng: np.random.Generator, center: Params | None, scale: float) -> Params:
    bounds = {
        "g_fc": (0.01, 0.20),
        "fmin": (0.005, 0.30),
        "fmax": (0.30, 3.00),
        "height_min_change": (0.0, 0.010),
        "height_step_min": (0.0, 0.000050),
        "confirm_samples": (1, 30),
    }

    def rfloat(k: str) -> float:
        lo, hi = bounds[k]
        if center is None:
            return float(rng.uniform(lo, hi))
        c = float(getattr(center, k))
        w = (hi - lo) * scale
        l2, h2 = max(lo, c - w / 2.0), min(hi, c + w / 2.0)
        if h2 <= l2:
            l2, h2 = lo, hi
        return float(rng.uniform(l2, h2))

    def rint(k: str) -> int:
        lo, hi = bounds[k]
        if center is None:
            return int(rng.integers(lo, hi + 1))
        c = int(getattr(center, k))
        w = max(1, int(round((hi - lo) * scale / 2.0)))
        l2, h2 = max(lo, c - w), min(hi, c + w)
        if h2 < l2:
            l2, h2 = lo, hi
        return int(rng.integers(l2, h2 + 1))

    p = Params(
        g_fc=rfloat("g_fc"),
        fmin=rfloat("fmin"),
        fmax=rfloat("fmax"),
        height_min_change=rfloat("height_min_change"),
        height_step_min=rfloat("height_step_min"),
        confirm_samples=rint("confirm_samples"),
    )
    if p.fmax <= p.fmin + 0.02:
        p.fmax = min(3.0, p.fmin + 0.02)
    return p


def load_center(path: Path) -> Params | None:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return Params(
            g_fc=float(d["g_fc"]),
            fmin=float(d["fmin"]),
            fmax=float(d["fmax"]),
            height_min_change=float(d["height_min_change"]),
            height_step_min=float(d["height_step_min"]),
            confirm_samples=int(d["confirm_samples"]),
        )
    except Exception:
        return None


def save_best(path: Path, p: Params, m: EvalMetrics, meta: dict[str, Any]) -> None:
    payload = {
        "g_fc": p.g_fc,
        "fmin": p.fmin,
        "fmax": p.fmax,
        "height_min_change": p.height_min_change,
        "height_step_min": p.height_step_min,
        "confirm_samples": p.confirm_samples,
        "score": m.score,
        "mean_hs": m.mean_hs,
        "std_hs": m.std_hs,
        "mean_tp": m.mean_tp,
        "std_tp": m.std_tp,
        "coverage": m.coverage,
        "mean_confirm_ratio": m.mean_confirm_ratio,
        **meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    t, az = load_session(args.file)
    chunks = split_chunks(t, az, args.chunk_seconds, args.fft_min_samples)
    if not chunks:
        raise ValueError("No hay bloques suficientes para tuning. Baja --fft-min-samples o sube --chunk-seconds.")

    state_path = Path(args.state_file)
    center = load_center(state_path)

    results: list[tuple[EvalMetrics, Params]] = []
    best_m: EvalMetrics | None = None
    best_p: Params | None = None

    for r in range(args.rounds):
        scale = 1.0 if r == 0 else max(0.10, 0.55 ** r)
        round_center = best_p if best_p is not None else center

        for _ in range(args.trials):
            p = sample_params(rng, round_center, scale)
            m = score_params(chunks, p, args.fft_min_samples, args.target_hs)
            results.append((m, p))
            if best_m is None or m.score > best_m.score:
                best_m = m
                best_p = p

        assert best_m is not None and best_p is not None
        print(
            f"[Ronda {r+1}/{args.rounds}] best_score={best_m.score:.4f} "
            f"Hs={best_m.mean_hs:.4f}±{best_m.std_hs:.4f} "
            f"Tp={best_m.mean_tp:.2f} coverage={best_m.coverage:.2f}"
        )

    results.sort(key=lambda x: x[0].score, reverse=True)

    print("\n=== TOP RESULTADOS ===")
    for i, (m, p) in enumerate(results[: args.topk], start=1):
        print(
            f"#{i} score={m.score:.4f} Hs={m.mean_hs:.4f}±{m.std_hs:.4f} "
            f"Tp={m.mean_tp:.2f} cr={m.mean_confirm_ratio:.3f} cov={m.coverage:.2f} | "
            f"g_fc={p.g_fc:.4f} fmin={p.fmin:.4f} fmax={p.fmax:.4f} "
            f"hmin={p.height_min_change:.6f} hstep={p.height_step_min:.8f} N={p.confirm_samples}"
        )

    assert best_m is not None and best_p is not None
    save_best(
        state_path,
        best_p,
        best_m,
        meta={
            "file": args.file,
            "chunk_seconds": args.chunk_seconds,
            "fft_min_samples": args.fft_min_samples,
            "trials": args.trials,
            "rounds": args.rounds,
            "target_hs": args.target_hs,
        },
    )

    print(f"\n[OK] Mejor set guardado en: {state_path}")
    print("\nComando recomendado:")
    print(
        "python3 process_imu_step_by_step.py "
        f"{args.file} --mode step4 "
        f"--g-fc {best_p.g_fc:.4f} --fmin {best_p.fmin:.4f} --fmax {best_p.fmax:.4f} "
        f"--height-min-change {best_p.height_min_change:.6f} "
        f"--height-step-min {best_p.height_step_min:.8f} --confirm-samples {best_p.confirm_samples}"
    )


if __name__ == "__main__":
    main()
