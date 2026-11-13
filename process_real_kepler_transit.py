#!/usr/bin/env python3
"""Download and analyze real Kepler photometry with a BLS transit search.

Install dependencies with:
    python -m pip install lightkurve astropy matplotlib

Example:
    python process_real_kepler_transit.py --target Kepler-8 --min-period 1 --max-period 10

The default target is Kepler-8 (KIC 6922244; Kepler-8b). Data are fetched
from the Mikulski Archive for Space Telescopes (MAST) by Lightkurve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import astropy.units as u
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from astropy.timeseries import BoxLeastSquares


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download real Kepler light curves and search for transits with BLS."
    )
    parser.add_argument("--target", default="Kepler-8", help="MAST target name or KIC (default: Kepler-8)")
    parser.add_argument("--min-period", type=float, default=0.5, help="Minimum BLS period in days")
    parser.add_argument("--max-period", type=float, default=20.0, help="Maximum BLS period in days")
    parser.add_argument("--period-samples", type=int, default=20000, help="Number of trial periods")
    parser.add_argument("--window-length", type=int, default=401, help="Odd Savitzky-Golay flattening window in cadences")
    parser.add_argument("--output", type=Path, default=None, help="Figure path (default: beside this script)")
    parser.add_argument("--show", action="store_true", help="Display the figure interactively")
    args = parser.parse_args()
    if args.min_period <= 0 or args.max_period <= args.min_period:
        parser.error("periods must satisfy 0 < --min-period < --max-period")
    if args.period_samples < 100:
        parser.error("--period-samples must be at least 100")
    if args.window_length < 5:
        parser.error("--window-length must be at least 5")
    if args.window_length % 2 == 0:
        args.window_length += 1
    return args


def download_kepler_light_curve(target: str) -> lk.LightCurve:
    """Fetch long-cadence PDCSAP quarters and return a sorted, normalized curve."""
    print(f"Searching MAST for Kepler long-cadence data: {target}")
    search = lk.search_lightcurve(target, mission="Kepler", cadence="long")
    if len(search) == 0:
        raise RuntimeError(f"No Kepler long-cadence light curves found for {target!r}.")
    # Prefer the pipeline-corrected flux column. Lightkurve will report a useful
    # error if the archive products do not contain it.
    collection = search.download_all(quality_bitmask="default", flux_column="pdcsap_flux")
    if collection is None or len(collection) == 0:
        raise RuntimeError(f"MAST returned no downloadable Kepler products for {target!r}.")

    curves = []
    for product in collection:
        curve = product.remove_nans().remove_outliers(sigma=5)
        if len(curve) < 10:
            continue
        curve = curve.normalize()
        curves.append(curve)
    if not curves:
        raise RuntimeError("Downloaded products contained no usable finite observations.")

    # Merge on absolute JD, sort chronologically, and remove any duplicate timestamps.
    jd = np.concatenate([np.asarray(curve.time.jd, dtype=float) for curve in curves])
    flux = np.concatenate([np.asarray(curve.flux.value, dtype=float) for curve in curves])
    order = np.argsort(jd)
    jd, flux = jd[order], flux[order]
    unique = np.concatenate(([True], np.diff(jd) > 1e-9))
    merged = lk.LightCurve(time=Time(jd[unique], format="jd"), flux=flux[unique])
    print(f"Downloaded {len(curves)} product(s); retained {len(merged)} quality-vetted observations.")
    return merged


def flatten_curve(raw: lk.LightCurve, window_length: int) -> tuple[lk.LightCurve, lk.LightCurve]:
    """Remove slow stellar/instrumental variability while retaining transit signals."""
    # A window longer than the data is invalid; use the largest usable odd window.
    effective_window = min(window_length, len(raw) if len(raw) % 2 else len(raw) - 1)
    if effective_window < 5:
        raise RuntimeError("Not enough observations for Savitzky-Golay detrending.")
    flat, trend = raw.flatten(
        window_length=effective_window,
        polyorder=2,
        return_trend=True,
        break_tolerance=5,
        sigma=5,
        niters=3,
    )
    flat = flat.remove_nans().remove_outliers(sigma=5)
    if len(flat) < 100:
        raise RuntimeError("Too few observations remain after detrending to run a reliable BLS search.")
    return flat, trend


def binned_median(phase: np.ndarray, flux: np.ndarray, bins: int = 120) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-0.5, 0.5, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    medians = np.full(bins, np.nan)
    index = np.digitize(phase, edges) - 1
    for i in range(bins):
        values = flux[index == i]
        if values.size:
            medians[i] = np.nanmedian(values)
    keep = np.isfinite(medians)
    return centers[keep], medians[keep]


def main() -> int:
    args = parse_args()
    output = args.output if args.output is not None else SCRIPT_DIR / "kepler_transit_analysis.png"
    if not output.is_absolute():
        output = SCRIPT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        raw = download_kepler_light_curve(args.target)
        flat, trend = flatten_curve(raw, args.window_length)
    except Exception as exc:
        print(f"Data download/preprocessing failed: {exc}", file=sys.stderr)
        print("Check your internet connection, target name, and MAST availability.", file=sys.stderr)
        return 1

    t = np.asarray(flat.time.jd, dtype=float)
    y = np.asarray(flat.flux.value, dtype=float)
    finite = np.isfinite(t) & np.isfinite(y)
    t, y = t[finite], y[finite]
    if len(t) < 100:
        print("Insufficient valid detrended data for BLS.", file=sys.stderr)
        return 1

    # A logarithmic trial-period grid resolves short and long periods efficiently.
    periods = np.geomspace(args.min_period, args.max_period, args.period_samples) * u.day
    durations = np.linspace(0.5, 8.0, 16) * u.hour
    durations = durations[durations.to_value(u.day) < args.min_period]
    if len(durations) == 0:
        durations = np.array([0.1 * args.min_period]) * u.day
    bls = BoxLeastSquares(t * u.day, y)
    try:
        result = bls.power(periods, durations)
    except Exception as exc:
        print(f"BLS search failed: {exc}", file=sys.stderr)
        return 1
    peak = int(np.nanargmax(result.power))
    best_period = result.period[peak].to_value(u.day)
    best_duration = result.duration[peak]
    best_t0 = result.transit_time[peak].to_value(u.day)
    best_power = float(result.power[peak])
    depth = float(result.depth[peak])
    print(f"BLS peak: period={best_period:.8f} d, duration={best_duration.to_value(u.hour):.2f} h, "
          f"depth={depth:.6g}, power={best_power:.6g}")

    # Fold about the detected transit epoch, centered on phase zero.
    phase = ((t - best_t0 + best_period / 2) % best_period - best_period / 2) / best_period
    bx, by = binned_median(phase, y)
    # Evaluate Astropy's fitted periodic box model over one complete phase.
    model_phase = np.linspace(-0.5, 0.5, 1500)
    reference_epoch = best_t0 + np.round((np.median(t) - best_t0) / best_period) * best_period
    model_times = reference_epoch + model_phase * best_period
    model_flux = bls.model(model_times * u.day, best_period * u.day,
                           best_duration, best_t0 * u.day)

    raw_t = np.asarray(raw.time.jd, dtype=float)
    raw_y = np.asarray(raw.flux.value, dtype=float)
    trend_t = np.asarray(trend.time.jd, dtype=float)
    trend_y = np.asarray(trend.flux.value, dtype=float)
    time_origin = float(np.nanmin(raw_t))

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    ax = axes[0]
    ax.plot(raw_t - time_origin, raw_y, ".", ms=1.8, alpha=0.35, color="0.45", label="Raw PDCSAP (normalized)")
    ax.plot(trend_t - time_origin, trend_y, color="tab:orange", lw=1.2, alpha=0.9, label="Estimated variability trend")
    ax.plot(t - time_origin, y, ".", ms=1.8, alpha=0.65, color="tab:blue", label="Detrended flux")
    ax.set(title=f"{args.target}: full Kepler light curve", xlabel="Time (days from first observation)", ylabel="Relative flux")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(result.period.to_value(u.day), result.power, color="tab:purple", lw=1)
    ax.axvline(best_period, color="tab:red", ls="--", label=f"Peak = {best_period:.5f} d")
    ax.set(title="BLS periodogram (real Kepler observations)", xlabel="Trial period (days)", ylabel="BLS power")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(phase, y, ".", ms=2, alpha=0.22, color="0.35", label="Individual observations")
    ax.plot(bx, by, "o", ms=4, color="tab:blue", label="Binned median")
    ax.plot(model_phase, model_flux, color="tab:red", lw=2, label="Best-fit BLS box model")
    ax.set_xlim(-0.15, 0.15)
    ax.set(title=f"Phase-folded transit: P={best_period:.6f} d, depth={depth:.4g}",
           xlabel="Orbital phase (transit at 0)", ylabel="Relative flux")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(f"Saved figure: {output}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
