# Kepler-exoplanet-analysis
Analyzing real NASA Kepler light curves to detect planetary transits using Box Least Squares (BLS)
# Kepler Exoplanet Transit Analysis

A fully reproducible analysis of **real NASA Kepler space photometry** that recovers the orbital and transit signatures of an exoplanet from raw stellar brightness curves.

## 🧭 Scientific Motivation

When a planet passes in front of its host star, it briefly blocks a small fraction of the starlight, producing a periodic, tiny dip in the measured brightness — a *transit*. The depth of that dip encodes the planet's size relative to the star, while the time between dips reveals its orbital period. Isolating these faint, periodic signals from the noisy, slowly varying light of the star itself is the central observational problem in exoplanet discovery.

This project turns real archival light curves from NASA's Kepler mission into concrete physical measurements:

- **Transit depth** → planet-to-star radius ratio, a direct probe of the planet's size.
- **Orbital period** → derived from the periodogram, which separates the periodic planet signal from stellar variability.
- **Phase-folded light curve** → the planetary signal averaged and stacked so the transit shape becomes clearly visible.

## 🔭 Physical Interpretation

- **Why detrend?** Stars vary on their own timescales (rotation, pulsation, activity), producing brightness changes far larger than the tiny planetary dip. Modeling this stellar variability first lets the small, true transit signal stand out.
- **Why a periodogram?** Detecting the *periodic* nature of the dips is what distinguishes a bona fide transiting planet from unrelated brightness fluctuations (e.g., a flare, a spotted surface, or an eclipsing star).
- **What we learn:** From the period and depth we connect the raw photometry to the planet's orbital configuration and approximate size — turning pixels into astrophysics.

## ⚙️ Pipeline (concise)

The analysis proceeds in clean, inspectable stages:

1. **Ingest** – Pull NASA Kepler long-cadence photometry (`pdcsap_flux`) from the MAST archive.
2. **Clean** – Remove statistical outliers to avoid spurious signals.
3. **Detrend** – Suppress stellar variability with a Savitzky–Golay filter so the transit is not buried in stellar noise.
4. **Detect** – Run a **Box Least Squares (BLS)** periodogram to find the recurring transit period.
5. **Measure & Fold** – Derive the transit depth and phase-fold the light curve to visualize the recovered transit.

## 🚀 Usage
```bash
python3 process_real_kepler_transit.py
# Optionally re-target a specific star, e.g. Kepler-10:
python3 process_real_kepler_transit.py --target "Kepler-10"
