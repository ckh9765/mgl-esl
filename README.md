# mgl-esl — Reproducibility Package

Reproducibility package for the IEEE Embedded Systems Letters submission:

> **A Trace-Driven Comparison of Binary, Smoothing, and Ternary Filtering
> for Memory-Side Data Governance in Edge Perception**
> Kuohuan Chao and Jin-Sheng Chen, National Taipei University of
> Technology (NTUT), Taipei, Taiwan.

The letter compares binary thresholding, temporal smoothing (boxcar and
EWMA), ternary classification with selective temporal verification, and a
hysteresis-with-timeout baseline at the memory-side filtering boundary,
under a single cost model, on surveillance-video traces (VIRAT, MEVA).

## Paper promises → repo paths

| # | Statement in the paper | Where to find it |
|---|---|---|
| 1 | "Simulation code, derived data, and analysis scripts" (Sec. IV-E) | `simulator/`, `results/`, `figures/`, `robustness/`, plus the analysis scripts in the repo root (`run_all.py`, `final_robustness_suite.py`, `sensitivity_defer_disable.py`) |
| 2 | "Per-trace results, including ground-truth densities" (Sec. IV-A) | `results/per_trace_results.md` |
| 3 | "Per-event inspection ... confirms real pedestrian activity" — MEVA G419 (Sec. IV-A) | `e12_inspection/` (montage, per-event crops, sampled-event list) |
| 4 | "A two-population Gaussian model ... script in the package" (Sec. IV-C) | `analytic_window_model.py` |
| 5 | "RTL for all three stateful mechanisms" (Sec. II-B) | `rtl/e9_verilog_mechanisms.v` |

## Repository layout

- `simulator/` — trace-driven simulator and salience preprocessing
- `results/` — derived salience arrays (`.npy`), sweep outputs (`.csv`,
  `.json`), and per-trace result tables
- `figures/` — plotting scripts for Fig. 1 and Fig. 2 of the letter
- `robustness/` — robustness and sensitivity checks (deferral ratio,
  alignment points, AR(1) noise, ground-truth threshold, metadata ratio)
- `calibration/` — pixel-domain and H.264 compression noise calibration
- `e12_inspection/` — per-event authenticity inspection for MEVA G419
- `rtl/` — synthesizable Verilog for the ternary, smoothing, and
  hysteresis-with-timeout per-tile state machines

## Reproducing the figures

Requirements: Python 3 with `numpy`, `scipy`, and `matplotlib`.
The derived salience arrays in `results/` are sufficient to regenerate
the paper's figures via the scripts in `figures/`; the analytic
internal-consistency check runs standalone via
`python analytic_window_model.py`.

## Data provenance and attribution

Raw videos are **not** redistributed in this repository.

- **VIRAT**: obtain from the official VIRAT Video Dataset release
  (scene S_0502 is used as the primary trace).
- **MEVA**: obtain from <https://mevadata.org> (scenes G436 and G419).
  The MEVA dataset is released under **CC BY 4.0**; credit: MEVA dataset,
  Kitware Inc. / IARPA DIVA program. The small frame crops in
  `e12_inspection/` are derived from MEVA under this license.

Salience arrays in `results/` were produced from the source videos with
`simulator/preprocess_salience.py`.

## Contact

Kuohuan Chao — t112669014@ntut.edu.tw
