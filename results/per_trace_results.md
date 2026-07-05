# Per-trace results (ESL letter, Sec. IV-A promise)

Protocol: 64x64 tiles; ground truth = clean salience p > 0.25; transient
event = maximal run of >=3 consecutive ground-truth frames on one tile;
sigma in [0.02, 0.30]; tau_U in [0.10, 0.55] (10 levels);
tau_B = 0.5 tau_U; N in {2, 3, 5}; n = 10 seeds; paired differences with
95% CI. Gains below are ternary-minus-binary normalized recall-budget
AUC at N = 2.

## Trace summary and ground-truth densities

| Trace | Scene | Frames | GT tile-frames | GT density | Events | Duration min/med/max | <5-frame share | =3-frame share |
|---|---|---|---|---|---|---|---|---|
| VIRAT (primary, extended) | S_0502 parking lot | 1615 | 2817 | 0.36% | 361 | 3/5/20 | 44% | 16% |
| MEVA hospital | G436, clip 2018-03-15.15-40-07 (500 fr) | 500 | 904 | 0.38% | 27 | 3/16/60 | 22% | 15% |
| MEVA school | G419, clip 2018-03-07.10-00-01 (500 fr) | 500 | -- | 1.27% | 335 | 3/5/66 | 37% | 19% |

## VIRAT S_0502 extended (primary)

Crossover sigma* ~= 0.064; peak at sigma = 0.20 (+0.030 +/- 0.002,
p = 1.1e-10). A 500-frame prefix places the peak at sigma = 0.15
(subsample variation; both MEVA scenes agree with 0.20).

| sigma | gain +/- 95% CI |
|---|---|
| 0.02 | -0.0180 +/- 0.0003 |
| 0.05 | -0.0051 +/- 0.0004 |
| 0.10 | +0.0134 +/- 0.0006 |
| 0.15 | +0.0285 +/- 0.0017 |
| 0.20 | +0.0297 +/- 0.0020 |
| 0.30 | +0.0136 +/- 0.0017 |

## MEVA hospital G436

Crossover sigma* ~= 0.088; peak at sigma = 0.20; debiased peak gain
+0.027 +/- 0.001 (CI [0.0252, 0.0280]).

| sigma | gain +/- 95% CI |
|---|---|
| 0.02 | -0.0485 +/- 0.0008 |
| 0.05 | -0.0149 +/- 0.0005 |
| 0.10 | +0.0048 +/- 0.0009 |
| 0.15 | +0.0205 +/- 0.0015 |
| 0.20 | +0.0266 +/- 0.0017 |
| 0.30 | +0.0139 +/- 0.0018 |

Three-way comparison on G436 (normalized AUC / event miss rate at
recall ~= 0.7; smoothing at per-sigma best W, EWMA at best lambda):

| sigma | Binary AUC | Ternary AUC | Smooth AUC | EWMA AUC | Tern miss | Smooth miss | EWMA miss |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.903 | 0.908 | 0.609 | 0.690 | 3.3% | 0.0% | 0.4% |
| 0.20 | 0.682 | 0.709 | 0.814 | 0.842 | 1.9% | 1.1% | 1.1% |
| 0.30 | 0.534 | 0.547 | 0.770 | 0.797 | 0.7% | 0.7% | 1.1% |

Coverage comparisons on G436 are not claimed in the paper: with 27
events, one event corresponds to 3.7 pp.

## MEVA school G419

Crossover sigma* ~= 0.109; peak at sigma = 0.20 (+0.024,
CI [0.0233, 0.0248]). Sign pattern (negative -> positive -> decay)
replicates.

Event-density note: 12x the event count of G436 reflects scene
content -- a busy indoor school corridor/stairwell vs. a sparse
parking view. Per-event inspection (e12_inspection/): 10 randomly
sampled events (seed = 0) cropped from the source video show 10/10
real pedestrian activity, 0/10 foliage, sky, or flicker; the duration
histogram decays smoothly with no spike at the 3-frame floor; the 335
events fall on 81/480 tiles (17%), with the top 10% of tiles carrying
79% of events (one contiguous walkway cluster; sky and wall regions
empty). Minor caveat: a few sampled events sit adjacent to bright
windows, so window-light variation may contribute marginally; the
dominant source is pedestrian traffic.

## Data provenance

Raw videos are not redistributed. VIRAT: official VIRAT Video Dataset
release. MEVA: https://mevadata.org (CC BY 4.0; credit Kitware Inc. /
IARPA DIVA program). Salience arrays in results/ were produced with
simulator/preprocess_salience.py.
