# DGMM 2027 submission: "How Round Can a Digital Granulometry Be?"

Deadlines: abstract **Oct 16, 2026**, paper **Oct 23, 2026** (LNCS, max 12 pages, EasyChair).

- `paper_granulo/` — **the submission** (`main.tex`, `refs.bib`, `figures/`; build with `latexmk -pdf main.tex`).
- `granulo/` — all code and data for it (see table below).

## Main results
1. Standard digital disks are not granulometric: scikit-image `disk` violates "D_r is D_s-open" for 73% of
   pairs s<r<=40 (30/39 consecutive pairs), OpenCV ellipse for 81%; size distributions become non-monotone
   (253/400 random images).
2. **Theorem 1**: any Minkowski chain (octagons, periodic lines, Bresenham decompositions, ...) whose elements
   are all eps-round with radius steps <= Delta has eps >= c rho^(1/3); with a fixed library eps grows linearly
   (Corollary 2, slope kappa/(1+2kappa), tight for octagons).
3. Constructions (radius steps < 2.2, r <= 48): LP-weighted periodic-line chains and MILP chains have max error ~1.0
   vs 1.85 for octagons and 3.27 for Bresenham/DSS decompositions; up to r = 1000: 3.7 vs 38 (bound 0.47).
   Real relaxation numerically bounded (0.43 up to r = 2000), not proven.
4. Experiments: on rotated polygonal scenes the periodic chain is 3x less orientation dependent than octagons;
   on three real images (nuclei, coins, gravel) all families are within resampling noise.

## Code (`granulo/`)
| file | purpose |
|---|---|
| `geom.py` | DSS generators (D4 orbits), support functions, polygon roundness |
| `explore2.py`, `explore4.py`, `explore3.py`, `explore1.py` | hole-freeness tests of Minkowski sums |
| `axiom_stats.py`, `nonmono.py` | Table 1 (axiom violations, non-monotone size distributions) |
| `gauss_chain2.py` | granulometric subsequences of digital disks |
| `kappa.py` | Table 2 (kappa of classical libraries) |
| `joint_lp.py` | real relaxation (monotone LP over radii) |
| `optchain.py`, `optchain2.py` | MILP chains (D4-orbit increments / single-segment increments, unit steps) |
| `round_track2.py` | coarse chains for large radii (planned rounding) |
| `curves.py` | Theorem 1 bound curve, octagon chain |
| `families.py`, `exp_rotation.py` | exact morphology by cascades; rotation experiment |
| `all_families.py` | all families of Table 3 (Gauss, octagons, Bresenham, periodic 8/13 dir., MILP), cached |
| `periodic_large.py` | LP-weighted periodic chains up to r = 1000 (Fig. 1b) |
| `exp_rotation2.py` | Table 4: rotation experiment, synthetic + real images |
| `fig_final.py` | final figures (error, shapes, rotation) |
| `verify_chain.py` | hole-freeness / openness / per-radius error of a computed chain |
| `fig_main.py` | all figures (`python fig_main.py fig_error fig_rotation fig_shapes fig_violations`) |
| `NOTES_theory.md` | proof notes |

Reproduce (approx. times on a laptop): `python axiom_stats.py` (5 min), `python nonmono.py`, `python kappa.py`,
`python joint_lp.py 32 1000 30`, `python optchain2.py 6 48 1500` (~40 min), `python verify_chain.py chain2_M6_R48.npz`,
`python curves.py`, `python exp_rotation.py chain2_M6_R48.npz`, `python round_track2.py joint_M32_R1000.npz 1000 _s1 1.0`,
then `python fig_main.py fig_error fig_rotation fig_shapes`.

## Before submitting
- Fill in authors/affiliations in `paper_granulo/main.tex` (currently anonymous; DGMM's anonymity policy was not stated).
- Read the open question in Sect. 5 (hole-freeness conjecture) — a proof would strengthen the paper.
- Optional: longer MILP runs (dual bound 0.32 vs 1.00 found for r<=48) could improve Fig. 2a.
