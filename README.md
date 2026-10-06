# DGMM 2027 submission: "Disk Granulometries on the Digital Grid: Why a Single Structuring Element Is Not Enough"

Deadlines: abstract **Oct 16, 2026**, paper **Oct 23, 2026** (LNCS, max 12 pages, EasyChair).

- `paper_granulo/` — **the submission** (`main.tex`, `refs.bib`, `figures/`; build with `latexmk -pdf main.tex`).
- `granulo/` — all code and data for it (see table below).

## Main results
1. Diagnosis: opening by each library disk (scikit-image, OpenCV) is NOT a granulometry (73% / 81% of radius pairs
   violate absorption); size distributions increase with r on 253/400 (241/400) random images.
2. Remedy (Prop. 1): the sup-closure gamma'_r = sup_{s>=r} opening by disk(s) is a granulometry, as round as the disks
   and equally cheap (0/400 non-monotone; same rotation sensitivity as plain disks). Recommended in practice.
3. Single structuring element per size (Minkowski chains) cannot be round: mean-radius steps <= delta -> every element
   has error >= rho/(16 pi^2 delta^2) (asymptotically); fitted-radius steps -> rho^(1/3); fixed library ->
   slope kappa/(1+kappa), attained exactly by octagons.
4. Matching upper bound: periodic lines on the boundary of [-m,m]^2: eps <= rho/(8m^2) + 2m^2, steps <= sqrt2 m/pi
   -> Theta(rho/delta^2) (factor 4), Theta(sqrt rho) with delta ~ rho^(1/4). Z^n: eps >= c_n rho delta^(-2n).
5. Experiments: practical families (Table 3), rotation + per-object sieve size on synthetic and 3 real images (Table 4).

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
| `exp_rotation2.py` | Table 4 (top): rotation experiment incl. sup-closure, synthetic + real images |
| `exp_objects.py` | Table 4 (bottom): per-object sieve size vs inscribed radius |
| `nonmono_sup.py` | sup-closure vs single openings on the 400 random images (Table 1) |
| `fig_rotation2.py` | Fig. 3 (rotation curves, 4 families) |
| `fig_final.py` | final figures (error, shapes, rotation) |
| `upper.py`, `upper_multi.py` | Theorem 3 construction: verification of bound, steps, gaps; multiscale O(sqrt rho) |
| `fig_idea.py` | Fig. 1 (axiom violation example, direction gap, proof idea) |
| `fig_bounds.py` | Fig. 1 (practical families; upper/lower bound sandwich) |
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
