# Internal review — "Correct or Round? Granulometries by Digital Disks" (DGMM 2027)

## Summary
The paper shows that standard digital disks (scikit-image, OpenCV) violate Matheron's absorption axiom,
proves that granulometric Minkowski chains with bounded mean-radius steps have error growing linearly
with the radius (rho/delta^2 in general, rho^(1/3) for fitted-radius steps, kappa*rho for a fixed library),
gives an explicit periodic-line chain matching the rate up to a constant, extends the lower bound to Z^n,
and illustrates with practical families and rotation experiments.

## Strengths
- Clear, relevant question at the core of DGMM (digital geometry x granulometries).
- The main lower bound is simple, correct and general (any increments, any centres).
- A matching explicit upper bound: Theta(rho/delta^2), Theta(sqrt rho).
- Concrete practical finding (axiom violations in standard libraries, non-monotone size distributions).
- Honest experiments, including negative results on real images.

## Major issues
1. **Wrong tightness claim for octagons.** The text says the octagon chain has error 38.05 "at rho = 1000",
   matching kappa/(1+2kappa)*rho = 38.06. The fitted radius of that element is 961.5, not 1000; the match is a
   coincidence of using the target radius and the weaker constant. (Verified: eps/rho = 0.039571.)
2. **Theorem 1 is weaker than necessary.** It assumes every element is eps-round and bounds the maximum error.
   Only the first and last elements enter Steps 3–4; with D bounded independently (mean-radius steps),
   the bound holds pointwise for every element, with the better constant rho c/(2+c) ~ rho/(16 D^2).
   The abstract's "error at radius rho" is only justified after this strengthening. With the pointwise
   version the fixed-library slope becomes kappa/(1+kappa), which octagons attain exactly (0.039571 vs 0.039570).
3. **Theory and experiments use different step notions.** Theory: mean-radius steps; Table 3/Fig 1a: fitted-radius
   steps. Report mean-radius steps (1.4–2.2) and the corresponding Corollary 1 bounds.
4. **Corollary 3, kappa > 0:** "by compactness" is not justified (the feasible set is unbounded); argue on the
   simplex with the ratio A/min. Also note that kappa computed on a direction grid is a lower approximation,
   hence the stated bounds remain valid.
5. **Theorem 3 (upper bound): ties.** Several counts can increase at the same t; the chain must break ties, and
   the bound must be shown for the intermediate elements (it holds with phi in [0,1]).
6. **Missing related work:** chamfer norms and balls (Borgefors 1986; Thiel), which are the classic polygonal
   disks of digital geometry; Normand (DGCI 2003), who builds convex structuring elements by unions of
   translates rather than Minkowski sums — exactly the class excluded by the theorem (open question 2).

## Minor issues
7. Z^n theorem: a sketch with unspecified constants; its exponent (2n) is not sharp in 2D (gives 4 vs 2). Say so.
8. Theorem order is confusing (Z^n theorem numbered before the upper bound); move Z^n after the upper bound.
9. "about 60%" -> 60–63%.
10. Table 2 columns are cramped.
11. Holes in the Theorem 3 construction: state that the bounds concern hulls and the granulometric property is
    unaffected (done), and quantify.
12. Upper/lower constant ratio must be updated after (2): factor 4 asymptotically.

## Recommendation (before fixes)
Weak accept: correct core, clear contribution; issues 1–3 must be fixed (1 is a factual error).

## Resolution (all issues addressed)
1. Octagon tightness: corrected — fitted radius 961.5, eps/rho = 0.039571 vs kappa/(1+kappa) = 0.039570.
2. Theorem 1 now needs only the first and last elements; bounds are pointwise for every element.
   Constants: rho c/(2+c) ~ rho/(16 D^2); Cor. 1 rho/(16 pi^2 delta^2); Cor. 3 kappa/(1+kappa) (tight).
   Upper/lower ratio now 4 asymptotically (8 including the constant term for rho >= 4 pi^4 delta^4).
3. Table 3 reports mean-radius steps (1.19–2.17); text states Cor. 1 forces only 0.06–0.14 at rho = 48.
4. Cor. 3 proof: argument on the simplex with f = A/min; grid-computed kappa noted as a lower approximation.
5. Theorem (upper bound): ties broken one line at a time; intermediate elements covered (phi in [0,1]);
   fixed a slip (rho >= t - 2 sqrt2 m^2) — final bound unchanged.
6. Related work: chamfer norms (Borgefors 1986, Thiel 2001), Normand (DGCI 2003) added.
7. Z^n theorem moved after the upper bound, stated as a sketch with non-sharp exponent.
8–11. 60–63% wording, table spacing, holes statement kept.
Recommendation after fixes: accept.
