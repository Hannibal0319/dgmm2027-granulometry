# Theory notes: roundness of granulometric digital disks

## Setting
- Digital set X ⊂ Z², finite. conv X its hull. Width function w_X(θ) = max_{p∈X}⟨p,u_θ⟩ − min_{p∈X}⟨p,u_θ⟩
  (= width of conv X, translation invariant, π-periodic).
- Anisotropy A(X) = max_θ w_X − min_θ w_X.
- ε-round: d_H(conv X, B(c, ρ)) ≤ ε for some centre c ⇒ w_X(θ) ∈ [2ρ − 2ε, 2ρ + 2ε] ⇒ A(X) ≤ 4ε.
- Granulometry (Matheron) on digital sets: family (γ_k) of openings by structuring elements D_k with
  D_l ∘ D_k = D_l for k ≤ l (D_l is D_k-open). Then γ_{D_l} ≤ γ_{D_k} (absorption).
- Minkowski chain: D_k = D_{k−1} ⊕ C_k (all classical constructions: neighbourhood sequences / octagons,
  periodic lines (Jones–Soille), Bresenham radial decompositions (Adams), MATLAB strel('disk',r,N>0)).
  Minkowski chains are granulometries (D ⊕ C is D-open).
- Key identity: w_{A⊕B} = w_A + w_B (support functions add).

## Theorem 1 (lower bound)
Let (D_k)_{k=0..K} be a Minkowski chain such that every D_k is ε-round with radius ρ_k, and
ρ_k − ρ_{k−1} ≤ Δ (bounded size steps). Let d = 2Δ + 4ε and α = arctan(1/d). Then
    8ε ≥ 2(ρ_K − ρ_0 − 2ε)(1 − cos(α/2)).
In particular, since 1 − cos(α/2) ≥ α²/8·(1 − α²/48) and α ≥ 1/d − 1/(3d³),
    ε ≥ c (ρ_K)^{1/3} for ρ_K large (c ≈ (1/512)^{1/3} ≈ 0.125 for Δ ≪ ε).

Proof.
1. w_{C_k} = w_{D_k} − w_{D_{k−1}} ≤ (2ρ_k + 2ε) − (2ρ_{k−1} − 2ε) ≤ 2Δ + 4ε = d for all θ, so
   diam(conv C_k) = max_θ w_{C_k} ≤ d. Every edge vector of conv C_k is a nonzero lattice vector of
   Euclidean norm ≤ d.
2. W := w_{D_K} − w_{D_0} = Σ_k w_{C_k} is the width function of P := Σ_k conv C_k, i.e. W(θ) = 2 h_Q(θ)
   with Q = ½(P − P) centrally symmetric polygon. The edge directions of Q are edge directions of the
   conv C_k: primitive directions of lattice vectors of norm ≤ d.
3. Gap lemma: no primitive vector (a,b) with a²+b² ≤ d² has direction angle in (0, arctan(1/d)):
   b ≥ 1 and a ≤ d imply b/a ≥ 1/d. So Q has no edge whose direction lies strictly within the angular
   interval (0, α); hence there is an interval I of outer-normal angles of length ≥ α on which h_Q is the
   support of a single vertex q: h_Q(θ) = |q| cos(θ − θ_q) on I.
4. On an interval of length α, a function |q| cos(θ − θ_q) varies by at least |q|(1 − cos(α/2)).
   |q| ≥ min h_Q = ½ min W ≥ ½[(2ρ_K − 2ε) − (2ρ_0 + 2ε)] = ρ_K − ρ_0 − 2ε.
   Hence A(W) = 2(max h_Q − min h_Q) ≥ 2(ρ_K − ρ_0 − 2ε)(1 − cos(α/2)).
5. A(W) ≤ A(D_K) + A(D_0) ≤ 8ε.  ∎

Corollary (fixed library). If all C_k belong to a fixed finite library L (up to translation), the error
grows linearly: ε ≥ κ(L) ρ_K − O(1), with κ(L) = the LP value
    min_{λ ≥ 0, Σλ w_C has mean 1} ½ A(Σ λ_C w_C)   (> 0 because no polygon has constant width).
(Exact constants computed for octagons, periodic lines, …)

Remarks.
- The theorem is about Minkowski chains; general granulometries only need D_l to be a union of translates
  of D_k. Empirically, Gauss digital disks almost never form such chains (see gauss_chain experiments).
- Real-valued relaxation (LP over weights, monotone) achieves bounded error (≈0.4 up to r=2000): the
  obstruction is integrality/granularity, exactly what the theorem captures (step 1 uses integrality
  through "edges are lattice vectors").

## Upper bounds (constructive / computational)
- Optimised chains (MILP, warm-started) – measure growth exponent; compare with R^{1/3} lower bound.
- Open problem: close the gap.

## Exactness (no holes)
Empirical: for D4-orbits of standard (4-conn) DSS, D = conv(D) ∩ Z² always (1200 random chains);
for 8-conn DSS when the unit square is a summand. For the published families verified directly.
