# PLAN — Hopfield network: capacity, phase diagram, spurious states, pseudo-inverse

Requested by: Oriphim. Planned by Ori (no attachments; standard textbook results only,
cited below, so no separate literature search).

## Question
For a Hopfield network of N binary (±1) neurons:
1. With Hebbian couplings, how does recall from a corrupted pattern (5–20 % flipped bits)
   degrade with load α = P/N ∈ [0.02, 0.30], for N = 200, 500, 1000? Where is the critical
   capacity α_c?
2. With finite-temperature Glauber dynamics, what is the (α, T) phase diagram — retrieval (R),
   spin-glass (SG) and paramagnetic (P) regions?
3. How many / what kind of spurious states (mixtures of patterns) does the network have?
4. With the pseudo-inverse (projection) rule instead of Hebb, what is the capacity?

## Model (fixed — part of what is agreed)
- Neurons s_i ∈ {−1,+1}; patterns ξ^μ_i i.i.d. ±1 with probability 1/2 (unbiased).
- Hebb: W_ij = (1/N) Σ_μ ξ^μ_i ξ^μ_j, W_ii = 0.
- Pseudo-inverse (Personnaz et al. 1985; Kanter & Sompolinsky 1987):
  W = (1/N) Ξ C⁻¹ Ξᵀ with C_μν = (1/N) Σ_i ξ^μ_i ξ^ν_i, then W_ii = 0 (zero self-coupling,
  as Kanter & Sompolinsky recommend; it is what gives the network finite basins of attraction).
- Field h_i = Σ_j W_ij s_j; no external field / thresholds.
- T = 0 dynamics: asynchronous, random sequential order (new random permutation each sweep),
  s_i ← sign(h_i), ties (h_i = 0) keep s_i. Converged when a full sweep flips nothing
  (cap 500 sweeps; non-converged runs are counted and reported).
- Finite T: Glauber heat bath, P(s_i = +1) = 1 / (1 + exp(−2 h_i / T)), random sequential order.
  With this normalisation the AGS units apply (T_g(α→0) = 1).

## Experiments
**E1 Hebbian recall sweep.** N ∈ {200, 500, 1000}; α = 0.02, 0.03, …, 0.30 (29 values);
initial noise f ∈ {0.05, 0.10, 0.15, 0.20} (exactly round(fN) distinct bits flipped).
For each (N, α) draw independent pattern sets until ≥ 400 recall trials per noise level
(each set tests up to 100 of its patterns). Record final overlap m = (1/N) Σ ξ_i s_i.
- "Correctly recalled" := final m ≥ 0.95 (just below the AGS overlap at α_c, m ≈ 0.967).
  Sensitivity also reported for m ≥ 0.90 and m = 1 (exact).
- α_c(N, f) := α at which the recalled fraction crosses 0.5 (linear interpolation).
  Compared with the AGS replica-symmetric value, computed in the code from the T = 0
  saddle-point equations (expected ≈ 0.138).
- A finite-size extrapolation α_c(N) = α_∞ + a N^(−1/2) is reported as indicative only
  (three N values).

**E2 Phase diagram (α, T).** N = 1000; α ∈ {0.01, 0.02, …, 0.30} (every 0.02 from 0.02, plus 0.01);
T ∈ {0.05, 0.10, …, 1.60}. At each point, 2 pattern sets:
- Run A from pattern ξ¹ (no noise): 100 equilibration + 200 measurement sweeps; time-averaged m.
- Run B from a random state: same sweeps; Edwards–Anderson q = (1/N) Σ_i ⟨s_i⟩² (time average).
- Classification: R if ⟨m⟩_A ≥ 0.5; else SG if q_B ≥ 0.1; else P. Thresholds are
  conventions; m and q maps are plotted so the user can see the raw data.
- Overlaid theory lines (computed by the code from the AGS RS equations, not typed in):
  T_g(α) = 1 + √α (SG instability of the m = 0 solution) and T_M(α) (highest T at which a
  retrieval solution m > 0 exists).

**E3 Spurious states.** Hebbian, T = 0, N = 1000, P ∈ {3, 5, 7, 10, 20, 50, 100, 138}.
(a) Basins: 1000 random initial states per P (spread over 4 pattern sets); classify each
fixed point by its overlaps: pattern (|m_μ| ≥ 0.9 for one μ; reversed patterns −ξ counted
separately), odd k-mixture (state overlaps ≥ 0.9 with sign(Σ_{μ∈S} ±ξ^μ) where S = patterns with
|m_μ| ≥ 0.2), or "other / spin-glass". Report fractions and the number of *distinct* spurious
fixed points (up to global sign) found.
(b) Stability of symmetric mixtures: for P ∈ {5, 10, 20, 50} and k ∈ {3, 5, 7}, start from up to
200 random symmetric k-mixtures sign(Σ ±ξ^μ) and run T = 0 dynamics; report the fraction that
are exact fixed points and that stay within overlap ≥ 0.9 of the start.
Reference: AGS 1985 — at α → 0, T = 0 all odd symmetric mixtures are stable (3-mixture
overlap 1/2 per pattern).

**E4 Pseudo-inverse capacity.** Same protocol as E1, rule replaced by pseudo-inverse,
N ∈ {200, 500, 1000}, f ∈ {0.05, …, 0.20}. Because the pseudo-inverse capacity is expected
far above 0.3, the α range is extended to 0.02 … 0.95 (step 0.01 to 0.30, then 0.05) —
otherwise the capacity could not be located. Same α_c definition. Reference: stability
limit α = 1 (Personnaz; Kanter & Sompolinsky), basins shrinking with α, so the noisy-recall
capacity depends on f.

## What counts as the answer
- Figures: overlap vs α and recalled fraction vs α for each N and f (Hebb); phase diagram
  with theory lines; spurious-state table; Hebb vs pseudo-inverse comparison.
- Numbers: α_c(N, f) for Hebb and pseudo-inverse, in results/summary.json.
- Sanity: Hebbian α_c should be in the neighbourhood of 0.14 (finite-size estimates with
  noisy starts are typically somewhat higher); pseudo-inverse α_c should be well above Hebb.
  A different outcome is reported, not tuned away.

## Scale-down
`python run.py --scale s` multiplies trial counts (and E2 grid density) by s for pilots;
the reported results use `--scale 1`. N values, α ranges and noise levels are not scaled.

## References (standard results)
- J. J. Hopfield, PNAS 79, 2554 (1982).
- D. J. Amit, H. Gutfreund, H. Sompolinsky (AGS), Phys. Rev. A 32, 1007 (1985) (spurious
  mixtures); Phys. Rev. Lett. 55, 1530 (1985); Ann. Phys. 173, 30 (1987) (α_c ≈ 0.138, phase diagram).
- L. Personnaz, I. Guyon, G. Dreyfus, J. Physique Lett. 46, L359 (1985) (projection rule).
- I. Kanter, H. Sompolinsky, Phys. Rev. A 35, 380 (1987) (pseudo-inverse, zero diagonal).
- Hertz, Krogh, Palmer, *Introduction to the Theory of Neural Computation* (1991), ch. 2.
