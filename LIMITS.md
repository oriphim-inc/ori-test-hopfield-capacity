# LIMITS: what this work does NOT show

Written by the Sceptic after the full run (`run.py --scale 1`, seed 20261113). Read these
alongside results/summary.json.

## Model assumptions (fixed in PLAN.md, not tested beyond)
- Unbiased i.i.d. ±1 patterns only. Correlated, sparse or biased patterns were not tested, and
  the pseudo-inverse advantage over Hebb is largest for correlated patterns, which this study never looks at.
- Zero self-coupling for both rules. For the pseudo-inverse this is the Kanter–Sompolinsky
  variant. With the diagonal kept (W_ii ≈ α), basins shrink much faster with α, and none of the PI numbers
  apply to that variant.
- Asynchronous random-sequential updates only. Synchronous (parallel) dynamics can show
  2-cycles and different basins, and it was not run.
- No thresholds, no external fields, no dilution, no synaptic noise, no clipping.

## E1 / E4 "capacity" is a noisy-cue recall capacity, not the AGS storage capacity
- α_c(N, f) is the load at which 50 % of cues with fraction f flipped end at m ≥ 0.95. It
  combines two things: whether a retrieval state exists, and how big its basin is. That is why it depends on f
  (Hebb, N=1000: 0.157 at 5 % noise down to 0.135 at 20 %). Only the f → 0, N → ∞ limit
  should be compared with α_c ≈ 0.138, and that limit was not measured.
- The finite-size fit α_c = α_∞ + a N^(-1/2) uses 3 points and 2 parameters (1 degree of
  freedom), and the N^(-1/2) form is assumed, not established. α_∞ ≈ 0.144 / 0.141 / 0.129 / 0.124
  (f = 5/10/15/20 %) is indicative only. It neither confirms nor refutes 0.138 to better than about ±0.01.
- Finite-N Hebbian retrieval at α ≈ 0.14–0.16 (m ≥ 0.95 above the RS limit) is a known
  finite-size effect. It does not show that retrieval survives above α_c in the thermodynamic limit.
- The exact-recall criterion (m = 1) gives α_c ≈ 0.10–0.14, and it falls with N, as expected
  (the exact-recall capacity → 0 like 1/(2 ln N)). It is a different quantity from the AGS α_c.
- The 16–84 % intervals on α_c come from a binomial bootstrap that treats every recall trial
  as independent. Trials that share a pattern set (up to 100 per set, so only 4 sets at N=1000 near
  the crossing) are correlated. A spot check at N=1000, α=0.155, f=5 % found a between-set
  spread about 1.5× the binomial one. The true uncertainty is therefore about 1.5× the reported
  interval or more, though it is still small (about ±0.002).
- Pseudo-inverse α_c (≈ 0.65 / 0.51 / 0.41 / 0.33 for f = 5–20 %) applies to the zero-diagonal
  rule and these noise levels only. The near-independence from N is plausible: the field
  decomposition is O(1) and N-independent. But only N ≤ 1000 was tested, and α was sampled every 0.05
  above 0.30, so a crossing there comes from linear interpolation across a 0.05-wide bin.
- Recall was judged against the cued pattern only. Ending in a different stored pattern
  counts as a failure.

## E2 phase diagram
- One size (N = 1000), 2 pattern sets per point, 100 + 200 sweeps, and a single initial
  condition per run. There is no annealing, no check that the runs reached equilibrium, and no finite-size scaling.
  The m and q maps are finite-time, finite-N observables, not equilibrium order parameters.
- The labels R / SG / P depend on the conventional thresholds m ≥ 0.5 and q ≥ 0.1.
  - Run A starts exactly on the pattern, so "R" means a retrieval state that is metastable
    over 300 sweeps, not thermodynamically stable (AGS distinguish T_M from T_c, the line where
    retrieval becomes the global minimum. T_c was not computed).
  - ⟨m⟩_A is averaged over 2 pattern sets, and the outcomes are bimodal (retrieved or lost).
    An "R" label with m_A ≈ 0.5–0.7 means one set retrieved and one did not. That is why α = 0.14 is
    labelled R, R, SG, R, SG for T = 0.05 … 0.25 and α = 0.16 is R only at T = 0.05. The R points beyond
    the RS line at α = 0.14–0.16 are finite-N, finite-time metastability, consistent with E1's
    finite-N crossing near 0.155. They do not show a retrieval phase beyond AGS.
  - q ≥ 0.1 places the SG boundary below T_g = 1 + √α, because q grows continuously from 0 at
    T_g. Near T_g the 200-sweep time average is also biased by slow relaxation.
    Agreement with T_g and T_M is only claimed within 0.25 in T (the pre-registered tolerance).
- The study does not separate a true spin-glass phase from trapping in mixture states. At small α
  a random start that ends in a mixture or a pattern also gives large q_B.
- The theory lines are replica-symmetric. Replica-symmetry breaking corrections (small,
  shifting α_c by at most a few thousandths to a few hundredths depending on the RSB level; not looked up) and the RS re-entrance near α_c are not resolved.

## E3 spurious states
- "mixture" means *symmetric* odd mixtures only: overlap ≥ 0.9 with sign(Σ ±ξ^μ) over the
  patterns with |m_μ| ≥ 0.2. *Asymmetric* mixtures (unequal overlaps, e.g. |m| ≈ 0.61, 0.38,
  0.36, 0.12, 0.11, seen at P = 5) fall into "other". At small P (P ≤ 10, α ≤ 0.01) "other"
  is therefore mostly asymmetric mixture states, **not spin-glass states**. "other" should not
  be read as "spin glass" except at large α, where no pattern has a large overlap.
- "distinct states" counts distinct microscopic fixed points (up to global sign) reached from
  the sampled random starts, summed over 4 pattern sets. For symmetric mixtures it matches the
  number of distinct mixture identities (spot check at P = 5: 41 identities, 41 microstates).
  For "other" at P ≥ 50 it equals about the number of starts (978/1000, 1000/1000), so it is
  limited by sample size. It is a lower bound that says "very many", not an estimate of how many
  fixed points exist (which grows exponentially in N).
- The basin fractions are fractions of uniformly random starts under one dynamics. They are not
  basin volumes in any other measure, and they say nothing about energies.
- Mixture stability (T = 0, N = 1000) is a finite-N statement. The AGS α-limits for mixture
  stability (about 0.03 for 3-mixtures) were not located precisely.

## Numerics
- Fields are updated incrementally in floating point. For even P, exact ties (h_i = 0) occur
  with Hebbian weights, and round-off can then decide the update instead of the "keep s_i" rule.
  This only affects tie spins and had no visible effect: 0 non-converged runs.
- Reproducibility was verified for the thread-pool backend: the same results with 1 and 8 workers,
  for both T = 0 and Glauber kernels, because each task re-seeds numba's per-thread generator
  right before its kernel. Bit-identical results across machines, numba versions or BLAS
  builds (the pseudo-inverse solve, W @ s) were not checked.
