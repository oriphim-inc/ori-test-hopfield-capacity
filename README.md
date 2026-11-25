> **Hopfield network memory capacity.** A project Ori built and ran on a problem from Oriphim's held-out test set, which Ori was never tuned on. Short paper: [paper/paper.pdf](paper/paper.pdf). The Sceptic's review is in [notes/review.md](notes/review.md); limits are in [LIMITS.md](LIMITS.md). [oriphim.com](https://oriphim.com)
>
> How this repo was made: Ori built and ran the project and wrote the paper. Afterwards, Oriphim added the licence, edited the author line and wrote this note, with help from Claude Code.

# Hopfield network: capacity, (α, T) phase diagram, spurious states, pseudo-inverse rule

By Oriphim, with Ori. Experiment as agreed: [PLAN.md](PLAN.md). What this does **not** show: [LIMITS.md](LIMITS.md).

## Answer
- **Hebbian critical capacity.** α_c is the load at which half of the noisy cues are recalled to m ≥ 0.95. It falls with N toward the AGS value α_c = 0.138, which `theory.py` computes from the replica-symmetric T = 0 equations (overlap 0.967 at α_c):

  | flipped bits | N=200 | N=500 | N=1000 | fit a + b/√N, value of a (indicative only) |
  |---|---|---|---|---|
  | 5 % | 0.176 | 0.165 | 0.157 | 0.144 |
  | 10 % | 0.167 | 0.157 | 0.153 | 0.141 |
  | 15 % | 0.162 | 0.147 | 0.145 | 0.129 |
  | 20 % | 0.148 | 0.138 | 0.135 | 0.124 |

  This α_c measures recall from a noisy cue. It mixes whether retrieval states exist with how large their basins are, so it depends on the cue noise. It is not identical to the AGS storage capacity. The intervals in `results/capacities.csv` are about ±0.001. Pattern-to-pattern correlation makes the true uncertainty about 1.5 times that; see LIMITS.md.
- **Pseudo-inverse rule (zero diagonal).** α_c ≈ **0.65, 0.51, 0.41, 0.33** for 5, 10, 15 and 20 % flipped bits, nearly the same for N = 200, 500 and 1000. That is 2.4 to 4 times the Hebbian value. Every pattern stays a fixed point up to α → 1. What limits recall is the basins of attraction, which shrink as the load rises.
- **Phase diagram (N = 1000, Glauber dynamics).**
  - **Retrieval** sits below a line that tracks the AGS T_M(α): from T ≈ 0.75 at α = 0.01 down to T ≈ 0.2 at α = 0.12. Theory, computed in the code: 0.80 and 0.22.
  - **Spin glass** fills the region from there up to T ≈ 0.95–1.45. That edge lies 0.1–0.2 below T_g = 1 + √α because of the q ≥ 0.1 cut-off at finite N.
  - **Paramagnet** is everything above.
  - Retrieval labels at α = 0.14–0.16 are finite-size states that survive for a limited time, not a real phase.
- **Spurious states (Hebbian, T = 0, N = 1000, 1000 random starts per P):**

  | P | pattern | reversed | 3-mix | 5-mix | 7-mix | other | distinct mixture states found |
  |---|---|---|---|---|---|---|---|
  | 3 | 0.35 | 0.37 | 0.28 | 0 | 0 | 0 | 16 = 4 per pattern set (all of them) |
  | 5 | 0.31 | 0.30 | 0.27 | 0.03 | 0 | 0.10 | 155 |
  | 7 | 0.32 | 0.31 | 0.19 | 0.07 | 0.002 | 0.10 | 223 |
  | 10 | 0.37 | 0.39 | 0.10 | 0.05 | 0.02 | 0.08 | 158 |
  | 20 | 0.38 | 0.39 | 0.01 | 0.003 | 0.001 | 0.21 | 10 |
  | 50 | 0.01 | 0.004 | 0 | 0 | 0 | 0.99 | 0 |
  | ≥100 | 0 | 0 | 0 | 0 | 0 | 1.00 | 0 |

  - At small P, roughly 30–40 % of random starts end in mixture or other spurious states.
  - "Other" means asymmetric mixtures at small P and spin-glass-like states at large P.
  - Stability of symmetric mixtures (`results/mixture_stability.csv`): 3-mixtures are exact fixed points 100 % of the time at P ≤ 10, 89 % at P = 20 and 2 % at P = 50. Higher-order mixtures lose stability sooner: 7-mixtures are 11 % at P = 20.

## Evidence
- Figures:
  - `results/overlap_vs_alpha_hebb.png` and `results/recalled_fraction_vs_alpha_hebb.png`: Hebbian recall vs α for each N.
  - `results/*_pinv.png`, `results/hebb_vs_pseudoinverse.png` and `results/capacity_vs_noise.png`: the pseudo-inverse rule and the comparison with Hebb.
  - `results/phase_diagram.png`: maps of m and q with the theory lines drawn on.
  - `results/spurious_states.png`.
- Numbers are in `results/*.csv` and `results/summary.json`.
- The 9 unit tests and the Sceptic's 34 pre-registered checks (`tests/checks/`) all pass. The Sceptic also confirmed the runs are bit-for-bit reproducible across worker counts.
- No run failed to converge.

## Re-run
```
uv venv .venv && uv pip install --python .venv numpy scipy matplotlib numba pytest
.venv/bin/python run.py            # full run, ~40 s on 15 cores; --scale 0.1 for a pilot
.venv/bin/python -m pytest -q tests
```
Code: `src/hopfield/core.py` (rules, T=0 async and Glauber kernels), `theory.py` (AGS RS equations), `experiments.py`.
