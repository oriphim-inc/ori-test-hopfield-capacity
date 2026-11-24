"""C1 integrity of every results file; C2 the AGS replica-symmetric theory numbers."""
import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.special import erf

from conftest import NOISES, NS


# ------------------------------------------------------------------ C1 integrity
def test_reported_run_is_full_scale(summary):
    # Pilots (--scale < 1) must never be reported as the answer.
    assert summary["scale"] == 1.0


@pytest.mark.parametrize("name", ["hebb", "pinv"])
def test_recall_tables_sane(name, request):
    rows = request.getfixturevalue(name)
    n_alpha = 29 if name == "hebb" else 29 + 13
    assert len(rows) == n_alpha * len(NS) * len(NOISES), "missing (N, alpha, f) grid points"
    for r in rows:
        for k in ("m_mean", "m_std", "mean_sweeps", "conv_frac", "frac_ge_0.90",
                  "frac_ge_0.95", "frac_ge_1.00"):
            assert math.isfinite(r[k]), f"non-finite {k} in {r}"
        assert -1 <= r["m_mean"] <= 1 and 0 <= r["m_std"] <= 1
        assert r["n"] >= 400, "PLAN: >= 400 recall trials per (N, alpha, f)"
        assert r["P"] == round(r["alpha"] * r["N"])
        # nested thresholds: fraction >= 0.90 >= fraction >= 0.95 >= fraction == 1
        assert 1 >= r["frac_ge_0.90"] >= r["frac_ge_0.95"] >= r["frac_ge_1.00"] >= 0
        # symmetric W, zero diagonal, asynchronous T=0: energy is a Lyapunov function, so
        # every run must converge; allow 1 % per row at most for pathological long transients.
        assert r["conv_frac"] >= 0.99, f"non-converged runs at {r}"
        assert r["mean_sweeps"] <= 100, f"implausibly slow convergence at {r}"


def test_nonconverged_negligible(summary):
    # Lyapunov argument => expect 0; tolerate <= 0.1 % of all T=0 recall runs.
    tot_h = 29 * 3 * 4 * 400
    tot_p = 42 * 3 * 4 * 400
    assert summary["hebb_nonconverged_runs"] <= 1e-3 * tot_h
    assert summary["pinv_nonconverged_runs"] <= 1e-3 * tot_p


def test_phase_table_sane(phase):
    assert len(phase) == 16 * 32, "PLAN grid: 16 alpha x 32 T"
    for r in phase:
        assert all(math.isfinite(r[k]) for k in ("m_A", "q_A", "q_B"))
        assert -1 <= r["m_A"] <= 1 and 0 <= r["q_A"] <= 1 and 0 <= r["q_B"] <= 1
        exp = "R" if r["m_A"] >= 0.5 else ("SG" if r["q_B"] >= 0.1 else "P")
        assert r["phase"] == exp, f"phase label disagrees with PLAN rule at {r}"


def test_spurious_tables_sane(basins, mixstab):
    cats = ["frac_pattern", "frac_reversed", "frac_mix3", "frac_mix5", "frac_mix7",
            "frac_mix_higher", "frac_other"]
    assert sorted(r["P"] for r in basins) == [3, 5, 7, 10, 20, 50, 100, 138]
    for r in basins:
        assert r["n_starts"] == 1000
        assert abs(sum(r[c] for c in cats) - 1) < 1e-9
        assert all(0 <= r[c] <= 1 for c in cats)
        assert r["nonconverged"] <= 1, "T=0 Hebbian dynamics must converge (Lyapunov)"
    for r in mixstab:
        assert r["n"] == 200
        assert 0 <= r["frac_exact_fixed"] <= r["frac_stay_near"] + 1e-12 <= 1 + 1e-12
        assert -1 <= r["mean_overlap_with_start"] <= 1


# ------------------------------------------------------------------ C2 theory
def _alpha_of_y(y):
    # Independent restatement of the AGS T=0 RS equations: m = erf(y),
    # C = 2 y exp(-y^2) / (sqrt(pi) m)... => sqrt(alpha) = (m - 2y e^{-y^2}/sqrt(pi)) / (sqrt(2) y)
    return (erf(y) - 2 * y * np.exp(-y * y) / np.sqrt(np.pi)) ** 2 / (2 * y * y)


def rs_m_t0(alpha):
    """RS retrieval overlap at T=0 on the stable (large-y) branch; nan above alpha_c."""
    yc = minimize_scalar(lambda y: -_alpha_of_y(y), bounds=(0.3, 5), method="bounded").x
    if alpha > _alpha_of_y(yc):
        return float("nan")
    return float(erf(brentq(lambda y: _alpha_of_y(y) - alpha, yc, 50)))


def test_alpha_c_RS(summary):
    # Literature: alpha_c = 0.138 (0.1379), m_c = 0.967 (AGS 1985/1987; Hertz-Krogh-Palmer ch.2)
    th = summary["theory"]
    assert 0.1375 <= th["alpha_c_RS_T0"] <= 0.1385
    assert 0.965 <= th["m_at_alpha_c_RS_T0"] <= 0.969


def test_theory_lines(lines):
    a = np.array([r["alpha"] for r in lines])
    tm = np.array([r["T_M"] for r in lines])
    tg = np.array([r["T_g"] for r in lines])
    assert np.allclose(tg, 1 + np.sqrt(a), atol=1e-9), "T_g must be 1 + sqrt(alpha)"
    fin = np.isfinite(tm)
    assert fin[a <= 0.12].all(), "retrieval solution must exist at T>0 for alpha <= 0.12"
    assert not np.any(fin & (a >= 0.16) & (tm > 0.05)), "T_M must vanish beyond ~alpha_c"
    assert 0.13 <= a[fin].max() <= 0.155, "T_M line must end near alpha_c = 0.138"
    assert 0.9 <= tm[0] <= 1.0 + 1e-3, "T_M(alpha -> 0) -> 1"
    assert np.all(tm[fin] <= tg[fin] + 1e-9), "retrieval line must lie below T_g"
    assert np.all(np.diff(tm[fin]) <= 0.01), "T_M must decrease with alpha"
    # Soft: small-alpha asymptote T_M ~ 1 - 1.95 sqrt(alpha) (AGS 1987). Warning only.
    small = fin & (a >= 0.002) & (a <= 0.02)
    dev = np.abs(tm[small] - (1 - 1.95 * np.sqrt(a[small])))
    if dev.size and dev.max() > 0.1:
        warnings.warn(f"T_M deviates from 1-1.95 sqrt(alpha) by up to {dev.max():.3f}")
