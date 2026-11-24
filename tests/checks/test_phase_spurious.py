"""C5 (alpha, T) phase diagram (E2); C6 spurious states and mixture stability (E3)."""
import math

import numpy as np


def Tg(a):
    return 1 + math.sqrt(a)


# ------------------------------------------------------------------ C5 phase diagram
def test_high_T_paramagnetic(phase):
    # 0.25 above T_g: finite-time q noise ~ (1 + 2 tau)/200 sweeps << 0.1, m ~ 1/sqrt(N).
    n = 0
    for r in phase:
        if r["T"] >= Tg(r["alpha"]) + 0.25:
            n += 1
            assert r["phase"] == "P" and abs(r["m_A"]) < 0.2, r
    assert n > 0


def test_low_T_low_alpha_retrieval(phase):
    for r in phase:
        if r["alpha"] <= 0.06 + 1e-9 and r["T"] <= 0.30 + 1e-9:
            assert r["phase"] == "R" and r["m_A"] >= 0.85, r
        if r["alpha"] <= 0.10 + 1e-9 and r["T"] <= 0.10 + 1e-9:
            assert r["phase"] == "R" and r["m_A"] >= 0.95, r


def test_low_T_high_alpha_spin_glass(phase):
    # alpha >= 0.20 is far beyond alpha_c even at N = 1000: no retrieval, frozen random state.
    for r in phase:
        if r["alpha"] >= 0.20 - 1e-9 and r["T"] <= 0.30 + 1e-9:
            assert r["phase"] == "SG", r


def test_ordered_below_Tg(phase):
    for r in phase:
        # 0.35 below T_g: q_EA well above the 0.1 threshold; margin covers N=1000 wandering
        if r["T"] <= Tg(r["alpha"]) - 0.35:
            assert r["phase"] in ("R", "SG"), r


def test_retrieval_region_connected_downward(phase):
    # From a pattern start, retrieval at T implies retrieval at all lower T, except within 0.1
    # of the top of the R region (stochastic escape near the spinodal).
    for a in sorted({r["alpha_nominal"] for r in phase}):
        col = sorted([r for r in phase if r["alpha_nominal"] == a], key=lambda r: r["T"])
        tR = [r["T"] for r in col if r["phase"] == "R"]
        if tR:
            for r in col:
                if r["T"] <= max(tR) - 0.10 - 1e-9:
                    assert r["phase"] == "R", (a, r["T"])


def test_boundaries_vs_theory(summary):
    pb = summary["phase_boundaries"]
    dev_sg = []
    for b in pb:
        a = b["alpha"]
        # SG/P boundary vs T_g = 1 + sqrt(alpha): grid 0.05, q threshold convention and N=1000
        assert b["T_SG_max_sim"] is not None, b
        d = b["T_SG_max_sim"] - b["T_g_theory"]
        dev_sg.append(abs(d))
        assert abs(d) <= 0.25, b
        # R boundary vs AGS T_M (metastability limit, which a pattern-start run probes)
        tm = b["T_M_theory"]
        if a <= 0.12 + 1e-9:
            assert b["T_R_max_sim"] is not None, b
            if tm is not None and math.isfinite(tm):
                assert abs(b["T_R_max_sim"] - tm) <= 0.25, b
        if a >= 0.20 - 1e-9:
            assert b["T_R_max_sim"] is None, b
    assert np.median(dev_sg) <= 0.15


# ------------------------------------------------------------------ C6 spurious states
def _row(basins, P):
    return [r for r in basins if r["P"] == P][0]


def test_small_P_mixtures_and_patterns_appear(basins):
    for P in (3, 5, 7, 10):
        r = _row(basins, P)
        mix = r["frac_mix3"] + r["frac_mix5"] + r["frac_mix7"] + r["frac_mix_higher"]
        assert mix >= 0.02, f"no mixture states found at P={P}"
        assert r["frac_pattern"] + r["frac_reversed"] >= 0.05, f"patterns absent at P={P}"


def test_P3_only_patterns_and_3mixtures(basins):
    # With 3 patterns every sign-mixture is a pattern or the symmetric 3-mixture; 2-mixtures
    # are unstable (zero field on disagreeing sites). So 'other' must be (nearly) empty.
    assert _row(basins, 3)["frac_other"] <= 0.02


def test_pattern_reversal_symmetry(basins):
    # s -> -s symmetry: reversed patterns as frequent as patterns, within 4 sigma + 0.01.
    for r in basins:
        fp, fr, n = r["frac_pattern"], r["frac_reversed"], r["n_starts"]
        assert abs(fp - fr) <= 4 * math.sqrt((fp + fr) / n) + 0.01, r


def test_large_P_spin_glass_dominates(basins):
    for P in (100, 138):
        r = _row(basins, P)
        assert r["frac_other"] >= 0.8 and r["frac_pattern"] + r["frac_reversed"] <= 0.1, r


def test_mixtures_stable_at_small_alpha(mixstab):
    # AGS: at alpha -> 0, T = 0 all odd symmetric mixtures are stable. alpha <= 0.01 here.
    for r in mixstab:
        if r["P"] in (5, 10):
            assert r["frac_stay_near"] >= 0.90, r
            if r["k"] == 3:  # margin 0.5 vs crosstalk ~0.1: essentially always exact
                assert r["frac_exact_fixed"] >= 0.90, r
