"""C3 Hebbian recall and capacity (E1); C4 pseudo-inverse recall and capacity (E4)."""
import math
import warnings

import numpy as np
import pytest

from conftest import NOISES, NS, binom_sigma, cap, series
from test_integrity_and_theory import rs_m_t0


# ------------------------------------------------------------------ shared shape checks
@pytest.mark.parametrize("name", ["hebb", "pinv"])
def test_recall_degrades_monotonically(name, request):
    # Recalled fraction may not *rise* with alpha by more than binomial noise:
    # 4 sigma (many comparisons) + 0.02 slack.
    rows = request.getfixturevalue(name)
    for N in NS:
        for f in NOISES:
            a, fr, rr = series(rows, N, f, "frac_ge_0.95")
            for i in range(len(a) - 1):
                n = min(rr[i]["n"], rr[i + 1]["n"])
                tol = 4 * binom_sigma(0.5 * (fr[i] + fr[i + 1]), n) + 0.02
                assert fr[i + 1] <= fr[i] + tol, f"{name} N={N} f={f}: rise at alpha={a[i+1]}"


@pytest.mark.parametrize("name", ["hebb", "pinv"])
def test_more_noise_never_helps(name, request):
    # Mean final overlap from a 5 % cue >= that from a 20 % cue, within 4 standard errors + 0.01.
    rows = request.getfixturevalue(name)
    for N in NS:
        _, m5, r5 = series(rows, N, 0.05, "m_mean")
        _, m20, r20 = series(rows, N, 0.20, "m_mean")
        for x, y in zip(r5, r20):
            se = math.sqrt(x["m_std"] ** 2 / x["n"] + y["m_std"] ** 2 / y["n"])
            assert x["m_mean"] >= y["m_mean"] - 4 * se - 0.01, f"{name} N={N} a={x['alpha']}"


# ------------------------------------------------------------------ C3 Hebb
def test_hebb_low_load_perfect(hebb):
    # alpha <= 0.05: crosstalk error rate Phi(-1/sqrt(alpha)) < 1e-5, basins large -> ~perfect.
    for r in hebb:
        if r["alpha"] <= 0.05 + 1e-9:
            assert r["frac_ge_0.95"] >= 0.95 and r["m_mean"] >= 0.97, r
        if r["alpha"] <= 0.02 + 1e-9:
            assert r["frac_ge_1.00"] >= 0.90, r


def test_hebb_high_load_fails(hebb):
    # alpha = 0.30 is > 2x alpha_c: recall (m >= 0.95) must be rare at every N.
    for r in hebb:
        if abs(r["alpha"] - 0.30) < 1e-9:
            assert r["frac_ge_0.95"] <= 0.10 and r["m_mean"] <= 0.80, r


def test_hebb_overlap_matches_RS(hebb):
    # N=1000, f=0.05, alpha <= 0.10 where recall succeeds: mean m within 0.015 of RS m(alpha)
    # (RS vs RSB and finite-N corrections are below 0.01 here).
    for r in hebb:
        if r["N"] == 1000 and r["f"] == 0.05 and r["alpha"] <= 0.10 + 1e-9 \
                and r["frac_ge_0.95"] >= 0.99:
            assert abs(r["m_mean"] - rs_m_t0(r["alpha"])) <= 0.015, r


def test_hebb_capacity_band(caps):
    # AGS alpha_c = 0.138. Finite N and noisy cues shift alpha_c(m >= 0.95) by a few 0.01.
    for N in NS:
        for f in NOISES:
            c = cap(caps, "hebb", N, f)["alpha_c"]
            assert math.isfinite(c), f"Hebb alpha_c not located N={N} f={f}"
            assert 0.08 <= c <= 0.22, (N, f, c)
            if f <= 0.10:
                assert 0.11 <= c <= 0.20, (N, f, c)
                if N == 1000:
                    assert 0.12 <= c <= 0.17, (N, f, c)


def test_hebb_capacity_precision(caps):
    # 400 trials per point and a steep curve: 16-84 % interval narrower than 0.03.
    for N in NS:
        for f in NOISES:
            c = cap(caps, "hebb", N, f)
            assert math.isfinite(c["ci16"]) and c["ci84"] - c["ci16"] <= 0.03, c


@pytest.mark.parametrize("rule", ["hebb", "pinv"])
def test_capacity_orderings(caps, rule):
    for N in NS:
        # larger cue noise cannot raise capacity (0.01 slack for sampling)
        assert cap(caps, rule, N, 0.20)["alpha_c"] <= cap(caps, rule, N, 0.05)["alpha_c"] + 0.01
        for f in NOISES:
            # nested thresholds => nested crossings, exactly
            c90 = cap(caps, rule, N, f, "frac_ge_0.90")["alpha_c"]
            c95 = cap(caps, rule, N, f, "frac_ge_0.95")["alpha_c"]
            c100 = cap(caps, rule, N, f, "frac_ge_1.00")["alpha_c"]
            if all(map(math.isfinite, (c90, c95, c100))):
                assert c90 >= c95 - 1e-9 and c95 >= c100 - 1e-9, (rule, N, f)


def test_hebb_extrapolation_soft(summary):
    # Indicative only (three N): warn, do not fail, if alpha_inf is far from 0.138.
    for f, d in summary.get("hebb_fss", {}).items():
        if not 0.10 <= d["alpha_inf"] <= 0.18:
            warnings.warn(f"Hebb alpha_inf(f={f}) = {d['alpha_inf']:.3f} outside [0.10, 0.18]")


# ------------------------------------------------------------------ C4 pseudo-inverse
def test_pinv_low_load_perfect(pinv):
    # Patterns are exact fixed points (h = (1 - C^-1_ii/N) xi); at alpha <= 0.10 basins are large.
    for r in pinv:
        if r["alpha"] <= 0.10 + 1e-9:
            assert r["frac_ge_0.95"] >= 0.95 and r["frac_ge_1.00"] >= 0.90, r


def test_pinv_capacity_located_and_above_hebb(caps):
    for N in NS:
        for f in NOISES:
            p = cap(caps, "pinv", N, f)["alpha_c"]
            h = cap(caps, "hebb", N, f)["alpha_c"]
            assert math.isfinite(p) and p < 0.95, f"pinv alpha_c not located N={N} f={f}"
            assert p >= h + 0.05, f"pinv not clearly above Hebb N={N} f={f}: {p} vs {h}"
            if f <= 0.10:
                assert p >= 1.5 * h, f"pinv < 1.5 x Hebb N={N} f={f}: {p} vs {h}"


def test_pinv_not_worse_than_hebb_pointwise(hebb, pinv):
    # On the common alpha grid, pinv recall fraction >= Hebb's within 4 sigma + 0.02.
    H = {(r["N"], r["f"], round(r["alpha"], 6)): r for r in hebb}
    for r in pinv:
        k = (r["N"], r["f"], round(r["alpha"], 6))
        if k in H:
            h = H[k]
            tol = 4 * binom_sigma(0.5 * (h["frac_ge_0.95"] + r["frac_ge_0.95"]),
                                  min(h["n"], r["n"])) + 0.02
            assert r["frac_ge_0.95"] >= h["frac_ge_0.95"] - tol, k
