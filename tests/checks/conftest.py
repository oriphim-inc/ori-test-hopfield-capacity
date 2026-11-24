"""Shared loaders for the Sceptic's result checks. Every check skips until results/ exist.

Tolerances were fixed before any result existed (stage "checks", 2026-11-13) and are
justified in the comment next to each check. Do not edit (Sceptic-only folder).
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
NS = [200, 500, 1000]
NOISES = [0.05, 0.10, 0.15, 0.20]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def read_csv(name):
    p = RES / name
    if not p.exists():
        pytest.skip(f"results/{name} not produced yet")
    with open(p, newline="") as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def binom_sigma(p, n):
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(max(p * (1 - p), 0.25 / n) / n)  # floor avoids sigma=0 at p in {0,1}


def series(rows, N, f, key):
    rr = sorted([r for r in rows if r["N"] == N and r["f"] == f], key=lambda r: r["alpha"])
    return np.array([r["alpha"] for r in rr]), np.array([r[key] for r in rr]), rr


@pytest.fixture(scope="session")
def summary():
    p = RES / "summary.json"
    if not p.exists():
        pytest.skip("results/summary.json not produced yet")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def hebb():
    return read_csv("recall_hebb.csv")


@pytest.fixture(scope="session")
def pinv():
    return read_csv("recall_pinv.csv")


@pytest.fixture(scope="session")
def caps():
    return read_csv("capacities.csv")


@pytest.fixture(scope="session")
def phase():
    return read_csv("phase_diagram.csv")


@pytest.fixture(scope="session")
def lines():
    return read_csv("theory_lines.csv")


@pytest.fixture(scope="session")
def basins():
    return read_csv("spurious_basins.csv")


@pytest.fixture(scope="session")
def mixstab():
    return read_csv("mixture_stability.csv")


def cap(caps, rule, N, f, crit="frac_ge_0.95"):
    c = [r for r in caps if r["rule"] == rule and r["N"] == N and r["f"] == f
         and r["criterion"] == crit]
    assert len(c) == 1, f"expected one capacity row for {rule} N={N} f={f} {crit}, got {len(c)}"
    return c[0]
