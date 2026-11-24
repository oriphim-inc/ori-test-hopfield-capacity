"""Experiments E1-E4 of PLAN.md. Each task is a pure function of its arguments and seed."""
import math
# Threads, not processes: the numba kernels release the GIL (and the sandbox blocks process pools)
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import core

RECALL_THRESHOLD = 0.95  # "correctly recalled" (PLAN.md)


def _rng(*key):
    return np.random.default_rng(np.random.SeedSequence(list(key)))


def _seed_int(rng):
    return int(rng.integers(0, 2**31 - 1))


# ---------------------------------------------------------------- E1 / E4 recall
def recall_task(rule, N, P, noises, min_trials, base_seed, max_tested=100):
    """Store P random patterns with `rule`; recall each tested pattern from round(f N) flips.

    Returns dict with per-noise arrays of final overlaps, sweeps and convergence flags."""
    rng = _rng(base_seed, 1 if rule == "hebb" else 2, N, P)
    n_test = min(P, max_tested)
    n_sets = max(1, math.ceil(min_trials / n_test))
    out = {f: {"m": [], "sweeps": [], "conv": []} for f in noises}
    for _ in range(n_sets):
        xi = core.random_patterns(rng, P, N)
        W = core.hebbian(xi) if rule == "hebb" else core.pseudo_inverse(xi)
        for mu in range(n_test):
            for f in noises:
                s0 = core.flip_bits(rng, xi[mu], int(round(f * N)))
                s, sw, conv = core.relax(W, s0, _seed_int(rng))
                out[f]["m"].append(float(xi[mu].astype(np.int64) @ s.astype(np.int64)) / N)
                out[f]["sweeps"].append(sw)
                out[f]["conv"].append(conv)
    return {"rule": rule, "N": N, "P": P, "alpha": P / N, "n_sets": n_sets,
            "res": {f: {k: np.array(v) for k, v in d.items()} for f, d in out.items()}}


def recall_sweep(rule, Ns, alphas, noises, min_trials, seed, workers):
    tasks = []
    for N in Ns:
        Ps = sorted({max(1, int(round(a * N))) for a in alphas})
        for P in Ps:
            tasks.append((rule, N, P, tuple(noises), min_trials, seed))
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(_recall_star, tasks))
    return results


def _recall_star(args):
    return recall_task(*args)


def summarise_recall(results, noises, thresholds=(0.90, RECALL_THRESHOLD, 1.0)):
    """Table rows: N, P, alpha, f, n, mean m, std m, fraction m>=thr for each thr, conv frac."""
    rows = []
    for r in results:
        for f in noises:
            d = r["res"][f]
            m = d["m"]
            row = {"N": r["N"], "P": r["P"], "alpha": r["alpha"], "f": f, "n": int(m.size),
                   "m_mean": float(m.mean()), "m_std": float(m.std()),
                   "mean_sweeps": float(d["sweeps"].mean()),
                   "conv_frac": float(d["conv"].mean())}
            for t in thresholds:
                row[f"frac_ge_{t:.2f}"] = float(np.mean(m >= t - 1e-12))
            rows.append(row)
    return rows


def crossing(alphas, frac, level=0.5):
    """First alpha where frac falls from >= level to < level (linear interpolation).
    nan if it never falls below level; alphas[0] if already below at the start."""
    alphas = np.asarray(alphas)
    frac = np.asarray(frac)
    if frac[0] < level:
        return float(alphas[0])
    for i in range(1, len(frac)):
        if frac[i] < level:
            a0, a1, f0, f1 = alphas[i - 1], alphas[i], frac[i - 1], frac[i]
            return float(a0 + (f0 - level) * (a1 - a0) / (f0 - f1))
    return float("nan")


def capacity_with_ci(alphas, frac, n, level=0.5, n_boot=400, seed=0):
    """alpha_c by crossing; 16-84 % parametric-bootstrap interval from binomial resampling."""
    rng = np.random.default_rng(seed)
    est = crossing(alphas, frac, level)
    boots = []
    for _ in range(n_boot):
        fb = rng.binomial(np.asarray(n), np.clip(frac, 0, 1)) / np.asarray(n)
        boots.append(crossing(alphas, fb, level))
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    if boots.size < n_boot // 2:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(boots, 16)), float(np.percentile(boots, 84))


# ---------------------------------------------------------------- E2 phase diagram
def phase_task(N, alpha, T, n_sets, n_eq, n_meas, base_seed):
    P = max(1, int(round(alpha * N)))
    rng = _rng(base_seed, 3, N, P, int(round(T * 1000)))
    mA, mabsA, qB, mmaxB, qA = [], [], [], [], []
    for _ in range(n_sets):
        xi = core.random_patterns(rng, P, N)
        W = core.hebbian(xi)
        # Run A: from pattern 1, overlap with pattern 1
        m_mean, m_abs, q_a = core.glauber(W, xi[0], T, xi[0], n_eq, n_meas, _seed_int(rng))
        mA.append(m_mean)
        mabsA.append(m_abs)
        qA.append(q_a)
        # Run B: random start -> Edwards-Anderson q
        s0 = (2 * rng.integers(0, 2, N) - 1).astype(np.int8)
        _, _, q = core.glauber(W, s0, T, xi[0], n_eq, n_meas, _seed_int(rng))
        qB.append(q)
    return {"N": N, "P": P, "alpha": P / N, "alpha_nominal": alpha, "T": T,
            "m_A": float(np.mean(mA)), "m_A_sets": mA, "q_A": float(np.mean(qA)),
            "q_B": float(np.mean(qB)), "q_B_sets": qB}


def _phase_star(args):
    return phase_task(*args)


def phase_diagram(N, alphas, Ts, n_sets, n_eq, n_meas, seed, workers):
    tasks = [(N, a, T, n_sets, n_eq, n_meas, seed) for a in alphas for T in Ts]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(_phase_star, tasks, chunksize=4))


def classify_phase(m_A, q_B, m_thr=0.5, q_thr=0.1):
    if m_A >= m_thr:
        return "R"
    if q_B >= q_thr:
        return "SG"
    return "P"


# ---------------------------------------------------------------- E3 spurious states
def classify_state(xi, s, pat_thr=0.9, member_thr=0.2, mix_thr=0.9):
    """Return (label, k): 'pattern', 'reversed', 'mixture' (k odd >= 3) or 'other'."""
    m = core.overlaps(xi, s)
    j = int(np.argmax(np.abs(m)))
    if abs(m[j]) >= pat_thr:
        return ("pattern" if m[j] > 0 else "reversed"), 1
    S = np.flatnonzero(np.abs(m) >= member_thr)
    k = S.size
    if k >= 3 and k % 2 == 1:
        mix = np.sign(np.sign(m[S]) @ xi[S].astype(np.float64))
        ov = abs(float(mix @ s.astype(np.float64)) / s.size)
        if ov >= mix_thr:
            return "mixture", int(k)
    return "other", int(k)


def spurious_task(N, P, n_starts, set_id, base_seed):
    rng = _rng(base_seed, 4, N, P, set_id)
    xi = core.random_patterns(rng, P, N)
    W = core.hebbian(xi)
    labels, ks, keys, nonconv = [], [], [], 0
    for _ in range(n_starts):
        s0 = (2 * rng.integers(0, 2, N) - 1).astype(np.int8)
        s, _, conv = core.relax(W, s0, _seed_int(rng))
        nonconv += (not conv)
        lab, k = classify_state(xi, s)
        labels.append(lab)
        ks.append(k)
        canon = s if s[0] > 0 else -s
        keys.append(canon.tobytes() if lab not in ("pattern", "reversed") else None)
    return {"N": N, "P": P, "set": set_id, "labels": labels, "k": ks, "keys": keys,
            "nonconv": nonconv}


def _spurious_star(args):
    return spurious_task(*args)


def mixture_stability_task(N, P, k, n_mix, base_seed):
    rng = _rng(base_seed, 5, N, P, k)
    xi = core.random_patterns(rng, P, N)
    W = core.hebbian(xi)
    exact, near, ov_list = 0, 0, []
    n = 0
    for _ in range(n_mix):
        S = rng.choice(P, size=k, replace=False)
        eps = 2 * rng.integers(0, 2, k) - 1
        mix = np.sign(eps @ xi[S].astype(np.float64)).astype(np.int8)
        s, _, _ = core.relax(W, mix, _seed_int(rng))
        ov = float(mix.astype(np.float64) @ s.astype(np.float64)) / N
        ov_list.append(ov)
        exact += bool(np.array_equal(s, mix))
        near += ov >= 0.9
        n += 1
    return {"N": N, "P": P, "k": k, "n": n, "frac_exact_fixed": exact / n,
            "frac_stay_near": near / n, "mean_overlap_with_start": float(np.mean(ov_list))}


def _mixstab_star(args):
    return mixture_stability_task(*args)


def spurious_experiment(N, Ps, n_starts_total, n_sets, mix_Ps, mix_ks, n_mix, seed, workers):
    per_set = max(1, n_starts_total // n_sets)
    tasks = [(N, P, per_set, j, seed) for P in Ps for j in range(n_sets)]
    mtasks = [(N, P, k, n_mix, seed) for P in mix_Ps for k in mix_ks if k <= P]
    with ThreadPoolExecutor(workers) as ex:
        basins = list(ex.map(_spurious_star, tasks))
        stab = list(ex.map(_mixstab_star, mtasks))
    rows = []
    for P in Ps:
        rs = [r for r in basins if r["P"] == P]
        labels = sum((r["labels"] for r in rs), [])
        ks = sum((r["k"] for r in rs), [])
        n = len(labels)
        row = {"N": N, "P": P, "alpha": P / N, "n_starts": n,
               "frac_pattern": labels.count("pattern") / n,
               "frac_reversed": labels.count("reversed") / n}
        for kk in (3, 5, 7):
            row[f"frac_mix{kk}"] = sum(1 for l, k in zip(labels, ks)
                                       if l == "mixture" and k == kk) / n
        row["frac_mix_higher"] = sum(1 for l, k in zip(labels, ks)
                                     if l == "mixture" and k > 7) / n
        row["frac_other"] = labels.count("other") / n
        # distinct spurious fixed points (up to global sign), per pattern set, summed
        n_distinct_mix, n_distinct_other = 0, 0
        for r in rs:
            mixk = {key for key, l in zip(r["keys"], r["labels"]) if l == "mixture"}
            oth = {key for key, l in zip(r["keys"], r["labels"]) if l == "other"}
            n_distinct_mix += len(mixk)
            n_distinct_other += len(oth)
        row["distinct_mixture_states"] = n_distinct_mix
        row["distinct_other_states"] = n_distinct_other
        row["n_pattern_sets"] = len(rs)
        row["nonconverged"] = sum(r["nonconv"] for r in rs)
        rows.append(row)
    return rows, stab
