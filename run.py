"""Reproduce every result: python run.py [--scale 1.0] [--workers 14] [--only E1,E2,E3,E4]

--scale multiplies trial counts (and thins the E2 grid when < 1) for pilots.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplcache"))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hopfield import experiments as ex  # noqa: E402
from hopfield import theory  # noqa: E402

NS = [200, 500, 1000]
NOISES = [0.05, 0.10, 0.15, 0.20]
ALPHAS_HEBB = np.round(np.arange(0.02, 0.3001, 0.01), 4)
ALPHAS_PI = np.round(np.concatenate([np.arange(0.02, 0.3001, 0.01),
                                     np.arange(0.35, 0.9501, 0.05)]), 4)
SEED = 20261113


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def capacities(rows, rule, key="frac_ge_0.95"):
    out = []
    for N in NS:
        for f in NOISES:
            rr = sorted([r for r in rows if r["N"] == N and r["f"] == f], key=lambda r: r["alpha"])
            a = [r["alpha"] for r in rr]
            est, lo, hi = ex.capacity_with_ci(a, [r[key] for r in rr], [r["n"] for r in rr],
                                              seed=N + int(f * 100))
            out.append({"rule": rule, "N": N, "f": f, "criterion": key,
                        "alpha_c": est, "ci16": lo, "ci84": hi})
    return out


def plot_recall(rows, rule, outdir, alpha_c_theory):
    for key, ylabel, fname in [("m_mean", "mean final overlap m", "overlap_vs_alpha"),
                               ("frac_ge_0.95", "fraction recalled (m ≥ 0.95)",
                                "recalled_fraction_vs_alpha")]:
        fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
        for ax, N in zip(axs, NS):
            for f in NOISES:
                rr = sorted([r for r in rows if r["N"] == N and r["f"] == f],
                            key=lambda r: r["alpha"])
                ax.plot([r["alpha"] for r in rr], [r[key] for r in rr], "o-", ms=3,
                        label=f"{int(f*100)}% flipped")
            ax.axvline(alpha_c_theory, color="k", ls="--", lw=1,
                       label=f"AGS α_c = {alpha_c_theory:.3f}")
            ax.set_title(f"{rule}, N = {N}")
            ax.set_xlabel("α = P/N")
            ax.grid(alpha=0.3)
        axs[0].set_ylabel(ylabel)
        axs[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{fname}_{rule}.png"), dpi=130)
        plt.close(fig)


def run_recall(rule, alphas, scale, workers, outdir, alpha_c_theory):
    t0 = time.time()
    min_trials = max(20, int(round(400 * scale)))
    res = ex.recall_sweep(rule, NS, alphas, NOISES, min_trials, SEED, workers)
    rows = ex.summarise_recall(res, NOISES)
    write_csv(os.path.join(outdir, f"recall_{rule}.csv"), rows)
    plot_recall(rows, rule, outdir, alpha_c_theory)
    caps = []
    for key in ("frac_ge_0.95", "frac_ge_0.90", "frac_ge_1.00"):
        caps += capacities(rows, rule, key)
    print(f"[{rule}] {len(res)} tasks in {time.time()-t0:.1f}s", flush=True)
    return rows, caps


def fss(caps, rule):
    """Indicative fit alpha_c(N) = a_inf + b / sqrt(N) per noise level (primary criterion)."""
    out = {}
    for f in NOISES:
        cc = [c for c in caps if c["rule"] == rule and c["f"] == f
              and c["criterion"] == "frac_ge_0.95" and np.isfinite(c["alpha_c"])]
        if len(cc) == len(NS):
            x = np.array([c["N"] for c in cc]) ** -0.5
            y = np.array([c["alpha_c"] for c in cc])
            b, a = np.polyfit(x, y, 1)
            out[str(f)] = {"alpha_inf": float(a), "slope": float(b)}
    return out


def run_phase(scale, workers, outdir):
    t0 = time.time()
    N = 1000
    alphas = np.round(np.concatenate([[0.01], np.arange(0.02, 0.3001, 0.02)]), 4)
    Ts = np.round(np.arange(0.05, 1.6001, 0.05), 4)
    if scale < 1:
        alphas, Ts = alphas[::3], Ts[::4]
    res = ex.phase_diagram(N, alphas, Ts, 2, 100, 200, SEED, workers)
    rows = []
    for r in res:
        rows.append({k: r[k] for k in ("N", "P", "alpha", "alpha_nominal", "T", "m_A", "q_A",
                                       "q_B")} | {"phase": ex.classify_phase(r["m_A"], r["q_B"])})
    write_csv(os.path.join(outdir, "phase_diagram.csv"), rows)
    # theory lines
    a_line = np.linspace(0.0005, 0.30, 120)
    tm = np.array([theory.T_M(a) for a in a_line])
    tg = theory.T_g(a_line)
    np.savetxt(os.path.join(outdir, "theory_lines.csv"),
               np.column_stack([a_line, tm, tg]), delimiter=",",
               header="alpha,T_M,T_g", comments="")
    A = np.array(sorted(set(r["alpha_nominal"] for r in rows)))
    T = np.array(sorted(set(r["T"] for r in rows)))
    M = np.full((len(T), len(A)), np.nan)
    Q = np.full_like(M, np.nan)
    C = np.full_like(M, np.nan)
    code = {"P": 0, "SG": 1, "R": 2}
    for r in rows:
        i, j = np.searchsorted(T, r["T"]), np.searchsorted(A, r["alpha_nominal"])
        M[i, j], Q[i, j], C[i, j] = r["m_A"], r["q_B"], code[r["phase"]]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, Z, title, cmap in [(axs[0], M, "⟨m⟩ from pattern start", "viridis"),
                               (axs[1], Q, "q_EA from random start", "magma"),
                               (axs[2], C, "phase: P (0) / SG (1) / R (2)", "coolwarm")]:
        im = ax.pcolormesh(_edges(A),
                           _edges(T), Z, cmap=cmap, shading="flat")
        fig.colorbar(im, ax=ax)
        ax.plot(a_line, tm, "w-" if cmap != "coolwarm" else "k-", lw=2, label="AGS T_M (RS)")
        ax.plot(a_line, tg, "w--" if cmap != "coolwarm" else "k--", lw=2,
                label="T_g = 1+√α")
        ax.set_xlabel("α = P/N")
        ax.set_ylabel("T")
        ax.set_title(f"{title}, N={N}")
        ax.set_xlim(0, A[-1] + 0.01)
        ax.set_ylim(0, T[-1] + 0.025)
    axs[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "phase_diagram.png"), dpi=130)
    plt.close(fig)
    # simulated boundaries: highest T with R, highest T with SG-or-R, per alpha
    bounds = []
    for j, a in enumerate(A):
        tr = T[C[:, j] == 2]
        tsg = T[C[:, j] >= 1]
        bounds.append({"alpha": float(a), "T_R_max_sim": float(tr.max()) if tr.size else None,
                       "T_SG_max_sim": float(tsg.max()) if tsg.size else None,
                       "T_M_theory": float(theory.T_M(a)), "T_g_theory": float(theory.T_g(a))})
    print(f"[phase] {len(res)} points in {time.time()-t0:.1f}s", flush=True)
    return bounds


def _edges(x):
    d = np.diff(x)
    mid = x[:-1] + d / 2
    return np.r_[x[0] - d[0] / 2, mid, x[-1] + d[-1] / 2]


def run_spurious(scale, workers, outdir):
    t0 = time.time()
    rows, stab = ex.spurious_experiment(
        1000, [3, 5, 7, 10, 20, 50, 100, 138], max(40, int(1000 * scale)), 4,
        [5, 10, 20, 50], [3, 5, 7], max(20, int(200 * scale)), SEED, workers)
    write_csv(os.path.join(outdir, "spurious_basins.csv"), rows)
    write_csv(os.path.join(outdir, "mixture_stability.csv"), stab)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cats = ["frac_pattern", "frac_reversed", "frac_mix3", "frac_mix5", "frac_mix7",
            "frac_mix_higher", "frac_other"]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for c in cats:
        v = np.array([r[c] for r in rows])
        ax.bar(x, v, bottom=bottom, label=c.replace("frac_", ""))
        bottom += v
    ax.set_xticks(x, [f"P={r['P']}\nα={r['alpha']:.3f}" for r in rows], fontsize=8)
    ax.set_ylabel("fraction of random starts")
    ax.set_title("Hebbian N=1000, T=0: where random initial states end up")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "spurious_states.png"), dpi=130)
    plt.close(fig)
    print(f"[spurious] done in {time.time()-t0:.1f}s", flush=True)
    return rows, stab


def plot_compare(rows_h, rows_p, caps, outdir):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, N in zip(axs, NS):
        for f, c in zip(NOISES, plt.cm.viridis(np.linspace(0, 0.9, len(NOISES)))):
            for rows, ls, lab in [(rows_h, "--", "Hebb"), (rows_p, "-", "PI")]:
                rr = sorted([r for r in rows if r["N"] == N and r["f"] == f],
                            key=lambda r: r["alpha"])
                ax.plot([r["alpha"] for r in rr], [r["frac_ge_0.95"] for r in rr], ls,
                        color=c, label=f"{lab} {int(f*100)}%")
        ax.set_title(f"N = {N}")
        ax.set_xlabel("α = P/N")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("fraction recalled (m ≥ 0.95)")
    axs[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "hebb_vs_pseudoinverse.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for rule, mk in [("hebb", "s--"), ("pinv", "o-")]:
        for N in NS:
            cc = [c for c in caps if c["rule"] == rule and c["N"] == N
                  and c["criterion"] == "frac_ge_0.95"]
            ax.errorbar([c["f"] for c in cc], [c["alpha_c"] for c in cc],
                        yerr=[[max(0, c["alpha_c"] - c["ci16"]) if np.isfinite(c["ci16"]) else 0
                               for c in cc],
                              [max(0, c["ci84"] - c["alpha_c"]) if np.isfinite(c["ci84"]) else 0
                               for c in cc]], fmt=mk, capsize=2, label=f"{rule} N={N}")
    ax.set_xlabel("fraction of bits flipped in the cue")
    ax.set_ylabel("α_c (50 % recalled, m ≥ 0.95)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capacity_vs_noise.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", default="E1,E2,E3,E4")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    only = set(a.only.split(","))
    t0 = time.time()
    ac, mc = theory.alpha_c_t0()
    summary = {"scale": a.scale, "seed": SEED,
               "theory": {"alpha_c_RS_T0": ac, "m_at_alpha_c_RS_T0": mc}}
    caps = []
    rows_h = rows_p = None
    if "E1" in only:
        rows_h, c = run_recall("hebb", ALPHAS_HEBB, a.scale, a.workers, a.out, ac)
        caps += c
        summary["hebb_fss"] = fss(caps, "hebb")
        summary["hebb_nonconverged_runs"] = int(sum(round((1 - r["conv_frac"]) * r["n"])
                                                    for r in rows_h))
    if "E4" in only:
        rows_p, c = run_recall("pinv", ALPHAS_PI, a.scale, a.workers, a.out, ac)
        caps += c
        summary["pinv_nonconverged_runs"] = int(sum(round((1 - r["conv_frac"]) * r["n"])
                                                    for r in rows_p))
    if caps:
        write_csv(os.path.join(a.out, "capacities.csv"), caps)
        summary["capacities"] = caps
    if rows_h is not None and rows_p is not None:
        plot_compare(rows_h, rows_p, caps, a.out)
    if "E2" in only:
        summary["phase_boundaries"] = run_phase(a.scale, a.workers, a.out)
    if "E3" in only:
        rows, stab = run_spurious(a.scale, a.workers, a.out)
        summary["spurious_basins"] = rows
        summary["mixture_stability"] = stab
    summary["wall_time_s"] = time.time() - t0
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print(f"total {summary['wall_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
