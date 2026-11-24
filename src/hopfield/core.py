"""Hopfield network core: patterns, learning rules, T=0 and Glauber dynamics.

Conventions (see PLAN.md): s_i in {-1,+1}, h_i = sum_j W_ij s_j, W symmetric, W_ii = 0.
Random numbers inside numba kernels come from numba's own generator, seeded per call.
"""
import numpy as np
from numba import njit

MAX_SWEEPS = 500


def random_patterns(rng, P, N):
    """P i.i.d. unbiased +-1 patterns, shape (P, N), int8."""
    return (2 * rng.integers(0, 2, size=(P, N), dtype=np.int8) - 1).astype(np.int8)


def hebbian(xi):
    """W_ij = (1/N) sum_mu xi_i xi_j, zero diagonal."""
    x = xi.astype(np.float64)
    N = x.shape[1]
    W = x.T @ x / N
    np.fill_diagonal(W, 0.0)
    return W


def pseudo_inverse(xi, zero_diag=True):
    """Projection rule W = (1/N) Xi^T C^{-1} Xi, C = Xi Xi^T / N (Personnaz et al. 1985).

    With zero_diag=True the self-couplings are removed (Kanter & Sompolinsky 1987)."""
    x = xi.astype(np.float64)
    N = x.shape[1]
    C = x @ x.T / N
    W = x.T @ np.linalg.solve(C, x) / N
    W = 0.5 * (W + W.T)  # remove round-off asymmetry
    if zero_diag:
        np.fill_diagonal(W, 0.0)
    return W


def flip_bits(rng, pattern, n_flip):
    s = pattern.copy()
    idx = rng.choice(pattern.size, size=n_flip, replace=False)
    s[idx] = -s[idx]
    return s


@njit(cache=True, nogil=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True, nogil=True)
def _relax_t0(W, s, max_sweeps):
    """Asynchronous T=0 dynamics in random sequential order until a sweep has no flip.

    s is modified in place (float64 +-1). Returns (sweeps, converged)."""
    N = s.size
    h = W @ s
    for sweep in range(max_sweeps):
        perm = np.random.permutation(N)
        flips = 0
        for k in range(N):
            i = perm[k]
            hi = h[i]
            if hi > 0.0:
                new = 1.0
            elif hi < 0.0:
                new = -1.0
            else:
                new = s[i]
            if new != s[i]:
                d = new - s[i]
                s[i] = new
                for j in range(N):
                    h[j] += d * W[i, j]
                flips += 1
        if flips == 0:
            return sweep + 1, True
    return max_sweeps, False


def relax(W, s0, seed, max_sweeps=MAX_SWEEPS):
    """Run T=0 asynchronous dynamics from s0; returns (final state int8, sweeps, converged)."""
    _seed(seed)
    s = s0.astype(np.float64).copy()
    sweeps, conv = _relax_t0(W, s, max_sweeps)
    return s.astype(np.int8), sweeps, conv


@njit(cache=True, nogil=True)
def _glauber(W, s, beta, target, n_eq, n_meas):
    """Glauber heat-bath dynamics, random sequential order.

    Returns (mean overlap with target over measurement sweeps, mean |overlap|,
    q_EA = mean_i <s_i>^2 over measurement sweeps). s modified in place."""
    N = s.size
    h = W @ s
    ssum = np.zeros(N)
    msum = 0.0
    mabs = 0.0
    for sweep in range(n_eq + n_meas):
        perm = np.random.permutation(N)
        for k in range(N):
            i = perm[k]
            x = -2.0 * beta * h[i]
            if x > 700.0:
                p = 0.0
            else:
                p = 1.0 / (1.0 + np.exp(x))
            new = 1.0 if np.random.random() < p else -1.0
            if new != s[i]:
                d = new - s[i]
                s[i] = new
                for j in range(N):
                    h[j] += d * W[i, j]
        if sweep >= n_eq:
            m = 0.0
            for i in range(N):
                ssum[i] += s[i]
                m += target[i] * s[i]
            m /= N
            msum += m
            mabs += abs(m)
    q = 0.0
    for i in range(N):
        a = ssum[i] / n_meas
        q += a * a
    return msum / n_meas, mabs / n_meas, q / N


def glauber(W, s0, T, target, n_eq, n_meas, seed):
    _seed(seed)
    s = s0.astype(np.float64).copy()
    return _glauber(W, s, 1.0 / T, target.astype(np.float64), n_eq, n_meas)


def overlaps(xi, s):
    """m_mu = (1/N) xi^mu . s for all mu."""
    return (xi.astype(np.float64) @ s.astype(np.float64)) / xi.shape[1]
