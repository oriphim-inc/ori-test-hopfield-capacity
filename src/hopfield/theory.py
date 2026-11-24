"""Replica-symmetric theory of Amit, Gutfreund & Sompolinsky (1985/1987), solved numerically.

Nothing here is a typed-in answer: alpha_c and the T_M(alpha) line come from solving the
saddle-point equations
    m = <tanh(beta (m + sqrt(alpha r) z))>_z
    q = <tanh^2(beta (m + sqrt(alpha r) z))>_z
    r = q / (1 - beta (1 - q))^2
with z ~ N(0,1).
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf

from scipy.optimize import fsolve

# Uniform grid quadrature over z ~ N(0,1): robust at low T where tanh(beta x) is step-like
_z = np.linspace(-9.0, 9.0, 6001)
_w = np.exp(-0.5 * _z ** 2)
_w = _w / _w.sum()


def alpha_of_y_t0(y):
    """T=0 limit: with y = m / sqrt(2 alpha r), m = erf(y) and
    alpha(y) = (erf(y) - 2 y exp(-y^2)/sqrt(pi))^2 / (2 y^2)."""
    return (erf(y) - 2.0 * y * np.exp(-y * y) / np.sqrt(np.pi)) ** 2 / (2.0 * y * y)


def alpha_c_t0():
    """Largest alpha with a retrieval solution at T=0, and the overlap there."""
    res = minimize_scalar(lambda y: -alpha_of_y_t0(y), bounds=(0.5, 4.0), method="bounded",
                          options={"xatol": 1e-10})
    return float(alpha_of_y_t0(res.x)), float(erf(res.x))


def _rhs(alpha, beta, m, q):
    denom = 1.0 - beta * (1.0 - q)
    r = q / denom ** 2
    t = np.tanh(beta * (m + np.sqrt(alpha * r) * _z))
    return float(_w @ t), float(_w @ (t * t)), denom


def solve_rs(alpha, T, m0=1.0, q0=1.0, iters=3000, tol=1e-10):
    """Solve the RS equations from (m0, q0): plain fixed-point iteration, then a Newton
    polish (fsolve). Returns (m, q, ok); ok=False when no solution is reached."""
    beta = 1.0 / T
    m, q = m0, q0
    for _ in range(iters):
        if 1.0 - beta * (1.0 - q) <= 0:
            return m, q, False
        m_new, q_new, _ = _rhs(alpha, beta, m, q)
        done = abs(m_new - m) < tol and abs(q_new - q) < tol
        m, q = m_new, q_new
        if done:
            return m, q, True

    def F(v):
        mm, qq = v
        if 1.0 - beta * (1.0 - qq) <= 0 or qq <= 0:
            return [1e3, 1e3]
        a, b, _ = _rhs(alpha, beta, mm, qq)
        return [a - mm, b - qq]
    sol, info, ier, _ = fsolve(F, [m, q], full_output=True, xtol=1e-12)
    ok = ier == 1 and max(abs(np.asarray(info["fvec"]))) < 1e-9
    return float(sol[0]), float(sol[1]), bool(ok)


def T_M(alpha, m_min=1e-2, tol=1e-4):
    """Highest T at which the retrieval (m > m_min) RS solution exists, by bisection."""
    def has_retrieval(T):
        m, q, ok = solve_rs(alpha, T)
        return ok and m > m_min
    lo, hi = 0.02, 1.2
    if not has_retrieval(lo):
        return np.nan
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if has_retrieval(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def T_g(alpha):
    """Spin-glass transition of the m=0 solution: linearising q = beta^2 alpha q/(1-beta)^2
    gives (T - 1)^2 = alpha, i.e. T_g = 1 + sqrt(alpha)."""
    return 1.0 + np.sqrt(np.asarray(alpha, dtype=float))
