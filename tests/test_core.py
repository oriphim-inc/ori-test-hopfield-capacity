import numpy as np

from hopfield import core, experiments as ex, theory


def test_hebbian_symmetric_zero_diag():
    rng = np.random.default_rng(0)
    xi = core.random_patterns(rng, 5, 100)
    W = core.hebbian(xi)
    assert np.allclose(W, W.T) and np.allclose(np.diag(W), 0)
    # W_ij = (1/N) sum_mu xi_i xi_j
    assert np.isclose(W[1, 2], (xi[:, 1].astype(float) @ xi[:, 2]) / 100)


def test_single_pattern_recalled_from_noise():
    rng = np.random.default_rng(1)
    xi = core.random_patterns(rng, 1, 200)
    W = core.hebbian(xi)
    s0 = core.flip_bits(rng, xi[0], 60)
    s, sweeps, conv = core.relax(W, s0, 3)
    assert conv and np.array_equal(s, xi[0])


def test_energy_never_increases_and_fixed_point():
    rng = np.random.default_rng(2)
    xi = core.random_patterns(rng, 40, 200)
    W = core.hebbian(xi)
    s0 = (2 * rng.integers(0, 2, 200) - 1).astype(np.int8)
    s, _, conv = core.relax(W, s0, 5)
    E = lambda x: -0.5 * x.astype(float) @ W @ x.astype(float)
    assert conv and E(s) <= E(s0) + 1e-12
    h = W @ s.astype(float)
    assert np.all((h * s >= 0))  # every spin aligned with (or neutral to) its field


def test_pseudo_inverse_projection():
    rng = np.random.default_rng(3)
    xi = core.random_patterns(rng, 50, 200)
    Wd = core.pseudo_inverse(xi, zero_diag=False)
    assert np.allclose(Wd @ xi.T, xi.T, atol=1e-8)  # projector onto pattern span
    W = core.pseudo_inverse(xi)
    for mu in range(50):  # all patterns are fixed points (alpha < 1, zero diag)
        h = W @ xi[mu].astype(float)
        assert np.all(h * xi[mu] > 0)


def test_glauber_high_T_paramagnet_low_T_retrieval():
    rng = np.random.default_rng(4)
    xi = core.random_patterns(rng, 5, 500)
    W = core.hebbian(xi)
    m, _, q = core.glauber(W, xi[0], 3.0, xi[0], 50, 200, 1)
    assert abs(m) < 0.1 and q < 0.05
    m, _, q = core.glauber(W, xi[0], 0.3, xi[0], 50, 100, 1)
    assert m > 0.9


def test_glauber_single_spin_detailed_statistics():
    # one spin with field h from a fixed neighbour: <s> = tanh(h/T)
    W = np.array([[0.0, 0.7], [0.7, 0.0]])
    # two coupled spins: <s1 s2> = tanh(J/T)
    T = 1.0
    s0 = np.array([1, 1], dtype=np.int8)
    _, mabs, _ = core.glauber(W, s0, T, np.array([1, 1], dtype=np.int8), 100, 200000, 7)
    # |m| for target (1,1): m = (s1+s2)/2 in {0, +-1}; P(aligned) = (1+tanh(J/T))/2
    assert abs(mabs - (1 + np.tanh(0.7)) / 2) < 0.01


def test_theory_alpha_c():
    a, m = theory.alpha_c_t0()
    assert abs(a - 0.138) < 0.001 and abs(m - 0.967) < 0.002
    assert abs(theory.T_M(1e-4) - 1) < 0.05


def test_crossing():
    a = [0.1, 0.2, 0.3]
    assert np.isclose(ex.crossing(a, [1.0, 0.6, 0.2]), 0.225)
    assert np.isnan(ex.crossing(a, [1, 1, 1]))


def test_classify_mixture():
    rng = np.random.default_rng(5)
    xi = core.random_patterns(rng, 10, 1000)
    mix = np.sign(xi[0].astype(int) + xi[3] - xi[7]).astype(np.int8)
    assert ex.classify_state(xi, mix) == ("mixture", 3)
    assert ex.classify_state(xi, -xi[2]) == ("reversed", 1)
