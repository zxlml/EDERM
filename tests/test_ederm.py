# -*- coding: utf-8 -*-
"""Unit tests for the EDERM implementation.

Run:  pytest tests/test_ederm.py -q
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from functions.models import (ederm_fit, mode_0, mode_1, mode_2, mode_3,
                              mode_4, mode_5, mode_6, kernel_function,
                              _kde_and_grad)
from functions.indicators import indicator, indicator_vec, indicator_double
from functions.baselines import erm_fit, huber_fit, mcc_fit, mom_fit, lssvr_fit
from functions.createdata import CreateData, CreateKernelData
from sklearn.metrics import r2_score


def _design(X):
    X = np.asarray(X, float).reshape(len(X), -1)
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


# ---------------------------------------------------------------- indicators
def test_correntropy_indicator_bounds_and_derivative():
    delta = 2.0
    rho = np.linspace(0.0, 3.0, 50)
    Ic, dIc = indicator_vec('correntropy', rho, lamb=1.0, delta=delta)
    # Algorithm 1: phi_i = (1-exp{-(1-lamb+rho)^2_+/(2 sigma^2)})/(1-exp{-1/(2 sigma^2)})
    # bounded in [0, 1/(1-exp(-1/(2 sigma^2)))], increasing in rho, phi(rho=lambda)=1,
    # and saturates at 1/norm for rho >> lambda (selected set -> uniform weight).
    norm = 1.0 - np.exp(-1.0 / (2.0 * delta * delta))
    assert np.all(Ic >= 0.0) and np.all(Ic <= 1.0 / norm + 1e-12)
    assert np.all(np.diff(Ic) >= -1e-12)
    # phi(rho = lambda) == 1 exactly (evaluate off-grid)
    Ic1, _ = indicator_vec('correntropy', np.array([1.0]), 1.0, delta)
    assert Ic1[0] == pytest.approx(1.0, rel=1e-6)
    # saturation: phi -> 1/norm for rho >> lambda
    IcS, _ = indicator_vec('correntropy', np.array([50.0]), 1.0, delta)
    assert IcS[0] == pytest.approx(1.0 / norm, rel=1e-3)
    # zero when 1 - lamb + rho <= 0
    Ic0, _ = indicator_vec('correntropy', np.array([0.0]), lamb=1.5, delta=delta)
    assert Ic0[0] == pytest.approx(0.0, abs=1e-12)
    # finite-difference check of dIc/drho
    eps = 1e-6
    for r in [0.5, 1.2, 2.0]:
        Ip, _ = indicator_vec('correntropy', np.array([r + eps]), 1.0, delta)
        Im, _ = indicator_vec('correntropy', np.array([r - eps]), 1.0, delta)
        fd = (Ip[0] - Im[0]) / (2 * eps)
        _, da = indicator_vec('correntropy', np.array([r]), 1.0, delta)
        assert da[0] == pytest.approx(fd, rel=1e-4)


def test_correntropy_normalization():
    # phi saturates (selected set gets uniform weight) when rho >> lambda
    norm = 1.0 - np.exp(-1.0 / 8.0)
    Ic, _ = indicator_vec('correntropy', np.array([50.0]), 1.0, 2.0)
    assert Ic[0] == pytest.approx(1.0 / norm, rel=1e-6)


def test_double_threshold_indicator():
    # Paper Remark 1: I = phi(lamb1 - rho) - phi(lamb2 - rho).
    # The correntropy surrogate is a SOFT band: it activates around rho > lamb1-1,
    # equals 1 exactly at rho = lamb1, and both terms saturate at 1/norm for
    # rho >> lamb2 so the difference vanishes again.
    rho = np.array([0.0, 1.0, 2.0, 12.0])
    Ic, dIc = indicator_double('correntropy', rho, 1.0, 3.0, 2.0)
    assert Ic[0] == pytest.approx(0.0, abs=1e-9)    # rho < lamb1 - 1: both off
    assert Ic[1] == pytest.approx(1.0, abs=1e-6)    # rho = lamb1: phi1=1, phi2=0
    assert Ic[2] > 1.0                              # inside the band, above phi1 peak
    assert Ic[3] == pytest.approx(0.0, abs=1e-4)    # rho >> lamb2: both saturated
    assert np.all(np.isfinite(dIc))


# ---------------------------------------------------------------- KDE / grad
def test_kde_matches_paper_definition():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 40)
    Phi = _design(rng.normal(0, 1, (40, 1)))
    h = 1.0
    rho, drho = _kde_and_grad(e, Phi, h)
    # manual check of Eq.(3) for sample 0
    n = len(e)
    manual = sum(np.exp(-(e[0] - e[j]) ** 2 / (2 * h * h)) for j in range(n)) / (n * h)
    assert rho[0] == pytest.approx(manual, rel=1e-10)


def test_kde_gradient_finite_difference():
    """Check d rho_i / d theta under the paper's mean-shift selection
    dynamics (Algorithm 1): only e_i is differentiated; the other residuals
    e_j are treated as fixed."""
    rng = np.random.default_rng(1)
    n = 15
    X = rng.normal(0, 1, (n, 1))
    y = 3 * X[:, 0] + rng.normal(0, 0.5, n)
    Phi = _design(X)
    h = 0.8
    theta0 = np.array([2.0, 0.5])
    e0 = y - Phi.dot(theta0)

    _, drho = _kde_and_grad(e0, Phi, h)
    eps = 1e-7
    for i in [0, 5, 14]:
        for d in range(2):
            # finite difference of rho_i w.r.t. theta_d with e_j (j != i) fixed
            def rho_i_partial(t):
                e = e0.copy()
                e[i] = y[i] - Phi[i].dot(t)
                return np.exp(-(e[i] - e) ** 2 / (2 * h * h)).sum() / (n * h)
            tp = theta0.copy(); tp[d] += eps
            tm = theta0.copy(); tm[d] -= eps
            fd = (rho_i_partial(tp) - rho_i_partial(tm)) / (2 * eps)
            assert drho[i, d] == pytest.approx(fd, rel=1e-3, abs=1e-8)


def test_ederm_objective_gradient_finite_difference():
    """Gradient check: the Algorithm-1 update direction equals the exact
    gradient of the mean-shift surrogate objective
        J_ms(theta) = (1/n) sum_i ell_i(theta) * phi(lambda - rho_i^ms(theta)),
    where rho_i^ms keeps the *other* residuals fixed at theta0 (this is the
    objective whose stationary points the paper's dynamics seek)."""
    rng = np.random.default_rng(2)
    n = 20
    X = rng.normal(0, 1, (n, 1))
    y = 2 * X[:, 0] + rng.normal(0, 1, n)
    Phi = _design(X)
    h, sigma, lamb = 1.0, 2.0, 1.0
    theta0 = np.array([1.5, -0.3])
    e0 = y - Phi.dot(theta0)

    def ms_objective(theta):
        e = y - Phi.dot(theta)
        ell = e ** 2
        # rho_i uses e_i(theta) but e_j (j != i) fixed at theta0
        D = e[:, None] - e0[None, :]
        rho = np.exp(-D * D / (2 * h * h)).sum(1) / (n * h)
        Ic, _ = indicator_vec('correntropy', rho, lamb, sigma)
        return np.mean(ell * Ic)

    from functions.models import _kde_and_grad
    e = e0
    rho, drho = _kde_and_grad(e, Phi, h)
    Ic, dIc = indicator_vec('correntropy', rho, lamb, sigma)
    grad = (-2.0 * (Phi.T.dot(e * Ic)) + (e * e * dIc).dot(drho)) / n

    eps = 1e-7
    for d in range(2):
        tp = theta0.copy(); tp[d] += eps
        tm = theta0.copy(); tm[d] -= eps
        fd = (ms_objective(tp) - ms_objective(tm)) / (2 * eps)
        assert grad[d] == pytest.approx(fd, rel=1e-3, abs=1e-8)


# ---------------------------------------------------------------- data
def test_createdata_shapes_and_outliers():
    for case in [1, 2, 3, 4, 5, 6, 7]:
        X, y, Xtst, ytst = CreateData(case, 'gaussian')
        assert len(X) == len(y) and len(Xtst) == len(ytst)
        assert X.ndim == 2


def test_createkerneldata_outlier_mechanism():
    Xtr, ytr, Xte, yte = CreateKernelData('f1', 100, 100, 'gaussian',
                                          outlier_ratio=0.1, seed=0)
    assert Xtr.shape == (100, 1) and Xte.shape == (100, 1)
    clean, _, _, _ = CreateKernelData('f1', 100, 100, 'gaussian',
                                      outlier_ratio=0.0, seed=0)
    assert np.max(np.abs(ytr - clean)) > 15      # outliers injected (f1: +N(20,1))
    # student noise option available
    X2, y2, _, _ = CreateKernelData('f2', 50, 50, 'student', seed=0)
    assert len(y2) == 50


# ---------------------------------------------------------------- convergence
def test_ederm_recovers_linear_on_clean_data():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (300, 1))
    y = 5 * X[:, 0] + rng.normal(0, 1, 300)
    Phi = _design(X)
    out = ederm_fit(Phi, y, lamb=1.0, Ictype='correntropy', losstype='mse',
                    h=4.0, delta=2.0, iters=1000, learning_step=0.002,
                    optimizer='fgd', seed=0)
    assert out['theta'][0] == pytest.approx(5.0, abs=0.2)
    assert out['theta'][1] == pytest.approx(0.0, abs=0.2)


def test_ederm_robust_vs_erm_under_outliers():
    """Paper Table 7, Type 3 scenario: ERM breaks, EDERM stays accurate."""
    rng = np.random.default_rng(4)
    n, m = 100, 30
    X = rng.uniform(-5, 5, n)
    y = 5 * X + rng.normal(0, 1, n)
    out_idx = rng.choice(n, m, replace=False)
    y[out_idx] = -10 * X[out_idx] + rng.normal(3, 1, m)   # Type 3 outliers

    Phi = _design(X)
    Xt = rng.uniform(-5, 5, 200)
    yt = 5 * Xt + rng.normal(0, 1, 200)
    Phi_t = _design(Xt)

    erm_theta = erm_fit(Phi, y, iters=5000, learning_step=0.01)
    erm_r2 = r2_score(yt, Phi_t.dot(erm_theta))
    ederm = ederm_fit(Phi, y, Phi_t, yt, lamb=1.0, Ictype='correntropy',
                      losstype='mse', h=4.0, delta=2.0, iters=2000,
                      learning_step=0.01, seed=0)
    assert erm_r2 < 0.5                             # ERM corrupted
    assert ederm['test_r2'] > 0.9                   # EDERM robust (paper: 0.9896)


def test_ederm_cc_better_than_erm_type3():
    rng = np.random.default_rng(100)
    X = rng.uniform(-5, 5, 100)
    y = 5 * X + rng.normal(0, 1, 100)
    out_idx = rng.choice(100, 30, replace=False)
    y[out_idx] = -10 * X[out_idx] + rng.normal(3, 1, 30)
    Phi = _design(X)
    Xt = rng.uniform(-5, 5, 200)
    yt = 5 * Xt + rng.normal(0, 1, 200)
    Phi_t = _design(Xt)
    cc = ederm_fit(Phi, y, Phi_t, yt, lamb=0.9, Ictype='correntropy',
                   losstype='closs', h=4.0, delta=2.0, iters=2000,
                   learning_step=0.01, seed=0)
    assert cc['test_r2'] > 0.9                      # paper: 0.9940


# ---------------------------------------------------------------- modes run
def test_all_modes_run_smoke():
    X, y, Xtst, ytst = CreateData(3, 'gaussian')
    kw = dict(plot=False, return_metrics=True, seed=0)
    r0 = mode_0(X, y, Xtst, ytst, iters=3000, learning_step=1e-5,
                online=False, **kw)
    r1 = mode_1(X, y, Xtst, ytst, iters=300, lamb=1.0, h=1.0, delta=2.0,
                learning_step=0.001, online=False, **kw)
    r2o = mode_2(X, y, Xtst, ytst, iters=300, lamb=1.0, lamb2=3.0, h=1.0,
                 delta=2.0, learning_step=0.001, online=False, **kw)
    r3 = mode_3(X, y, Xtst, ytst, iters=300, lamb=1.0, learning_step=0.001,
                online=False, **kw)
    r4 = mode_4(X, y, Xtst, ytst, iters=300, lamb=1.0, lamb2=3.0,
                learning_step=0.001, online=False, **kw)
    r5 = mode_5(X[:100], y[:100], Xtst[:100], ytst[:100], dimensions=4,
                iters=2000, learning_step=1e-5, online=False, **kw)
    r6 = mode_6(X[:60], y[:60], Xtst[:60], ytst[:60], delta_kernel=0.8,
                iters=200, learning_step=0.0002, online=False, **kw)
    for r in [r0, r1, r2o, r3, r4, r5, r6]:
        assert np.isfinite(r['test_mse'])


def test_all_optimizers_run():
    rng = np.random.default_rng(6)
    X = rng.normal(0, 1, (80, 1))
    y = 3 * X[:, 0] + rng.normal(0, 1, 80)
    Phi = _design(X)
    for opt in ['fgd', 'sgd', 'adaGrad', 'adam', 'amsgd', 'rmsprop', 'NAG']:
        out = ederm_fit(Phi, y, lamb=0.9, h=1.0, delta=2.0, iters=500,
                        learning_step=0.002, optimizer=opt, batchsize=16,
                        tol=0.0, seed=0)
        assert np.all(np.isfinite(out['theta'])), opt
    # adaGrad should still recover the slope reasonably
    out = ederm_fit(Phi, y, lamb=0.9, h=1.0, delta=2.0, iters=2000,
                    learning_step=0.05, optimizer='adaGrad', batchsize=32,
                    tol=0.0, seed=0)
    assert abs(out['theta'][0] - 3.0) < 0.5


def test_kernel_mode6_no_broadcast_crash():
    """Regression test for the original (n,) - (n,1) broadcasting bug."""
    rng = np.random.default_rng(7)
    X = rng.uniform(-4, 4, (50, 1))
    y = np.sinc(X[:, 0] / np.pi) + 0.1 * rng.normal(0, 1, 50)
    Xt = np.linspace(-4, 4, 40).reshape(-1, 1)
    yt = np.sinc(Xt[:, 0] / np.pi)
    out = mode_6(X, y, Xt, yt, delta_kernel=0.8, iters=2000, lamb=1.0,
                 h=1.0, learning_step=0.0002, return_metrics=True, seed=0,
                 plot=False)
    assert np.isfinite(out['test_mse'])
    # sinc is smooth: R2 should be decently positive even with outliers absent
    assert out['test_r2'] > 0.3


# ---------------------------------------------------------------- baselines
def test_baselines_recover_linear():
    rng = np.random.default_rng(8)
    X = rng.normal(0, 1, (200, 1))
    y = 2 * X[:, 0] + rng.normal(0, 0.5, 200)
    Phi = _design(X)
    for f, kw in [(erm_fit, {}), (huber_fit, dict(rho=1.0)),
                  (mcc_fit, dict(sigma=0.5)), (mom_fit, dict(blocks=3, seed=0))]:
        th = f(Phi, y, learning_step=0.02, iters=4000, **kw)
        assert th[0] == pytest.approx(2.0, abs=0.15), f.__name__


def test_lssvr_kernel_ridge():
    rng = np.random.default_rng(9)
    X = rng.uniform(-3, 3, (60, 1))
    y = np.sin(X[:, 0])
    Xq = rng.uniform(-3, 3, (30, 1))
    G = kernel_function('gaussian', 0.5, X, X)      # (60, 60)
    Gq = kernel_function('gaussian', 0.5, X, Xq)    # (30, 60)
    alpha = lssvr_fit(G, y, reg=0.01)
    pred = Gq.dot(alpha)
    assert np.all(np.isfinite(alpha))
    # kernel ridge interpolates a smooth function well
    assert np.mean((pred - np.sin(Xq[:, 0])) ** 2) < 0.1


# ---------------------------------------------------------------- extended
def test_01_indicator_mode3_type3():
    rng = np.random.default_rng(10)
    X = rng.uniform(-5, 5, 100)
    y = 5 * X + rng.normal(0, 1, 100)
    out_idx = rng.choice(100, 30, replace=False)
    y[out_idx] = -10 * X[out_idx] + rng.normal(3, 1, 30)
    Phi = _design(X)
    out = ederm_fit(Phi, y, lamb=1.0, indicator01=True, h=1.0, iters=300,
                    learning_step=0.005, seed=0)
    assert np.isfinite(out['train_mse'])
