# -*- coding: utf-8 -*-
"""
Baseline robust regression methods compared in the EDERM paper
(Tables 3, 4, 5, 7): ERM(MSE), Huber regression, MCC, MoM and LSSVR.

All methods share the same design-matrix interface as ederm_fit:
    f = Phi @ theta,  Phi = [X, 1] (linear) or Gram matrix (kernel RKHS).
Optimization is full-batch gradient descent (matches the paper's setup,
step size gamma selected by cross-validation).
"""
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2


def _gd(Phi, y, grad_fn, iters, learning_step, tol, theta=None, record=False):
    n, d = Phi.shape
    theta = np.zeros(d) if theta is None else theta.copy()
    prev = None
    for k in range(iters):
        grad = grad_fn(theta) / n
        theta = theta - learning_step * grad
        # convergence on gradient norm (cheap and robust)
        if np.linalg.norm(grad) < tol:
            break
    return theta, k + 1


def erm_fit(Phi, y, iters=2000, learning_step=0.01, tol=1e-10):
    """ERM with squared loss."""
    y = np.asarray(y, float).ravel()
    def grad_fn(theta):
        e = y - Phi.dot(theta)
        return -2.0 * (Phi.T.dot(e))
    theta, _ = _gd(Phi, y, grad_fn, iters, learning_step, tol)
    return theta


def huber_fit(Phi, y, rho=1.0, iters=2000, learning_step=0.01, tol=1e-10):
    """Huber regression (Huber, 1973): l(e)=e^2 if |e|<rho else 2 rho |e|-rho^2."""
    y = np.asarray(y, float).ravel()
    def grad_fn(theta):
        e = y - Phi.dot(theta)
        dl = np.where(np.abs(e) < rho, -2.0 * e, -2.0 * rho * np.sign(e))
        return Phi.T.dot(dl)
    theta, _ = _gd(Phi, y, grad_fn, iters, learning_step, tol)
    return theta


def mcc_fit(Phi, y, sigma=0.5, iters=2000, learning_step=0.01, tol=1e-10):
    """Maximum Correntropy Criterion (Feng et al., 2015):
    l(e) = sigma^2 (1 - exp(-e^2/(2 sigma^2)))."""
    y = np.asarray(y, float).ravel()
    def grad_fn(theta):
        e = y - Phi.dot(theta)
        dl = -e * np.exp(-e * e / (2.0 * sigma * sigma))
        return Phi.T.dot(dl)
    theta, _ = _gd(Phi, y, grad_fn, iters, learning_step, tol)
    return theta


def mom_fit(Phi, y, blocks=3, iters=2000, learning_step=0.01, tol=1e-10, seed=None):
    """Median-of-Means (Lugosi & Mendelson, 2019; Lecue & Lerasle, 2020):
    minimize the median over B blocks of the block-averaged squared loss.
    Subgradient: mean gradient of the median block."""
    y = np.asarray(y, float).ravel()
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    parts = np.array_split(perm, blocks)

    def grad_fn(theta):
        e = y - Phi.dot(theta)
        block_means = np.array([np.mean(e[p] ** 2) for p in parts])
        b = int(np.argsort(block_means)[len(block_means) // 2])  # median block
        # sum-form gradient; _gd divides by n, giving the median-block mean gradient
        return -2.0 * Phi[parts[b]].T.dot(e[parts[b]])

    theta, _ = _gd(Phi, y, grad_fn, iters, learning_step, tol)
    return theta


def lssvr_fit(Phi, y, reg=0.01):
    """Least-squares SVR in kernel space = kernel ridge:
    alpha = (K + r I)^{-1} y.  (Phi is the Gram matrix here.)"""
    y = np.asarray(y, float).ravel()
    n = Phi.shape[0]
    alpha = np.linalg.solve(Phi + reg * n * np.eye(n), y)
    return alpha


def _safe_r2(y, pred):
    """R2 that tolerates diverged iterates (non-finite -> worst score)."""
    if not np.all(np.isfinite(pred)):
        return -1e6
    return float(r2(y, pred))


def evaluate(theta, Phi, y, Phi_tst, ytst):
    pred = Phi.dot(theta)
    pred_t = Phi_tst.dot(theta)
    finite = np.all(np.isfinite(pred)) and np.all(np.isfinite(pred_t))
    return {'train_mse': float(mse(y, pred)) if finite else 1e12,
            'test_mse': float(mse(ytst, pred_t)) if finite else 1e12,
            'test_r2': _safe_r2(ytst, pred_t),
            'train_r2': _safe_r2(y, pred)}
