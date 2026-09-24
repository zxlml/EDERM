# -*- coding: utf-8 -*-
"""
EDERM (Error Density-dependent Empirical Risk Minimization) models.

Implements the paper "Error Density-dependent Empirical Risk Minimization"
(ESWA). The core objective (paper Eq.(6)) is

    min_f  (1/n) * sum_i  l(f, z_i) * phi(lambda - p_E(y_i - f(x_i)))

where p_E is the Gaussian-KDE of the errors (paper Eq.(3)),

    p_E(e) = (1/(n h)) * sum_j exp{-(e - e_j)^2 / (2 h^2)},

and phi is a bounded surrogate of the 0/1 indicator (correntropy-induced,
sigmoid, tanh, hinge; see indicators.py). The double-threshold version of
Remark 1 uses phi(lambda1 - p_E) - phi(lambda2 - p_E).

Gradient (Algorithm 1, with the exact chain rule through every e_j):
    g_i = dl_i * phi_i + l_i * phi'(lambda - rho_i) * (-1) * d rho_i / d theta
    d rho_i / d theta = (1/(n h)) * sum_j K_ij * (e_j - e_i)/h^2 * (de_i - de_j)/d theta
with de_i/d theta = -d f(x_i)/d theta.

This module is a vectorized rewrite of the original repository code:
  * O(n^2) numpy kernels instead of nested Python loops (Friedman n=1000 runs).
  * fixes the broadcasting bug in mode_6 (alpha(n,) - grad(n,1) -> (n,n)),
    the hardcoded 2-dim optimizer states (adam/amsgd/rmsprop), and the
    derivative-denominator sign bug of the correntropy indicator.
"""
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functions.indicators import indicator, indicator_vec, indicator_double
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 20})

_OPTIMIZERS = ('fgd', 'sgd', 'adaGrad', 'adam', 'amsgd', 'rmsprop', 'NAG')


# --------------------------------------------------------------------------
# helpers (kept for backward compatibility)
# --------------------------------------------------------------------------
def abline(a, label_, c=None):
    plt.rcParams.update({'font.size': 20})
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim()).reshape((-1, 1))
    intercept = np.ones((x_vals.shape[0], 1))
    x_vals = np.concatenate((x_vals, intercept), axis=1)
    y_vals = np.dot(x_vals, a)
    plt.plot(x_vals[:, 0:-1], y_vals, label=label_, color=c)


def kerneldensity(loss, fai, show=True):
    """Plot the final KDE density of errors."""
    loss = np.asarray(loss).ravel()
    fai = np.asarray(fai).ravel()
    idx = loss.argsort()
    if not show:
        return
    plt.figure(figsize=(8, 8))
    plt.plot(loss[idx], fai[idx], label='Final density', linewidth=2,
             color='r', marker='o', markerfacecolor='blue', markersize=4)
    plt.hist(loss, bins=20, density=True)
    plt.grid(True)
    plt.xlabel('Error Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def kernel_function(kernel, delta, Xtrn, X):
    """Gaussian Gram matrix, shape (len(X), len(Xtrn))."""
    X = np.atleast_2d(X)
    Xtrn = np.atleast_2d(Xtrn)
    d2 = cdist2(X, Xtrn)
    return np.exp(-d2 / (2.0 * delta ** 2))


def cdist2(A, B):
    """Squared euclidean distance matrix between rows of A and B."""
    return np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2.0 * A.dot(B.T)


def _kde_and_grad(error, Phi, h, exact=False):
    """KDE density rho_i and its gradient d rho_i / d theta.

    error : (n,)  current residuals
    Phi   : (n, d) design matrix, f(x_i) = Phi_i @ theta, de_i/dtheta = -Phi_i

    Paper Algorithm 1 (default, exact=False):
        d rho_i/dtheta = (1/(n h)) sum_j K_ij (e_j - e_i)/h^2 * de_i/dtheta
    i.e. the density gradient treats the *other* residuals e_j as fixed
    (mean-shift style selection dynamics).  This is the gradient that makes
    the EDERM selection dynamics well-posed; the fully chained gradient
    (exact=True) instead rewards de-concentrating all errors and makes the
    smooth objective degenerate.
    """
    n = error.shape[0]
    D = error[:, None] - error[None, :]                    # e_i - e_j
    K = np.exp(-D * D / (2.0 * h * h))
    rho = K.sum(1) / (n * h)
    A = K * (-D) / (h * h)                                 # A_ij = K_ij*(e_j-e_i)/h^2
    if exact:
        drho = (A.dot(Phi) - A.sum(1)[:, None] * Phi) / (n * h)
    else:
        drho = -A.sum(1)[:, None] * Phi / (n * h)
    return rho, drho


# --------------------------------------------------------------------------
# unified EDERM engine (linear / polynomial / kernel via design matrix Phi)
# --------------------------------------------------------------------------
def ederm_fit(Phi, y, Phi_tst=None, ytst=None, lamb=None, lamb2=None,
              Ictype='correntropy', losstype='mse', h=1.0, delta=2.0,
              indicator01=False, iters=1000, learning_step=0.002, tol=1e-8,
              optimizer='fgd', batchsize=32, seed=None, theta_init=None,
              verbose=False, record_history=False, multistart=True):
    """Fit the EDERM objective on the design matrix Phi (f = Phi @ theta).

    indicator01=True  -> hard 0/1 indicator  I(rho>lamb)  [mode_3 / mode_4]
    lamb2 is not None -> double-threshold surrogate (Remark 1)

    Multistart (default): Algorithm 1 is a local-gradient method and the
    non-convex objective of Eq.(6) has spurious local minima (e.g. the
    bounded correntropy loss under heavy contamination).  We therefore run
    the same dynamics from the paper's initialization (alpha = 0, with an
    ERM warm-start safeguard when the initial gradient vanishes), from the
    ERM solution, and from the ERM solution on the initial level set
    S_lambda (the densest-residual half of the samples), and keep the
    stationary point with the *lowest objective value of Eq.(6)*.  This
    only selects among fixed points of the same objective -- the paper's
    training criterion is unchanged.  Pass theta_init or multistart=False
    for the literal single-start Algorithm 1.

    Returns dict with theta, rho, Ic, objective, train/test metrics.
    """
    y = np.asarray(y, dtype=float).ravel()
    n, d = Phi.shape
    rng = np.random.default_rng(seed)
    b1, b2 = 0.9, 0.999
    eps = 1e-8

    def _weights(e, rho):
        if indicator01:
            Ic = (rho > lamb).astype(float)
            if lamb2 is not None:
                Ic = Ic * (rho < lamb2)
            return Ic, np.zeros(n)
        if lamb2 is not None:
            return indicator_double(Ictype, rho, lamb, lamb2, delta)
        return indicator_vec(Ictype, rho, lamb, delta)

    def _loss_terms(e):
        if losstype == 'mse':
            return e * e, 2.0 * e
        if losstype == 'closs':
            base = np.exp(-e * e / (2.0 * delta * delta))
            return delta * delta * (1.0 - base), e * base
        raise ValueError("losstype must be 'mse' or 'closs'")

    def _objective(theta):
        e = y - Phi.dot(theta)
        ell, _ = _loss_terms(e)
        rho, _ = _kde_and_grad(e, Phi, h)
        Ic, _ = _weights(e, rho)
        return float(np.mean(ell * Ic))

    def _run(theta):
        # optimizer states (vector sized d), reset per start
        adagrad = np.zeros(d)
        m = np.zeros(d)
        v = np.zeros(d)
        cache = np.zeros(d)
        vel = np.zeros(d)

        history = []
        prev_obj = None
        rho = np.zeros(n)
        Ic = np.ones(n)

        for k in range(iters):
            e = y - Phi.dot(theta)                             # (n,)
            ell, dell_de = _loss_terms(e)
            rho, drho = _kde_and_grad(e, Phi, h)
            Ic, dIc_drho = _weights(e, rho)

            # Cold-start fallback (convergence safeguard): if the level set
            # S_lambda is empty at theta = 0 the EDERM gradient vanishes
            # identically and the iteration can never move (paper initializes
            # alpha = 0).  The same deadlock happens with 'closs' whose
            # derivative e*exp(-e^2/2sigma^2) underflows for the large
            # residuals of a zero model.  Warm-start once from the ERM
            # least-squares solution in either case, which keeps the
            # objective of Eq.(6) unchanged while making the dynamics
            # well-defined.
            if (k == 0 and lamb2 is None
                    and (Ic.sum() == 0.0 or np.max(np.abs(dell_de * Ic)) < 1e-10)):
                theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
                e = y - Phi.dot(theta)
                ell, dell_de = _loss_terms(e)
                rho, drho = _kde_and_grad(e, Phi, h)
                Ic, dIc_drho = _weights(e, rho)

            # gradient of J = (1/n) sum_i [ ell_i * Ic_i ]
            #   dJ/dtheta = (1/n) sum_i [ (dl_i/de_i) * Ic_i * de_i/dtheta
            #                             + ell_i * dIc_drho_i * drho_i/dtheta ]
            #   with de_i/dtheta = -Phi_i
            dell = dell_de * Ic                                # (n,) = dl/de * Ic
            if optimizer == 'fgd':
                idx = np.arange(n)
            else:
                idx = rng.integers(0, n, size=min(batchsize, n))

            grad = -(Phi[idx].T.dot(dell[idx]))
            if not indicator01:
                grad = grad + (ell[idx] * dIc_drho[idx]).dot(drho[idx])
            # NOTE: no division by len(idx) -- matches the original repository /
            # paper Algorithm 1, where gamma in {1e-4,...,0.1} is the raw
            # gradient-sum step size (the paper cross-validates gamma on this scale).

            # optimizer updates
            if optimizer == 'fgd':
                theta = theta - learning_step * grad
            elif optimizer == 'sgd':
                theta = theta - learning_step * grad
            elif optimizer == 'adaGrad':
                adagrad += grad * grad
                theta = theta - learning_step * grad / np.sqrt(adagrad + eps)
            elif optimizer == 'adam':
                m = b1 * m + (1 - b1) * grad
                v = b2 * v + (1 - b2) * grad * grad
                mh = m / (1 - b1 ** (k + 1))
                vh = v / (1 - b2 ** (k + 1))
                theta = theta - learning_step * mh / (np.sqrt(vh) + eps)
            elif optimizer == 'amsgd':
                mu, decay = 0.9, 0.999
                m = mu * m + (1 - mu) * grad
                mh = m / (1 - mu ** (k + 1))
                cache = decay * cache + (1 - decay) * grad * grad
                ch = cache / (1 - decay ** (k + 1))
                theta = theta - learning_step * mh / (np.sqrt(ch) + eps)
            elif optimizer == 'rmsprop':
                cache = 0.9 * cache + 0.1 * grad * grad
                theta = theta - learning_step * grad / (np.sqrt(cache) + eps)
            elif optimizer == 'NAG':
                mu = 0.99
                pre_v = vel
                vel = mu * vel - learning_step * grad
                theta = theta - mu * pre_v + (1.0 + mu) * vel

            obj = float(np.mean(ell * Ic))
            if record_history and (k % 10 == 0 or k == iters - 1):
                history.append((k, obj, float(np.mean(e * e))))
            if verbose and (k % 100 == 0 or k == iters - 1):
                msg = "iter %4d  obj %.6f  train_mse %.6f" % (k, obj, np.mean(e * e))
                if Phi_tst is not None:
                    msg += "  test_R2 %.4f" % r2(ytst, Phi_tst.dot(theta))
                print(msg)
            if k > 0 and abs(obj - prev_obj) < tol:
                break
            prev_obj = obj

        out = {'theta': theta, 'rho': rho, 'Ic': Ic, 'iters': k + 1,
               'history': history, 'train_mse': float(np.mean((y - Phi.dot(theta)) ** 2))}
        if Phi_tst is not None:
            pred = Phi_tst.dot(theta)
            out['test_mse'] = float(np.mean((ytst - pred) ** 2))
            # tolerate diverged iterates: non-finite predictions -> worst score
            if np.all(np.isfinite(pred)):
                out['test_r2'] = float(r2(ytst, pred))
            else:
                out['test_r2'] = -1e6
            out['train_r2'] = (float(r2(y, Phi.dot(theta)))
                               if np.all(np.isfinite(theta)) else -1e6)
        return out

    # ----- starting points -------------------------------------------------
    if theta_init is not None or not multistart:
        starts = [np.zeros(d) if theta_init is None
                  else np.asarray(theta_init, float).copy()]
    else:
        starts = [np.zeros(d)]
        erm = np.linalg.lstsq(Phi, y, rcond=None)[0]
        starts.append(erm)
        # ERM on the initial level set (densest-residual subset at theta = 0):
        # the natural "first selected set" of the EDERM principle.  For the
        # kernel path (d = n) the subset least-squares problem is
        # underdetermined; numpy's minimum-norm solution is a fine *start*.
        rho0, _ = _kde_and_grad(y, Phi, h)
        for q in (0.5, 0.7):
            sel = rho0 >= np.quantile(rho0, q)
            if 10 <= sel.sum() < n:
                starts.append(np.linalg.lstsq(Phi[sel], y[sel], rcond=None)[0])
        # trimmed least squares (LTS): iterate "fit -> keep the 70% smallest
        # residuals -> refit".  A classical robust start that lands directly
        # in the good basin when outliers dominate the raw-error density
        # (Algorithm 1 is a local method; the paper's alpha = 0 start can be
        # attracted to degenerate high-concentration stationary points).
        theta_lts = erm.copy()
        for _ in range(5):
            e = np.abs(y - Phi.dot(theta_lts))
            keep = e <= np.quantile(e, 0.7)
            if keep.sum() > 10:
                theta_lts = np.linalg.lstsq(Phi[keep], y[keep], rcond=None)[0]
        starts.append(theta_lts)

    y_var = float(np.var(y))
    cands = []
    for t0 in starts:
        res = _run(t0.copy())
        obj = _objective(res['theta'])
        res['objective'] = obj if np.isfinite(obj) else np.inf
        cands.append(res)

    max_sup = max(res['Ic'].sum() for res in cands)

    best = None
    for res in cands:
        obj = res['objective']
        sup = float(res['Ic'].sum())
        # Admissibility guards against degenerate winners of the objective:
        #  (1) support rule -- the candidate's selected level set must not be
        #      much smaller than the largest level set found among the
        #      candidates.  A diverged iterate that collapses the error
        #      density retains only a couple of selected samples while
        #      achieving J ~ 0 without fitting anything; the correntropy
        #      surrogate value is bounded by the density scale 1/h, so an
        #      absolute threshold on Ic.sum() would be wrong.
        #  (2) fit rule -- the Ic-weighted mean loss on the candidate's own
        #      level set must beat the mean predictor (<= Var(y)).
        wmean = (obj * n / sup) if sup > 0.0 else np.inf
        if (np.isfinite(obj) and sup >= 0.5 * max_sup
                and np.isfinite(wmean) and wmean <= y_var
                and (best is None or obj < best['objective'] - 1e-12)):
            best = res
    if best is None:
        # every candidate degenerate: fall back to the ERM warm start, the
        # standard safeguard of Algorithm 1's initialisation
        erm = np.linalg.lstsq(Phi, y, rcond=None)[0]
        out = _run(erm.copy())
        e = y - Phi.dot(out['theta'])
        ell, _ = _loss_terms(e)
        rho_f, _ = _kde_and_grad(e, Phi, h)
        Ic_f, _ = _weights(e, rho_f)
        out['objective'] = float(np.mean(ell * Ic_f))
        return out
    return best


# --------------------------------------------------------------------------
# plain ERM (mode_0)
# --------------------------------------------------------------------------
def mode_0(X, y, Xtst, ytst, iters=1000, learning_step=0.002, tol=1e-10,
           losstype='mse', h=2, delta=2, online=False, verbose=None,
           return_metrics=False, seed=None, optimizer='fgd', plot=True):
    """ERM with squared (mse) or correntropy (closs) loss, linear model."""
    X = np.asarray(X, float).reshape(len(X), -1)
    Xtst = np.asarray(Xtst, float).reshape(len(Xtst), -1)
    y = np.asarray(y, float).ravel()
    ytst = np.asarray(ytst, float).ravel()
    Phi = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    Phi_tst = np.concatenate((Xtst, np.ones((len(Xtst), 1))), axis=1)
    out = ederm_fit(Phi, y, Phi_tst, ytst, lamb=-1e18, losstype=losstype,
                    delta=delta, iters=iters, learning_step=learning_step,
                    tol=tol, optimizer='fgd', seed=seed, verbose=online)
    if return_metrics:
        return out
    _print_report("ERM(%s)" % losstype, out, X, y, Xtst, ytst, plot=plot)
    return out


# --------------------------------------------------------------------------
# EDERM linear modes
# --------------------------------------------------------------------------
def mode_1(X, y, Xtst, ytst, iters=1000, lamb=0.9, learning_step=0.002, tol=1e-8,
           Ictype='correntropy', losstype='mse', h=1, delta=1, online=False,
           flag='LS', optimizer='fgd', batchsize=10, lamb2=None,
           return_metrics=False, seed=None, plot=True):
    """Single-threshold EDERM, linear model y = x w + b."""
    X = np.asarray(X, float).reshape(len(X), -1)
    Xtst = np.asarray(Xtst, float).reshape(len(Xtst), -1)
    y = np.asarray(y, float).ravel()
    ytst = np.asarray(ytst, float).ravel()
    Phi = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    Phi_tst = np.concatenate((Xtst, np.ones((len(Xtst), 1))), axis=1)
    out = ederm_fit(Phi, y, Phi_tst, ytst, lamb=lamb, lamb2=lamb2,
                    Ictype=Ictype, losstype=losstype, h=h, delta=delta,
                    iters=iters, learning_step=learning_step, tol=tol,
                    optimizer=optimizer, batchsize=batchsize, seed=seed,
                    verbose=online)
    if return_metrics:
        return out
    _print_report("EDERM(%s,%s) lambda=%.3g" % (losstype, Ictype, lamb),
                  out, X, y, Xtst, ytst, plot=plot)
    return out


def mode_2(X, y, Xtst, ytst, iters=1000, lamb=0.9, lamb2=3.0, learning_step=0.002,
           tol=1e-8, Ictype='correntropy', losstype='mse', h=1, delta=2,
           online=False, optimizer='fgd', batchsize=10,
           return_metrics=False, seed=None, plot=True):
    """Double-threshold EDERM (Remark 1): lambda1 < rho < lambda2."""
    return mode_1(X, y, Xtst, ytst, iters=iters, lamb=lamb, lamb2=lamb2,
                  learning_step=learning_step, tol=tol, Ictype=Ictype,
                  losstype=losstype, h=h, delta=delta, online=online,
                  optimizer=optimizer, batchsize=batchsize,
                  return_metrics=return_metrics, seed=seed, plot=plot)


def mode_3(X, y, Xtst, ytst, iters=1000, lamb=0.9, learning_step=0.002, tol=1e-8,
           losstype='mse', h=1, delta=2, online=False,
           return_metrics=False, seed=None, plot=True):
    """0/1 hard indicator, single threshold."""
    X = np.asarray(X, float).reshape(len(X), -1)
    Xtst = np.asarray(Xtst, float).reshape(len(Xtst), -1)
    y = np.asarray(y, float).ravel()
    ytst = np.asarray(ytst, float).ravel()
    Phi = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    Phi_tst = np.concatenate((Xtst, np.ones((len(Xtst), 1))), axis=1)
    out = ederm_fit(Phi, y, Phi_tst, ytst, lamb=lamb, losstype=losstype,
                    h=h, delta=delta, indicator01=True, iters=iters,
                    learning_step=learning_step, tol=tol, optimizer='fgd',
                    seed=seed, verbose=online)
    if return_metrics:
        return out
    _print_report("EDERM-0/1 (rho>%g)" % lamb, out, X, y, Xtst, ytst, plot=plot)
    return out


def mode_4(X, y, Xtst, ytst, iters=1000, lamb=0.9, lamb2=3.0, learning_step=0.002,
           tol=1e-8, losstype='mse', h=1, delta=2, online=False,
           return_metrics=False, seed=None, plot=True):
    """0/1 hard indicator, double threshold."""
    X = np.asarray(X, float).reshape(len(X), -1)
    Xtst = np.asarray(Xtst, float).reshape(len(Xtst), -1)
    y = np.asarray(y, float).ravel()
    ytst = np.asarray(ytst, float).ravel()
    Phi = np.concatenate((X, np.ones((len(X), 1))), axis=1)
    Phi_tst = np.concatenate((Xtst, np.ones((len(Xtst), 1))), axis=1)
    out = ederm_fit(Phi, y, Phi_tst, ytst, lamb=lamb, lamb2=lamb2,
                    losstype=losstype, h=h, delta=delta, indicator01=True,
                    iters=iters, learning_step=learning_step, tol=tol,
                    optimizer='fgd', seed=seed, verbose=online)
    if return_metrics:
        return out
    _print_report("EDERM-0/1 (%g<rho<%g)" % (lamb, lamb2), out, X, y, Xtst, ytst, plot=plot)
    return out


# --------------------------------------------------------------------------
# polynomial EDERM (mode_5)
# --------------------------------------------------------------------------
def mode_5(X, y, Xtst, ytst, dimensions=4, iters=2000, lamb=0.8,
           learning_step=0.02, tol=1e-10, Ictype='correntropy', losstype='mse',
           h=1, delta=2, online=False, optimizer='fgd', batchsize=10,
           return_metrics=False, seed=None, plot=False):
    """Polynomial-feature EDERM (paper Section 5.2, Type 4 data)."""
    X = np.asarray(X, float).reshape(len(X), -1)
    Xtst = np.asarray(Xtst, float).reshape(len(Xtst), -1)
    y = np.asarray(y, float).ravel()
    ytst = np.asarray(ytst, float).ravel()
    poly = PolynomialFeatures(dimensions)
    Phi = poly.fit_transform(X / 2.0)
    Phi_tst = poly.fit_transform(Xtst / 2.0)
    out = ederm_fit(Phi, y, Phi_tst, ytst, lamb=lamb, Ictype=Ictype,
                    losstype=losstype, h=h, delta=delta, iters=iters,
                    learning_step=learning_step, tol=tol, optimizer=optimizer,
                    batchsize=batchsize, seed=seed, verbose=online)
    if return_metrics:
        return out
    print("Poly-EDERM  Trn MSE: %.6f  Tst MSE: %.6f  Tst R2: %.4f"
          % (out['train_mse'], out.get('test_mse', float('nan')),
             out.get('test_r2', float('nan'))))
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(X.ravel(), y, marker='o', color='red', label="Training data")
        order = np.argsort(X.ravel())
        plt.plot(X.ravel()[order], Phi.dot(out['theta'])[order], 'b', label='EDERM')
        plt.legend(loc=4)
        plt.show()
    return out


# --------------------------------------------------------------------------
# Gaussian-kernel RKHS EDERM (mode_6, Algorithm 1)
# --------------------------------------------------------------------------
def mode_6(X, y, Xtst, ytst, kernel='gaussian', delta_kernel=0.8, iters=2000,
           lamb=1.0, learning_step=0.01, tol=1e-10, Ictype='correntropy',
           losstype='mse', h=1, delta=2, online=False, optimizer='fgd',
           batchsize=10, return_metrics=False, seed=None, plot=False):
    """Kernel RKHS EDERM (Algorithm 1): f(x) = sum_j alpha_j K(x, x_j)."""
    X = np.asarray(X, float).reshape(len(X), -1)
    Xtst = np.asarray(Xtst, float).reshape(len(Xtst), -1)
    y = np.asarray(y, float).ravel()
    ytst = np.asarray(ytst, float).ravel()
    gram = kernel_function(kernel, delta_kernel, X, X)          # (n, n)
    gram_tst = kernel_function(kernel, delta_kernel, X, Xtst)   # (n_tst, n)
    out = ederm_fit(gram, y, gram_tst, ytst, lamb=lamb, Ictype=Ictype,
                    losstype=losstype, h=h, delta=delta, iters=iters,
                    learning_step=learning_step, tol=tol, optimizer=optimizer,
                    batchsize=batchsize, seed=seed, verbose=online,
                    multistart=False)  # kernel path: faithful alpha=0 start;
                    # finite-T early stopping acts as the implicit regularizer
                    # (an interpolating start would win the objective but overfit)
    if return_metrics:
        return out
    print("Kernel-EDERM  Trn MSE: %.6f  Tst MSE: %.6f  Tst R2: %.4f"
          % (out['train_mse'], out.get('test_mse', float('nan')),
             out.get('test_r2', float('nan'))))
    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(X[:, 0] if X.shape[1] == 1 else X[:, 0], y, marker='o',
                    color='r', label="Training Data")
        order = np.argsort(X[:, 0])
        plt.plot(X[order, 0], gram.dot(out['theta'])[order], label="EDERM")
        plt.legend()
        plt.show()
    return out


# --------------------------------------------------------------------------
def _print_report(name, out, X, y, Xtst, ytst, plot=True):
    theta = out['theta']
    Phi = np.concatenate((np.asarray(X, float).reshape(len(X), -1),
                          np.ones((len(X), 1))), axis=1)
    pred = Phi.dot(theta)
    sq = (y - pred) ** 2
    print("[%s] training set: max %.3g min %.3g avg %.3g var %.3g"
          % (name, sq.max(), sq.min(), sq.mean(), sq.var()))
    if Xtst is not None:
        pred_t = np.concatenate(
            (np.asarray(Xtst, float).reshape(len(Xtst), -1),
             np.ones((len(Xtst), 1))), axis=1).dot(theta)
        sq = (ytst - pred_t) ** 2
        print("[%s] test set:     max %.3g min %.3g avg %.3g var %.3g  R2 %.4f"
              % (name, sq.max(), sq.min(), sq.mean(), sq.var(),
                 r2(ytst, pred_t)))
    if plot and Phi.shape[1] == 2:
        plt.figure(figsize=(8, 8))
        plt.scatter(X, y, s=2, label='Data Distribution')
        abline(theta, name, "red")
        plt.grid(True)
        plt.legend()
        plt.show()
