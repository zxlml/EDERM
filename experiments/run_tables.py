# -*- coding: utf-8 -*-
"""Reproduce the paper's simulation experiments:

  * Table 7: linear regression on contaminated Types 1-3 (50 repeats, R2).
  * Table 3: Gaussian-kernel regression on f1-f4 with Gaussian/Student-t
    noise and 0/10/20% outliers (R2).

Hyperparameters follow the paper's protocol (Table 6 grids, reduced for
runtime) and are cross-validated on separate CV seeds, then fixed for all
repeats -- exactly the paper's selection procedure.

Run:  python experiments/run_tables.py
Outputs: experiments/table7.csv, experiments/table3.csv
"""
import sys, os, math, time, csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from functions.models import ederm_fit, kernel_function
from functions.baselines import (erm_fit, huber_fit, mcc_fit, mom_fit,
                                 lssvr_fit, evaluate)
from functions.createdata import CreateData, CreateKernelData

REPEATS_T7 = 30
REPEATS_T3 = 10


# --------------------------------------------------------------------- data
def type_data(case, seed):
    """Paper Section 5.1 Types 1-3 contaminated linear data."""
    rng = np.random.default_rng(seed)
    if case == 1:
        X = rng.normal(0, 1, (270, 1))
        y = 5 * X[:, 0] + rng.normal(0, 1, 270)
        Xo = rng.normal(-2, 1, (30, 1))
        yo = rng.normal(2, 0.5, 30)
        Xt = rng.normal(0, 1, (300, 1))
        yt = 5 * Xt[:, 0] + rng.normal(0, 1, 300)
    elif case == 2:
        X = rng.normal(0, 10, (70, 1))
        y = 2 * X[:, 0] + rng.normal(0, 1, 70)
        Xo = rng.normal(-15, 10, (30, 1))
        yo = 2 * Xo[:, 0] + rng.normal(0, 1, 30) + rng.normal(5, 2, 30)
        Xt = np.linspace(-5, 5, 100)
        yt = 2 * Xt + rng.normal(0, 1, 100)
    else:  # Type 3
        X = rng.uniform(-5, 5, 100)
        y = 5 * X + rng.normal(0, 1, 100)
        oi = rng.choice(100, 30, replace=False)
        y[oi] = -10 * X[oi] + rng.normal(3, 1, 30)
        Xo = X[oi]
        yo = y[oi]
        Xt = rng.uniform(-5, 5, 100)
        yt = 5 * Xt + rng.normal(0, 1, 100)
    X = np.concatenate((np.asarray(X, float).ravel(), np.asarray(Xo, float).ravel()))
    y = np.concatenate((np.asarray(y, float).ravel(), np.asarray(yo, float).ravel()))
    return X.reshape(-1, 1), y, np.asarray(Xt, float).reshape(-1, 1), np.asarray(yt, float).ravel()


def design(X):
    X = np.asarray(X, float).reshape(len(X), -1)
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


def ridge_fit(Phi, y, reg):
    n = Phi.shape[0]
    A = Phi.T.dot(Phi) + reg * n * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T.dot(y))


# ------------------------------------------------------------- T7 methods
def t7_methods(X, y, Xt, yt, hp):
    Phi, Phi_t = design(X), design(Xt)
    res = {}
    th = erm_fit(Phi, y, iters=4000, learning_step=hp['erm_lr'])
    res['ERM(MSE)'] = evaluate(th, Phi, y, Phi_t, yt)['test_r2']
    th = ridge_fit(Phi, y, hp['ridge'])
    res['LSSVR'] = evaluate(th, Phi, y, Phi_t, yt)['test_r2']
    th = huber_fit(Phi, y, rho=1.0, iters=4000, learning_step=hp['erm_lr'])
    res['Huber'] = evaluate(th, Phi, y, Phi_t, yt)['test_r2']
    th = mcc_fit(Phi, y, sigma=hp['mcc_sigma'], iters=4000, learning_step=hp['erm_lr'])
    res['MCC'] = evaluate(th, Phi, y, Phi_t, yt)['test_r2']
    th = mom_fit(Phi, y, blocks=hp['mom_blocks'], iters=4000,
                 learning_step=hp['erm_lr'], seed=0)
    res['MoM'] = evaluate(th, Phi, y, Phi_t, yt)['test_r2']
    for name, losstype in [('EDERM(S,C)', 'mse'), ('EDERM(C,C)', 'closs')]:
        out = ederm_fit(Phi, y, Phi_t, yt, lamb=hp['lamb'], Ictype='correntropy',
                        losstype=losstype, h=hp['h'], delta=2.0, iters=1500,
                        learning_step=hp['ederm_lr'], tol=0.0, seed=0)
        res[name] = out['test_r2']
    return res


def t7_cv(case):
    """Cross-validate hyperparams on 3 dedicated CV seeds (paper protocol):
    coordinate-wise search, baselines scored with their own method,
    EDERM scored with the mean of its two variants."""
    # NOTE: the GD step size must respect the spectral stability bound
    # lr < 1/lambda_max(Phi'Phi): Type 2 has X ~ N(0, 10), so 0.01 diverges
    # violently (NaN) while 1e-3 converges within 4000 iterations.
    grid = [('erm_lr', [0.0001, 0.001], 'ERM(MSE)'),
            ('ridge', [0.01, 0.1], 'LSSVR'),
            ('mcc_sigma', [0.5, 2.0], 'MCC'),
            ('mom_blocks', [3, 9], 'MoM'),
            ('ederm_lr', [0.001, 0.01], None),
            ('lamb', [0.9, 1.0, 1.1], None)]
    hp = dict(erm_lr=0.001, ridge=0.01, mcc_sigma=2.0, mom_blocks=9,
              ederm_lr=0.01, lamb=1.0, h=4.0)
    for key, values, method in grid:
        scores = {}
        for v in values:
            trial = dict(hp)
            trial[key] = v
            s = []
            for seed in (9001, 9002):
                X, y, Xt, yt = type_data(case, seed)
                r = t7_methods(X, y, Xt, yt, trial)
                s.append(np.mean([r['EDERM(S,C)'], r['EDERM(C,C)']])
                         if method is None else r[method])
            scores[v] = float(np.mean(s))
        hp[key] = max(scores, key=scores.get)
    return hp


def run_table7():
    print("=" * 70)
    print("Table 7: linear regression, Types 1-3, %d repeats" % REPEATS_T7)
    paper = {1: {'ERM(MSE)': 0.3499, 'LSSVR': 0.8426, 'Huber': 0.9301,
                 'MCC': 0.9503, 'MoM': 0.9497, 'EDERM(S,C)': 0.9416,
                 'EDERM(C,C)': 0.9624},
             2: {'ERM(MSE)': 0.9339, 'LSSVR': 0.9429, 'Huber': 0.9708,
                 'MCC': 0.9624, 'MoM': 0.9666, 'EDERM(S,C)': 0.9678,
                 'EDERM(C,C)': 0.9749},
             3: {'ERM(MSE)': 0.1007, 'LSSVR': 0.9517, 'Huber': 0.9904,
                 'MCC': 0.9909, 'MoM': 0.3196, 'EDERM(S,C)': 0.9896,
                 'EDERM(C,C)': 0.9940}}
    methods = ['ERM(MSE)', 'LSSVR', 'Huber', 'MCC', 'MoM', 'EDERM(S,C)', 'EDERM(C,C)']
    rows = []
    for case in (1, 2, 3):
        t0 = time.time()
        hp = t7_cv(case)
        print("-- Type %d  hyperparams: %s  (CV %.0fs)" % (case, hp, time.time() - t0))
        acc = {m: [] for m in methods}
        for rep in range(REPEATS_T7):
            X, y, Xt, yt = type_data(case, 1000 + rep)
            r = t7_methods(X, y, Xt, yt, hp)
            for m in methods:
                acc[m].append(r[m])
            plt.close('all')
        for m in methods:
            mean, std = float(np.mean(acc[m])), float(np.std(acc[m]))
            rows.append(['Type%d' % case, m, mean, std, paper[case][m]])
            print("  %-11s %-11s R2=%.4f(%.4f)   paper %.4f"
                  % ('Type%d' % case, m, mean, std, paper[case][m]))
    with open(os.path.join(os.path.dirname(__file__), 'table7.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['data', 'method', 'r2_mean', 'r2_std', 'paper_r2'])
        w.writerows(rows)


# ------------------------------------------------------------- T3 methods
def t3_methods(G, y, Gt, yt, hp, multistart=False):
    res = {}
    th = erm_fit(G, y, iters=3000, learning_step=hp['gd_lr'])
    res['ERM(MSE)'] = evaluate(th, G, y, Gt, yt)['test_r2']
    th = huber_fit(G, y, rho=hp['huber_rho'], iters=3000, learning_step=hp['gd_lr'])
    res['Huber'] = evaluate(th, G, y, Gt, yt)['test_r2']
    th = mcc_fit(G, y, sigma=hp['mcc_sigma'], iters=3000, learning_step=hp['gd_lr'])
    res['MCC'] = evaluate(th, G, y, Gt, yt)['test_r2']
    th = mom_fit(G, y, blocks=3, iters=3000, learning_step=hp['gd_lr'], seed=0)
    res['MoM'] = evaluate(th, G, y, Gt, yt)['test_r2']
    alpha = lssvr_fit(G, y, reg=hp['lssvr_reg'])
    res['LSSVR'] = float(1 - np.mean((yt - Gt.dot(alpha)) ** 2) / np.var(yt))
    for name, losstype in [('EDERM(S,C)', 'mse'), ('EDERM(C,C)', 'closs')]:
        out = ederm_fit(G, y, Gt, yt, lamb=hp['lamb'], Ictype='correntropy',
                        losstype=losstype, h=hp['h'], delta=2.0, iters=2000,
                        learning_step=hp['ederm_lr'], tol=0.0, seed=0,
                        multistart=multistart)
        res[name] = out['test_r2']
    return res


def run_table3(funcs=('f1', 'f2', 'f3', 'f4'), out_name='table3.csv'):
    print("=" * 70)
    print("Table 3: kernel regression f1-f4, %d repeats" % REPEATS_T3)
    paper = {('f1', 'gaussian', 0.0): 0.9917, ('f1', 'gaussian', 0.1): 0.9855,
             ('f1', 'gaussian', 0.2): 0.9689,
             ('f2', 'gaussian', 0.0): 0.9823, ('f2', 'gaussian', 0.1): 0.9535,
             ('f2', 'gaussian', 0.2): 0.9410,
             ('f3', 'gaussian', 0.0): 0.9721, ('f3', 'gaussian', 0.1): 0.9604,
             ('f3', 'gaussian', 0.2): 0.9597}
    methods = ['ERM(MSE)', 'LSSVR', 'Huber', 'MCC', 'MoM', 'EDERM(S,C)', 'EDERM(C,C)']
    # Paper: data size 200 (f1/f2) and 2000 (f3/f4), split evenly into
    # train/test; kernel parameter mu = 0.05 optimal for most settings
    # (K = exp{-mu ||x-x'||^2}, i.e. bandwidth delta = 1/sqrt(2 mu)).
    # f3/f4 live on raw scales up to 560*pi, so inputs are standardized
    # (train statistics) -- otherwise the Gram matrix degenerates to the
    # identity for any reasonable bandwidth.  n=300 for f3/f4 is a runtime
    # compromise with the paper's 1000 (KDE is O(n^2) per iteration).
    # per-function EDERM CV grids: f1 needs the paper's operating point
    # (h=0.5, lambda~0.95-1.2, moderate lr); the density-gradient term of
    # Algorithm 1 is scale-sensitive and diverges for h>=1 & lambda=1 at
    # kernel scale (verified by finite-difference probing)
    cfg = {'f1': dict(n=100, std=False, stdy=False, ms=False,
                      mus=(0.5, 1.0, 2.0, 3.0),
                      lrs=(1e-3, 3e-3), hs=(0.5, 1.0), lams=(0.95, 1.0, 1.2)),
           'f2': dict(n=100, std=False, stdy=False, ms=False,
                      mus=(0.3, 0.5, 1.0, 2.0),
                      lrs=(1e-4, 1e-3), hs=(1.0, 4.0), lams=(0.95, 1.0)),
           'f3': dict(n=250, std=True, stdy=True, ms=True,
                      mus=(0.5, 1.0, 2.0, 4.0),
                      lrs=(1e-4, 1e-3), hs=(1.0, 4.0), lams=(1.0,)),
           'f4': dict(n=250, std=True, stdy=True, ms=True,
                      mus=(4.0, 8.0, 16.0, 32.0),
                      lrs=(1e-4, 1e-3), hs=(1.0, 4.0), lams=(1.0,))}
    rows = []
    for func in funcs:
        for noise in ('gaussian', 'student'):
            for orat in (0.0, 0.1, 0.2):
                t0 = time.time()
                cf = cfg[func]

                def _data(seed):
                    Xtr, ytr, Xte, yte = CreateKernelData(
                        func, cf['n'], cf['n'], noise, outlier_ratio=orat,
                        seed=seed)
                    if cf['std']:
                        m_, s_ = Xtr.mean(0), Xtr.std(0) + 1e-12
                        Xtr, Xte = (Xtr - m_) / s_, (Xte - m_) / s_
                    if cf['stdy']:
                        # f3/f4 raw y spans hundreds; the paper's lambda in
                        # [0.9, 1.2] presumes O(1) error-density scale, i.e.
                        # standardized targets.  R2 is invariant under the
                        # common affine transform of (prediction, truth).
                        m2_, s2_ = ytr.mean(), ytr.std() + 1e-12
                        ytr, yte = (ytr - m2_) / s2_, (yte - m2_) / s2_
                    return Xtr, ytr, Xte, yte

                def _gram(Xtr, Xte, mu):
                    return (kernel_function('gaussian', mu, Xtr, Xtr),
                            kernel_function('gaussian', mu, Xtr, Xte))

                # -- stage 1: pick the kernel bandwidth mu (coarse EDERM).
                #    NOTE: the GD step size on the kernel coefficients is
                #    stability-limited (lr < ~2/lambda_max(G)^2), which is why
                #    the paper selects eta = 1e-4; the grid only keeps
                #    near-stable values and CV uses the same T as the final
                #    runs (larger lr only "wins" through an early-stopping
                #    artifact that diverges at full T).
                best_mu, best_mu_s = None, -np.inf
                for mu in cf['mus']:
                    s = []
                    for seed in (7001, 7002):
                        Xtr, ytr, Xte, yte = _data(seed)
                        G, Gt = _gram(Xtr, Xte, mu)
                        out = ederm_fit(G, ytr, Gt, yte, lamb=1.0,
                                        Ictype='correntropy', losstype='mse',
                                        h=2.0, delta=2.0, iters=1000,
                                        learning_step=1e-4, tol=0.0,
                                        seed=0, multistart=False)
                        s.append(out['test_r2'])
                    if float(np.mean(s)) > best_mu_s:
                        best_mu_s, best_mu = float(np.mean(s)), mu
                mu = best_mu

                # -- stage 2: pick EDERM (lr, h, lamb) with the best mu
                best, best_s = None, -np.inf
                for ederm_lr in cf['lrs']:
                    for h in cf['hs']:
                        for lamb in cf['lams']:
                            s = []
                            for seed in (7001, 7002):
                                Xtr, ytr, Xte, yte = _data(seed)
                                G, Gt = _gram(Xtr, Xte, mu)
                                out = ederm_fit(G, ytr, Gt, yte, lamb=lamb,
                                                Ictype='correntropy',
                                                losstype='mse', h=h, delta=2.0,
                                                iters=2000, learning_step=ederm_lr,
                                                tol=0.0, seed=0, multistart=False)
                                s.append(out['test_r2'])
                            m = float(np.mean(s))
                            if m > best_s:
                                best_s, best = m, (ederm_lr, h, lamb)

                # -- stage 3: pick the shared GD step size of the baselines
                #    by the mean test R2 of the four GD-based baselines
                best_gd, best_gd_s = None, -np.inf
                for gd_lr in (0.001, 0.01):
                    s = []
                    for seed in (7001, 7002):
                        Xtr, ytr, Xte, yte = _data(seed)
                        G, Gt = _gram(Xtr, Xte, mu)
                        r1 = evaluate(erm_fit(G, ytr, iters=3000,
                                              learning_step=gd_lr),
                                      G, ytr, Gt, yte)['test_r2']
                        r2_ = evaluate(huber_fit(G, ytr, rho=1.0, iters=3000,
                                                 learning_step=gd_lr),
                                       G, ytr, Gt, yte)['test_r2']
                        r3_ = evaluate(mcc_fit(G, ytr, sigma=0.5, iters=3000,
                                               learning_step=gd_lr),
                                       G, ytr, Gt, yte)['test_r2']
                        r4_ = evaluate(mom_fit(G, ytr, blocks=3, iters=3000,
                                               learning_step=gd_lr, seed=0),
                                       G, ytr, Gt, yte)['test_r2']
                        s.append(np.mean([r1, r2_, r3_, r4_]))
                    if float(np.mean(s)) > best_gd_s:
                        best_gd_s, best_gd = float(np.mean(s)), gd_lr

                hp = dict(gd_lr=best_gd, huber_rho=1.0, mcc_sigma=0.5,
                          lssvr_reg=0.01, ederm_lr=best[0], h=best[1],
                          lamb=best[2], mu=mu)
                acc = {m: [] for m in methods}
                for rep in range(REPEATS_T3):
                    Xtr, ytr, Xte, yte = _data(2000 + rep)
                    G, Gt = _gram(Xtr, Xte, mu)
                    r = t3_methods(G, ytr, Gt, yte, hp, multistart=cf['ms'])
                    for m in methods:
                        acc[m].append(r[m])
                key = (func, noise, orat)
                for m in methods:
                    mean, std = float(np.mean(acc[m])), float(np.std(acc[m]))
                    pf = paper.get(key, '') if m == 'EDERM(S,C)' else ''
                    rows.append(['%s/%s/%d%%' % (func, noise, int(orat * 100)),
                                 m, mean, std, pf])
                    print("  %-16s %-11s R2=%8.4f(%.4f) %s"
                          % (rows[-1][0], m, mean, std,
                             '  paper %.4f' % pf if pf != '' else ''))
                print("  [setting done in %.0fs, hp=%s]" % (time.time() - t0, hp))
    with open(os.path.join(os.path.dirname(__file__), out_name), 'w',
              newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'method', 'r2_mean', 'r2_std', 'paper_r2_EDERM(SC)'])
        w.writerows(rows)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('both', 't7'):
        run_table7()
    if which in ('both', 't3'):
        run_table3()
    if which == 't3f12':
        run_table3(funcs=('f1', 'f2'), out_name='table3_f12.csv')
    if which == 't3f4':
        run_table3(funcs=('f4',), out_name='table3_f4.csv')
    if which == 't3f34':
        run_table3(funcs=('f3', 'f4'), out_name='table3_f34.csv')
    print("done.")
