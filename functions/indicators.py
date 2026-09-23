# -*- coding: utf-8 -*-
"""
Indicator surrogate functions for EDERM (Error Density-dependent ERM).

The indicator function I{p_E(e) >= lambda} is replaced by a bounded smooth
surrogate phi(lambda - p_E(e)) (paper Eq.(6) and Section 2):

    correntropy : phi(t) = (1 - exp{-(1-t)^2_+ / (2 sigma^2)}) / (1 - exp{-1/(2 sigma^2)})
    sigmoid     : phi(t) = 1 / (1 + exp(sigma_s * t))          (modified, Fig.6)
    tanh        : phi(t) = 1 + tanh(-sigma_s * t)              (modified, Fig.6)
    hinge       : phi(t) = min(max(sigma_h * (-t), 0), 1)      (modified, Fig.6)

Two APIs are provided:
  * scalar API ``indicator(name, fai, dfai, lamb, delta)`` (kept for backward
    compatibility with the original repository) -- returns (Ic, dIc) where dIc
    is already multiplied by dfai = d rho / d theta.
  * vectorized API ``indicator_vec(name, rho, lamb, delta)`` -- returns
    (Ic, dIc_drho) as numpy arrays; the caller multiplies by d rho / d theta.
"""
import math
import numpy as np

_VALID = ('correntropy', 'sigmoid', 'tanh', 'hinge', 'hingeIC',
          'modifiedsquare', 'exponential', 'noIC')


def _normalize(name):
    name = name.lower()
    if name == 'hingeic':
        return 'hinge'
    if name == 'noic':
        return 'noIC'
    if name not in _VALID:
        raise ValueError("Unknown indicator type: %s (valid: %s)" % (name, _VALID))
    return name


# ---------------------------------------------------------------- scalar API
def indicator(name, fai, dfai, lamb, delta):
    """Scalar surrogate value and derivative (dIc already times dfai)."""
    name = _normalize(name)
    if name == 'correntropy':
        u = 1.0 - lamb + fai                      # 1 - t = 1 - lambda + rho
        if u > 0:
            norm = 1.0 - math.exp(-1.0 / (2.0 * delta * delta))
            Ic = (1.0 - math.exp(-u * u / (2.0 * delta * delta))) / norm
            dIc = math.exp(-u * u / (2.0 * delta * delta)) * (u / (delta * delta)) / norm * dfai
            return Ic, dIc
        return 0.0, 0.0
    if name == 'modifiedsquare':
        if fai > lamb:
            return delta * (fai - lamb) ** 2, 2.0 * delta * (fai - lamb) * dfai
        return 0.0, 0.0
    if name == 'exponential':
        v = math.exp(delta * (fai - lamb))
        return v, v * delta * dfai
    if name == 'sigmoid':
        s = 1.0 / (1.0 + math.exp(-delta * (fai - lamb)))
        return s, delta * s * (1.0 - s) * dfai
    if name == 'tanh':
        t = math.tanh(delta * (fai - lamb))
        return 1.0 + t, delta * (1.0 - t * t) * dfai
    if name == 'hinge':
        if fai > lamb:
            return min(delta * (fai - lamb), 1.0), delta * dfai
        return 0.0, 0.0
    if name == 'noIC':
        return 1.0, 0.0
    raise ValueError(name)


# ------------------------------------------------------------- vectorized API
def indicator_vec(name, rho, lamb, delta):
    """Vectorized surrogate.

    Returns
    -------
    Ic      : (n,)  surrogate values phi(lambda - rho)
    dIc_drho: (n,)  derivative d phi / d rho
    """
    name = _normalize(name)
    rho = np.asarray(rho, dtype=float)
    n = rho.shape[0]
    ones = np.ones(n)
    zeros = np.zeros(n)

    if name == 'correntropy':
        u = 1.0 - lamb + rho                       # (1 - t)_+ with t = lambda - rho
        active = u > 0
        ua = np.where(active, u, 0.0)
        norm = 1.0 - math.exp(-1.0 / (2.0 * delta * delta))
        Ic = np.where(active,
                      (1.0 - np.exp(-ua * ua / (2.0 * delta * delta))) / norm,
                      0.0)
        dIc_drho = np.where(active,
                            np.exp(-ua * ua / (2.0 * delta * delta)) * (ua / (delta * delta)) / norm,
                            0.0)
        return Ic, dIc_drho

    if name == 'modifiedsquare':
        d = rho - lamb
        active = d > 0
        da = np.where(active, d, 0.0)
        return np.where(active, delta * da * da, 0.0), np.where(active, 2.0 * delta * da, 0.0)

    if name == 'exponential':
        v = np.exp(delta * (rho - lamb))
        return v, delta * v

    if name == 'sigmoid':
        s = 1.0 / (1.0 + np.exp(-delta * (rho - lamb)))
        return s, delta * s * (1.0 - s)

    if name == 'tanh':
        t = np.tanh(delta * (rho - lamb))
        return 1.0 + t, delta * (1.0 - t * t)

    if name == 'hinge':
        d = delta * (rho - lamb)
        active = d > 0
        return np.where(active, np.minimum(d, 1.0), 0.0), np.where(active & (d < 1.0), delta, 0.0)

    if name == 'noIC':
        return ones, zeros

    raise ValueError(name)


def indicator_double(name, rho, lamb, lamb2, delta):
    """Double-threshold surrogate of paper Remark 1:
        phi(lambda1 - rho) - phi(lambda2 - rho),  lambda1 < rho < lambda2.
    Returns (Ic, dIc_drho)."""
    Ic1, d1 = indicator_vec(name, rho, lamb, delta)
    Ic2, d2 = indicator_vec(name, rho, lamb2, delta)
    return Ic1 - Ic2, d1 - d2
