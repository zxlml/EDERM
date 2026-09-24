<div align="center">

# EDERM

**Error Density-dependent Empirical Risk Minimization**

[English](README.md) | [简体中文](README_zh.md)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Journal](https://img.shields.io/badge/Journal-Expert%20Systems%20With%20Applications-9cf.svg)

Official implementation of the paper *"Error Density-dependent Empirical Risk Minimization"* (accepted by **Expert Systems With Applications**, ESWA).

</div>

## 📑 Table of Contents

- [📢 News](#-news)
- [✨ Introduction](#-introduction)
- [🔧 Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [🏗️ Project Architecture](#️-project-architecture)
- [⚙️ Key API](#️-key-api)
- [☑️ Todo List](#️-todo-list)
- [ 🔗 Citation](#-citation)
- [📄 License](#-license)

## 📢 News

- **[2025/09]** Major refactoring released: fully vectorized `numpy` engine (O(n²) KDE kernels instead of nested Python loops), fixed gradient bugs of the original repository, robust multistart training, 17 unit tests, and one-command reproduction scripts for Tables 3 & 7 of the paper.

## ✨ Introduction

ERM (Empirical Risk Minimization) is notoriously fragile under heavy contamination: a few large errors dominate the average loss. EDERM instead **weights each sample's loss by the density of its own error** — samples whose errors lie in low-density regions (i.e., outliers) are automatically discarded. The EDERM objective (Eq. 6 of the paper) is:

```math
\min_f \;\; \frac{1}{n}\sum_{i=1}^{n} \ell\big(f, z_i\big)\;\phi\big(\lambda - \hat{p}_E(e_i)\big),
\qquad e_i = y_i - f(x_i)
```

where `p̂_E` is the Gaussian-KDE of the errors (Eq. 3) and `φ` is a bounded, smooth surrogate of the 0/1 level-set indicator `I{p̂_E ≥ λ}`. Samples outside the level set **S_λ = {i : p̂_E(e_i) ≥ λ}** receive (approximately) zero weight, so the model is fitted only on the "dense-error" majority.

Highlights of this implementation:

- **Indicator surrogates**: `correntropy`, `sigmoid`, `tanh`, `hinge`, plus the double-threshold surrogate of Remark 1 (`EDERM(S,C)` vs `EDERM(C,C)` variants).
- **Paper-faithful Algorithm 1**: mean-shift style density gradient (`∇ρ_i` through `∇e_i` only), gradient-sum update `α ← α − γ·Σ g_i / n`, with an optional fully-chained gradient (`exact=True`).
- **Convergence safeguards**: cold-start fallback (the level set can be empty at `α = 0`), and multistart selection among stationary points of the *same* objective (zero / ERM / initial-level-set / trimmed-least-squares starts), with an admissibility rule that rejects degenerate density-collapse solutions.
- **Unified engine**: linear, polynomial and kernel models all reduce to a design matrix `Φ` with `f = Φθ`.
- **Baselines included**: ERM(MSE), LSSVR (ridge), Huber, MCC, MoM — reimplemented with the same API for fair comparison.

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/zxlml/EDERM.git
cd EDERM

# (Optional) create a virtual environment
conda create -n ederm python=3.10 -y
conda activate ederm

# Install dependencies
pip install numpy scipy scikit-learn matplotlib seaborn pytest
```

## ⚡ Quick Start

### 1. Run the demo

```bash
python Demo.py
```

`Demo.py` walks through all EDERM modes (`mode_0` … `mode_6`): plain ERM, single/double-threshold EDERM on linear and polynomial models, kernel regression, and the hard 0/1-indicator variant.

### 2. Run the unit tests

```bash
python -m pytest tests/ -q        # 17 tests
```

### 3. Reproduce the paper's tables

```bash
cd experiments

python run_tables.py both    # Table 7 (linear, Types 1-3) + Table 3 (kernel, f1-f4)
python run_tables.py t7      # Table 7 only
python run_tables.py t3      # Table 3 only
python run_tables.py t3f12   # Table 3, functions f1/f2 only
python run_tables.py t3f34   # Table 3, functions f3/f4 only
```

Each setting is cross-validated over a small hyperparameter grid and then averaged over repeated runs (`30` repeats for Table 7, `10` for Table 3). Results are written to `experiments/table7.csv` / `experiments/table3*.csv`, and progress is streamed to the console.

## 🏗️ Project Architecture

```
EDERM/
├── functions/                # Core library
│   ├── models.py             # Unified vectorized EDERM engine (linear / poly / kernel)
│   ├── indicators.py         # Indicator surrogates: correntropy / sigmoid / tanh / hinge
│   ├── createdata.py         # Data generators: linear Types 1-3, kernel f1-f4 (noise + outliers)
│   ├── baselines.py          # Baselines: ERM(MSE), LSSVR, Huber, MCC, MoM
│   ├── clossfunction.py      # Correntropy loss
│   └── comparison.py         # Closed-form / gradient-descent reference implementations
├── experiments/
│   ├── run_tables.py         # One-command reproduction of paper Tables 3 & 7
│   └── table3.csv            # Reproduced results (paper-matched settings)
├── Extended_Experiments/     # Original extended notebooks (noisy-label classification,
│                             #   KDE kernels, sensitivity analysis, square-error-variable)
├── tests/
│   └── test_ederm.py         # 17 unit tests (gradients, data, modes, robustness)
├── Demo.py                   # Demo of all EDERM modes
└── LICENSE                   # MIT License
```

## ⚙️ Key API

All EDERM variants are exposed through one entry point:

```python
from functions.models import ederm_fit, kernel_function

# Gaussian-kernel design matrix: Phi[i, j] = exp(-||x_i - x_j||^2 / (2 mu^2))
G  = kernel_function('gaussian', mu, X_trn, X_trn)
Gt = kernel_function('gaussian', mu, X_trn, X_tst)

out = ederm_fit(G, y_trn, Gt, y_tst,
                lamb=1.0,          # error-density threshold lambda
                Ictype='correntropy',  # 'correntropy' | 'sigmoid' | 'tanh' | 'hinge'
                losstype='mse',    # 'mse' | 'closs'
                h=0.5,             # KDE bandwidth of the error density
                delta=2.0,         # bandwidth of the indicator surrogate
                iters=2000,
                learning_step=1e-3,
                seed=0)
print(out['theta'], out['test_r2'], out['objective'])
```

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `lamb` | – | Threshold `λ` on the error density (the level set `S_λ`) |
| `lamb2` | `None` | If set, enables the double-threshold surrogate of Remark 1 (`φ(λ1−ρ) − φ(λ2−ρ)`) |
| `Ictype` | `'correntropy'` | Indicator surrogate: `correntropy` / `sigmoid` / `tanh` / `hinge` |
| `losstype` | `'mse'` | Sample loss: `mse` (squared) or `closs` (correntropy-induced) |
| `h` | `1.0` | Gaussian-KDE bandwidth of the error density |
| `delta` | `2.0` | Bandwidth `σ` of the indicator surrogate |
| `indicator01` | `False` | If `True`, use the hard 0/1 indicator `I{ρ > λ}` instead of a smooth surrogate |
| `iters` / `learning_step` | `1000` / `0.002` | GD budget and step size `γ` (gradient-sum scale, see Algorithm 1) |
| `optimizer` | `'fgd'` | `fgd` / `sgd` / `adam` / `adaGrad` / `rmsprop` / `NAG` / `amsgd` |
| `multistart` | `True` | Multistart selection among stationary points of the same objective (set `False` or pass `theta_init` for the literal single-start Algorithm 1) |

> **Note on the density scale.** `ρ_i = (1/nh) Σ_j K(e_i − e_j)` is bounded by `1/h`, so the threshold `λ` must satisfy `λ ≲ 1/h` for the level set to be reachable. Keep `λ ∈ [0.5, 1.2]` with `h ∈ [0.5, 4]` as in the paper's experiments, and make sure the step size respects `lr < 1/λ_max(ΦᵀΦ)`.

## ☑️ Todo List

- [ ] Real-world benchmark datasets (paper §4.4)
- [ ] Classification with noisy labels (PyTorch notebook integration)
- [ ] GPU acceleration of the O(n²) KDE kernel
- [ ] `pip` packaging

## 🔗 Citation

If you find this code useful for your research, please cite:

```bibtex
@article{chen2025ederm,
  title   = {Error Density-dependent Empirical Risk Minimization},
  author  = {Chen, Hong and Zhang, Xuelin and Gong, Tieliang and Gu, Bin and Zheng, Feng},
  journal = {Expert Systems With Applications},
  year    = {2025},
  note    = {Accepted}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).
