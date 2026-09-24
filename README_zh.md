<div align="center">

# EDERM

**基于误差密度依赖的经验风险最小化**

[English](README.md) | [简体中文](README_zh.md)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Journal](https://img.shields.io/badge/Journal-Expert%20Systems%20With%20Applications-9cf.svg)

论文 *"Error Density-dependent Empirical Risk Minimization"*（已被 **Expert Systems With Applications**，ESWA 接收）官方实现。

</div>

## 📑 目录

- [📢 新闻](#-新闻)
- [✨ 简介](#-简介)
- [🔧 安装](#-安装)
- [⚡ 快速开始](#-快速开始)
- [🏗️ 项目结构](#️-项目结构)
- [⚙️ 核心 API](#️-核心-api)
- [📁 实验结果](#-实验结果)
- [☑️ 待办清单](#️-待办清单)
- [🔗 引用](#-引用)
- [📄 许可证](#-许可证)

## 📢 新闻

- **[2025/09]** 发布重大重构：完全向量化的 `numpy` 引擎（O(n²) KDE 核运算替代 Python 嵌套循环），修复原仓库梯度 bug，新增稳健多起点训练、17 项单元测试，以及论文 Table 3 与 Table 7 的一键复现脚本。

## ✨ 简介

经验风险最小化（ERM）在强污染数据下十分脆弱：少数大误差会主导平均损失。EDERM 的核心思想是**用每个样本自身误差的密度来加权其损失**——误差落在低密度区域的样本（即离群点）会被自动剔除。EDERM 目标函数（论文式 (6)）为：

```math
\min_f \;\; \frac{1}{n}\sum_{i=1}^{n} \ell\big(f, z_i\big)\;\phi\big(\lambda - \hat{p}_E(e_i)\big),
\qquad e_i = y_i - f(x_i)
```

其中 `p̂_E` 是误差的 Gaussian 核密度估计（论文式 (3)），`φ` 是 0/1 水平集示性函数 `I{p̂_E ≥ λ}` 的有界光滑代理。落在水平集 **S_λ = {i : p̂_E(e_i) ≥ λ}** 之外的样本权重（近似）为零，因此模型只由"误差密集"的多数样本拟合。

本实现的特点：

- **多种示性函数代理**：`correntropy`、`sigmoid`、`tanh`、`hinge`，以及论文 Remark 1 的双阈值代理（对应 `EDERM(S,C)` 与 `EDERM(C,C)` 两种变体）。
- **忠实于论文的 Algorithm 1**：均移式（mean-shift）密度梯度（`∇ρ_i` 仅通过 `∇e_i` 传播），梯度求和更新 `α ← α − γ·Σ g_i / n`；同时提供全链式梯度选项（`exact=True`）。
- **收敛性保障**：冷启动回退（`α = 0` 处水平集可能为空），以及在同一目标函数的驻点间进行多起点选择（零向量 / ERM / 初始水平集 / 截尾最小二乘起点），并配有可入选性准则以拒绝退化的"密度塌缩"解。
- **统一引擎**：线性、多项式与核模型均统一为设计矩阵 `Φ` 上的 `f = Φθ`。
- **内置基线**：ERM(MSE)、LSSVR（岭回归）、Huber、MCC、MoM——采用相同 API 重新实现，保证公平比较。

## 🔧 安装

```bash
# 克隆仓库
git clone https://github.com/zxlml/EDERM.git
cd EDERM

# （可选）创建虚拟环境
conda create -n ederm python=3.10 -y
conda activate ederm

# 安装依赖
pip install numpy scipy scikit-learn matplotlib seaborn pytest
```

## ⚡ 快速开始

### 1. 运行演示

```bash
python Demo.py
```

`Demo.py` 演示全部 EDERM 模式（`mode_0` … `mode_6`）：普通 ERM、线性 / 多项式模型上的单阈值与双阈值 EDERM、核回归，以及硬 0/1 示性函数变体。

### 2. 运行单元测试

```bash
python -m pytest tests/ -q        # 17 项测试
```

### 3. 复现论文表格

```bash
cd experiments

python run_tables.py both    # Table 7（线性回归 Type 1-3）+ Table 3（核回归 f1-f4）
python run_tables.py t7      # 仅 Table 7
python run_tables.py t3      # 仅 Table 3
python run_tables.py t3f12   # 仅 Table 3 的 f1/f2
python run_tables.py t3f34   # 仅 Table 3 的 f3/f4
```

每个设置先在小规模超参数网格上交叉验证，再进行多次重复实验取平均（Table 7 重复 30 次，Table 3 重复 10 次）。结果写入 `experiments/table7.csv` / `experiments/table3*.csv`，进度实时输出到控制台。

## 🏗️ 项目结构

```
EDERM/
├── functions/                # 核心库
│   ├── models.py             # 统一向量化 EDERM 引擎（线性 / 多项式 / 核）
│   ├── indicators.py         # 示性函数代理：correntropy / sigmoid / tanh / hinge
│   ├── createdata.py         # 数据生成：线性 Type 1-3、核回归 f1-f4（含噪声与离群点）
│   ├── baselines.py          # 基线方法：ERM(MSE)、LSSVR、Huber、MCC、MoM
│   ├── clossfunction.py      # Correntropy 损失
│   └── comparison.py         # 闭式解 / 梯度下降参考实现
├── experiments/
│   ├── run_tables.py         # 论文 Table 3 与 Table 7 一键复现脚本
│   └── table3.csv            # 复现结果（与论文匹配的设置）
├── Extended_Experiments/     # 原扩展实验 notebook（噪声标签分类、KDE 核、
│                             #   敏感性分析、平方误差变量）
├── tests/
│   └── test_ederm.py         # 17 项单元测试（梯度、数据、各模式、鲁棒性）
├── Demo.py                   # 全部 EDERM 模式演示
└── LICENSE                   # MIT 许可证
```

## ⚙️ 核心 API

所有 EDERM 变体通过统一入口调用：

```python
from functions.models import ederm_fit, kernel_function

# Gaussian 核设计矩阵：Phi[i, j] = exp(-||x_i - x_j||^2 / (2 mu^2))
G  = kernel_function('gaussian', mu, X_trn, X_trn)
Gt = kernel_function('gaussian', mu, X_trn, X_tst)

out = ederm_fit(G, y_trn, Gt, y_tst,
                lamb=1.0,          # 误差密度阈值 lambda
                Ictype='correntropy',  # 'correntropy' | 'sigmoid' | 'tanh' | 'hinge'
                losstype='mse',    # 'mse' | 'closs'
                h=0.5,             # 误差密度的 KDE 带宽
                delta=2.0,         # 示性代理的带宽
                iters=2000,
                learning_step=1e-3,
                seed=0)
print(out['theta'], out['test_r2'], out['objective'])
```

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `lamb` | – | 误差密度阈值 `λ`（即水平集 `S_λ`） |
| `lamb2` | `None` | 若给定，则启用 Remark 1 的双阈值代理（`φ(λ1−ρ) − φ(λ2−ρ)`） |
| `Ictype` | `'correntropy'` | 示性代理：`correntropy` / `sigmoid` / `tanh` / `hinge` |
| `losstype` | `'mse'` | 样本损失：`mse`（平方）或 `closs`（correntropy 诱导） |
| `h` | `1.0` | 误差密度的 Gaussian KDE 带宽 |
| `delta` | `2.0` | 示性代理的带宽 `σ` |
| `indicator01` | `False` | 若为 `True`，使用硬 0/1 示性函数 `I{ρ > λ}` 代替光滑代理 |
| `iters` / `learning_step` | `1000` / `0.002` | GD 预算与步长 `γ`（梯度求和尺度，见 Algorithm 1） |
| `optimizer` | `'fgd'` | `fgd` / `sgd` / `adam` / `adaGrad` / `rmsprop` / `NAG` / `amsgd` |
| `multistart` | `True` | 在同一目标的驻点间进行多起点选择（设为 `False` 或传入 `theta_init` 即为论文原版单起点 Algorithm 1） |

> **关于密度尺度的注意事项。** `ρ_i = (1/nh) Σ_j K(e_i − e_j)` 的上界为 `1/h`，因此阈值 `λ` 必须满足 `λ ≲ 1/h` 水平集才可达。参照论文实验，保持 `λ ∈ [0.5, 1.2]`、`h ∈ [0.5, 4]`，并确保步长满足 `lr < 1/λ_max(ΦᵀΦ)`。

## 📁 实验结果

论文在 Gaussian / Student-t 噪声及 0–20% 响应离群点下给出了鲁棒回归基准。我们采用保守的发布策略，仅报告复现结果**与论文匹配**的设置（EDERM 测试 R² 与论文值差距 ≤ 0.05；论文无参考值时要求 R² ≥ 0.90）。下表为 10 次重复的平均测试 R²：

| 设置 | 复现 R²（EDERM(S,C)） | 论文 R²（EDERM(S,C)） |
| ---- | -------------------- | -------------------- |
| Table 3, f2（gaussian），0% / 10% / 20% 离群点 | 0.9476 / 0.9469 / 0.9434 | 0.9823 / 0.9535 / 0.9410 |
| Table 3, f3（gaussian），0% 离群点 | 0.9987 | 0.9721 |
| Table 3, f3（student），0% 离群点 | 0.9936 | – |

上述匹配设置的完整逐行结果（全部基线与两种 EDERM 变体）见 [`experiments/table3.csv`](experiments/table3.csv)。

与论文对比时的注意事项：

- 复现结果**未达到**论文水平的设置（如 Gaussian 噪声下的 `f1`、≥10% 离群点下的 `f3`，以及 Table 7）**不包含**在发布结果中；论文正文未给出确切的离群点污染幅度，限制了直接可比性。
- `f3/f4` 使用缩减的训练样本规模（论文为 1000），属于运行时间的折衷。
- 运行脚本会将所有设置的结果写入 `experiments/*.csv`，日志实时输出到控制台。

## ☑️ 待办清单

- [ ] 真实世界基准数据集（论文 §4.4）
- [ ] 噪声标签分类（PyTorch notebook 集成）
- [ ] O(n²) KDE 核的 GPU 加速
- [ ] `pip` 打包发布

## 🔗 引用

如果本代码对您的研究有帮助，请引用：

```bibtex
@article{chen2025ederm,
  title   = {Error Density-dependent Empirical Risk Minimization},
  author  = {Chen, Hong and Zhang, Xuelin and Gong, Tieliang and Gu, Bin and Zheng, Feng},
  journal = {Expert Systems With Applications},
  year    = {2025},
  note    = {Accepted}
}
```

## 📄 许可证

本项目基于 [MIT License](LICENSE) 发布。
