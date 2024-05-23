# Demo for paper "Error Density-dependent Empirical Risk Minimization" submitted to ESWA Journal.

# Getting Started

## Dependencies

The dependencies for different application may change.

The following dependencies are needed:

1. python3

2. math

3. scipy

4. sklearn

5. numpy

6. random

7. seaborn

8. matplotlib

9. Jupyter

10. Pytorch (just for classification task)

## Data Source

### Synthetic Data

We provide 7 types of synthetic data for simulation experiments.

### Real Data

3 types of real data (including "Yacht Hydrodynamics ", "Airfoil Self-Noise" and "Concrete Compressive Strength") are used in this paper, and can be download from the public UCI repository [1]. Additional classification task is conducted on Fashion-MNIST image dataset [2].



## How to run the code

### Run the Demo.py

Please choose the data type, regression task and other parameters (density threshold lamb1 & lamb2, optimization tools) before running the Demo.py:

```python
if __name__=='__main__':
    # Generate data for regression
    # Type 1 2 3 -> linear regression; Type 4 -> polynomial regression; Type 4 5 6 7 -> gaussian kernel regression
    # Type 1 2 3 4 5 -> the curves are visible   6 7 are high-dimensional Friedman curves
    data_type = 1
    
    # Regression Task: Linear, polynomial or gaussian kernel regression
    task = 'linear'  # 'linear', 'kernel' or 'polynomial'
    
    # Error density threshold \lambda1 ( \lambda1 and \lambda2 for extended EDRM)
    lamb1=1
    lamb2=3
    
    # Extended accelerate optimizers (fgd,sgd,adam...)
    optimizer = 'adaGrad'  # 'fgd','sgd','adaGrad','adam','amsgd','rmsprop','NAG'
    
    # Run demo
    EDRM(data_type, task, lamb1, lamb2,optimizer)
```

Besides, the loss function and indicator surrogate function can also be selected:

```python
'''
Ic type:  'correntropy','sigmoid','tanh','hinge',('modifiedsquare','exponential')
loss type:  'mse','closs'
'''
mode_1(X, y, Xtst, ytst,iters=1000,lamb=0.9,learning_step=0.002,tol = 1/10**8,Ictype='correntropy',losstype='closs',h=1,delta=2,online =True, optimizer=opt)
```



## Extended Experiments

We also conduct additional experiments with Type 1 linear data with square error variable
$$
e_i = (y_i - f(x_i))^2,
$$
instead of 
$$
e_i = (y_i - f(x_i)).
$$
Besides, we also try other kernels for kernel density estimation (KDE), including the Exponential kernel and Epanechnikov kernel instead of gaussian kernel.



The sensitivity analysis of the EDRM target functions are also provided in the 'Extended_Exp' file.



In addition, we further exploit EDERM for classification task on Fashion-MNIST [2].

## Reference

[1]  UCI machine learning repository

```
@misc{asuncion2007uci,
  title={UCI machine learning repository},
  author={Asuncion, Arthur and Newman, David},
  year={2007},
  publisher={Irvine, CA, USA}
}
```

[2]  Fashion-MNIST dataset

```
@article{xiao2017fashion,
  title={Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms},
  author={Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  journal={arXiv preprint arXiv:1708.07747},
  year={2017}
}
```

