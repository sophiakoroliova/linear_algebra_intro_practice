# Linear algebra introduction practice

The repository contains practical homework from the course "Incorrect Data Processing Problems".

Please, use NumPy, SciPy, scikit-learn and similar libs to implement the tasks.

## Practices

1. Vectors;
2. Matrices;
3. Linear and affine mappings;
4. Matrix decompositions;
5. Regularization.

## Lab 5 Conclusions 

1. Regression (Diabetes)

Regularization provided a modest but clear improvement.

Linear (Base): R^2 = 0.4526

Ridge (L2): R^2 = 0.4572

Lasso (L1): R^2 = 0.4670

L1 (Lasso) provided the best performance, slightly edging out L2 and the baseline. It also achieved this while simplifying the model by zeroing out 1 of the 10 features.

2. Classification (Breast Cancer)

Here, the impact of regularization was highly significant.

No Reg (Base): Accuracy = 93.86%

L2 (Ridge): Accuracy = 97.37%

L1 (Lasso): Accuracy = 97.37%

The regularized models (L1 and L2) were dramatically more accurate than the baseline model.

The most important finding is that L1 and L2 achieved the exact same top accuracy, but the L1 (Lasso) model was far simpler, removing 15 out of 30 features (half!). This makes L1 the clear winner, as it provides top performance with a much more interpretable and less complex model.

## Useful links

* [Introduction to Linear Algebra for Applied Machine Learning with Python](https://pabloinsente.github.io/intro-linear-algebra)
* [Regularization 1](https://github.com/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)
* [Regularization 2](https://nbviewer.org/github/justmarkham/DAT8/blob/master/notebooks/20_regularization.ipynb)
