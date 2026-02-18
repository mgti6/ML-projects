# Linear Regression From Scratch

This notebook shows how to implement **linear regression from scratch** in Python, without using a ready-made model from `scikit-learn`.

## Goal

Train a regression line of the form:

$$
\hat{y} = ax + b
$$

using gradient descent on the Mean Squared Error (MSE) loss.

## What This Notebook Covers

- Data generation / preparation
- Data visualization (scatter plot)
- Manual implementation of the MSE loss
- Derivation of gradients $(\partial L / \partial a$ and $(\partial L / \partial b$.
- Implementation of `gradient_descent(...)`
- Training loop to update \(a\) and \(b\)
- Plot of the final fitted regression line

## Mathematical Formulas

Loss function:

$$
L(a,b) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - (ax_i + b)\right)^2
$$

Gradients:

$$
\frac{\partial L}{\partial a}
= -\frac{2}{n}\sum_{i=1}^{n}x_i\left(y_i-(ax_i+b)\right)
$$

$$
\frac{\partial L}{\partial b}
= -\frac{2}{n}\sum_{i=1}^{n}\left(y_i-(ax_i+b)\right)
$$

Parameter updates:

$$
a \leftarrow a - \eta \frac{\partial L}{\partial a}, \quad
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

## Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## How to Run

```bash
pip install numpy pandas matplotlib notebook
jupyter notebook

