import numpy as np
from eddington.fit_functions.fit_function import fit_function


@fit_function(
    n=2,
    syntax="a[0] + a[1] * x",
    x_derivative=lambda a, x: np.full(shape=x.shape, fill_value=a[1]),
    a_derivative=lambda a, x: np.stack((np.ones(shape=x.shape), x)),
)
def linear(a, x):
    return a[0] + a[1] * x


@fit_function(
    n=1,
    syntax="a[0]",
    x_derivative=lambda a, x: np.zeros(shape=x.shape),
    a_derivative=lambda a, x: np.ones(shape=x.shape),
)
def constant(a, x):
    return np.full(fill_value=a[0], shape=x.shape)


@fit_function(
    n=3,
    syntax="a[0] + a[1] * x + a[2] * x ^ 2",
    x_derivative=lambda a, x: a[1] + 2 * a[2] * x,
    a_derivative=lambda a, x: np.stack([np.ones(shape=x.shape), x, x ** 2]),
)
def parabolic(a, x):
    return a[0] + a[1] * x + a[2] * x ** 2


@fit_function(
    n=3,
    syntax="a[0] / (x + a[1]) + a[2]",
    x_derivative=lambda a, x: -a[0] / ((x + a[1]) ** 2),
    a_derivative=lambda a, x: np.stack(
        [1 / (x + a[1]), -a[0] / ((x + a[1]) ** 2), np.ones(shape=x.shape)]
    ),
)
def hyperbolic(a, x):
    return a[0] / (x + a[1]) + a[2]


@fit_function(
    n=3,
    syntax="a[0] * exp(a[1] * x) + a[2]",
    x_derivative=lambda a, x: a[0] * a[1] * np.exp(a[1] * x),
    a_derivative=lambda a, x: np.stack(
        [np.exp(a[1] * x), a[0] * x * np.exp(a[1] * x), np.ones(x.shape)]
    ),
)
def exponential(a, x):
    return a[0] * np.exp(a[1] * x) + a[2]
