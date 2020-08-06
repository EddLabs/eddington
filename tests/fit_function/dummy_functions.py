import numpy as np

from eddington import fit_function


@fit_function(n=2, syntax="a[0] + a[1] * x ** 2", save=False)
def dummy_func1(a, x):
    return a[0] + a[1] * x ** 2


@fit_function(
    n=4,
    syntax="a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3",
    x_derivative=lambda a, x: a[1] + 2 * a[2] * x + 3 * a[3] * x ** 2,
    a_derivative=lambda a, x: np.stack([np.ones(shape=np.shape(x)), x, x ** 2, x ** 3]),
    save=False,
)
def dummy_func2(a, x):
    return a[0] + a[1] * x + a[2] * x ** 2 + a[3] * x ** 3
