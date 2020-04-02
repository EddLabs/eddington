import numpy as np
from eddington.exceptions import InvalidGeneratorInitialization
from eddington.fit_functions.fit_function import fit_function
from eddington.fit_functions.fit_function_generator import fit_function_generator
from eddington.fit_functions.fit_functions_list import linear, parabolic, hyperbolic


@fit_function_generator(parameters="n", syntax="a[0] + a[1] * x + ... + a[n] * x ^ n")
def polynom(n):
    n = int(n)
    if n <= 0:
        raise InvalidGeneratorInitialization(f"n must be positive, got {n}")

    if n == 1:
        return linear

    if n == 2:
        return parabolic

    arange = np.arange(1, n + 1)

    @fit_function(
        n=n + 1,
        name=f"polynom_{n}",
        x_derivative=lambda a, x: polynom(n - 1)(arange * a[1:], x),
        a_derivative=lambda a, x: np.stack([x ** i for i in range(n + 1)]),
        save=False,
    )
    def func(a, x):
        return sum([a[i] * x ** i for i in range(n + 1)])

    return func


@fit_function_generator(parameters="n", syntax="a[0] * (x + a[1]) ^ n + a[2]")
def straight_power(n):
    n = int(n)
    if n <= 0:
        raise InvalidGeneratorInitialization(f"n must be positive, got {n}")

    if n == 1:
        raise InvalidGeneratorInitialization('n cannot be 1. use "linear" fit instead.')

    @fit_function(
        n=3,
        name=f"straight_power_{n}",
        x_derivative=lambda a, x: n * a[0] * (x + a[1]) ** (n - 1),
        a_derivative=lambda a, x: np.stack(
            [(x + a[1]) ** n, n * a[0] * (x + a[1]) ** (n - 1), np.ones(shape=x.shape)]
        ),
        save=False,
    )
    def func(a, x):
        return a[0] * (x + a[1]) ** n + a[2]

    return func


@fit_function_generator(parameters="n", syntax="a[0] / (x + a[1]) ^ n + a[2]")
def inverse_power(n):
    n = int(n)
    if n <= 0:
        raise InvalidGeneratorInitialization(f"n must be positive, got {n}")

    if n == 1:
        return hyperbolic

    @fit_function(
        n=3,
        name=f"inverse_power_{n}",
        x_derivative=lambda a, x: -n * a[0] / (x + a[1]) ** (n + 1),
        a_derivative=lambda a, x: np.stack(
            [
                1 / (x + a[1]) ** n,
                -n * a[0] / (x + a[1]) ** (n + 1),
                np.ones(shape=x.shape),
            ]
        ),
        save=False,
    )
    def func(a, x):
        return a[0] / (x + a[1]) ** n + a[2]

    return func
