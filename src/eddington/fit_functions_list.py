"""List of common fit functions."""
from typing import Union

import numpy as np
import scipy.special

from eddington.exceptions import FitFunctionLoadError
from eddington.fit_function_class import FitFunction, fit_function


@fit_function(
    n=2,
    syntax="a[0] + a[1] * x",
    x_derivative=lambda a, x: np.full(shape=np.shape(x), fill_value=a[1]),
    a_derivative=lambda a, x: np.stack([np.ones(shape=np.shape(x)), x]),
)  # pylint: disable=C0103
def linear(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Simple linear fit function."""
    return a[0] + a[1] * x


@fit_function(
    n=1,
    syntax="a[0]",
    x_derivative=lambda a, x: np.zeros(shape=np.shape(x)),
    a_derivative=lambda a, x: np.stack([np.ones(shape=np.shape(x))]),
)  # pylint: disable=C0103
def constant(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Constant fit function."""
    return np.full(fill_value=a[0], shape=np.shape(x))


@fit_function(
    n=3,
    syntax="a[0] + a[1] * x + a[2] * x ^ 2",
    x_derivative=lambda a, x: a[1] + 2 * a[2] * x,
    a_derivative=lambda a, x: np.stack([np.ones(shape=np.shape(x)), x, x ** 2]),
)  # pylint: disable=C0103
def parabolic(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Parabolic fit function."""
    return a[0] + a[1] * x + a[2] * x ** 2


@fit_function(
    n=4,
    syntax="a[0] * (x + a[1]) ^ a[2] + a[3]",
    x_derivative=lambda a, x: a[2] * a[0] * (x + a[1]) ** (a[2] - 1),
    a_derivative=lambda a, x: np.stack(
        [
            (x + a[1]) ** a[2],
            a[2] * a[0] * (x + a[1]) ** (a[2] - 1),
            a[0] * np.log(x + a[1]) * (x + a[1]) ** a[2],
            np.ones(shape=np.shape(x)),
        ]
    ),
)  # pylint: disable=C0103
def straight_power(
    a: np.ndarray, x: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:  # pylint: disable=C0103
    """Represent fitting of y ~ x^n."""
    return a[0] * (x + a[1]) ** a[2] + a[3]


@fit_function(
    n=4,
    syntax="a[0] / (x + a[1]) ^ a[2] + a[3]",
    x_derivative=lambda a, x: -a[2] * a[0] / (x + a[1]) ** (a[2] + 1),
    a_derivative=lambda a, x: np.stack(
        [
            1 / (x + a[1]) ** a[2],
            -a[2] * a[0] / (x + a[1]) ** (a[2] + 1),
            -a[0] * np.log(x + a[1]) * (x + a[1]) ** a[2],
            np.ones(shape=np.shape(x)),
        ]
    ),
)  # pylint: disable=C0103
def inverse_power(
    a: np.ndarray, x: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:  # pylint: disable=C0103
    """Represent fitting of y ~ x^(-n)."""
    return a[0] / (x + a[1]) ** a[2] + a[3]


@fit_function(
    n=3,
    syntax="a[0] / (x + a[1]) + a[2]",
    x_derivative=lambda a, x: -a[0] / ((x + a[1]) ** 2),
    a_derivative=lambda a, x: np.stack(
        [1 / (x + a[1]), -a[0] / ((x + a[1]) ** 2), np.ones(shape=np.shape(x))]
    ),
)  # pylint: disable=C0103
def hyperbolic(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Hyperbolic fit function."""
    return a[0] / (x + a[1]) + a[2]


@fit_function(
    n=3,
    syntax="a[0] * exp(a[1] * x) + a[2]",
    x_derivative=lambda a, x: a[0] * a[1] * np.exp(a[1] * x),
    a_derivative=lambda a, x: np.stack(
        [np.exp(a[1] * x), a[0] * x * np.exp(a[1] * x), np.ones(np.shape(x))]
    ),
)  # pylint: disable=C0103
def exponential(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Exponential fit function."""
    return a[0] * np.exp(a[1] * x) + a[2]


@fit_function(
    n=4,
    syntax="a[0] * cos(a[1] * x + a[2]) + a[3]",
    x_derivative=lambda a, x: -a[0] * a[1] * np.sin(a[1] * x + a[2]),
    a_derivative=lambda a, x: np.stack(
        [
            np.cos(a[1] * x + a[2]),
            -a[0] * x * np.sin(a[1] * x + a[2]),
            -a[0] * np.sin(a[1] * x + a[2]),
            np.ones(shape=np.shape(x)),
        ]
    ),
)  # pylint: disable=C0103
def cos(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Cosines fit function."""
    return a[0] * np.cos(a[1] * x + a[2]) + a[3]


@fit_function(
    n=4,
    syntax="a[0] * sin(a[1] * x + a[2]) + a[3]",
    x_derivative=lambda a, x: a[0] * a[1] * np.cos(a[1] * x + a[2]),
    a_derivative=lambda a, x: np.stack(
        [
            np.sin(a[1] * x + a[2]),
            a[0] * x * np.cos(a[1] * x + a[2]),
            a[0] * np.cos(a[1] * x + a[2]),
            np.ones(shape=np.shape(x)),
        ]
    ),
)  # pylint: disable=C0103
def sin(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Sine fit function."""
    return a[0] * np.sin(a[1] * x + a[2]) + a[3]


@fit_function(
    n=4,
    syntax="a[0] * exp( - ((x - a[1]) / a[2]) ^ 2) + a[3]",
    x_derivative=lambda a, x: a[0]
    * np.exp(-(((x - a[1]) / a[2]) ** 2))  # noqa: W503
    * (-2 * (x - a[1]) / a[2]),  # noqa: W503
    a_derivative=lambda a, x: np.stack(
        [
            np.exp(-(((x - a[1]) / a[2]) ** 2)),
            a[0] * np.exp(-(((x - a[1]) / a[2]) ** 2)) * (2 * (x - a[1]) / a[2]),
            a[0] * np.exp(-(((x - a[1]) / a[2]) ** 2)) * (2 * (x - a[1]) / (a[2] ** 2)),
            np.ones(shape=np.shape(x)),
        ]
    ),
)  # pylint: disable=C0103
def normal(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Normal distribution fit function."""
    return a[0] * np.exp(-(((x - a[1]) / a[2]) ** 2)) + a[3]


@fit_function(
    n=3,
    syntax="a[0] * (a[1] ^ x) * exp(-a[1]) / gamma(x+1) + a[2]",
    x_derivative=lambda a, x: (
        a[0] * np.power(a[1], x) * np.exp(-a[1]) / scipy.special.gamma(x + 1)
    )  # noqa: W503
    * (np.log(a[1]) - scipy.special.digamma(x + 1)),  # noqa: W503
    a_derivative=lambda a, x: np.stack(
        [
            np.power(a[1], x) * np.exp(-a[1]) / scipy.special.gamma(x + 1),
            (a[0] * np.exp(-a[1]) / scipy.special.gamma(x + 1))
            * (x * np.power(a[1], x - 1) - np.power(a[1], x)),  # noqa: W503
            np.ones(shape=np.shape(x)),
        ]
    ),
)  # pylint: disable=C0103
def poisson(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Poisson fit function."""
    return a[0] * np.power(a[1], x) * np.exp(-a[1]) / scipy.special.gamma(x + 1) + a[2]


def polynomial(n: int) -> FitFunction:  # pylint: disable=C0103
    """
    Creates a polynomial fit function with parameters as coefficients.

    :param n: Degree of the polynom.
    :return: :class:`FitFunction`
    """
    n = int(n)
    if n <= 0:
        raise FitFunctionLoadError(f"n must be positive, got {n}")

    if n == 1:
        return linear

    arange = np.arange(1, n + 1)

    syntax = "a[0] + a[1] * x + " + " + ".join(
        [f"a[{i}] * x ^ {i}" for i in arange[1:]]
    )

    @fit_function(
        n=n + 1,
        name=f"polynomial_{n}",
        syntax=syntax,
        x_derivative=lambda a, x: polynomial(n - 1)(arange * a[1:], x),
        a_derivative=lambda a, x: np.stack([x ** i for i in range(n + 1)]),
        save=False,
    )  # pylint: disable=C0103
    def func(a: np.ndarray, x: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
        return sum([a[i] * x ** i for i in range(n + 1)])

    return func
