import numpy as np
import pytest
from pytest_cases import THIS_MODULE, parametrize_with_cases, case

from eddington import FittingData, linear, plot_fitting
from eddington.exceptions import PlottingError
from tests.util import assert_calls

HAS_LEGEND = "has_legend"

EPSILON = 1e-5

FUNC = linear
X = np.arange(1, 11)
A1, A2, A3 = np.array([1, 1]), np.array([3, 2]), np.array([3.924356, 1.2345e-5])
A1_REPR, A2_REPR, A3_REPR = (
    "[a[0]=1.000, a[1]=1.000]",
    "[a[0]=3.000, a[1]=2.000]",
    "[a[0]=3.924, a[1]=1.234e-05]",
)
FIT_DATA = FittingData.random(FUNC, x=X, a=np.array([1, 2]), measurements=X.shape[0])
TITLE_NAME = "Title"


def case_no_args(mock_plt):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=A1)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=A1_REPR))
    ]
    return kwargs, plot_calls, mock_plt


def case_xmin(mock_plt):
    x = np.arange(-10, 10.9, step=0.0209)

    kwargs = dict(a=A1, xmin=-10)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=A1_REPR))
    ]
    return kwargs, plot_calls, mock_plt


def case_xmax(mock_plt):
    x = np.arange(0.1, 20, step=0.0199)

    kwargs = dict(a=A1, xmax=20)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=A1_REPR))
    ]
    return kwargs, plot_calls, mock_plt


def case_step(mock_plt):
    x = np.arange(0.1, 10.9, step=0.1)

    kwargs = dict(a=A1, step=0.1)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=A1_REPR))
    ]
    return kwargs, plot_calls, mock_plt


@case(tags=[HAS_LEGEND])
def case_a_list_with_legend(mock_plt):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=[A1, A2])
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=A1_REPR)),
        ([x, FUNC(A2, x)], dict(figure=mock_plt.figure.return_value, label=A2_REPR)),
    ]
    return kwargs, plot_calls, mock_plt


def case_a_list_without_legend(mock_plt):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=[A1, A2], legend=False)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=A1_REPR)),
        ([x, FUNC(A2, x)], dict(figure=mock_plt.figure.return_value, label=A2_REPR)),
    ]
    return kwargs, plot_calls, mock_plt


@case(tags=[HAS_LEGEND])
def case_a_dict_with_legend(mock_plt):
    x = np.arange(0.1, 10.9, step=0.0108)
    one = "one"
    two = "two"
    kwargs = dict(a={one: A1, two: A2})
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=one)),
        ([x, FUNC(A2, x)], dict(figure=mock_plt.figure.return_value, label=two)),
    ]
    return kwargs, plot_calls, mock_plt


def case_a_dict_without_legend(mock_plt):
    x = np.arange(0.1, 10.9, step=0.0108)
    one = "one"
    two = "two"
    kwargs = dict(a={one: A1, two: A2}, legend=False)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(figure=mock_plt.figure.return_value, label=one)),
        ([x, FUNC(A2, x)], dict(figure=mock_plt.figure.return_value, label=two)),
    ]
    return kwargs, plot_calls, mock_plt


def case_a_redundent_precision(mock_plt):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=A3)
    plot_calls = [
        ([x, FUNC(A3, x)], dict(figure=mock_plt.figure.return_value, label=A3_REPR))
    ]
    return kwargs, plot_calls, mock_plt


@parametrize_with_cases(argnames="kwargs, plot_calls, plt", cases=THIS_MODULE)
def test_plot_fitting_without_boundaries(kwargs, plot_calls, plt):
    plot_fitting(data=FIT_DATA, func=FUNC, title_name=TITLE_NAME, **kwargs)
    assert_calls(plt.plot, plot_calls, rel=EPSILON)


@parametrize_with_cases(
    argnames="kwargs, plot_calls, plt", cases=THIS_MODULE, has_tag=HAS_LEGEND
)
def test_legend_was_called(kwargs, plot_calls, plt):
    plot_fitting(data=FIT_DATA, func=FUNC, title_name=TITLE_NAME, **kwargs)
    plt.legend.assert_called_once()


@parametrize_with_cases(
    argnames="kwargs, plot_calls, plt",
    cases=THIS_MODULE,
    filter=lambda case: HAS_LEGEND not in case._pytestcase.id,
)
def test_legend_was_not_called(kwargs, plot_calls, plt):
    plot_fitting(data=FIT_DATA, func=FUNC, a=A1, title_name=TITLE_NAME, legend=False)
    plt.legend.assert_not_called()


def test_plot_unknown_a_type(mock_plt):
    with pytest.raises(
        PlottingError,
        match=(
            "^3.4 has unmatching type. Can except only numpy arrays, "
            "lists of numpy arrays and dictionaries.$"
        ),
    ):
        plot_fitting(
            data=FIT_DATA, func=FUNC, a=3.4, title_name=TITLE_NAME, legend=False
        )
