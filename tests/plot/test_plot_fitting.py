import numpy as np
import pytest
from pytest_cases import THIS_MODULE, case, parametrize_with_cases

from eddington import FittingData, linear, plot_fitting
from eddington.exceptions import PlottingError
from eddington.plot import LineStyle
from tests.util import assert_calls

HAS_LEGEND = "has_legend"
DOES_NOT_HAVE_LEGEND = "does_not_have_legend"

EPSILON = 1e-5

FUNC = linear
X = np.arange(1, 11)
A1, A2, A3 = np.array([1, 1]), np.array([3, 2]), np.array([3.924356, 1.2345e-5])
A1_REPR, A2_REPR, A3_REPR = (
    "[a[0]=1.000, a[1]=1.000]",
    "[a[0]=3.000, a[1]=2.000]",
    "[a[0]=3.924, a[1]=1.234e-5]",
)
FIT_DATA = FittingData.random(FUNC, x=X, a=np.array([1, 2]), measurements=X.shape[0])
TITLE_NAME = "Title"


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_no_args(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=A1)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="solid"))
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_xmin(mock_figure):
    x = np.arange(-10, 10.9, step=0.0209)

    kwargs = dict(a=A1, xmin=-10)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="solid"))
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_xmax(mock_figure):
    x = np.arange(0.1, 20, step=0.0199)

    kwargs = dict(a=A1, xmax=20)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="solid"))
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_step(mock_figure):
    x = np.arange(0.1, 10.9, step=0.1)

    kwargs = dict(a=A1, step=0.1)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="solid"))
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[HAS_LEGEND])
def case_a_list_with_legend(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=[A1, A2])
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="solid")),
        ([x, FUNC(A2, x)], dict(label=A2_REPR, color=None, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_list_without_legend(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=[A1, A2], legend=False)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="solid")),
        ([x, FUNC(A2, x)], dict(label=A2_REPR, color=None, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[HAS_LEGEND])
def case_a_dict_with_legend(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    one = "one"
    two = "two"
    kwargs = dict(a={one: A1, two: A2})
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=one, color=None, linestyle="solid")),
        ([x, FUNC(A2, x)], dict(label=two, color=None, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_dict_without_legend(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    one = "one"
    two = "two"
    kwargs = dict(a={one: A1, two: A2}, legend=False)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=one, color=None, linestyle="solid")),
        ([x, FUNC(A2, x)], dict(label=two, color=None, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_redundent_precision(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)

    kwargs = dict(a=A3)
    plot_calls = [
        ([x, FUNC(A3, x)], dict(label=A3_REPR, color=None, linestyle="solid"))
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_string_color(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    color = "yellow"

    kwargs = dict(a=A1, color=color)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=color, linestyle="solid"))
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_list_with_colors_list(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    color1, color2 = "green", "yellow"

    kwargs = dict(a=[A1, A2], legend=False, color=[color1, color2])
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=color1, linestyle="solid")),
        ([x, FUNC(A2, x)], dict(label=A2_REPR, color=color2, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_list_with_colors_list_shorter(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    color1, color2 = "green", "yellow"

    kwargs = dict(a=[A1, A2, A3], legend=False, color=[color1, color2])
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=color1, linestyle="solid")),
        ([x, FUNC(A2, x)], dict(label=A2_REPR, color=color2, linestyle="solid")),
        ([x, FUNC(A3, x)], dict(label=A3_REPR, color=None, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_dict_with_color_dict(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    label1, label2 = "label1", "label2"
    color1, color2 = "green", "yellow"
    kwargs = dict(a={label1: A1, label2: A2}, color={label1: color1, label2: color2})
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=label1, color=color1, linestyle="solid")),
        ([x, FUNC(A2, x)], dict(label=label2, color=color2, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_enum_linestyle(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    linestyle = LineStyle.DOTTED

    kwargs = dict(a=A1, linestyle=linestyle)
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="dotted"))
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_list_with_linestyles_list(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    linestyle1, linestyle2 = LineStyle.DOTTED, LineStyle.DASHED

    kwargs = dict(a=[A1, A2], legend=False, linestyle=[linestyle1, linestyle2])
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="dotted")),
        ([x, FUNC(A2, x)], dict(label=A2_REPR, color=None, linestyle="dashed")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_list_with_linestyles_list_shorter(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    linestyle1, linestyle2 = LineStyle.DOTTED, LineStyle.DASHED

    kwargs = dict(a=[A1, A2, A3], legend=False, linestyle=[linestyle1, linestyle2])
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=A1_REPR, color=None, linestyle="dotted")),
        ([x, FUNC(A2, x)], dict(label=A2_REPR, color=None, linestyle="dashed")),
        ([x, FUNC(A3, x)], dict(label=A3_REPR, color=None, linestyle="solid")),
    ]
    return kwargs, plot_calls, mock_figure


@case(tags=[DOES_NOT_HAVE_LEGEND])
def case_a_dict_with_linestyle_dict(mock_figure):
    x = np.arange(0.1, 10.9, step=0.0108)
    label1, label2 = "label1", "label2"
    linestyle1, linestyle2 = LineStyle.DOTTED, LineStyle.DASHED
    kwargs = dict(
        a={label1: A1, label2: A2}, linestyle={label1: linestyle1, label2: linestyle2}
    )
    plot_calls = [
        ([x, FUNC(A1, x)], dict(label=label1, color=None, linestyle="dotted")),
        ([x, FUNC(A2, x)], dict(label=label2, color=None, linestyle="dashed")),
    ]
    return kwargs, plot_calls, mock_figure


@parametrize_with_cases(argnames=["kwargs", "plot_calls", "figure"], cases=THIS_MODULE)
def test_plot_fitting(kwargs, plot_calls, figure):
    plot_fitting(data=FIT_DATA, func=FUNC, title_name=TITLE_NAME, **kwargs)
    ax = figure.add_subplot.return_value
    assert_calls(ax.plot, plot_calls, rel=EPSILON)


@parametrize_with_cases(
    argnames=["kwargs", "plot_calls", "figure"], cases=THIS_MODULE, has_tag=HAS_LEGEND
)
def test_legend_was_called(kwargs, plot_calls, figure):
    plot_fitting(data=FIT_DATA, func=FUNC, title_name=TITLE_NAME, **kwargs)
    ax = figure.add_subplot.return_value
    ax.legend.assert_called_once_with()


@parametrize_with_cases(
    argnames=["kwargs", "plot_calls", "figure"],
    cases=THIS_MODULE,
    has_tag=DOES_NOT_HAVE_LEGEND,
)
def test_legend_was_not_called(kwargs, plot_calls, figure):
    plot_fitting(data=FIT_DATA, func=FUNC, a=A1, title_name=TITLE_NAME, legend=False)
    ax = figure.add_subplot.return_value
    ax.legend.assert_not_called()


def test_plot_unknown_a_type(mock_figure):
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


@parametrize_with_cases(argnames=["kwargs", "plot_calls", "figure"], cases=THIS_MODULE)
def test_plot_with_data_color(kwargs, plot_calls, figure):
    color = "yellow"
    plot_fitting(
        data=FIT_DATA, func=FUNC, title_name=TITLE_NAME, data_color=color, **kwargs
    )
    ax = figure.add_subplot.return_value
    assert_calls(
        ax.errorbar,
        [
            (
                [],
                dict(
                    x=FIT_DATA.x,
                    y=FIT_DATA.y,
                    xerr=FIT_DATA.xerr,
                    yerr=FIT_DATA.yerr,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=color,
                    mec=color,
                ),
            )
        ],
        rel=EPSILON,
    )
