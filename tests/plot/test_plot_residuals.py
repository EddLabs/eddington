from pytest_cases import parametrize_with_cases

from tests.plot import cases
from tests.util import assert_calls

EPSILON = 1e-5


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[cases.case_plot_residuals]
)
def test_plot_residuals_without_boundaries(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert (
        fig._actual_fig == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.hlines, [([0], dict(xmin=0.1, xmax=10.9, linestyles="dashed"))], rel=EPSILON
    )
    y_residuals = cases.FIT_DATA.y - cases.FUNC(cases.A, cases.FIT_DATA.x)
    assert_calls(
        ax.errorbar,
        [
            (
                [],
                dict(
                    x=cases.FIT_DATA.x,
                    y=y_residuals,
                    xerr=cases.FIT_DATA.xerr,
                    yerr=cases.FIT_DATA.yerr,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=None,
                    mec=None,
                ),
            ),
        ],
        rel=EPSILON,
    )


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[cases.case_plot_residuals]
)
def test_plot_residuals_with_xmin(base_dict, plot_method, mock_figure):
    xmin = 4
    fig = plot_method(**base_dict, xmin=xmin)
    assert (
        fig._actual_fig == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.hlines, [([0], dict(xmin=xmin, xmax=10.9, linestyles="dashed"))], rel=EPSILON
    )
    y_residuals = cases.FIT_DATA.y - cases.FUNC(cases.A, cases.FIT_DATA.x)
    data_filter = [val >= xmin for val in cases.FIT_DATA.x]
    assert_calls(
        mock_figure.add_subplot.return_value.errorbar,
        [
            (
                [],
                dict(
                    x=cases.FIT_DATA.x[data_filter],
                    y=y_residuals[data_filter],
                    xerr=cases.FIT_DATA.xerr[data_filter],
                    yerr=cases.FIT_DATA.yerr[data_filter],
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=None,
                    mec=None,
                ),
            ),
        ],
        rel=EPSILON,
    )


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[cases.case_plot_residuals]
)
def test_plot_residuals_with_xmax(base_dict, plot_method, mock_figure):
    xmax = 7
    fig = plot_method(**base_dict, xmax=xmax)
    assert (
        fig._actual_fig == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.hlines, [([0], dict(xmin=0.1, xmax=xmax, linestyles="dashed"))], rel=EPSILON
    )
    y_residuals = cases.FIT_DATA.y - cases.FUNC(cases.A, cases.FIT_DATA.x)
    data_filter = [val <= xmax for val in cases.FIT_DATA.x]
    assert_calls(
        mock_figure.add_subplot.return_value.errorbar,
        [
            (
                [],
                dict(
                    x=cases.FIT_DATA.x[data_filter],
                    y=y_residuals[data_filter],
                    xerr=cases.FIT_DATA.xerr[data_filter],
                    yerr=cases.FIT_DATA.yerr[data_filter],
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=None,
                    mec=None,
                ),
            ),
        ],
        rel=EPSILON,
    )


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[cases.case_plot_residuals]
)
def test_plot_residuals_with_color(base_dict, plot_method, mock_figure):
    color = "blue"
    fig = plot_method(**base_dict, color=color)
    assert (
        fig._actual_fig == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.hlines, [([0], dict(xmin=0.1, xmax=10.9, linestyles="dashed"))], rel=EPSILON
    )
    y_residuals = cases.FIT_DATA.y - cases.FUNC(cases.A, cases.FIT_DATA.x)
    assert_calls(
        mock_figure.add_subplot.return_value.errorbar,
        [
            (
                [],
                dict(
                    x=cases.FIT_DATA.x,
                    y=y_residuals,
                    xerr=cases.FIT_DATA.xerr,
                    yerr=cases.FIT_DATA.yerr,
                    markersize=1,
                    marker="o",
                    linestyle="None",
                    label=None,
                    ecolor=color,
                    mec=color,
                ),
            ),
        ],
        rel=EPSILON,
    )
