from pytest_cases import parametrize_with_cases

from tests.plot import cases
from tests.util import assert_calls

EPSILON = 1e-5


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[cases.case_plot_residuals]
)
def test_residuals_error_bar(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert (
        fig._actual_fig == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    y_residuals = cases.FIT_DATA.y - cases.FUNC(cases.A, cases.FIT_DATA.x)
    ax = mock_figure.add_subplot.return_value
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
                ),
            ),
        ],
        rel=EPSILON,
    )


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


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[cases.case_plot_residuals]
)
def test_plot_residuals_with_xmin(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, xmin=-10)
    assert (
        fig._actual_fig == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.hlines, [([0], dict(xmin=-10, xmax=10.9, linestyles="dashed"))], rel=EPSILON
    )


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[cases.case_plot_residuals]
)
def test_plot_residuals_with_xmax(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, xmax=20)
    assert (
        fig._actual_fig == mock_figure  # pylint: disable=protected-access
    ), "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.hlines, [([0], dict(xmin=0.1, xmax=20, linestyles="dashed"))], rel=EPSILON
    )
