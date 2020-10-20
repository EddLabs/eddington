from pytest_cases import parametrize_with_cases

from tests.plot import cases
from tests.util import assert_calls

EPSILON = 1e-5


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[cases.case_plot_data])
def test_plot_data_without_boundaries(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict)
    assert fig._actual_fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.set_xlim, [([], dict(left=0.1)), ([], dict(right=10.9))], rel=EPSILON
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[cases.case_plot_data])
def test_plot_data_with_xmin(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, xmin=-10)
    assert fig._actual_fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(
        ax.set_xlim, [([], dict(left=-10)), ([], dict(right=10.9))], rel=EPSILON
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[cases.case_plot_data])
def test_plot_data_with_xmax(base_dict, plot_method, mock_figure):
    fig = plot_method(**base_dict, xmax=20)
    assert fig._actual_fig == mock_figure, "Figure is different than expected"
    ax = mock_figure.add_subplot.return_value
    assert_calls(ax.set_xlim, [([], dict(left=0.1)), ([], dict(right=20))], rel=EPSILON)
