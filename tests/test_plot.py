import numpy as np
from pytest_cases import fixture, parametrize_with_cases, THIS_MODULE

from eddington import plot_data, FitData, linear, plot_fitting, plot_residuals
from tests.util import assert_dict_equal, assert_list_equal

EPSILON = 1e-5

FUNC = linear
X = np.arange(1, 11)
A = np.array([1, 2])
FIT_DATA = FitData.random(FUNC, x=X, a=A, measurements=X.shape[0])
TITLE_NAME = "Title"


@fixture
def mock_plt(mocker):
    return mocker.patch("eddington.plot.plt")


def case_plot_data():
    return dict(data=FIT_DATA, title_name=TITLE_NAME), plot_data


def case_plot_fitting():
    return dict(func=FUNC, data=FIT_DATA, a=A, title_name=TITLE_NAME), plot_fitting


def case_plot_residuals():
    return dict(func=FUNC, data=FIT_DATA, a=A, title_name=TITLE_NAME), plot_residuals


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_simple_plot(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict)
    assert fig == mock_plt.figure.return_value, "Figure is different than expected"
    mock_plt.title.assert_called_once_with(TITLE_NAME, figure=fig)
    mock_plt.xlabel.assert_not_called()
    mock_plt.ylabel.assert_not_called()
    mock_plt.grid.assert_not_called()


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_plot_with_xlabel(base_dict, plot_method, mock_plt):
    xlabel = "X Label"
    fig = plot_method(**base_dict, xlabel=xlabel)
    mock_plt.xlabel.assert_called_once_with(xlabel, figure=fig)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_plot_with_ylabel(base_dict, plot_method, mock_plt):
    ylabel = "Y Label"
    fig = plot_method(**base_dict, ylabel=ylabel)
    mock_plt.ylabel.assert_called_once_with(ylabel, figure=fig)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
def test_plot_with_grid(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict, grid=True)
    mock_plt.grid.assert_called_once_with(True, figure=fig)


@parametrize_with_cases(
    argnames="base_dict, plot_method", cases=[case_plot_data, case_plot_fitting]
)
def test_error_bar(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict)
    assert_dict_equal(
        mock_plt.errorbar.call_args_list[0][1],
        dict(
            x=FIT_DATA.x,
            y=FIT_DATA.y,
            xerr=FIT_DATA.xerr,
            yerr=FIT_DATA.yerr,
            markersize=1,
            marker="o",
            linestyle="None",
            figure=fig,
        ),
        rel=1e-5,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_residuals_error_bar(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict)
    y_residuals = FIT_DATA.y - FUNC(A, FIT_DATA.x)
    assert_dict_equal(
        mock_plt.errorbar.call_args_list[0][1],
        dict(
            x=FIT_DATA.x,
            y=y_residuals,
            xerr=FIT_DATA.xerr,
            yerr=FIT_DATA.yerr,
            markersize=1,
            marker="o",
            linestyle="None",
            figure=fig,
        ),
        rel=1e-5,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_fitting])
def test_plot_fitting_without_boundaries(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict)
    x = np.arange(0.1, 10.9, step=0.0108)
    assert mock_plt.plot.call_count == 1
    assert_list_equal(mock_plt.plot.call_args_list[0][0], [x, FUNC(A, x)], rel=EPSILON)
    assert_dict_equal(mock_plt.plot.call_args_list[0][1], dict(figure=fig), rel=EPSILON)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_fitting])
def test_plot_fitting_with_xmin(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict, xmin=-10)
    x = np.arange(-10, 10.9, step=0.0209)
    assert mock_plt.plot.call_count == 1
    assert_list_equal(mock_plt.plot.call_args_list[0][0], [x, FUNC(A, x)], rel=EPSILON)
    assert_dict_equal(mock_plt.plot.call_args_list[0][1], dict(figure=fig), rel=EPSILON)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_fitting])
def test_plot_fitting_with_xmax(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict, xmax=20)
    x = np.arange(0.1, 20, step=0.0199)
    assert mock_plt.plot.call_count == 1
    assert_list_equal(mock_plt.plot.call_args_list[0][0], [x, FUNC(A, x)], rel=EPSILON)
    assert_dict_equal(mock_plt.plot.call_args_list[0][1], dict(figure=fig), rel=EPSILON)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_fitting])
def test_plot_fitting_with_step(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict, step=0.1)
    x = np.arange(0.1, 10.9, step=0.1)
    assert mock_plt.plot.call_count == 1
    assert_list_equal(mock_plt.plot.call_args_list[0][0], [x, FUNC(A, x)], rel=EPSILON)
    assert_dict_equal(mock_plt.plot.call_args_list[0][1], dict(figure=fig), rel=EPSILON)


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_plot_residuals_without_boundaries(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict)
    assert mock_plt.hlines.call_count == 1
    assert_list_equal(mock_plt.hlines.call_args_list[0][0], [0], rel=EPSILON)
    assert_dict_equal(
        mock_plt.hlines.call_args_list[0][1],
        dict(xmin=0.1, xmax=10.9, linestyles="dashed", figure=fig),
        rel=EPSILON,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_plot_residuals_with_xmin(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict, xmin=-10)
    assert mock_plt.hlines.call_count == 1
    assert_list_equal(mock_plt.hlines.call_args_list[0][0], [0], rel=EPSILON)
    assert_dict_equal(
        mock_plt.hlines.call_args_list[0][1],
        dict(xmin=-10, xmax=10.9, linestyles="dashed", figure=fig),
        rel=EPSILON,
    )


@parametrize_with_cases(argnames="base_dict, plot_method", cases=[case_plot_residuals])
def test_plot_residuals_with_xmax(base_dict, plot_method, mock_plt):
    fig = plot_method(**base_dict, xmax=20)
    assert mock_plt.hlines.call_count == 1
    assert_list_equal(mock_plt.hlines.call_args_list[0][0], [0], rel=EPSILON)
    assert_dict_equal(
        mock_plt.hlines.call_args_list[0][1],
        dict(xmin=0.1, xmax=20, linestyles="dashed", figure=fig),
        rel=EPSILON,
    )
