import numpy as np
from pytest_cases import fixture, parametrize_with_cases, THIS_MODULE

from eddington import plot_data, FitData, linear
from tests.util import assert_dict_equal

FUNC = linear
A = np.array([1, 2])
FIT_DATA = FitData.random(FUNC)
TITLE_NAME = "Title"


@fixture
def mock_plt(mocker):
    return mocker.patch("eddington.plot.plt")


def case_plot_data():
    return dict(data=FIT_DATA, title_name=TITLE_NAME), plot_data


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


@parametrize_with_cases(argnames="base_dict, plot_method", cases=THIS_MODULE)
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
