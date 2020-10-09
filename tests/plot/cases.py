import numpy as np

from eddington import FittingData, linear, plot_data, plot_fitting, plot_residuals

FUNC = linear
X = np.arange(1, 11)
A = np.array([1, 2])
FIT_DATA = FittingData.random(FUNC, x=X, a=A, measurements=X.shape[0])
TITLE_NAME = "Title"


def case_plot_data():
    return dict(data=FIT_DATA, title_name=TITLE_NAME), plot_data


def case_plot_fitting():
    return dict(func=FUNC, data=FIT_DATA, a=A, title_name=TITLE_NAME), plot_fitting


def case_plot_residuals():
    return dict(func=FUNC, data=FIT_DATA, a=A, title_name=TITLE_NAME), plot_residuals
