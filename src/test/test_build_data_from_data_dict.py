from collections import OrderedDict
from unittest import TestCase
import numpy as np

from eddington_core import FitData


class TestBuildDataFromDataFrame(TestCase):

    decimal = 5

    x = np.array([4, -2, 6, 8.2, 9])
    xerr = np.array([0.1, 2.0, 1.6, 0.23, 1.0])
    y = np.array([1, 3, -9, -20, 6.58])
    yerr = np.array([1.3, 0.3, 0.62, 0.58, 2.1])
    data_frame = OrderedDict([("x", x), ("xerr", xerr), ("y", y), ("yerr", yerr)])
    data = FitData.build_from_data_dict(data_dict=data_frame)

    def test_x(self):
        np.testing.assert_almost_equal(
            self.data.x,
            self.x,
            decimal=self.decimal,
            err_msg="X is different than expected",
        )

    def test_y(self):
        np.testing.assert_almost_equal(
            self.data.y,
            self.y,
            decimal=self.decimal,
            err_msg="Y is different than expected",
        )

    def test_xerr(self):
        np.testing.assert_almost_equal(
            self.data.xerr,
            self.xerr,
            decimal=self.decimal,
            err_msg="X is different than expected",
        )

    def test_yerr(self):
        np.testing.assert_almost_equal(
            self.data.yerr,
            self.yerr,
            decimal=self.decimal,
            err_msg="Y is different than expected",
        )
