from unittest import TestCase

import numpy as np
from mock import patch, DEFAULT, call

from eddington_core import FitData
from eddington_core.consts import (
    DEFAULT_MIN_COEFF,
    DEFAULT_MAX_COEFF,
    DEFAULT_XMIN,
    DEFAULT_XMAX,
    DEFAULT_XSIGMA,
    DEFAULT_YSIGMA,
    DEFAULT_MEASUREMENTS,
)
from tests.dummy_functions import dummy_func1


class RandomFitDataTestCase:
    decimal = 5

    func = dummy_func1
    amin = DEFAULT_MIN_COEFF
    amax = DEFAULT_MAX_COEFF
    xmin = DEFAULT_XMIN
    xmax = DEFAULT_XMAX
    xsigma = DEFAULT_XSIGMA
    ysigma = DEFAULT_YSIGMA
    measurements = DEFAULT_MEASUREMENTS
    args = {}

    a = None

    def setUp(self):
        random_utils = patch.multiple(
            "eddington_core.fit_data",
            random_array=DEFAULT,
            random_sigma=DEFAULT,
            random_error=DEFAULT,
        )
        random_utils_obj = random_utils.start()
        self.random_array = random_utils_obj["random_array"]
        self.random_sigma = random_utils_obj["random_sigma"]
        self.random_error = random_utils_obj["random_error"]

        self.addCleanup(random_utils.stop)

        if self.a is None:
            self.a = np.random.randint(1, 5, size=self.func.n)
        self.x = np.arange(0, self.measurements)
        self.xerr = np.random.normal(size=self.measurements)
        self.yerr = np.random.normal(size=self.measurements)
        self.real_xerr = np.random.normal(size=self.measurements)
        self.real_yerr = np.random.normal(size=self.measurements)
        self.y = self.func(self.a, self.x + self.real_xerr) + self.real_yerr

        self.set_random_array_side_effect()
        self.set_random_sigma_side_effect()
        self.set_random_error_side_effect()

        self.data = FitData.random(fit_func=self.func, **self.args)

    def set_random_array_side_effect(self):
        self.random_array.side_effect = [self.a, self.x]

    def set_random_sigma_side_effect(self):
        self.random_sigma.side_effect = [self.xerr, self.yerr]

    def set_random_error_side_effect(self):
        self.random_error.side_effect = [self.real_xerr, self.real_yerr]

    def test_x_data(self):
        np.testing.assert_almost_equal(
            self.x,
            self.data.x,
            decimal=self.decimal,
            err_msg="Random x value of data is different than expected",
        )

    def test_xerr_data(self):
        np.testing.assert_almost_equal(
            self.xerr,
            self.data.xerr,
            decimal=self.decimal,
            err_msg="Random x error value of data is different than expected",
        )

    def test_y_data(self):
        np.testing.assert_almost_equal(
            self.y,
            self.data.y,
            decimal=self.decimal,
            err_msg="Random y value of data is different than expected",
        )

    def test_yerr_data(self):
        np.testing.assert_almost_equal(
            self.yerr,
            self.data.yerr,
            decimal=self.decimal,
            err_msg="Random y error value of data is different than expected",
        )

    def test_random_array_calls(self):
        self.assertEqual(self.random_array.call_count, 2)
        self.assertEqual(
            self.random_array.call_args_list[0],
            call(min_val=self.amin, max_val=self.amax, n=self.func.n),
        )
        self.assertEqual(
            self.random_array.call_args_list[1],
            call(min_val=self.xmin, max_val=self.xmax, n=self.measurements),
        )

    def test_random_sigma_calls(self):
        self.assertEqual(self.random_sigma.call_count, 2)
        self.assertEqual(
            self.random_sigma.call_args_list[0],
            call(average_sigma=self.xsigma, n=self.measurements),
        )
        self.assertEqual(
            self.random_sigma.call_args_list[1],
            call(average_sigma=self.ysigma, n=self.measurements),
        )

    def test_random_error_calls(self):
        self.assertEqual(self.random_error.call_count, 2)
        self.assertEqual(
            self.random_error.call_args_list[0], call(scales=self.xerr),
        )
        self.assertEqual(
            self.random_error.call_args_list[1], call(scales=self.yerr),
        )


class TestDefaultRandomFitData(TestCase, RandomFitDataTestCase):
    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithMeasurements(TestCase, RandomFitDataTestCase):
    measurements = 50
    args = dict(measurements=measurements)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithMinCoeff(TestCase, RandomFitDataTestCase):
    amin = 2
    args = dict(min_coeff=amin)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithMaxCoeff(TestCase, RandomFitDataTestCase):
    amax = 50
    args = dict(max_coeff=amax)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithMinX(TestCase, RandomFitDataTestCase):
    xmin = 2
    args = dict(xmin=xmin)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithMaxX(TestCase, RandomFitDataTestCase):
    xmax = 45
    args = dict(xmax=xmax)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithXSigma(TestCase, RandomFitDataTestCase):
    xsigma = 2
    args = dict(xsigma=xsigma)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithYSigma(TestCase, RandomFitDataTestCase):
    ysigma = 2
    args = dict(ysigma=ysigma)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)


class TestDefaultRandomFitDataWithActualA(TestCase, RandomFitDataTestCase):
    a = np.random.randint(1, 5, size=RandomFitDataTestCase.func.n)
    args = dict(a=a)

    def setUp(self):
        RandomFitDataTestCase.setUp(self)

    def set_random_array_side_effect(self):
        self.random_array.side_effect = [self.x]

    def test_random_array_calls(self):
        self.assertEqual(self.random_array.call_count, 1)
        self.assertEqual(
            self.random_array.call_args_list[0],
            call(min_val=self.xmin, max_val=self.xmax, n=self.measurements),
        )
