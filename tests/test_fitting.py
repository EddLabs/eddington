from typing import Dict

import numpy as np
from eddington_test import MetaTestCase
from mock import DEFAULT, patch

from eddington_core import FitData, fit_function, fit_to_data


def dummy_func_x_derivative(a, x):
    return 2 * a[0] * x


def dummy_func_a_derivative(a, x):  # pylint: disable=W0613
    return np.stack([x ** 2, np.ones(shape=x.shape)])


@fit_function(n=2, save=False)
def dummy_func(a, x):
    return a[0] * x ** 2 + a[1]


@fit_function(n=2, x_derivative=dummy_func_x_derivative, save=False)
def dummy_func_with_x_derivative(a, x):
    return a[0] * x ** 2 + a[1]


@fit_function(n=2, a_derivative=dummy_func_a_derivative, save=False)
def dummy_func_with_a_derivative(a, x):
    return a[0] * x ** 2 + a[1]


@fit_function(
    n=2,
    x_derivative=dummy_func_x_derivative,
    a_derivative=dummy_func_a_derivative,
    save=False,
)
def dummy_func_with_both_derivatives(a, x):
    return a[0] * x ** 2 + a[1]


class FittingBaseTestCase(MetaTestCase):
    a0 = np.array([8, 5])
    a = np.array([3, 4])
    aerr = np.array([0.1, 0.2])
    acov = np.array([[0.1, 0.2], [0.2, 0.3]])
    chi2 = 1.5

    kwargs: Dict = {}
    model_extra_kwargs: Dict = {}

    def setUp(self):
        odr_patcher = patch.multiple(
            "eddington_core.fitting", ODR=DEFAULT, RealData=DEFAULT, Model=DEFAULT
        )
        odr_patches_dict = odr_patcher.start()
        self.data = FitData.random(self.func)
        self.model = odr_patches_dict["Model"]
        self.real_data = odr_patches_dict["RealData"]
        self.odr = odr_patches_dict["ODR"]

        self.addCleanup(odr_patcher.stop)

        self.odr.return_value.run.return_value = type(
            "",
            (object,),
            dict(
                beta=self.a,
                sum_square=self.chi2,
                sd_beta=self.aerr,
                cov_beta=self.acov,
            ),
        )()

        self.result = fit_to_data(
            data=self.data, func=self.func, a0=self.a0, **self.kwargs
        )

    def test_fit_result(self):
        np.testing.assert_equal(
            self.result.a, self.a, err_msg="Result is different than expected"
        )

    def test_model(self):
        self.model.assert_called_once_with(fcn=self.func, **self.model_extra_kwargs)

    def test_real_data(self):
        self.assertEqual(
            self.real_data.call_args_list[0].kwargs.keys(),
            {"x", "y", "sx", "sy"},
            msg="Real data arguments are different than expected",
        )
        np.testing.assert_equal(
            self.real_data.call_args_list[0].kwargs["x"],
            self.data.x,
            err_msg="X is different than expected",
        )
        np.testing.assert_equal(
            self.real_data.call_args_list[0].kwargs["sx"],
            self.data.xerr,
            err_msg="X error is different than expected",
        )
        np.testing.assert_equal(
            self.real_data.call_args_list[0].kwargs["y"],
            self.data.y,
            err_msg="Y is different than expected",
        )
        np.testing.assert_equal(
            self.real_data.call_args_list[0].kwargs["sy"],
            self.data.yerr,
            err_msg="Y error is different than expected",
        )

    def test_odr(self):
        self.odr.assert_called_once_with(
            data=self.real_data.return_value,
            model=self.model.return_value,
            beta0=self.a0,
        )


class TestFittingWithoutDerivatives(metaclass=FittingBaseTestCase):
    func = dummy_func


class TestFittingWithXDerivative(metaclass=FittingBaseTestCase):
    func = dummy_func_with_x_derivative
    model_extra_kwargs = dict(fjacd=func.x_derivative)


class TestFittingWithADerivative(metaclass=FittingBaseTestCase):
    func = dummy_func_with_a_derivative
    model_extra_kwargs = dict(fjacb=func.a_derivative)


class TestFittingWithBothDerivatives(metaclass=FittingBaseTestCase):
    func = dummy_func_with_both_derivatives
    model_extra_kwargs = dict(fjacd=func.x_derivative, fjacb=func.a_derivative)


class TestFittingWithoutUseXDerivative(metaclass=FittingBaseTestCase):
    func = dummy_func_with_both_derivatives
    kwargs = dict(use_x_derivative=False)
    model_extra_kwargs = dict(fjacb=func.a_derivative)


class TestFittingWithoutUseADerivative(metaclass=FittingBaseTestCase):
    func = dummy_func_with_both_derivatives
    kwargs = dict(use_a_derivative=False)
    model_extra_kwargs = dict(fjacd=func.x_derivative)


class TestFittingWithA0None(metaclass=FittingBaseTestCase):
    a0 = None
    func = dummy_func

    def test_odr(self):
        self.odr.assert_called_once()
        self.assertEqual(
            list(self.odr.call_args.kwargs.keys()),
            ["data", "model", "beta0"],
            msg="Kwargs keys are different than expected",
        )
        self.assertEqual(
            self.real_data.return_value,
            self.odr.call_args.kwargs["data"],
            msg="Data is different than expected",
        )
        self.assertEqual(
            self.model.return_value,
            self.odr.call_args.kwargs["model"],
            msg="Model is different than expected",
        )
        np.testing.assert_equal(
            self.odr.call_args.kwargs["beta0"],
            np.array([1, 1]),
            err_msg="beta0 is different than expected",
        )
