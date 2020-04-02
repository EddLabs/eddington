import numpy as np

from eddington.fit_data import FitData
from eddington.fit_functions.fit_function import FitFunction
from eddington.fit_util import fit_to_data


class FitResultBaseTestCase:
    default_decimal = 2

    def __init__(
        self,
        name: str,
        func: FitFunction,
        data: FitData,
        a0,
        a,
        aerr,
        arerr,
        chi2,
        degrees_of_freedom,
        chi2_reduced,
        p_probability,
        decimal=default_decimal,
    ):
        self.name = name
        self.func = func
        self.data = data
        self.a0 = a0
        self.a = a
        self.aerr = aerr
        self.arerr = arerr
        self.chi2 = chi2
        self.degrees_of_freedom = degrees_of_freedom
        self.chi2_reduced = chi2_reduced
        self.p_probability = p_probability
        self.result = fit_to_data(func=self.func, data=self.data, a0=a0)
        self.decimal = decimal


def add_integration_tests(cls, case: FitResultBaseTestCase):
    def test_a(self):
        np.testing.assert_almost_equal(
            case.a,
            case.result.a,
            decimal=case.decimal,
            err_msg="A is different than expected",
        )

    def test_aerr(self):
        np.testing.assert_almost_equal(
            case.aerr,
            case.result.aerr,
            decimal=case.decimal,
            err_msg="A relative error is different than expected",
        )

    def test_arerr(self):
        np.testing.assert_almost_equal(
            case.arerr,
            case.result.arerr,
            decimal=case.decimal,
            err_msg="A relative error is different than expected",
        )

    def test_chi2(self):
        self.assertAlmostEqual(
            case.chi2,
            case.result.chi2,
            places=case.decimal,
            msg="Chi2 is different than expected",
        )

    def test_degrees_of_freedom(self):
        self.assertEqual(
            case.degrees_of_freedom,
            case.result.degrees_of_freedom,
            msg="Degrees of freedom are different than expected",
        )

    def test_chi2_reduced(self):
        self.assertAlmostEqual(
            case.chi2_reduced,
            case.result.chi2_reduced,
            places=case.decimal,
            msg="Chi2 reduced is different than expected",
        )

    def test_p_probability(self):
        self.compare_small(
            case.p_probability,
            case.result.p_probability,
            title="P probability",
            decimal=case.decimal,
        )

    setattr(cls, f"test_a___{case.name}", test_a)
    setattr(cls, f"test_aerr___{case.name}", test_aerr)
    setattr(cls, f"test_arerr___{case.name}", test_arerr)
    setattr(cls, f"test_chi2___{case.name}", test_chi2)
    setattr(cls, f"test_degrees_of_freedom___{case.name}", test_degrees_of_freedom)
    setattr(cls, f"test_chi2_reduced___{case.name}", test_chi2_reduced)
    setattr(cls, f"test_p_probability___{case.name}", test_p_probability)
