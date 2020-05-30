import numpy as np
from eddington_test import MetaTestCase

from eddington_core import fit_to_data


class FittingAlgorithmMetaTestCase(MetaTestCase):
    def setUp(self):
        if self.fix is not None:
            for index, value in self.fix:
                self.func.fix(index, value)
        self.result = fit_to_data(data=self.data, func=self.func)

    def tearDown(self):
        self.func.clear_fixed()

    def test_aerr(self):  # pylint: disable=W0613
        np.testing.assert_almost_equal(
            self.result.aerr,
            self.expected_result["aerr"],
            decimal=self.decimal,
            err_msg="A relative error is different than expected",
        )

    def test_arerr(self):  # pylint: disable=W0613
        np.testing.assert_almost_equal(
            self.result.arerr,
            self.expected_result["arerr"],
            decimal=self.decimal,
            err_msg="A relative error is different than expected",
        )

    def test_acov(self):
        old_precision = np.get_printoptions()["precision"]
        np.set_printoptions(precision=8)
        expected_acov = np.array(self.expected_result["acov"])
        self.assertEqual(
            self.result.acov.shape,
            expected_acov.shape,
            msg="Covariance matrix shape is different than expected.",
        )
        for i in range(self.result.acov.shape[0]):
            for j in range(self.result.acov.shape[0]):
                self.assertSmallEqual(
                    self.result.acov[i, j],
                    expected_acov[i, j],
                    places=self.decimal,
                    msg=f"""Covariance matrix is different in coordinate ({i}, {j}).
    Expected matrix:
    {self.result.acov}
    Actual matrix:
    {expected_acov}""",
                )
        np.set_printoptions(precision=old_precision)

    def test_chi2(self):
        self.assertAlmostEqual(
            self.result.chi2,
            self.expected_result["chi2"],
            places=self.decimal,
            msg="Chi2 is different than expected",
        )

    def test_degrees_of_freedom(self):
        self.assertEqual(
            self.result.degrees_of_freedom,
            self.expected_result["degrees_of_freedom"],
            msg="Degrees of freedom are different than expected",
        )

    def test_chi2_reduced(self):
        self.assertAlmostEqual(
            self.result.chi2_reduced,
            self.expected_result["chi2_reduced"],
            places=self.decimal,
            msg="Chi2 reduced is different than expected",
        )

    def test_p_probability(self):
        self.assertSmallEqual(
            self.result.p_probability,
            self.expected_result["p_probability"],
            places=self.decimal,
            msg="P probability is different than expected. "
            f"{self.result.p_probability} != {self.expected_result['p_probability']}",
        )

    def test_a(self):  # pylint: disable=W0613
        np.testing.assert_almost_equal(
            self.result.a,
            self.expected_result["a"],
            decimal=self.decimal,
            err_msg="A is different than expected",
        )
