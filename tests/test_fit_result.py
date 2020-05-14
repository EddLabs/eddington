import warnings
from unittest import TestCase

import numpy as np
from eddington_core import FitResult
from mock import mock_open, patch


class FitResultBaseTestCase:

    decimal = 5

    def setUp(self):
        self.fit_result = FitResult(
            a0=self.a0,
            a=self.a,
            aerr=self.aerr,
            acov=self.acov,
            chi2=self.chi2,
            degrees_of_freedom=self.degrees_of_freedom,
        )

    def test_a0(self):
        np.testing.assert_almost_equal(
            self.fit_result.a0,
            self.a0,
            decimal=self.decimal,
            err_msg="Initial Guess is different than expected",
        )

    def test_a(self):
        np.testing.assert_almost_equal(
            self.fit_result.a,
            self.a,
            decimal=self.decimal,
            err_msg="Calculated parameters are different than expected",
        )

    def test_aerr(self):
        np.testing.assert_almost_equal(
            self.fit_result.aerr,
            self.aerr,
            decimal=self.decimal,
            err_msg="Parameters errors are different than expected",
        )

    def test_arerr(self):
        np.testing.assert_almost_equal(
            self.fit_result.arerr,
            self.arerr,
            decimal=self.decimal,
            err_msg="Parameters relative errors are different than expected",
        )

    def test_acov(self):
        np.testing.assert_almost_equal(
            self.fit_result.acov,
            self.acov,
            decimal=self.decimal,
            err_msg="Parameters covariance is different than expected",
        )

    def test_chi2(self):
        self.assertAlmostEqual(
            self.fit_result.chi2,
            self.chi2,
            places=self.decimal,
            msg="Chi2 is different than expected",
        )

    def test_chi2_reduced(self):
        self.assertAlmostEqual(
            self.fit_result.chi2_reduced,
            self.chi2_reduced,
            places=self.decimal,
            msg="Chi2 reduced is different than expected",
        )

    def test_degrees_of_freedom(self):
        self.assertEqual(
            self.fit_result.degrees_of_freedom,
            self.degrees_of_freedom,
            msg="Degrees of freedom are different than expected",
        )

    def test_p_probability(self):
        self.assertAlmostEqual(
            self.fit_result.p_probability,
            self.p_probability,
            places=self.decimal,
            msg="Chi2 reduced is different than expected",
        )

    def test_representation(self):
        self.assertEqual(
            self.repr_string,
            str(self.fit_result),
            msg="Representation is different than expected",
        )

    def test_export_to_file(self):
        path = "/path/to/output"
        mock_open_obj = mock_open()
        with patch("eddington_core.fit_result.open", mock_open_obj):
            self.fit_result.export_to_file(path)
            mock_open_obj.assert_called_once_with(path, mode="w")
            mock_open_obj.return_value.write.assert_called_with(self.repr_string)


class TestStandardFitResult(TestCase, FitResultBaseTestCase):

    a0 = [1.0, 3.0]
    a = [1.1, 2.98]
    aerr = [0.1, 0.76]
    acov = [[0.01, 2.3], [2.3, 0.988]]
    chi2 = 8.276
    degrees_of_freedom = 5
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [9.09091, 25.50336]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 1.100 \u00B1 0.1000 (9.091% error)
\ta[1] = 2.980 \u00B1 0.7600 (25.503% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""

    def setUp(self):
        FitResultBaseTestCase.setUp(self)


class TestFitResultWithZeroError(TestCase, FitResultBaseTestCase):

    a0 = [1.0, 3.0]
    a = [1.1, 2.98]
    aerr = [0.0, 0.0]
    acov = [[0.0, 0.0], [0.0, 0.0]]
    chi2 = 8.276
    degrees_of_freedom = 5
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [0.0, 0.0]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 1.100 ± 0.000 (0.000% error)
\ta[1] = 2.980 ± 0.000 (0.000% error)
Fitted parameters covariance:
[[0. 0.]
 [0. 0.]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""

    def setUp(self):
        FitResultBaseTestCase.setUp(self)


class TestFitResultWithZeroValue(TestCase, FitResultBaseTestCase):

    a0 = [1.0, 3.0]
    a = [0.0, 0.0]
    aerr = [0.1, 0.76]
    acov = [[0.01, 2.3], [2.3, 0.988]]
    chi2 = 8.276
    degrees_of_freedom = 5
    chi2_reduced = 1.6552
    p_probability = 0.14167
    arerr = [np.inf, np.inf]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 0.000 \u00B1 0.1000 (inf% error)
\ta[1] = 0.000 \u00B1 0.7600 (inf% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 8.276
Degrees of freedom: 5
Chi squared reduced: 1.655
P-probability: 0.1417
"""

    def setUp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            FitResultBaseTestCase.setUp(self)


class TestFitResultWithSmallPProbability(TestCase, FitResultBaseTestCase):

    a0 = [1.0, 3.0]
    a = [1.1, 2.98]
    aerr = [0.1, 0.76]
    acov = [[0.01, 2.3], [2.3, 0.988]]
    chi2 = 43.726
    degrees_of_freedom = 5
    chi2_reduced = 8.7452
    p_probability = 2.63263e-8
    arerr = [9.09091, 25.50336]
    repr_string = """Results:
========

Initial parameters' values:
\t1.0 3.0
Fitted parameters' values:
\ta[0] = 1.100 \u00B1 0.1000 (9.091% error)
\ta[1] = 2.980 \u00B1 0.7600 (25.503% error)
Fitted parameters covariance:
[[0.01  2.3  ]
 [2.3   0.988]]
Chi squared: 43.726
Degrees of freedom: 5
Chi squared reduced: 8.745
P-probability: 2.633e-08
"""

    def setUp(self):
        FitResultBaseTestCase.setUp(self)
