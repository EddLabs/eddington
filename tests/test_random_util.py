from unittest import TestCase

import numpy as np
from mock import patch

from eddington_core.random_util import random_array, random_sigma, random_error


class TestRandomUtil(TestCase):

    n = 35
    returned = np.random.uniform(size=n)

    @patch("numpy.random.uniform")
    def test_random_array(self, uniform):
        uniform.return_value = self.returned
        min_val = 1
        max_val = 5
        actual = random_array(min_val=min_val, max_val=max_val, n=self.n)
        np.testing.assert_equal(
            actual, self.returned, err_msg="Random array returned unexpected array"
        )
        uniform.assert_called_once_with(min_val, max_val, size=self.n)

    @patch("numpy.random.exponential")
    def test_random_sigma(self, exponential):
        exponential.return_value = self.returned
        average_sigma = 0.5
        actual = random_sigma(average_sigma=average_sigma, n=self.n)
        np.testing.assert_equal(
            actual, self.returned, err_msg="Random sigma returned unexpected array"
        )
        exponential.assert_called_once_with(average_sigma, size=self.n)

    @patch("numpy.random.normal")
    def test_random_error(self, normal):
        normal.return_value = self.returned
        scales = np.arange(0, self.n)
        actual = random_error(scales=scales)
        np.testing.assert_equal(
            actual, self.returned, err_msg="Random error returned unexpected array"
        )
        normal.assert_called_once_with(scale=scales)
