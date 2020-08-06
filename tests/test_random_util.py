import pytest
import numpy as np

from eddington.random_util import random_array, random_error, random_sigma


n = 35
returned = np.random.uniform(size=n)


@pytest.fixture
def uniform(mocker):
    mock = mocker.patch("numpy.random.uniform")
    mock.return_value = returned
    return mock


@pytest.fixture
def exponential(mocker):
    mock = mocker.patch("numpy.random.exponential")
    mock.return_value = returned
    return mock


@pytest.fixture
def normal(mocker):
    mock = mocker.patch("numpy.random.normal")
    mock.return_value = returned
    return mock


def test_random_array(uniform):
    min_val = 1
    max_val = 5
    actual = random_array(min_val=min_val, max_val=max_val, size=n)
    assert actual == pytest.approx(returned), "Random array returned unexpected array"
    uniform.assert_called_once_with(min_val, max_val, size=n)


def test_random_sigma(exponential):
    average_sigma = 0.5
    actual = random_sigma(average_sigma=average_sigma, size=n)
    assert actual == pytest.approx(returned), "Random sigma returned unexpected array"
    exponential.assert_called_once_with(average_sigma, size=n)


def test_random_error(normal):
    scales = np.arange(0, n)
    actual = random_error(scales=scales)
    assert actual == pytest.approx(returned), "Random error returned unexpected array"
    normal.assert_called_once_with(scale=scales)
