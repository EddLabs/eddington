"""Functions for data randomization."""
import numpy as np


def random_array(min_val, max_val, size):
    """
    Creates a random array.

    :param min_val: Minimum value for array item.
    :param max_val: Maximum value for array item.
    :param size: Size of the array.
    :return: array
    """
    return np.random.uniform(min_val, max_val, size=size)


def random_sigma(average_sigma, size):
    """
    Creates random standard deviation (aka. sigma).

    :param average_sigma: average value for sigma.
    :param size: Size of the returned array.
    :return: array of sigma values.
    """
    return np.random.exponential(average_sigma, size=size)


def random_error(scales):
    """
    Generates random errors based on an array of scales.

    :param scales: array of standard deviations.
    :return: errors array.
    """
    return np.random.normal(scale=scales)
