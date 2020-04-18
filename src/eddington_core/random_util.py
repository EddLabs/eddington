import numpy as np


def random_array(min_val, max_val, n):
    return np.random.uniform(min_val, max_val, size=n)


def random_sigma(average_sigma, n):
    return np.random.exponential(average_sigma, size=n)


def random_error(scales):
    return np.random.normal(scale=scales)
