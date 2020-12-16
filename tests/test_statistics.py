import pytest

from eddington.statistics import Statistics
from tests.util import assert_statistics

EPSILON = 1e-3


def test_calculate_statistics_for_one_value():
    k = 6
    values = [k]
    stats = Statistics.from_array(values)
    assert_statistics(
        stats,
        Statistics(
            mean=k,
            median=k,
            variance=0,
            standard_deviation=0,
            maximum_value=k,
            minimum_value=k,
        ),
        rel=EPSILON,
    )


def test_calculate_statistics_for_two_values():
    a, b = 5, 8
    values = [a, b]
    stats = Statistics.from_array(values)
    assert_statistics(
        stats,
        Statistics(
            mean=(a + b) / 2,
            median=(a + b) / 2,
            variance=(b - a) ** 2 / 4,
            standard_deviation=(b - a) / 2,
            maximum_value=b,
            minimum_value=a,
        ),
        rel=EPSILON,
    )


def test_calculate_statistics_for_three_values():
    values = [5, 9, 8]
    stats = Statistics.from_array(values)
    assert_statistics(
        stats,
        Statistics(
            mean=7.3333,
            median=8,
            variance=2.888,
            standard_deviation=1.699,
            maximum_value=9,
            minimum_value=5,
        ),
        rel=EPSILON,
    )


def test_calculate_statistics_raises_error_for_no_values():
    with pytest.raises(ValueError, match="^Cannot calculate statistics of no values.$"):
        Statistics.from_array([])
