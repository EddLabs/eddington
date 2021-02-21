import numpy as np
import pytest
from pytest_cases import parametrize

from eddington.exceptions import IntervalError, IntervalIntersectionError
from eddington.interval import Interval
from tests.util import assert_numpy_array_equal

EPSILON = 1e-5


def test_finite_interval_constructor():
    interval = Interval(0.5, 7)
    assert interval.min_val == 0.5
    assert interval.max_val == 7
    assert interval.mid_point == 3.75
    assert interval.is_finite()
    assert not interval.is_all()


def test_lower_bounded_interval_constructor():
    interval = Interval(0.5, None)
    assert interval.min_val == 0.5
    assert interval.max_val is None
    assert interval.mid_point is None
    assert not interval.is_finite()
    assert not interval.is_all()


def test_upper_bounded_interval_constructor():
    interval = Interval(None, 7)
    assert interval.min_val is None
    assert interval.max_val == 7
    assert interval.mid_point is None
    assert not interval.is_finite()
    assert not interval.is_all()


def test_all_interval_constructor():
    interval = Interval(None, None)
    assert interval.min_val is None
    assert interval.max_val is None
    assert interval.mid_point is None
    assert not interval.is_finite()
    assert interval.is_all()


@parametrize(
    argnames=["min_val", "max_val"],
    argvalues=[
        (0, 10),
        (0, None),
        (None, 10),
        (None, None),
    ],
)
def test_interval_unpacking(min_val, max_val):
    interval = Interval(min_val=min_val, max_val=max_val)
    interval_min, interval_max = interval
    assert interval_min == min_val
    assert interval_max == max_val


@parametrize(
    argnames=["interval", "min_val", "result"],
    argvalues=[
        (Interval.all(), 3, Interval(3, None)),
        (Interval(0.5, 7), 2, Interval(2, 7)),
        (Interval(0.5, 7), -1, Interval(-1, 7)),
        (Interval(0.5, 7), 7, Interval(7, 7)),
        (Interval(0.5, None), 2, Interval(2, None)),
        (Interval(0.5, None), -1, Interval(-1, None)),
        (Interval(None, 7), 2, Interval(2, 7)),
    ],
)
def test_interval_successful_set_minimum(interval, min_val, result):
    assert interval != result
    interval.min_val = min_val
    assert interval == result


@parametrize(
    argnames=["interval", "max_val", "result"],
    argvalues=[
        (Interval.all(), 3, Interval(None, 3)),
        (Interval(0.5, 7), 2, Interval(0.5, 2)),
        (Interval(0.5, 7), 8, Interval(0.5, 8)),
        (Interval(0.5, 7), 0.5, Interval(0.5, 0.5)),
        (Interval(0.5, None), 2, Interval(0.5, 2)),
        (Interval(None, 7), 2, Interval(None, 2)),
        (Interval(None, 7), 8, Interval(None, 8)),
    ],
)
def test_interval_successful_set_maximum(interval, max_val, result):
    assert interval != result
    interval.max_val = max_val
    assert interval == result


@parametrize(
    argnames=["interval", "size"],
    argvalues=[
        (Interval(0.5, 7), 6.5),
        (Interval.all(), None),
        (Interval(0.5, None), None),
        (Interval(None, 7), None),
    ],
)
def test_interval_size(interval, size):
    assert interval.size() == size


@parametrize(
    argnames=["interval", "value"],
    argvalues=[
        (Interval(0.5, 7), 3),
        (Interval(0.5, 7), 0.5),
        (Interval(0.5, 7), 7),
        (Interval(0.5, None), 3),
        (Interval(0.5, None), 0.5),
        (Interval(None, 7), 3),
        (Interval(None, 7), 7),
        (Interval.all(), 3),
        (Interval.all(), 0.5),
        (Interval.all(), 7),
    ],
)
def test_interval_contains(interval, value):
    assert value in interval, f"{value} is not in {interval}"


@parametrize(
    argnames=["interval", "value"],
    argvalues=[
        (Interval(0.5, 7), -3),
        (Interval(0.5, 7), 9),
        (Interval(0.5, None), -3),
        (Interval(None, 7), 8),
    ],
)
def test_interval_not_contains(interval, value):
    assert value not in interval, f"{value} is in {interval}"


@parametrize(
    argnames=["smaller", "bigger"],
    argvalues=[
        (Interval(2.5, 5), Interval(0.5, 7)),
        (Interval(2.5, 5), Interval(0.5, None)),
        (Interval(2.5, 5), Interval(None, 7)),
        (Interval(2.5, 5), Interval.all()),
        (Interval(2.5, None), Interval(0.5, None)),
        (Interval(2.5, None), Interval.all()),
        (Interval(None, 5), Interval(None, 7)),
        (Interval(None, 5), Interval.all()),
    ],
)
def test_intervals_gt_lt_relations(smaller, bigger):
    # Positive

    assert smaller <= bigger
    assert smaller < bigger
    assert bigger >= smaller
    assert bigger > smaller
    assert smaller != bigger

    # Negative

    assert not bigger <= smaller
    assert not bigger < smaller
    assert not smaller >= bigger
    assert not smaller > bigger


@parametrize(
    argnames=["interval1", "interval2"],
    argvalues=[
        (Interval(2.5, 5), Interval(2.5, 5)),
        (Interval(2.5, None), Interval(2.5, None)),
        (Interval(None, 5), Interval(None, 5)),
        (Interval.all(), Interval.all()),
    ],
)
def test_intervals_equality_relations(interval1, interval2):
    assert interval1 <= interval2
    assert interval1 >= interval2
    assert interval1 == interval2


def test_interval_not_equal_to_tuple():
    assert Interval(1, 2) != (1, 2)


@parametrize(
    argnames=["interval", "val", "result"],
    argvalues=[
        (Interval(2.5, 5), 0, Interval(2.5, 5)),
        (Interval(2.5, 5), 1, Interval(3.5, 6)),
        (Interval(2.5, 5), -1, Interval(1.5, 4)),
        (Interval(2.5, None), 0, Interval(2.5, None)),
        (Interval(2.5, None), 1, Interval(3.5, None)),
        (Interval(2.5, None), -1, Interval(1.5, None)),
        (Interval(None, 5), 0, Interval(None, 5)),
        (Interval(None, 5), 1, Interval(None, 6)),
        (Interval(None, 5), -1, Interval(None, 4)),
        (Interval.all(), 0, Interval.all()),
        (Interval.all(), 1, Interval.all()),
        (Interval.all(), -1, Interval.all()),
    ],
)
def test_intervals_add(interval, val, result):
    assert interval + val == result
    assert val + interval == result

    interval += val
    assert interval == result


@parametrize(
    argnames=["interval", "val", "result"],
    argvalues=[
        (Interval(2.5, 5), 0, Interval(2.5, 5)),
        (Interval(2.5, 5), 1, Interval(1.5, 4)),
        (Interval(2.5, 5), -1, Interval(3.5, 6)),
        (Interval(2.5, None), 0, Interval(2.5, None)),
        (Interval(2.5, None), 1, Interval(1.5, None)),
        (Interval(2.5, None), -1, Interval(3.5, None)),
        (Interval(None, 5), 0, Interval(None, 5)),
        (Interval(None, 5), 1, Interval(None, 4)),
        (Interval(None, 5), -1, Interval(None, 6)),
        (Interval.all(), 0, Interval.all()),
        (Interval.all(), 1, Interval.all()),
        (Interval.all(), -1, Interval.all()),
    ],
)
def test_intervals_subtract(interval, val, result):
    assert interval - val == result

    interval -= val
    assert interval == result


@parametrize(
    argnames=["interval", "val", "result"],
    argvalues=[
        (Interval(2.5, 5), 0, Interval(3.75, 3.75)),
        (Interval(2.5, 5), 1, Interval(2.5, 5)),
        (Interval(2.5, 5), 2, Interval(1.25, 6.25)),
        (Interval(2.5, 5), 0.5, Interval(3.125, 4.375)),
    ],
)
def test_intervals_multiply(interval, val, result):
    assert interval * val == result
    assert val * interval == result
    assert interval.mid_point == result.mid_point

    interval *= val
    assert interval == result


@parametrize(
    argnames=["interval", "val", "result"],
    argvalues=[
        (Interval(2.5, 5), 1, Interval(2.5, 5)),
        (Interval(2.5, 5), 2, Interval(3.125, 4.375)),
        (Interval(2.5, 5), 0.5, Interval(1.25, 6.25)),
    ],
)
def test_intervals_divide(interval, val, result):
    assert interval / val == result
    assert interval.mid_point == result.mid_point

    interval /= val
    assert interval == result


@parametrize(
    argnames=["interval", "num", "ticks"],
    argvalues=[
        (Interval(0, 1), 2, np.array([0, 1])),
        (Interval(0, 1), 3, np.array([0, 0.5, 1])),
        (Interval(-2, 4), 7, np.array([-2, -1, 0, 1, 2, 3, 4])),
        (Interval(0, 0.5), 6, np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])),
    ],
)
def test_intervals_ticks(interval, num, ticks):
    actual_ticks = interval.ticks(num)
    assert isinstance(actual_ticks, np.ndarray)
    assert actual_ticks.size == num
    assert_numpy_array_equal(actual_ticks, ticks, rel=EPSILON)


@parametrize(
    argnames=["midpoint", "size", "result"],
    argvalues=[
        (2.5, 1, Interval(2, 3)),
        (2, 6, Interval(-1, 5)),
    ],
)
def test_neighbourhood_interval(midpoint, size, result):
    assert Interval.neighbourhood(midpoint, size) == result


@parametrize(
    argnames=["intervals", "result"],
    argvalues=[
        ([Interval(2.5, 5)], Interval(2.5, 5)),
        ([Interval(2.5, 5), Interval(1.5, 4)], Interval(2.5, 4)),
        ([Interval(2.5, 5), Interval(1.5, 4), Interval(3.5, 6)], Interval(3.5, 4)),
        ([Interval(2.5, 5), Interval(3.5, None)], Interval(3.5, 5)),
        ([Interval(2.5, 5), Interval(None, 4.5)], Interval(2.5, 4.5)),
        ([Interval(None, 4.5), Interval(3.5, None)], Interval(3.5, 4.5)),
        ([Interval(2.5, 5), Interval.all()], Interval(2.5, 5)),
    ],
)
def test_intervals_intersect(intervals, result):
    assert Interval.intersect(*intervals) == result

    for interval in intervals:
        assert result <= interval


@parametrize(
    argnames=["intervals", "result"],
    argvalues=[
        ([Interval(2.5, 5)], Interval(2.5, 5)),
        ([Interval(2.5, 5), Interval(1.5, 4)], Interval(1.5, 5)),
        ([Interval(2.5, 5), Interval(1.5, 4), Interval(3.5, 6)], Interval(1.5, 6)),
        ([Interval(2.5, 5), Interval(3.5, None)], Interval(2.5, None)),
        ([Interval(2.5, 5), Interval(1.5, None)], Interval(1.5, None)),
        ([Interval(2.5, 5), Interval(None, 4.5)], Interval(None, 5)),
        ([Interval(2.5, 4.5), Interval(None, 5)], Interval(None, 5)),
        ([Interval(None, 4.5), Interval(3.5, None)], Interval.all()),
        ([Interval(None, 3.5), Interval(4.5, None)], Interval.all()),
        ([Interval(2.5, 5), Interval.all()], Interval.all()),
    ],
)
def test_intervals_unify(intervals, result):
    assert (
        Interval.unify(*intervals) == result
    ), f"Unification of {intervals} is {result}"

    for interval in intervals:
        assert interval <= result


def test_interval_constructor_failure():
    with pytest.raises(
        IntervalError,
        match=r"^Minimum value should be greater than maximum value, got \(2.5, 1\)$",
    ):
        Interval(2.5, 1)


def test_interval_set_minimum_failure():
    interval = Interval(0.5, 1)
    with pytest.raises(
        IntervalError,
        match=r"^Minimum value should be greater than maximum value, got \(2.5, 1\)$",
    ):
        interval.min_val = 2.5


def test_interval_set_maximum_failure():
    interval = Interval(2.5, 5)
    with pytest.raises(
        IntervalError,
        match=r"^Minimum value should be greater than maximum value, got \(2.5, 1\)$",
    ):
        interval.max_val = 1


def test_interval_neighborhood_negative_size_failure():
    with pytest.raises(
        IntervalError,
        match=r"^Minimum value should be greater than maximum value, got \(3.0, 2.0\)$",
    ):
        Interval.neighbourhood(2.5, -1)


def test_interval_multiply_by_negative_failure():
    with pytest.raises(
        IntervalError,
        match=(
            "^Minimum value should be greater than maximum value, "
            r"got \(3.25, 0.25\)$"
        ),
    ):
        -2 * Interval(1, 2.5)  # pylint: disable=expression-not-assigned


def test_infinite_interval_multiply_failure():
    with pytest.raises(
        IntervalError, match="^Can multiply only finite interval by value.$"
    ):
        2 * Interval(1, None)  # pylint: disable=expression-not-assigned


def test_infinite_ticks_failure():
    with pytest.raises(
        IntervalError, match="^Can get ticks of only finite intervals.$"
    ):
        Interval(1, None).ticks(3)


def test_negative_ticks_num_failure():
    with pytest.raises(
        IntervalError, match="^Number of ticks must be at least 2, got -3$"
    ):
        Interval(1, 2).ticks(-3)


@parametrize(
    argnames=["intervals"],
    argvalues=[
        ([Interval(2.5, 5), Interval(6, 8)]),
        ([Interval(2.5, 5), Interval(1.5, 4), Interval(6, 8)]),
        ([Interval(2.5, 5), Interval(5.5, None)]),
        ([Interval(2.5, 5), Interval(None, 2)]),
        ([Interval(None, 3.5), Interval(4.5, None)]),
    ],
)
def test_intervals_intersection_failure(intervals):
    with pytest.raises(IntervalIntersectionError):
        Interval.intersect(*intervals)
