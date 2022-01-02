"""Interval class for various purposes."""
from typing import Optional

import numpy as np

from eddington.exceptions import IntervalError, IntervalIntersectionError


class Interval:
    """Class representing an interval between two values."""

    EPSILON = 1e-5

    def __init__(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ):
        """
        Interval constructor.

        :param min_val: Lower bound of the interval. If None, no lower bound.
        :type min_val: Optional[float]
        :param max_val: Upper bound of the interval. If None, no upper bound.
        :type max_val: Optional[float]
        """
        self.__set_min_and_max(min_val=min_val, max_val=max_val)

    def __getitem__(self, index: int) -> Optional[float]:
        """
        Get interval border in tuple-like API.

        :param index: index of the item
        :type index: int
        :return: min_value if index is 0, max_value if index is 1
        :rtype: Optional[int]
        :raises IndexError: Raised when index is not 0 or 1
        """
        if index == 0:
            return self.min_val
        if index == 1:
            return self.max_val
        raise IndexError(f"Index {index} is invalid for Interval")

    @property
    def min_val(self) -> Optional[float]:
        """
        Interval lower bound.

        :return: Lower bound of the interval. If None, no lower bound.
        :rtype: Optional[float]
        """
        return self._min_val

    @min_val.setter
    def min_val(self, min_val: Optional[float]):
        self.__set_min_and_max(min_val=min_val, max_val=self.max_val)

    @property
    def max_val(self) -> Optional[float]:
        """
        Interval upper bound.

        :return: Upper bound of the interval. If None, no upper bound.
        :rtype: Optional[float]
        """
        return self._max_val

    @max_val.setter
    def max_val(self, max_val: Optional[float]):
        self.__set_min_and_max(min_val=self.min_val, max_val=max_val)

    def __contains__(self, item: float) -> bool:
        """
        Checks if a float is within the interval.

        :param item: Value to check.
        :type item: float
        :return: Whether the value is within the interval.
        :rtype: bool
        """
        if self.min_val is not None and item < self.min_val:
            return False
        if self.max_val is not None and item > self.max_val:
            return False
        return True

    def __le__(self, other: "Interval") -> bool:
        """
        Checks this interval contained inside another interval, or equal to it.

        :param other: Interval to compare to
        :type other: Interval
        :return: Whether this interval contained inside the other interval.
        :rtype: bool
        """
        if self.min_val is not None and self.min_val not in other:
            return False
        if self.min_val is None and other.min_val is not None:
            return False
        if self.max_val is not None and self.max_val not in other:
            return False
        if self.max_val is None and other.max_val is not None:
            return False
        return True

    def __lt__(self, other: "Interval") -> bool:
        """
        Checks this interval contained inside another interval.

        :param other: Interval to compare to
        :type other: Interval
        :return: Whether this interval contained inside the other interval.
        :rtype: bool
        """
        return self <= other and self != other

    def __ge__(self, other: "Interval") -> bool:
        """
        Checks this interval contains another interval, or equal to it.

        :param other: Interval to compare to
        :type other: Interval
        :return: Whether this interval contain the other interval.
        :rtype: bool
        """
        return other <= self

    def __gt__(self, other: "Interval") -> bool:
        """
        Checks this interval contains another interval.

        :param other: Interval to compare to
        :type other: Interval
        :return: Whether this interval contain the other interval.
        :rtype: bool
        """
        return other < self

    def __eq__(self, other) -> bool:
        """
        Checks this interval equals to another interval.

        :param other: Interval to compare to
        :type other: Interval
        :return: Whether this interval equals other interval
        :rtype: bool
        """
        if not isinstance(other, Interval):
            return False
        return self.__equal_values(self.min_val, other.min_val) and self.__equal_values(
            self.max_val, other.max_val
        )

    def __ne__(self, other) -> bool:
        """
        Checks this interval not equals to another interval.

        :param other: Interval to compare to
        :type other: Interval
        :return: Whether this interval not equals other interval.
        :rtype: bool
        """
        return not self == other

    def __add__(self, other: float) -> "Interval":
        """
        Create a copy of this interval, shifted upwards by a value.

        :param other: value to shift this interval according to.
        :type other: float
        :return: Shifted interval
        :rtype: Interval
        """
        min_val = None if self.min_val is None else self.min_val + other
        max_val = None if self.max_val is None else self.max_val + other
        return Interval(min_val=min_val, max_val=max_val)

    def __radd__(self, other: float) -> "Interval":
        """
        Create a copy of this interval, shifted upwards by a value.

        :param other: value to shift this interval according to.
        :type other: float
        :return: Shifted interval
        :rtype: Interval
        """
        return self + other

    def __iadd__(self, other: float) -> "Interval":
        """
        Shift this interval upwards by a value.

        :param other: value to shift this interval according to.
        :type other: float
        :return: self
        :rtype: Interval
        """
        self.__set_from_interval(self + other)
        return self

    def __sub__(self, other: float) -> "Interval":
        """
        Create a copy of this interval, shifted downwards by a value.

        :param other: value to shift this interval according to.
        :type other: float
        :return: Shifted interval
        :rtype: Interval
        """
        return self + (-other)

    def __isub__(self, other: float) -> "Interval":
        """
        Shift this interval downwards by a value.

        :param other: value to shift this interval according to.
        :type other: float
        :return: self
        :rtype: Interval
        """
        self.__set_from_interval(self - other)
        return self

    def __mul__(self, other: float) -> "Interval":
        """
        Create a copy of this interval, with size multiplied by a value.

        This method keeps the midpoint unchanged.

        :param other: Value to multiply this interval size according to.
        :type other: float
        :return: New interval instance, with new size.
        :rtype: Interval
        :raises IntervalError: raised if interval is not finite.
        """
        if not self.is_finite():
            raise IntervalError("Can multiply only finite interval by value.")
        return self.neighbourhood(
            val=self.mid_point, size=self.size() * other  # type: ignore
        )

    def __rmul__(self, other: float) -> "Interval":
        """
        Create a copy of this interval, with size multiplied by a value.

        This method keeps the midpoint unchanged.

        :param other: Value to multiply this interval size according to.
        :type other: float
        :return: New interval instance, with new size.
        :rtype: Interval
        """
        return self * other

    def __imul__(self, other: float) -> "Interval":
        """
        Multiply the size of this interval by a value.

        This method keeps the midpoint unchanged.

        :param other: Value to multiply this interval size according to.
        :type other: float
        :return: Self
        :rtype: Interval
        """
        self.__set_from_interval(self * other)
        return self

    def __truediv__(self, other: float) -> "Interval":
        """
        Create a copy of this interval, with size divided by a value.

        This method keeps the midpoint unchanged.

        :param other: Value to divide this interval size according to.
        :type other: float
        :return: New interval instance, with new size.
        :rtype: Interval
        """
        return self * (1 / other)

    def __itruediv__(self, other: float) -> "Interval":
        """
        Divide the size of this interval by a value.

        This method keeps the midpoint unchanged.

        :param other: Value to multiply this interval size according to.
        :type other: float
        :return: Self
        :rtype: Interval
        """
        self.__set_from_interval(self / other)
        return self

    def __repr__(self) -> str:
        """
        Pretty representation string.

        :return: Representation string
        :rtype: str
        """
        return f"Interval({self.min_val}, {self.max_val})"

    @property
    def mid_point(self) -> Optional[float]:
        """
        Middle point of this interval. If interval is not finite, returns None.

        :return: Interval middle point
        :rtype: float or None
        """
        if not self.is_finite():
            return None
        return (self.min_val + self.max_val) / 2  # type: ignore

    def size(self) -> Optional[float]:
        """
        Size of this interval. If interval is not finite, returns None.

        :return: Interval size
        :rtype: float or None
        """
        if not self.is_finite():
            return None
        return self.max_val - self.min_val  # type: ignore

    def is_finite(self) -> bool:
        """
        Is this interval finite.

        :return: Interval is finite.
        :rtype: bool
        """
        return self.min_val is not None and self.max_val is not None

    def is_all(self):
        """
        Is this interval contains all floats.

        :return: Interval is R.
        :rtype: bool
        """
        return self.min_val is None and self.max_val is None

    def ticks(self, num: int) -> np.ndarray:
        """
        Split the interval into ticks. Available only for finite intervals.

        :param num: Number of ticks
        :type num: int
        :return: Array of tick values.
        :rtype: numpy.ndarray
        :raises IntervalError: raised if interval is not finite or if num < 2.
        """
        if not self.is_finite():
            raise IntervalError("Can get ticks of only finite intervals.")
        if num < 2:
            raise IntervalError(f"Number of ticks must be at least 2, got {num}")
        return np.linspace(self.min_val, self.max_val, num=num)  # type: ignore

    @classmethod
    def all(cls) -> "Interval":
        """
        Interval of all values.

        :return: Interval with no bounds.
        :rtype: Interval
        """
        return Interval()

    @classmethod
    def neighbourhood(cls, val: float, size: float) -> "Interval":
        """
        Get a neighborhood of a value with given size.

        :param val: midpoint of the result interval
        :type val: float
        :param size: Size of the result interval
        :type size: float
        :return: neighborhood interval
        :rtype: Interval
        """
        return Interval(val - size / 2, val + size / 2)

    @classmethod
    def intersect(cls, *intervals: "Interval") -> "Interval":
        """
        Intersect the given intervals into single interval.

        :param intervals: Intervals to intersect
        :type intervals: List of Intervals
        :return: Intersection interval
        :rtype: Interval
        :raises IntervalIntersectionError: Raised when intervals intersection is empty
        """
        min_vals = [
            interval.min_val for interval in intervals if interval.min_val is not None
        ]
        max_vals = [
            interval.max_val for interval in intervals if interval.max_val is not None
        ]
        min_val = None if len(min_vals) == 0 else np.max(min_vals)
        max_val = None if len(max_vals) == 0 else np.min(max_vals)
        try:
            return Interval(min_val=min_val, max_val=max_val)
        except IntervalError as error:
            raise IntervalIntersectionError(*intervals) from error

    @classmethod
    def unify(cls, *intervals: "Interval") -> "Interval":
        """
        Smallest interval containing all given intervals.

        :param intervals: Intervals to unify
        :type intervals: List of Intervals
        :return: Unification interval
        :rtype: Interval
        """
        min_vals = [
            interval.min_val for interval in intervals if interval.min_val is not None
        ]
        max_vals = [
            interval.max_val for interval in intervals if interval.max_val is not None
        ]
        min_val = None if len(min_vals) < len(intervals) else np.min(min_vals)
        max_val = None if len(max_vals) < len(intervals) else np.max(max_vals)
        return Interval(min_val=min_val, max_val=max_val)

    def __set_from_interval(self, other: "Interval"):
        self.__set_min_and_max(min_val=other.min_val, max_val=other.max_val)

    def __set_min_and_max(self, min_val: Optional[float], max_val: Optional[float]):
        self.__validate(min_val=min_val, max_val=max_val)
        self._min_val = min_val
        self._max_val = max_val

    @classmethod
    def __validate(cls, min_val: Optional[float], max_val: Optional[float]):
        if min_val is None or max_val is None:
            return
        if min_val > max_val:
            raise IntervalError(
                "Minimum value should be greater than maximum value, "
                f"got ({min_val}, {max_val})"
            )

    @classmethod
    def __equal_values(cls, value1: Optional[float], value2: Optional[float]):
        if value1 is None:
            return value2 is None
        if value2 is None:
            return False
        return np.fabs(value1 - value2) < cls.EPSILON
