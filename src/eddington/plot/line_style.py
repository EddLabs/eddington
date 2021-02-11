"""Enum representing the different line style options."""
from enum import Enum
from typing import List


class LineStyle(Enum):
    """Enum class for line style options."""

    SOLID = "solid"
    DASHED = "dashed"
    DASHDOT = "dashdot"
    DOTTED = "dotted"
    NONE = "none"

    @classmethod
    def all(cls) -> List[str]:
        """
        Get all line style values.

        :return: Possible values of line styles
        :rtype: List[str]
        """
        return [linestyle.value for linestyle in cls]
