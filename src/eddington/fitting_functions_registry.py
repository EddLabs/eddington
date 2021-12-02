"""Registry containing predefined fitting functions and fitting functions generators."""
from typing import List

from eddington.exceptions import FittingFunctionLoadError, FittingFunctionSaveError


class FittingFunctionsRegistry:  # noqa: D415,D213,D205
    """A singleton class containing all saved :class:`FittingFunction` instances."""

    __name_to_func_dict = {}  # type: ignore

    @classmethod
    def add(cls, func):  # type: ignore
        """
        Add a fitting function.

        :param func: fitting function to add to registry
        :type func: FittingFunction
        :raises FittingFunctionSaveError: Raised when trying to add a function with a
            name which already exists.
        """
        if func.name in cls.__name_to_func_dict:
            raise FittingFunctionSaveError(
                f'Cannot save "{func.name}" to registry'
                " since there is another fitting function with this name"
            )
        cls.__name_to_func_dict[func.name] = func

    @classmethod
    def remove(cls, func_name: str) -> None:
        """
        Remove a fitting function.

        :param func_name: Name of the function to remove.
        :type func_name: str
        """
        del cls.__name_to_func_dict[func_name]

    @classmethod
    def clear(cls) -> None:
        """Clear all fitting functions."""
        cls.__name_to_func_dict.clear()

    @classmethod
    def all(cls):  # type: ignore
        """
        Get all fitting functions.

        :returns: list of all fitting functions
        :rtype: List[FittingFunction]
        """
        return list(cls.__name_to_func_dict.values())

    @classmethod
    def names(cls) -> List[str]:
        """
        Property of the names of all fitting functions.

        :return: Names of all fitting functions in registry
        :rtype: List[str]
        """
        return list(cls.__name_to_func_dict.keys())

    @classmethod
    def load(cls, func_name: str):  # type: ignore
        """
        Get a fitting function by name.

        :param func_name: Name of the function to load.
        :type func_name: str
        :returns: a fitting function from the registry
        :rtype: FittingFunction
        :raises FittingFunctionLoadError: Raised when trying to load a function which
            does not exist in the registry
        """
        if not cls.exists(func_name):
            raise FittingFunctionLoadError(f"No fitting function named {func_name}")
        return cls.__name_to_func_dict[func_name]

    @classmethod
    def exists(cls, func_name: str) -> bool:
        """
        Checks whether a fitting function exist.

        :param func_name: Name of the function to load.
        :type func_name: str
        :returns: bool
        """
        return func_name in cls.__name_to_func_dict
