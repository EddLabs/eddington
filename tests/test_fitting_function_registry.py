import pytest
from pytest_cases import fixture, unpack_fixture

from eddington import FittingFunctionLoadError, FittingFunctionsRegistry
from eddington.exceptions import FittingFunctionSaveError
from tests.util import dummy_function


@fixture
def dummy_functions(clear_functions_registry):
    func1 = dummy_function("dummy_function1", "Dummy function 1")
    func2 = dummy_function("dummy_function2", "Dummy function 2", save=False)
    func3 = dummy_function("dummy_function3", "Dummy function 3")
    func4 = dummy_function("dummy_function4", "Dummy function 4")
    func5 = dummy_function("dummy_function5", "Dummy function 5", save=False)

    saved_funcs = [func1, func3, func4]
    unsaved_funcs = [func2, func5]
    return saved_funcs, unsaved_funcs


(saved_dummy_functions, unsaved_dummy_functions,) = unpack_fixture(
    argnames="saved_dummy_functions, unsaved_dummy_functions", fixture=dummy_functions
)


def test_load_function(saved_dummy_functions):
    for func in saved_dummy_functions:
        actual_func = FittingFunctionsRegistry.load(func.name)
        assert (
            func == actual_func
        ), "Registry get name function returns different function than expected."


def test_exists(saved_dummy_functions):
    for func in saved_dummy_functions:
        assert FittingFunctionsRegistry.exists(
            func.name
        ), f"Expected {func.name} to exists. It does not."


def test_remove(saved_dummy_functions):
    for func in saved_dummy_functions:
        FittingFunctionsRegistry.remove(func.name)
        assert not FittingFunctionsRegistry.exists(
            func.name
        ), f"Expected {func.name} to not exist. It does."


def test_all(saved_dummy_functions):
    assert list(saved_dummy_functions) == list(
        FittingFunctionsRegistry.all()
    ), "Functions are different than expected"


def test_names(saved_dummy_functions):
    assert [func.name for func in saved_dummy_functions] == list(
        FittingFunctionsRegistry.names()
    ), "Functions names are different than expected"


def test_add_dummy_function_without_saving(unsaved_dummy_functions):
    for func in unsaved_dummy_functions:
        assert not FittingFunctionsRegistry.exists(
            func.name
        ), "Function was saved, even though it wasn't supposed to"


def test_load_non_existing_function(unsaved_dummy_functions):
    for func in unsaved_dummy_functions:
        with pytest.raises(
            FittingFunctionLoadError, match=f"^No fitting function named {func.name}$"
        ):
            FittingFunctionsRegistry.load(func.name)


def test_saving_two_fitting_functions_with_the_same_name(saved_dummy_functions):
    for func in saved_dummy_functions:
        name = func.name
        with pytest.raises(
            FittingFunctionSaveError,
            match=(
                f'^Cannot save "{name}" to registry '
                "since there is another fitting function with this name$"
            ),
        ):
            dummy_function(name, "new syntax", save=True)
