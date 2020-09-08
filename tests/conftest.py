import pytest

from eddington import FitFunctionsRegistry


@pytest.fixture
def json_dump_mock(mocker):
    return mocker.patch("json.dump")


@pytest.fixture
def clear_functions_registry():
    backup = list(FitFunctionsRegistry.all())
    FitFunctionsRegistry.clear()
    yield
    FitFunctionsRegistry.clear()
    for func in backup:
        FitFunctionsRegistry.add(func)
