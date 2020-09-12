from pytest import fixture

from eddington import FittingData, FittingFunctionsRegistry


@fixture
def json_dump_mock(mocker):
    return mocker.patch("json.dump")


@fixture
def clear_functions_registry():
    backup = list(FittingFunctionsRegistry.all())
    FittingFunctionsRegistry.clear()
    yield
    FittingFunctionsRegistry.clear()
    for func in backup:
        FittingFunctionsRegistry.add(func)


@fixture
def mock_load_function(mocker):
    return mocker.patch.object(FittingFunctionsRegistry, "load")


@fixture
def mock_read_from_csv(mocker):
    return mocker.patch.object(FittingData, "read_from_csv")


@fixture
def mock_read_from_json(mocker):
    return mocker.patch.object(FittingData, "read_from_json")


@fixture
def mock_read_from_excel(mocker):
    return mocker.patch.object(FittingData, "read_from_excel")
