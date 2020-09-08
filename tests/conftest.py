from pytest import fixture

from eddington import FitFunctionsRegistry, FitData


@fixture
def json_dump_mock(mocker):
    return mocker.patch("json.dump")


@fixture
def clear_functions_registry():
    backup = list(FitFunctionsRegistry.all())
    FitFunctionsRegistry.clear()
    yield
    FitFunctionsRegistry.clear()
    for func in backup:
        FitFunctionsRegistry.add(func)


@fixture
def mock_load_function(mocker):
    return mocker.patch.object(FitFunctionsRegistry, "load")


@fixture
def mock_read_from_csv(mocker):
    return mocker.patch.object(FitData, "read_from_csv")


@fixture
def mock_read_from_json(mocker):
    return mocker.patch.object(FitData, "read_from_json")


@fixture
def mock_read_from_excel(mocker):
    return mocker.patch.object(FitData, "read_from_excel")
