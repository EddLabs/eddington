from pytest import fixture

from eddington import FittingData, FittingFunctionsRegistry, io_util


@fixture
def json_dumps_mock(mocker):
    return mocker.patch("json.dumps")


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


@fixture
def mock_save_as_excel(mocker):
    return mocker.patch.object(io_util, "save_as_excel")


@fixture
def mock_save_as_csv(mocker):
    return mocker.patch.object(io_util, "save_as_csv")
