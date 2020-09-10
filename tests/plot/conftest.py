from pytest_cases import fixture


@fixture
def mock_plt(mocker):
    return mocker.patch("eddington.plot.plt")
