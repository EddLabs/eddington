import pytest

from eddington import FittingFunctionsRegistry


@pytest.fixture
def clear_fix():
    yield
    for fit_func in FittingFunctionsRegistry.all():
        fit_func.clear_fixed()
