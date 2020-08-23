import pytest

from eddington import FitFunctionsRegistry


@pytest.fixture
def clear_fix():
    yield
    for fit_func in FitFunctionsRegistry.all():
        fit_func.clear_fixed()
