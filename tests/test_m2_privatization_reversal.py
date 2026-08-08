import importlib

import pytest

PUBLIC_MODULES = [
    "seroepi.estimators.base",
    "seroepi.estimators.prevalence",
    "seroepi.estimators.diversity",
    "seroepi.estimators.incidence",
    "seroepi.estimators.serology",
    "app.app",
    "app.burden",
    "app.coverage",
    "app.dataset",
    "app.forecasting",
    "app.formulation",
    "app.utils",
]

PRIVATE_MODULES = [
    "seroepi.estimators._base",
    "seroepi.estimators._core",
    "seroepi.estimators._modelled",
    "app._app",
    "app._burden",
    "app._coverage",
    "app._dataset",
    "app._forecasting",
    "app._formulation",
    "app._utils",
]


@pytest.mark.parametrize("mod_name", PUBLIC_MODULES)
def test_public_modules_importable(mod_name: str):
    """Verify that all 10 public modules can be dynamically imported and accessed."""
    mod = importlib.import_module(mod_name)
    assert mod is not None
    assert hasattr(mod, "__file__")
    assert mod.__file__ is not None


@pytest.mark.parametrize("mod_name", PRIVATE_MODULES)
def test_private_modules_raise_import_error(mod_name: str):
    """Verify that old private module paths raise ModuleNotFoundError/ImportError."""
    with pytest.raises((ModuleNotFoundError, ImportError)):
        importlib.import_module(mod_name)


def test_estimators_public_instantiation():
    """Verify key estimators and result classes from public base and core modules can be accessed/instantiated."""
    base_mod = importlib.import_module("seroepi.estimators.base")
    core_mod = importlib.import_module("seroepi.estimators.prevalence")

    assert hasattr(base_mod, "BaseEstimator")
    assert hasattr(base_mod, "Estimates")
    assert hasattr(core_mod, "UnpooledPrevalenceEstimator")
    assert hasattr(core_mod, "UnpooledPrevalenceEstimator")

    # Instantiate estimator from public module
    estimator = core_mod.UnpooledPrevalenceEstimator()
    assert estimator is not None


def test_app_public_exports():
    """Verify main UI/server exports in app modules are present."""
    app_mod = importlib.import_module("app.app")
    assert hasattr(app_mod, "main_ui")
    assert hasattr(app_mod, "main_server")
