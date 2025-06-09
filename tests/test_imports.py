import importlib

import pytest


def test_main_import():
    import minerva

    assert minerva is not None


def test_version():
    import minerva

    version = minerva.__version__
    assert version and version != "0.0.0"


submodules = [
    "minerva.analysis",
    "minerva.callback",
    "minerva.data",
    "minerva.engines",
    "minerva.losses",
    "minerva.models",
    "minerva.optimizers",
    "minerva.pipelines",
    "minerva.samplers",
    "minerva.transforms",
    "minerva.utils",
]


@pytest.mark.parametrize("module", submodules)
def test_submodule_import(module):
    assert importlib.import_module(module)
