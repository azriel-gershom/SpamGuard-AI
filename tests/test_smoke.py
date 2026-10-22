"""Smoke tests for SpamGuard AI Phase 01 Foundation."""

import importlib
import py_compile
from pathlib import Path

from src import config


def test_project_directories_exist():
    """Verify all required project directories are initialized."""
    assert config.BASE_DIR.exists() and config.BASE_DIR.is_dir()
    assert config.SRC_DIR.exists() and config.SRC_DIR.is_dir()
    assert config.DATA_DIR.exists() and config.DATA_DIR.is_dir()
    assert config.RAW_DATA_DIR.exists() and config.RAW_DATA_DIR.is_dir()
    assert config.PROCESSED_DATA_DIR.exists() and config.PROCESSED_DATA_DIR.is_dir()
    assert config.MODELS_DIR.exists() and config.MODELS_DIR.is_dir()
    assert config.REPORTS_DIR.exists() and config.REPORTS_DIR.is_dir()
    assert config.FIGURES_DIR.exists() and config.FIGURES_DIR.is_dir()
    assert config.DOCS_DIR.exists() and config.DOCS_DIR.is_dir()


def test_config_constants():
    """Verify configuration settings and constants."""
    assert config.RANDOM_STATE == 42
    assert 0.0 < config.TEST_SIZE < 1.0
    assert config.APP_TITLE == "SpamGuard AI"
    assert config.APP_VERSION == "0.1.0"


def test_core_dependencies_importable():
    """Verify all primary dependencies are installed and importable."""
    required_modules = [
        "streamlit",
        "sklearn",
        "pandas",
        "numpy",
        "plotly",
        "matplotlib",
        "joblib",
        "pytest",
    ]
    for module_name in required_modules:
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Module {module_name} failed to import"


def test_app_syntax_compilation():
    """Verify app.py and source files compile without syntax errors."""
    app_file = config.BASE_DIR / "app.py"
    assert app_file.exists(), "app.py does not exist"
    compiled = py_compile.compile(str(app_file), doraise=True)
    assert compiled is not None, "Failed to compile app.py"

    src_init = config.SRC_DIR / "__init__.py"
    src_config = config.SRC_DIR / "config.py"
    py_compile.compile(str(src_init), doraise=True)
    py_compile.compile(str(src_config), doraise=True)
