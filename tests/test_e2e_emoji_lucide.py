"""E2E Test Suite for Emoji Replacement with Lucide Icons in SeroEpi.

This suite performs opaque-box end-to-end testing across four verification tiers:
- Tier 1: Verification that pyproject.toml lists lucide in [project.optional-dependencies] app.
- Tier 2: Verification that icon_lucide(name) produces valid Lucide SVG elements wrapped in htmltools.HTML.
- Tier 3: Programmatic verification that zero emoji characters (unicode category So / code points > 0x1F000)
         remain in any .py file under app/.
- Tier 4: Verification that app:app and all app modules can be imported cleanly without errors.
"""

import importlib
import pathlib
import tomllib
import unicodedata

import pytest
from htmltools import HTML
from shiny import App


# -----------------------------------------------------------------------------
# Tier 1: Pyproject Dependency Configuration Verification
# -----------------------------------------------------------------------------
def test_tier1_pyproject_toml_app_optional_dependencies_contains_lucide() -> None:
    """Tier 1: Verify pyproject.toml lists 'lucide' under [project.optional-dependencies] app."""
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), f"pyproject.toml not found at {pyproject_path}"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    optional_deps = project.get("optional-dependencies", {})
    app_deps = optional_deps.get("app", [])

    lucide_deps = [dep for dep in app_deps if "lucide" in dep.lower()]
    assert len(lucide_deps) > 0, (
        f"lucide package missing from [project.optional-dependencies] app in pyproject.toml. "
        f"Found app optional dependencies: {app_deps}"
    )


# -----------------------------------------------------------------------------
# Tier 2: Lucide Icon Helper Contract Verification
# -----------------------------------------------------------------------------
def _get_icon_lucide_helper():
    """Dynamically resolve icon_lucide function from app.icons or app.utils."""
    for mod_name in ["app.icons", "app.utils"]:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "icon_lucide"):
                return getattr(mod, "icon_lucide")
        except (ImportError, ModuleNotFoundError):
            continue

    pytest.fail("icon_lucide helper function not found in app.icons or app.utils")


def test_tier2_icon_lucide_renders_html_svg() -> None:
    """Tier 2: Verify icon_lucide produces valid SVG markup wrapped in htmltools.HTML."""
    icon_fn = _get_icon_lucide_helper()
    result = icon_fn("database")

    assert isinstance(result, HTML), f"Expected htmltools.HTML return type, got {type(result)}"
    rendered = str(result)
    assert "<svg" in rendered.lower(), f"Rendered output missing <svg> tag: {rendered}"
    assert "</svg>" in rendered.lower(), f"Rendered output missing </svg> tag: {rendered}"
    assert "xmlns=" in rendered.lower(), f"Rendered output missing xmlns attribute: {rendered}"


def test_tier2_icon_lucide_custom_attributes() -> None:
    """Tier 2: Verify icon_lucide respects size, class_, and extra kwargs."""
    icon_fn = _get_icon_lucide_helper()
    result = icon_fn("sliders", size=24, class_="custom-test-icon")
    rendered = str(result)

    assert "custom-test-icon" in rendered, f"Class 'custom-test-icon' missing from rendered SVG: {rendered}"
    assert ("width=" in rendered) or ("height=" in rendered) or ("24" in rendered), (
        f"Size 24 not reflected in rendered SVG: {rendered}"
    )


def test_tier2_icon_lucide_invalid_icon_name() -> None:
    """Tier 2: Verify icon_lucide handles non-existent icon names gracefully or raises an error."""
    icon_fn = _get_icon_lucide_helper()
    with pytest.raises((ValueError, KeyError, AttributeError, Exception)):
        icon_fn("non_existent_invalid_icon_xyz_99999")


# -----------------------------------------------------------------------------
# Tier 3: Zero-Emoji Codebase Audit Verification
# -----------------------------------------------------------------------------
def test_tier3_zero_emojis_in_app_python_files() -> None:
    """Tier 3: Programmatically verify zero emoji characters remain in any .py file under app/."""
    app_dir = pathlib.Path(__file__).parent.parent / "app"
    assert app_dir.is_dir(), f"app directory not found at {app_dir}"

    py_files = sorted(app_dir.rglob("*.py"))
    assert len(py_files) > 0, "No .py files found under app/"

    emoji_violations = []

    for py_file in py_files:
        rel_path = py_file.relative_to(app_dir.parent)
        lines = py_file.read_text(encoding="utf-8").splitlines()

        for line_num, line in enumerate(lines, 1):
            for col_num, char in enumerate(line, 1):
                code_point = ord(char)
                category = unicodedata.category(char)

                # Criteria: Category 'So' (Symbol, other) OR code points > 0x1F000 OR common emoji ranges
                is_emoji = (
                    category == "So"
                    or code_point >= 0x1F000
                    or (0x1F300 <= code_point <= 0x1F9FF)
                    or (0x2600 <= code_point <= 0x27BF)
                )

                if is_emoji:
                    emoji_violations.append(
                        f"{rel_path}:{line_num}:{col_num}: "
                        f"Found emoji '{char}' (U+{code_point:04X}, category={category})"
                    )

    if emoji_violations:
        formatted_violations = "\n".join(emoji_violations)
        pytest.fail(
            f"Zero-Emoji Audit Failed: Found {len(emoji_violations)} emoji character(s) in app/*.py files:\n"
            f"{formatted_violations}"
        )


# -----------------------------------------------------------------------------
# Tier 4: App Importability & Submodule Verification
# -----------------------------------------------------------------------------
def test_tier4_app_module_importable() -> None:
    """Tier 4: Verify app:app (Shiny App instance) can be imported cleanly without errors."""
    try:
        from app import app
    except Exception as e:
        pytest.fail(f"Failed to import app from app package: {e}")

    assert app is not None, "app object imported from app is None"
    assert isinstance(app, App), f"Expected shiny.App instance, got {type(app)}"


def test_tier4_app_main_ui_and_server_importable() -> None:
    """Tier 4: Verify app.app exports main_ui and main_server."""
    try:
        from app.app import main_server, main_ui
    except Exception as e:
        pytest.fail(f"Failed to import main_ui / main_server from app.app: {e}")

    assert main_ui is not None
    assert main_server is not None


def test_tier4_all_app_submodules_importable() -> None:
    """Tier 4: Verify all app submodules can be imported cleanly."""
    modules = [
        "app.burden",
        "app.coverage",
        "app.dataset",
        "app.forecasting",
        "app.formulation",
        "app.utils",
    ]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            pytest.fail(f"Failed to import module {mod_name}: {e}")
