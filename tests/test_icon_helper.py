import pytest
from htmltools import HTML

from app import icon_lucide as icon_app
from app.icons import icon_lucide
from app.utils import icon_lucide as icon_utils


def test_icon_lucide_basic():
    h1 = icon_lucide("house")
    assert isinstance(h1, HTML)
    h1_str = str(h1)
    assert "<svg" in h1_str
    assert 'xmlns="http://www.w3.org/2000/svg"' in h1_str
    assert 'class="lucide-icon me-1"' in h1_str
    assert 'width="18"' in h1_str
    assert 'height="18"' in h1_str


def test_icon_lucide_custom_attributes():
    h2 = icon_utils("database", size=20, class_="text-primary")
    assert isinstance(h2, HTML)
    h2_str = str(h2)
    assert "<svg" in h2_str
    assert 'xmlns="http://www.w3.org/2000/svg"' in h2_str
    assert 'class="lucide-icon text-primary"' in h2_str
    assert 'width="20"' in h2_str
    assert 'height="20"' in h2_str


def test_icon_lucide_normalization():
    h3 = icon_lucide("  CHART_BAR  ")
    assert isinstance(h3, HTML)
    h3_str = str(h3)
    assert "<svg" in h3_str


def test_icon_lucide_exports():
    h_app = icon_app("house")
    h_utils = icon_utils("house")
    assert isinstance(h_app, HTML)
    assert isinstance(h_utils, HTML)
    assert str(h_app) == str(h_utils)


def test_icon_lucide_invalid_name_raises_value_error():
    with pytest.raises(ValueError, match="does not exist in Lucide icon set"):
        icon_lucide("nonexistent_icon_xyz_123")


def test_icon_lucide_alias_mapping():
    from app.icons import ICON_ALIASES

    assert "sliders" in ICON_ALIASES
    assert ICON_ALIASES["sliders"] == "sliders-horizontal"

    h = icon_lucide("sliders", size=24, class_="custom-test-icon")
    assert isinstance(h, HTML)
    h_str = str(h)
    assert "<svg" in h_str
    assert 'xmlns="http://www.w3.org/2000/svg"' in h_str
    assert "custom-test-icon" in h_str
