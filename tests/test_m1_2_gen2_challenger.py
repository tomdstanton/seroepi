"""Adversarial stress test suite for Lucide icon Shiny UI component integration and title preservation (Milestone 1, Iteration 2).

Challenger 2 Harness:
- Verifies icon_lucide inside Shiny UI components (ui.accordion_panel, ui.nav_panel, ui.card, ui.input_action_button).
- Verifies accordion panel title preservation (data-value="Load Data" and clean title "Load Data").
- Verifies SVG unescaped HTML injection.
- Verifies custom attributes and alias resolution within Shiny UI wrappers.
"""

import html
import re

from shiny import ui

from app.icons import ICON_ALIASES, icon_lucide


def test_accordion_panel_icon_compatibility_and_title_preservation():
    """Verify ui.accordion_panel with icon=icon_lucide preserves clean data-value and title."""
    panel_title = "Load Data"
    panel = ui.accordion_panel(panel_title, "Sample Content", icon=icon_lucide("database"))
    acc = ui.accordion(panel, id="dataset_accordion")
    html_str = str(acc)

    # 1. Check data-value attribute is clean title without SVG noise
    assert f'data-value="{panel_title}"' in html_str
    # 2. Check title div contains clean title
    assert f'<div class="accordion-title">{panel_title}</div>' in html_str
    # 3. Check icon is placed inside accordion-icon wrapper
    assert '<div class="accordion-icon"><svg xmlns="http://www.w3.org/2000/svg"' in html_str
    # 4. Check SVG is not HTML escaped
    assert "&lt;svg" not in html_str


def test_accordion_panel_multiple_titles_preservation():
    """Verify title preservation across complex accordion panel titles."""
    titles = [
        "Prevalence & Burden",
        "Formulation & Evaluation",
        "Incidence Aggregation",
    ]
    for t in titles:
        panel = ui.accordion_panel(t, f"Content for {t}", icon=icon_lucide("sliders"))
        clean_t = re.sub(r"\W+", "_", t)
        safe_id = f"acc_{clean_t}"
        acc = ui.accordion(panel, id=safe_id)
        html_str = str(acc)
        # Check data-value attribute contains clean title (escaped attribute value as required by HTML spec)
        escaped_title = html.escape(t)
        assert f'data-value="{escaped_title}"' in html_str or f'data-value="{t}"' in html_str
        assert escaped_title in html_str or t in html_str
        assert '<div class="accordion-icon"><svg' in html_str


def test_nav_panel_icon_compatibility():
    """Verify ui.nav_panel with icon=icon_lucide renders unescaped SVG and preserves data-value."""
    nav = ui.nav_panel("Home Tab", "Home Content", icon=icon_lucide("house"))
    navset = ui.navset_tab(nav)
    html_str = str(navset.tagify())

    assert 'data-value="Home Tab"' in html_str
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in html_str
    assert "Home Tab</a>" in html_str
    assert "&lt;svg" not in html_str


def test_card_header_icon_compatibility():
    """Verify ui.card with ui.card_header and icon_lucide renders clean HTML."""
    card = ui.card(
        ui.card_header(icon_lucide("sliders"), "Settings Header"),
        "Card content body",
    )
    html_str = str(card)

    assert '<div class="card-header">' in html_str
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in html_str
    assert "Settings Header" in html_str
    assert "&lt;svg" not in html_str


def test_action_button_icon_compatibility():
    """Verify ui.input_action_button with icon=icon_lucide renders action-icon and action-label."""
    btn = ui.input_action_button("submit_btn", "Submit Data", icon=icon_lucide("check"))
    html_str = str(btn)

    assert 'id="submit_btn"' in html_str
    assert '<span class="action-icon"><svg xmlns="http://www.w3.org/2000/svg"' in html_str
    assert '<span class="action-label">Submit Data</span>' in html_str
    assert "&lt;svg" not in html_str


def test_icon_lucide_custom_attributes_in_shiny_ui():
    """Verify custom size and class attributes propagate properly when embedded in Shiny UI."""
    btn = ui.input_action_button(
        "custom_btn",
        "Custom Icon",
        icon=icon_lucide("settings", size=24, class_="text-danger me-3"),
    )
    html_str = str(btn)

    assert 'width="24"' in html_str
    assert 'height="24"' in html_str
    assert (
        'class="lucide-icon text-danger me-3"' in html_str
        or 'class="lucide lucide-settings lucide-icon text-danger me-3"' in html_str
    )


def test_icon_lucide_aliases_in_shiny_ui():
    """Verify icon aliases render valid icons within Shiny UI components."""
    for alias in ICON_ALIASES.keys():
        safe_alias_id = f"btn_{alias.replace('-', '_')}"
        btn = ui.input_action_button(safe_alias_id, f"Button {alias}", icon=icon_lucide(alias))
        html_str = str(btn)
        assert '<svg xmlns="http://www.w3.org/2000/svg"' in html_str
        assert f"Button {alias}" in html_str
