# ruff: noqa: I001
from typing import Any

from htmltools import HTML
import lucide

ICON_ALIASES: dict[str, str] = {
    "sliders": "sliders-horizontal",
    "bar-chart": "chart-bar",
    "warning": "triangle-alert",
    "alert": "circle-alert",
    "gear": "settings",
    "edit": "pencil",
}
_ICON_ALIASES = ICON_ALIASES


def icon_lucide(name: str, size: int = 18, class_: str = "me-1", **kwargs: Any) -> HTML:
    """Render a Lucide SVG icon wrapped in htmltools.HTML for Shiny UI integration."""
    icon_name = str(name).strip().lower().replace("_", "-")
    icon_name = ICON_ALIASES.get(icon_name, icon_name)

    combined_class = f"lucide-icon {class_}".strip()
    icon_kwargs = {"class": combined_class, **kwargs}
    try:
        svg_str = lucide._render_icon(icon_name, size, **icon_kwargs)
    except (lucide.IconDoesNotExist, FileNotFoundError, AttributeError, ValueError) as err:
        raise ValueError(f"Icon '{name}' (normalized as '{icon_name}') does not exist in Lucide icon set.") from err

    if "xmlns=" not in svg_str.lower():
        svg_str = svg_str.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)

    return HTML(svg_str)
