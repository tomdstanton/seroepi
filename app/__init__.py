from shiny import App

from .app import main_server, main_ui
from .icons import icon_lucide

app = App(main_ui, main_server)

__all__ = ["app", "icon_lucide", "main_server", "main_ui"]
