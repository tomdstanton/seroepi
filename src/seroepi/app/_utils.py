from typing import Callable, Union, AsyncGenerator, Any, get_origin, Literal, get_args
import inspect
from shiny import ui, module, reactive, render
import pandas as pd
from asyncio import to_thread
from re import compile as re_compile
from contextlib import asynccontextmanager
import tempfile
from pathlib import Path

from shinywidgets import render_widget, output_widget

from seroepi.constants import PlotType, Domain, BayesianInferenceMethod
from seroepi.plotting import render_plot


# Functions ------------------------------------------------------------------------------------------------------------
# Pre-sort prefixes descending to catch 'spatial_res_' before 'spatial_' without recomputing on every call
_DOMAIN_PREFIXES = sorted([f"{d.value}_" for d in Domain], key=len, reverse=True)

def _clean_ui_label(text: Any) -> str:
    """Strips domain prefixes and title-cases strings for clean UI rendering."""
    if not isinstance(text, str):
        return str(text)
    
    for prefix in _DOMAIN_PREFIXES:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1)
            break
            
    return text.replace('_', ' ').title()

def build_grouped_choices(columns: list[str], fallback_group: str = "Other", include_empty: bool = False) -> dict:
    domain_map = {
        Domain.GENOTYPE.value: "Genotypes 🧬",
        Domain.PHENOTYPE.value: "Phenotypes 🔬",
        Domain.AMR.value: "AMR Determinants 💊",
        Domain.VIRULENCE.value: "Virulence Markers ⚔️",
        Domain.SPATIAL.value: "Spatial 🌍",
        Domain.TEMPORAL.value: "Temporal 📅",
        Domain.CLUSTER.value: "Clusters 🕸️"
    }
    
    choices = {"": "Select..."} if include_empty else {}
    for col in columns:
        prefix = col.split('_')[0]
        
        group_name = domain_map.get(prefix, fallback_group)
        if prefix in domain_map:
            clean_name = col.split('_', 1)[-1].replace('_', ' ').title()
        else:
            clean_name = col.replace('_', ' ').title()
            
        if group_name not in choices:
            choices[group_name] = {}
        choices[group_name][col] = clean_name
        
    return choices

def format_metadata_ui(meta_dict: dict) -> list:
    """Helper to cleanly format a dictionary into bolded UI paragraphs."""
    formatted_ui = []
    for k, v in meta_dict.items():
        if isinstance(v, list):
            val_str = ", ".join(map(_clean_ui_label, v)) if v else "None"
        elif hasattr(v, 'value'):
            val_str = _clean_ui_label(v.value)
        else:
            val_str = _clean_ui_label(v)
        formatted_ui.append(ui.p(ui.tags.strong(f"{k.replace('_', ' ').title()}: "), val_str, class_="mb-1"))
    return formatted_ui


class ColMapper:
    _aliases = {  # Define aliases for intelligent column guessing
        "map_id": ["sample", "id", "isolate", "name", "run", "barcode"],
        "map_date": ["date", "year", "collection", "time"],
        "map_spatial": ["spatial", "country", "nation", "region", "location", "site"],
        "map_lat": ["lat", "latitude"],
        "map_lon": ["lon", "long", "longitude"]
    }
    _id_regex = re_compile(r'\bid\b')  # ID regex
    __slots__ = ('_col_map',)
    def __init__(self, available_cols: list[str]):
        self._col_map = {c.lower().strip(): c for c in available_cols}

    def guess(self, field_id: str) -> str:
        terms = self._aliases.get(field_id, [])

        # 1. Exact match first (e.g. 'sample_id' == 'sample_id')
        for t in terms:
            if col := self._col_map.get(t, ""):
                return col

        # 2. Substring match (e.g. 'collection' in 'collection_date')
        for t in terms:
            for i, (c, col) in enumerate(self._col_map.items()):
                if t == 'id' and not self._id_regex.search(c.replace('_', ' ')):
                    continue  # Prevent 'id' matching 'width'
                if t in c:
                    return col
        return ""


@asynccontextmanager
async def ui_task(error_prefix: str = "Task Error"):
    """
    Standardizes background task execution in the Shiny app by wrapping it
    in a Progress bar and safely catching/notifying unhandled exceptions.
    """
    with ui.Progress(min=0, max=100) as p:
        try:
            yield p
        except Exception as e:
            ui.notification_show(f"{error_prefix}: {str(e)}", type="error", duration=15)


async def generate_temp_download(save_func: Callable[[Path], Any], suffix: str,
                                 error_prefix: str = "Export Error") -> AsyncGenerator[bytes, None]:
    """
    Standardizes the creation, writing, and byte-yielding of temporary files for Shiny downloads.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        await to_thread(save_func, path)
        data = await to_thread(path.read_bytes)
        yield data
    except Exception as e:
        ui.notification_show(f"{error_prefix}: {str(e)}", type="error", duration=15)
    finally:
        path.unlink(missing_ok=True)

def update_registry(reactive_val: reactive.Value, key: str, value: Any) -> None:
    """Safely copies, updates, and sets a dictionary inside a reactive.Value."""
    reg = reactive_val.get().copy() if reactive_val.get() is not None else {}
    reg[key] = value
    reactive_val.set(reg)

def export_settings_ui(prefix: str = "") -> ui.Tag:
    """Reusable UI snippet for plot export configuration."""
    p = f"{prefix}_" if prefix else ""
    return ui.div(
        ui.h6("Export Settings", class_="mb-2"),
        ui.tooltip(ui.input_select(f"{p}plot_format", "Format", choices=["png", "pdf", "svg", "jpeg"]),
                   "The image format for the exported plot."),
        ui.tooltip(ui.input_numeric(f"{p}plot_width", "Width (px)", value=1200),
                   "Exported image width in pixels."),
        ui.tooltip(ui.input_numeric(f"{p}plot_height", "Height (px)", value=800),
                   "Exported image height in pixels.")
    )

HYPERPARAM_TOOLTIPS = {
    "method": "The inference engine. MCMC is mathematically rigorous but slower; SVI is a fast approximation.",
    "num_samples": "Number of posterior samples to draw from the parameter distributions.",
    "num_chains": "Number of independent MCMC chains to run in parallel. Recommended: 4.",
    "num_warmup": "Number of initial warmup steps used to tune the sampler before drawing valid samples.",
    "svi_steps": "Number of optimization steps used to converge the SVI ELBO loss.",
    "seed": "Random seed to ensure the model produces mathematically reproducible results.",
    "use_relative_incidence": "Adjust raw incidence counts by the total number of genomes sequenced in that period (models relative burden).",
    "forecast_horizon": "Number of future time steps to project the model forward past the observed data."
}

class EstimatorIntrospector:
    """Stateful cache for estimator reflection and hyperparameter parsing."""
    _signature_cache = {}

    def __init__(self, estimator_class: Any):
        self.cls = estimator_class
        if self.cls not in self._signature_cache:
            self._signature_cache[self.cls] = inspect.signature(self.cls.__init__)
        self.sig = self._signature_cache[self.cls]

    def build_ui(self, prefix: str = "est_param_", exclude: list[str] = None, default_overrides: dict = None) -> ui.Tag:
        if exclude is None: exclude = ['self']
        if default_overrides is None: default_overrides = {}

        elements = []
        for name, param in self.sig.parameters.items():
            if name in exclude:
                continue

            input_id = f"{prefix}{name}"
            label = name.replace("_", " ").title()
            default = default_overrides.get(name, param.default if param.default != inspect.Parameter.empty else None)
            origin = get_origin(param.annotation)

            inp = None
            if param.annotation is int or param.annotation is float:
                inp = ui.input_numeric(input_id, label, value=default)
            elif param.annotation is bool:
                inp = ui.input_checkbox(input_id, label, value=default)
            elif origin is Literal:
                choices = get_args(param.annotation)
                choices_dict = {c: str(c).replace('_', ' ').title() for c in choices}
                inp = ui.input_select(input_id, label, choices=choices_dict, selected=default)
            elif 'InferenceMethod' in str(param.annotation):
                default_val = default.value if hasattr(default, 'value') else default
                inp = ui.input_select(input_id, label, choices=BayesianInferenceMethod.ui_labels(), selected=default_val)

            if inp is not None:
                tooltip_text = HYPERPARAM_TOOLTIPS.get(name)
                if tooltip_text:
                    inp = ui.tooltip(inp, tooltip_text)
                elements.append(inp)

        if elements:
            return ui.div(ui.hr(), ui.p("Hyperparameters", class_="text-muted small mb-1"), *elements)
        return ui.div()

    def extract_kwargs(self, shiny_input: Any, prefix: str = "est_param_", exclude: list[str] = None) -> dict:
        if exclude is None: exclude = ['self']
        kwargs = {}

        for name, param in self.sig.parameters.items():
            if name in exclude:
                continue

            input_id = f"{prefix}{name}"
            if input_id in shiny_input:
                val = shiny_input[input_id]()
                if val == "":  # Skip empty inputs to fallback on defaults
                    continue

                origin = get_origin(param.annotation)
                if param.annotation is int: kwargs[name] = int(val)
                elif param.annotation is float: kwargs[name] = float(val)
                elif param.annotation is bool: kwargs[name] = bool(val)
                elif origin is Literal: kwargs[name] = val
                elif 'InferenceMethod' in str(param.annotation): kwargs[name] = BayesianInferenceMethod(val)
                else: kwargs[name] = val

        return kwargs

# UIs ------------------------------------------------------------------------------------------------------------------
@module.ui
def safe_plot_ui():
    """Reusable UI module for a Plotly widget with standard error handling."""
    return output_widget("plot")

@module.ui
def dt_download_ui(title: str):
    """Reusable UI module for a DataTable with a CSV download button."""
    return ui.card(
        ui.card_header(
            ui.div(
                title,
                ui.download_button("btn_download", "Download CSV", class_="btn-sm btn-outline-primary"),
                class_="d-flex justify-content-between align-items-center w-100"
            )
        ),
        ui.output_data_frame("table")
    )


# Servers --------------------------------------------------------------------------------------------------------------
@module.server
def safe_plot_server(input, output, session, data_reactive: Union[reactive.Value, Callable], plot_type: Union[PlotType, str, Callable]):
    """A universal plotting server module that handles null-checks and error boundaries."""

    @render_widget
    def plot():
        data = data_reactive() if callable(data_reactive) else data_reactive.get()
        if data is None:
            return None

        # Resolve the plot type if it's passed as a reactive function (e.g. dropdowns)
        p_type = plot_type() if callable(plot_type) else plot_type

        try:
            return render_plot(data, p_type)
        except Exception as e:
            ui.notification_show(f"Plotting Error: {str(e)}", type="error", duration=10)
            return None


@module.server
def dt_download_server(input, output, session, data_callable: Callable, filename: str, height: str = "600px"):
    """A universal module for rendering interactive DataTables with an attached CSV export."""
    @render.data_frame
    def table():
        if (df := data_callable()) is None:
            return pd.DataFrame()
        return render.DataTable(df, selection_mode="none", height=height, filters=True)

    @render.download(filename=filename)
    async def btn_download():
        if (df := data_callable()) is not None:
            yield await to_thread(df.to_csv, index=False)
