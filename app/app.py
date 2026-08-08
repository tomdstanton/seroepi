import importlib.metadata
import os
from asyncio import sleep, to_thread
from pathlib import Path

import shinyswatch
from google import genai
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from shiny import module, reactive, render, ui

from .burden import burden_server, burden_ui
from .coverage import coverage_server, coverage_ui
from .dataset import dataset_server, dataset_ui
from .forecasting import forecasting_server, forecasting_ui
from .formulation import formulation_server, formulation_ui
from .serology import serology_server, serology_ui
from .icons import icon_lucide
from .utils import generate_temp_download, ui_task

# Constants ------------------------------------------------------------------------------------------------------------
_package = "SeroEpi"
_app_name = _package # f"{_package} App"
_meta = importlib.metadata.metadata(_package)
__version__ = _meta.get("Version", "dev")

import email.utils

# Dynamically extract author info mapped from pyproject.toml
_author_raw = _meta.get("Author-email")
__author_name, __author_email = email.utils.parseaddr(_author_raw)

# Extract GitHub/Repository URL
_urls = dict(u.split(", ", 1) for u in _meta.get_all("Project-URL") or [])
__github_url = _urls.get("Repository") or _urls.get("Source") or _urls.get("GitHub") or f"https://github.com/tomdstanton/{_package}"


# UI -------------------------------------------------------------------------------------------------------------------
@module.ui
def home_ui():
    return ui.div(
        # Hero Section
        ui.div(
            ui.h1("SeroEpi", class_="display-3 fw-bold text-primary mb-3"),
            ui.p("An advanced computational suite for genomic sero-epidemiology and vaccine formulation.", class_="lead text-muted mb-4"),
            ui.p("Transforming vast pathogen datasets into actionable public health intelligence using cutting-edge Bayesian inference and high-performance data engines.", class_="mb-5"),
            class_="text-center py-5 px-3 rounded-3 bg-light mb-5 shadow-sm"
        ),
        
        # Analytical Pillars
        ui.h3("Core Analytical Pillars", class_="text-primary mb-4 fw-bold border-bottom pb-2"),
        ui.layout_columns(
            ui.value_box(
                "Burden & Dynamics",
                ui.p("Track pathogen prevalence and variant diversity across time and space.", class_="fs-6"),
                showcase=icon_lucide("biohazard", size="3em"),
                theme="primary",
            ),
            ui.value_box(
                "Vaccine Formulation",
                ui.p("Design optimal multivalent vaccine formulations targeting circulating lineages.", class_="fs-6"),
                showcase=icon_lucide("syringe", size="3em"),
                theme="success",
            ),
            ui.value_box(
                "Bayesian Forecasting",
                ui.p("Predict future incidence and calculate effective reproduction numbers.", class_="fs-6"),
                showcase=icon_lucide("sparkles", size="3em"),
                theme="info",
            ),
            ui.value_box(
                "Classical Serology",
                ui.p("Estimate Force of Infection and Basic Reproduction Number.", class_="fs-6"),
                showcase=icon_lucide("droplet", size="3em"),
                theme="warning",
            ),
            col_widths=(6, 6, 6, 6),
            class_="mb-5"
        ),

        # Technical Architecture
        ui.h3("Technical Scale & Architecture", class_="text-primary mb-4 fw-bold border-bottom pb-2"),
        ui.layout_columns(
            ui.value_box(
                "Polars Data Engine",
                ui.p("Lightning-fast, multi-threaded data processing for massive genomics datasets.", class_="fs-6 text-dark"),
                showcase=icon_lucide("database", size="3em"),
                theme="light",
            ),
            ui.value_box(
                "JAX & NumPyro",
                ui.p("Hardware-accelerated Bayesian inference and Probabilistic Programming.", class_="fs-6 text-dark"),
                showcase=icon_lucide("cpu", size="3em"),
                theme="light",
            ),
            ui.value_box(
                "Geospatial Intelligence",
                ui.p("Built-in resolution of global boundaries and dynamic demographic metadata.", class_="fs-6 text-dark"),
                showcase=icon_lucide("globe", size="3em"),
                theme="light",
            ),
            col_widths=(4, 4, 4),
            class_="mb-5"
        ),
        
        class_="container mt-4",
        style="max-width: 1200px;",
    )


main_ui = ui.page_navbar(
    # Notice we pass a unique ID string to each module function!
    ui.nav_panel("Home", home_ui("tab_home"), icon=icon_lucide("house")),
    ui.nav_panel("Dataset", dataset_ui("tab_dataset"), icon=icon_lucide("database")),
    ui.nav_panel("Burden", burden_ui("tab_burden"), icon=icon_lucide("biohazard")),
    ui.nav_panel("Formulation", formulation_ui("tab_formulation"), icon=icon_lucide("syringe")),
    ui.nav_panel("Coverage", coverage_ui("tab_coverage"), icon=icon_lucide("shield")),
    ui.nav_panel("Forecasting", forecasting_ui("tab_forecasting"), icon=icon_lucide("sparkles")),
    ui.nav_panel("Serology", serology_ui("tab_serology"), icon=icon_lucide("droplet")),
    ui.nav_spacer(),  # Pushes everything after this to the right side of the navbar
    ui.nav_control(
        ui.div(
            ui.tags.button(
                icon_lucide("save"),
                " Workspace",
                class_="btn btn-sm btn-outline-primary mt-1 me-2 dropdown-toggle",
                type="button",
                **{"data-bs-toggle": "dropdown", "aria-expanded": "false"},
            ),
            ui.div(
                ui.h6("Workspace Summary", class_="dropdown-header px-0 text-primary fw-bold"),
                ui.output_ui("session_state_summary"),
                ui.hr(class_="my-2"),
                ui.h6("Manage Session", class_="dropdown-header px-0 text-primary fw-bold"),
                ui.download_button(
                    "btn_save_workspace", "Save Workspace (.sero)", class_="btn-outline-primary w-100 mb-2"
                ),
                ui.input_file("workspace_file", "Restore Workspace (.sero)", accept=[".sero"]),
                class_="dropdown-menu dropdown-menu-end p-3 shadow border-0",
                style="min-width: 280px;",
            ),
            class_="dropdown",
        )
    ),
    ui.nav_control(
        ui.div(
            ui.tags.button(
                icon_lucide("book-open"),
                " Active Context",
                class_="btn btn-sm btn-outline-success mt-1 me-2 dropdown-toggle",
                type="button",
                **{"data-bs-toggle": "dropdown", "aria-expanded": "false"},
            ),
            ui.div(
                ui.h6("Select Active Items", class_="dropdown-header px-0 text-primary fw-bold"),
                ui.input_select("global_dataset", "Dataset", choices={"": "None"}),
                ui.input_select("global_run", "Burden Run", choices={"": "None"}),
                ui.input_select("global_vac", "Formulation", choices={"": "None"}),
                class_="dropdown-menu dropdown-menu-end p-3 shadow border-0",
                style="min-width: 260px;",
            ),
            class_="dropdown",
        )
    ),
    ui.nav_control(
        ui.tags.button(
            icon_lucide("bot"),
            " Ask AI",
            class_="btn btn-sm btn-outline-info mt-1",
            type="button",
            **{"data-bs-toggle": "offcanvas", "data-bs-target": "#chat_offcanvas"},
        )
    ),
    ui.nav_control(ui.input_dark_mode(id="dark_mode")),
    theme=shinyswatch.theme.simplex(),
    title=_app_name,
    id="main_nav",
    window_title=_app_name,
    footer=ui.TagList(
        ui.div(
            ui.HTML(f"""
                <div style="text-align: center; font-size: 0.85rem; color: #64748B; border-top: 1px solid #1E293B; padding-top: 15px; padding-bottom: 15px; margin-top: auto;">
                    <strong>{_package}</strong> v{__version__} &bull; 
                    Built by <a href="mailto:{__author_email}" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-envelope-fill"></i> {__author_name}</a> &bull; 
                    <a href="{__github_url}" trait="_blank" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-github"></i> GitHub</a> &bull; 
                    <a href="https://pypi.org/project/{_package}/" trait="_blank" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-box-seam-fill"></i> PyPI</a>
                </div>
            """),
            class_="container-fluid d-flex flex-column justify-content-end",
        ),
        ui.div(
            ui.div(
                ui.h5(icon_lucide("bot"), f" {_app_name} Assistant", class_="offcanvas-title"),
                ui.tags.button(type="button", class_="btn-close text-reset", **{"data-bs-dismiss": "offcanvas"}),
                class_="offcanvas-header",
            ),
            ui.div(ui.chat_ui("ai_assistant"), class_="offcanvas-body pb-3"),
            class_="offcanvas offcanvas-end",
            tabindex="-1",
            id="chat_offcanvas",
            style="width: 500px;",
            **{"data-bs-scroll": "true", "data-bs-backdrop": "false"},  # Allows interacting with dashboard while open!
        ),
    ),
    header=ui.tags.head(ui.tags.script(src="https://code.jquery.com/ui/1.13.2/jquery-ui.min.js")),
)


# Server ---------------------------------------------------------------------------------------------------------------
def main_server(input, output, session):
    app_state = {
        "shared_df": reactive.Value(None),
        "shared_dist": reactive.Value(None),
        "shared_agg_df": reactive.Value(None),
        "prev_results": reactive.Value(None),
        "fitted_estimator": reactive.Value(None),
        "pw_collections_cache": reactive.Value({}),
        "baseline_res": reactive.Value(None),
        "current_formulation": reactive.Value(None),
        "results_registry": reactive.Value({}),
        "formulation_registry": reactive.Value({}),
        "dataset_registry": reactive.Value({}),
        "active_dataset_name": reactive.Value(None),
        "active_run_name": reactive.Value(None),
        "active_vac_name": reactive.Value(None),
    }

    # --- GLOBAL REGISTRY ROUTERS ---
    @reactive.Effect
    def sync_global_dataset_dropdown():
        reg = app_state["dataset_registry"].get()
        active = app_state["active_dataset_name"].get()
        if not reg:
            ui.update_select("global_dataset", choices={"": "None"}, selected="")
            return
        choices = {k: k for k in reg.keys()}
        selected = active if active in choices else list(choices.keys())[-1]
        ui.update_select("global_dataset", choices=choices, selected=selected)

    @reactive.Effect
    @reactive.event(input.global_dataset)
    def handle_global_dataset_change():
        selected = input.global_dataset()
        reg = app_state["dataset_registry"].get()
        if selected and reg and selected in reg:
            ds_dict = reg[selected]
            app_state["active_dataset_name"].set(selected)
            if app_state["shared_df"].get() is not ds_dict["df"]:
                app_state["shared_df"].set(ds_dict["df"])
                app_state["shared_dist"].set(ds_dict["dist"])
                app_state["shared_trans_dist"].set(ds_dict["trans_dist"])
                app_state["shared_agg_df"].set(None)

    @reactive.Effect
    def sync_global_run_dropdown():
        reg = app_state["results_registry"].get()
        active = app_state["active_run_name"].get()
        if not reg:
            ui.update_select("global_run", choices={"": "None"}, selected="")
            return
        choices = {k: k for k in reg.keys()}
        selected = active if active in choices else list(choices.keys())[-1]
        ui.update_select("global_run", choices=choices, selected=selected)

    @reactive.Effect
    @reactive.event(input.global_run)
    def handle_global_run_change():
        selected = input.global_run()
        reg = app_state["results_registry"].get()
        if selected and reg and selected in reg:
            run_dict = reg[selected]
            app_state["active_run_name"].set(selected)
            if app_state["prev_results"].get() is not run_dict["res"]:
                app_state["prev_results"].set(run_dict["res"])
                app_state["fitted_estimator"].set(run_dict["est"])
                app_state["shared_agg_df"].set(run_dict["agg_df"])

    @reactive.Effect
    def sync_global_vac_dropdown():
        reg = app_state["formulation_registry"].get()
        active = app_state["active_vac_name"].get()
        if not reg:
            ui.update_select("global_vac", choices={"": "None"}, selected="")
            return
        choices = {k: k for k in reg.keys()}
        selected = active if active in choices else list(choices.keys())[-1]
        ui.update_select("global_vac", choices=choices, selected=selected)

    @reactive.Effect
    @reactive.event(input.global_vac)
    def handle_global_vac_change():
        selected = input.global_vac()
        reg = app_state["formulation_registry"].get()
        if selected and reg and selected in reg:
            app_state["active_vac_name"].set(selected)
            if app_state["current_formulation"].get() is not reg[selected]:
                app_state["current_formulation"].set(reg[selected])

    dataset_server("tab_dataset", app_state=app_state)
    burden_server("tab_burden", app_state=app_state)
    formulation_server("tab_formulation", app_state=app_state)
    coverage_server("tab_coverage", app_state=app_state)
    forecasting_server("tab_forecasting", app_state=app_state)
    serology_server("tab_serology", app_state=app_state)

    @render.download_button(filename=f"{_package}_workspace.sero")
    def btn_save_workspace():
        export_state = {k: v.get() for k, v in app_state.items()}
        return generate_temp_download(lambda p: joblib_dump(export_state, p), ".sero", "Workspace Export Error")

    @reactive.Effect
    @reactive.event(input.workspace_file)
    async def load_workspace():
        if not (file_info := input.workspace_file()):
            return

        async with ui_task("Workspace Restoration Error") as p:
            p.set(message="Restoring Workspace...", value=50)
            await sleep(0)

            imported_state = await to_thread(joblib_load, Path(file_info[0]["datapath"]))

            # Safely restore all reactive values that match our state dictionary
            for key, r_val in app_state.items():
                if key in imported_state:
                    r_val.set(imported_state[key])

            p.set(message="Workspace Restored!", value=100)
            ui.notification_show("Session successfully restored.", type="message")

            # Expand the dataset accordion to show data loaded (using the namespace ID!)
            ui.update_accordion("tab_dataset-dataset_accordion", show="Cluster Generation")

    @render.ui
    def session_state_summary():
        df = app_state["shared_df"].get()
        res = app_state["prev_results"].get()
        est = app_state["fitted_estimator"].get()
        vac = app_state["current_formulation"].get()

        elements = []

        if ds_reg := app_state["dataset_registry"].get():
            elements.append(
                ui.p(ui.tags.strong("Loaded Datasets: "), f"{len(ds_reg)} available", class_="mb-1 text-success small")
            )
        else:
            elements.append(ui.p(ui.tags.strong("Loaded Datasets: "), "None", class_="mb-1 text-muted small"))

        if df is not None:
            elements.append(
                ui.p(
                    ui.tags.strong("Active Dataset: "),
                    f"{len(df):,} rows, {len(df.columns)} cols",
                    class_="mb-2 text-success small",
                )
            )
        else:
            elements.append(ui.p(ui.tags.strong("Active Dataset: "), "None loaded", class_="mb-2 text-muted small"))

        if res is not None:
            strata = ", ".join(res.stratified_by) if res.stratified_by else "Global"
            elements.append(
                ui.p(ui.tags.strong("Burden: "), f"'{res.trait}' by {strata}", class_="mb-2 text-success small")
            )
        else:
            elements.append(ui.p(ui.tags.strong("Burden: "), "Not calculated", class_="mb-2 text-muted small"))

        if est is not None and getattr(est, "is_fitted_", False):
            est_name = type(est).__name__.replace("PrevalenceEstimator", "")
            elements.append(ui.p(ui.tags.strong("Model: "), f"{est_name} (Fitted)", class_="mb-2 text-success small"))
        else:
            elements.append(ui.p(ui.tags.strong("Model: "), "No active model", class_="mb-2 text-muted small"))

        if vac is not None:
            elements.append(
                ui.p(
                    ui.tags.strong("Formulation: "),
                    f"{vac.max_valency}-valent formulation",
                    class_="mb-2 text-success small",
                )
            )
        else:
            elements.append(ui.p(ui.tags.strong("Formulation: "), "Not generated", class_="mb-2 text-muted small"))

        if reg := app_state["results_registry"].get():
            elements.append(
                ui.p(ui.tags.strong("Cached Models: "), f"{len(reg)} runs available", class_="mb-1 text-success small")
            )
        else:
            elements.append(ui.p(ui.tags.strong("Cached Models: "), "None", class_="mb-1 text-muted small"))

        if v_reg := app_state["formulation_registry"].get():
            elements.append(
                ui.p(
                    ui.tags.strong("Cached Formulations: "),
                    f"{len(v_reg)} formulations available",
                    class_="mb-0 text-success small",
                )
            )
        else:
            elements.append(ui.p(ui.tags.strong("Cached Formulations: "), "None", class_="mb-0 text-muted small"))

        return ui.div(*elements)

    # =====================================================================================
    # AI ASSISTANT (Context-Aware Chatbot)
    # =====================================================================================
    chat = ui.Chat(id="ai_assistant")

    @chat.on_user_submit
    async def handle_chat_message():
        # Build a Structured Markdown Payload of the Analytical Workspace
        sys_prompt_parts = [
            "You are a world-class bioinformatics and epidemiology AI assistant. ",
            "You are embedded directly within the 'seroepi' Shiny dashboard. ",
            "Answer the user's questions based strictly on the current state of their analytical workspace below:\n",
            "### WORKSPACE STATE\n",
        ]

        if (df := app_state["shared_df"].get()) is not None:
            sys_prompt_parts.append(f"**Active Dataset**: {df.shape[0]} rows, {df.shape[1]} columns.")

        if (res := app_state["prev_results"].get()) is not None:
            sys_prompt_parts.append(
                f"**Active burden Run** ({res.method}, Trait: '{res.trait}'):\n```text\n{str(res.data.head(10))}\n```"
            )

        if (est := app_state["fitted_estimator"].get()) is not None and getattr(est, "is_fitted_", False):
            sys_prompt_parts.append(f"**Active Model**: {type(est).__name__}")
            if hasattr(est, "diagnostics"):
                try:
                    diag_df = est.diagnostics()
                    sys_prompt_parts.append(f"- **MCMC Diagnostics**:\n```text\n{str(diag_df)}\n```")
                except Exception:
                    pass

        if (vac := app_state["current_formulation"].get()) is not None:
            sys_prompt_parts.append(f"**Active Formulation Formulation** ({vac.max_valency}-valent):")
            sys_prompt_parts.append(f"- **Targets**: {', '.join(vac.get_formulation())}")
            sys_prompt_parts.append(f"- **Top Rankings**:\n```text\n{str(vac.rankings.head(vac.max_valency))}\n```")
            if not vac.stability_metrics.is_empty():
                sys_prompt_parts.append(f"- **LOO Stability Metrics**:\n```text\n{str(vac.stability_metrics)}\n```")

        if reg := app_state["results_registry"].get():
            sys_prompt_parts.append(f"**Cached burden Runs**: {', '.join(reg.keys())}")

        if v_reg := app_state["formulation_registry"].get():
            sys_prompt_parts.append(f"**Cached Formulations**: {', '.join(v_reg.keys())}")

        if ds_reg := app_state["dataset_registry"].get():
            sys_prompt_parts.append(f"**Loaded Datasets**: {', '.join(ds_reg.keys())}")

        sys_prompt = "\n".join(sys_prompt_parts)

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            await chat.append_message(
                "GEMINI_API_KEY environment variable is missing. Please set it to use the AI assistant."
            )
            return

        # Translate Shiny's message history to Gemini's expected format
        gemini_messages = [
            {"role": "model" if msg["role"] == "assistant" else "user", "parts": [{"text": msg["content"]}]}
            for msg in chat.messages()
        ]

        client = genai.Client(api_key=api_key)

        try:
            # 3. Query the LLM via its async interface, injecting the system prompt via config
            response = await client.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents=gemini_messages, config={"system_instruction": sys_prompt}
            )

            # Efficiently unwrap the Gemini text chunks for Shiny's streaming UI
            async def stream_generator():
                async for chunk in response:
                    if chunk.text:
                        yield chunk.text

            await chat.append_message_stream(stream_generator())
        except Exception as e:
            await chat.append_message(f"Error communicating with AI: {str(e)}")
