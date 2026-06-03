from asyncio import sleep, get_running_loop, to_thread
from pathlib import Path

from shiny import ui, module, reactive, render

from seroepi.app._utils import safe_plot_ui, dt_download_ui, dt_download_server, safe_plot_server, ui_task, generate_temp_download, update_registry, export_settings_ui
from seroepi.constants import PlotType, AggregationType
from seroepi.formulation import Formulation, CVFormulationDesigner, PostHocFormulationDesigner, CustomFormulationDesigner
from seroepi.plotting import render_plot


@module.ui
def formulation_ui():
    """UI layout for the algorithmic and manual formulation designer."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Formulation Design 💊",
                    ui.tooltip(ui.input_slider("max_valency", "Trait Valency",
                                               min=2, max=30, value=10, step=1),
                               "The maximum number of distinct variants to include in the optimal formulation formulation."),
                    ui.tooltip(ui.input_selectize("form_holdout", "Cross-Validation Stratum", choices=["Awaiting Data..."]),
                               "The spatial or demographic level used to evaluate stability via Leave-One-Out Cross-Validation."),
                    ui.tooltip(ui.input_select("form_designer", "Designer Type",
                                               choices={"posthoc": "Post-Hoc (Fast)", "cv": "Cross-Validated (Rigorous)"}),
                               "Post-Hoc is faster but assumes independence. Cross-Validated re-fits the model iteratively for rigorous stability testing."),
                    ui.input_action_button("btn_run_designer", "Generate Formulation 🚀", class_="btn-primary w-100 mt-3")
                ),
                ui.accordion_panel(
                    "Manual Override 🛠️",
                    ui.p("Drag to reorder, click 'x' to remove.", class_="text-muted small"),
                    ui.tooltip(
                        ui.input_selectize(
                            "custom_traits",
                            "Custom Formulation Pipeline:",
                            choices=[],
                            multiple=True,
                            options={"plugins": ["remove_button", "drag_drop"], "placeholder": "e.g. K1, K2..."}
                        ),
                        "Drag to reorder, or manually type and add variants to evaluate a custom formulation formulation."
                    ),
                    ui.div(
                        ui.input_action_button("btn_eval_custom", "Evaluate Custom", class_="btn-outline-primary w-50"),
                        ui.input_action_button("btn_copy_optimal", "Copy Optimal", class_="btn-outline-primary w-50"),
                        class_="d-flex gap-2 mt-3"
                    )
                ),
                ui.accordion_panel(
                    "Import / Export 💾",
                    ui.p("Save rigorous CV formulations to avoid recalculating, or load an existing one.", class_="text-muted small"),
                    ui.input_file("formulation_upload", "Load Formulation (.pkl)", accept=[".pkl"]),
                    ui.output_ui("formulation_export_ui")
                ),
                id="formulation_accordion",
                open=["Formulation Design 💊"], multiple=True
            ),
            width=350
        ),
        ui.navset_card_tab(
            ui.nav_panel("Rankings 🏆", ui.output_ui("form_rankings_content"), value="tab_form_rankings"),
            ui.nav_panel("Stability Metrics ⚖️", ui.output_ui("form_stability_content"), value="tab_form_stability"),
            ui.nav_panel("Permutations 🔄", ui.output_ui("form_history_content"), value="tab_form_permutations"),
            ui.nav_panel("Plots 📊", ui.output_ui("form_plots_content"), value="tab_form_plots"),
            id="formulation_tabs"
        )
    )


@module.server
def formulation_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    shared_agg_df = app_state["shared_agg_df"]
    prev_results = app_state["prev_results"]
    fitted_estimator = app_state["fitted_estimator"]
    baseline_res = app_state["baseline_res"]
    current_formulation = app_state["current_formulation"]
    results_registry = app_state["results_registry"]
    formulation_registry = app_state["formulation_registry"]

    @reactive.Effect
    def manage_accordion_state():
        if prev_results.get() is None:
            ui.update_accordion("formulation_accordion", show="Formulation Design 💊")

    @reactive.Effect
    def update_form_inputs():
        res = prev_results.get()
        est = fitted_estimator.get()

        if res is None or est is None:
            ui.update_selectize("form_holdout", choices=["Awaiting Data..."])
            ui.update_select("form_designer", choices={"posthoc": "Post-Hoc (Fast)", "cv": "Cross-Validated (Rigorous)"}, selected="cv")
            return

        # 1. Populate the Stratum dropdown exactly from the prior calculation metadata
        holdouts = res.stratified_by
        if holdouts:
            ui.update_selectize("form_holdout", choices=holdouts, selected=holdouts[0])
        else:
            ui.update_selectize("form_holdout", choices=["Global (CV Not Possible)"],
                                selected="Global (CV Not Possible)")

        # 2. Select default Designer Type based on the estimator instance
        if est and hasattr(est, 'is_fitted_'):
            ui.update_select("form_designer", selected="cv")
        else:
            ui.update_select("form_designer", selected="posthoc")

    @reactive.Effect
    def update_custom_traits_choices():
        res = prev_results.get()
        if res is not None:
            all_traits = res.data['target'].unique().tolist()
            ui.update_selectize("custom_traits", choices=all_traits)

    @reactive.Effect
    @reactive.event(input.btn_run_designer)
    async def generate_optimal():
        res, est, agg_df = prev_results.get(), fitted_estimator.get(), shared_agg_df.get()
        if res is None or est is None or agg_df is None:
            ui.notification_show("Please calculate burden and ensure a valid dataset is active.", type="warning")
            return

        if res.aggregation_type != AggregationType.COMPOSITIONAL:
            ui.notification_show(
                "Note: Evaluating a single binary trait. For multi-antigen formulations, use Compositional mode in Tab 1.",
                type="message", duration=8)

        holdout = input.form_holdout()

        if not holdout or "Awaiting Data" in holdout:
            ui.notification_show("Please wait for valid burden estimates before designing a formulation.", type="warning")
            return
            
        is_global = "Not Possible" in holdout
        loo_col = None if is_global else holdout

        async with ui_task("Designer Error") as p:
                # We can instantly use the already-calculated results from Tab 1 as the Baseline!
                baseline_res.set(res)

                designer_type = input.form_designer()
                target_str = "Global Data (No CV)" if is_global else holdout
                p.set(message=f"Running {designer_type.upper()} Designer on {target_str}...", value=50)
                await sleep(0)

                loop = get_running_loop()

                def ui_progress_callback(current, total):
                    # Safely update the Shiny UI from the background thread
                    percentage = 50 + int(45 * (current / total))
                    if is_global:
                        message = "Generating unstratified baseline formulation..."
                    else:
                        message = f"Running CV Fold {current}/{total}..." if designer_type != 'posthoc' else f"Running Permutation {current}/{total}..."
                    loop.call_soon_threadsafe(p.set, percentage, message)

                if designer_type == 'posthoc':
                    designer = PostHocFormulationDesigner(valency=input.max_valency(), n_jobs=-1)
                    await to_thread(designer.fit, res, loo_col=loo_col, progress_callback=ui_progress_callback)
                else:
                    designer = CVFormulationDesigner(valency=input.max_valency(), n_jobs=-1)
                    await to_thread(designer.fit, est, agg_df, loo_col=loo_col,
                                            progress_callback=ui_progress_callback)

                optimal_formulation = designer.formulation_

                # Update the active formulation state
                current_formulation.set(optimal_formulation)

                # Cache the formulation dynamically
                run_name = app_state["active_run_name"].get() or "Unknown Run"
                vac_name = f"Optimal {optimal_formulation.max_valency}-valent ({designer_type.upper()}) | {run_name}"
                update_registry(formulation_registry, vac_name, optimal_formulation)
                app_state["active_vac_name"].set(vac_name)

                p.set(message="Rendering dashboards...", value=95)
                await sleep(0)
                ui.notification_show("Optimal Formulation Designed!", type="message")
                ui.update_navset("formulation_tabs", selected="tab_form_plots")


    # --- THE CUSTOM OVERRIDE LOGIC ---
    @reactive.Effect
    @reactive.event(input.btn_copy_optimal)
    def copy_to_custom():
        if formulation := current_formulation.get():  # type: Formulation
            ui.update_selectize("custom_traits", selected=formulation.get_formulation())
            ui.notification_show("Copied to manual editor.", type="message")

    @reactive.Effect
    @reactive.event(input.btn_eval_custom)
    def evaluate_custom():
        if not (traits := input.custom_traits()):
            ui.notification_show("Please select at least one custom trait.", type="warning")
            return

        res = prev_results.get()
        if res is None:
            ui.notification_show("Please select a valid active burden run.", type="warning")
            return

        designer = CustomFormulationDesigner(targets=list(traits))
        designer.fit(res)
        current_formulation.set(designer.formulation_)
        
        # Cache the custom formulation
        run_name = app_state["active_run_name"].get() or "Unknown Run"
        vac_name = f"Custom {designer.valency}-valent | {run_name}"
        update_registry(formulation_registry, vac_name, designer.formulation_)
        app_state["active_vac_name"].set(vac_name)

        ui.notification_show("Custom formulation evaluated.", type="message")
        ui.update_navset("formulation_tabs", selected="tab_form_plots")

    @render.ui
    def formulation_export_ui():
        if current_formulation.get() is not None:
            return ui.download_button("btn_download_formulation", "Download Formulation", class_="btn-outline-primary w-100 mt-2")
        return ui.div()

    @render.download(filename=lambda: "vaccine_formulation.pkl")
    def btn_download_formulation():
        vac = current_formulation.get()
        if vac:
            return generate_temp_download(vac.save, ".pkl", "Formulation Export Error")

    @reactive.Effect
    @reactive.event(input.formulation_upload)
    async def load_formulation_file():
        if not (file_info := input.formulation_upload()):
            return
        async with ui_task("Load Formulation Error") as p:
            p.set(message="Loading formulation...", value=50)
            await sleep(0)
            vac = await to_thread(Formulation.load, Path(file_info[0]["datapath"]))
            current_formulation.set(vac)
            run_name = app_state["active_run_name"].get() or "Imported Run"
            vac_name = f"Imported {vac.max_valency}-valent | {run_name}"
            update_registry(formulation_registry, vac_name, vac)
            app_state["active_vac_name"].set(vac_name)
            ui.notification_show("Formulation loaded successfully!", type="message")
            ui.update_navset("formulation_tabs", selected="tab_form_plots")

    # --- TAB 2: RENDERING THE DASHBOARD ---

    @render.ui
    def form_plots_content():
        if current_formulation.get() is None:
            return ui.div("Configure and generate an optimal formulation in the sidebar.",
                          class_="text-center mt-5 text-muted fs-4")
                          
        return ui.layout_sidebar(
            ui.sidebar(
                export_settings_ui("form"),
                ui.output_ui("dl_stability_btn_ui"),
                width=280
            ),
            ui.output_ui("stability_plot_card")
        )

    @render.ui
    def form_rankings_content():
        if current_formulation.get() is None:
            return ui.div("Generate a formulation to view rankings.", class_="text-center mt-5 text-muted fs-4")
        return dt_download_ui("form_rankings", "Formulation Rankings")

    @render.ui
    def form_stability_content():
        vac = current_formulation.get()
        if vac is None or vac.stability_metrics.empty:
            return ui.div("Run Cross-Validated designer to view stability metrics.", class_="text-center mt-5 text-muted fs-4")
        return dt_download_ui("form_stability", "LOO Stability Metrics")

    @render.ui
    def form_history_content():
        vac = current_formulation.get()
        if vac is None or vac.stability_metrics.empty:
            return ui.div("Run Cross-Validated designer to view permutation history.", class_="text-center mt-5 text-muted fs-4")
        return dt_download_ui("form_history", "Permutation History")

    @render.ui
    def stability_plot_card():
        vac = current_formulation.get()
        if vac is not None and not vac.stability_metrics.empty:
            return ui.card(
                ui.card_header("Cross-Validation Stability Matrix"),
                safe_plot_ui("stability_plot")
            )
        return ui.div()

    @render.ui
    def dl_stability_btn_ui():
        vac = current_formulation.get()
        if vac is not None and not vac.stability_metrics.empty:
            return ui.download_button("btn_download_stability", "Download Stability Plot", class_="btn-outline-primary w-100")
        return ui.div()

    @render.download(filename=lambda: f"formulation_stability.{input.form_plot_format()}")
    def btn_download_stability():
        vac = current_formulation.get()
        if vac is None or vac.stability_metrics.empty:
            return
        fig = render_plot(vac, PlotType.STABILITY_BUMP)
        def save_fig(p: Path):
            fig.write_image(p, format=input.form_plot_format(), width=input.form_plot_width(), height=input.form_plot_height())
        return generate_temp_download(save_fig, f".{input.form_plot_format()}", "Plot Export Error")

    dt_download_server("form_rankings",
                       data_callable=lambda: current_formulation.get().rankings if current_formulation.get() else None,
                       filename="formulation_rankings.csv")
    dt_download_server("form_stability",
                       data_callable=lambda: current_formulation.get().stability_metrics.reset_index() if current_formulation.get() else None,
                       filename="formulation_stability_metrics.csv")
    dt_download_server("form_history",
                       data_callable=lambda: current_formulation.get().permutation_history if current_formulation.get() else None,
                       filename="formulation_permutation_history.csv")

    safe_plot_server("stability_plot", data_reactive=current_formulation, plot_type=PlotType.STABILITY_BUMP)
