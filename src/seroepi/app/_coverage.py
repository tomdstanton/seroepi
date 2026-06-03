from pathlib import Path
from shiny import ui, module, reactive, render
from asyncio import to_thread, sleep
from dataclasses import fields

from seroepi.constants import PlotType
from seroepi.app._utils import ui_task, build_grouped_choices, safe_plot_ui, safe_plot_server, export_settings_ui, generate_temp_download, dt_download_ui, dt_download_server, format_metadata_ui
from seroepi.plotting import render_plot
from seroepi import estimators

@module.ui
def coverage_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Evaluate Coverage 🛡️",
                    ui.p("Evaluate how well your Formulation covers different sub-populations.", class_="small text-muted mb-2"),
                    ui.input_radio_buttons(
                        "coverage_mode",
                        "Evaluation Mode",
                        choices={
                            "binary": "Binary (Covered vs Not Covered)",
                            "compositional": "Compositional (Breakdown of formulation components)"
                        }
                    ),
                    ui.tooltip(ui.input_selectize("coverage_stratify", "Stratify By (e.g. AMR Gene, Region)", choices=[], multiple=True),
                               "Variables to slice the data by to see coverage differences."),
                    ui.input_action_button("btn_evaluate_coverage", "Calculate Coverage", class_="btn-primary w-100 mt-3")
                ),
                id="coverage_accordion",
                open=["Evaluate Coverage 🛡️"],
            ),
            width=350
        ),
        ui.navset_card_tab(
            ui.nav_panel("Data 🧮", ui.output_ui("coverage_data_content"), value="tab_coverage_data"),
            ui.nav_panel(
                "Plots 📊",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_select("coverage_plot_type", "Plot Type", choices={
                            PlotType.CUMULATIVE_COVERAGE.value: "Cumulative Coverage (Pareto)",
                            PlotType.FOREST.value: "Forest Plot (Stratified Coverage)",
                            PlotType.COMPOSITION_BAR.value: "Composition Bar Plot",
                            PlotType.CHOROPLETH.value: "Spatial Coverage"
                        }),
                        ui.hr(),
                        export_settings_ui("cov"),
                        ui.download_button("btn_dl_cov_plot", "Download Plot", class_="btn-outline-primary w-100"),
                        width=280
                    ),
                    ui.output_ui("coverage_plot_wrapper")
                ),
                value="tab_coverage_plots"
            ),
            id="coverage_tabs"
        )
    )

@module.server
def coverage_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    prev_results = app_state["prev_results"]
    current_formulation = app_state["current_formulation"]
    coverage_results = reactive.Value(None)

    @reactive.Effect
    def update_dropdowns():
        if (df := shared_df.get()) is not None:
            strat_choices = build_grouped_choices(df.epi.stratify_cols + df.epi.genotypes, "Metadata & Traits")
            ui.update_selectize("coverage_stratify", choices=strat_choices)

    @reactive.Effect
    @reactive.event(input.btn_evaluate_coverage)
    async def evaluate_coverage():
        df = shared_df.get()
        vac = current_formulation.get()
        if df is None or vac is None:
            ui.notification_show("Please ensure a dataset is loaded and a formulation is generated.", type="warning")
            return
            
        stratify = list(input.coverage_stratify())
        mode = input.coverage_mode()
        
        async with ui_task("Coverage Error") as p:
            p.set(message="Aggregating coverage...", value=30)
            await sleep(0)
            
            def run_calculation():
                if mode == "binary":
                    # Evaluate binary True/False coverage
                    cov_name = f"{vac.max_valency}-valent {vac.trait} Coverage"
                    cov_df = vac.assess_coverage(df, col_name=cov_name)
                    agg_df = cov_df.epi.aggregate_prevalence(stratify_by=stratify, trait_col=cov_name)
                else:
                    # Evaluate compositional breakdown of the variants IN the formulation
                    targets = vac.get_formulation()
                    filtered_df = df[df[vac.trait].isin(targets)].copy()
                    
                    strat = stratify + [vac.trait] if vac.trait not in stratify else stratify
                    agg_df = filtered_df.epi.aggregate_prevalence(stratify_by=strat, trait_col=None)
                
                # Run a fast Frequentist prevalence estimator for coverage insights
                return estimators.GLMPrevalenceEstimator().calculate(agg_df)
                
            res = await to_thread(run_calculation)
            coverage_results.set(res)
            
            p.set(message="Done!", value=100)
            ui.notification_show("Coverage evaluated successfully!", type="message")

    @reactive.Calc
    def coverage_plot_router():
        p_type = input.coverage_plot_type()
        if p_type == PlotType.CUMULATIVE_COVERAGE.value:
            res = prev_results.get()
            vac = current_formulation.get()
            if res is not None and vac is not None:
                return {"res": res, "formulation": vac}
            return None
        return coverage_results.get()

    @render.ui
    def coverage_data_content():
        if (res := coverage_results.get()) is None:
            return ui.div("Calculate coverage to view the results data.", class_="text-center mt-5 text-muted fs-4")

        # Dynamically extract instance attributes into a dictionary for metadata display
        meta_dict = {f.name: getattr(res, f.name) for f in fields(res) if f.name not in ['data', 'model_results']}
        meta_ui = format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Coverage Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("coverage_data", "Coverage Estimates")
        )

    dt_download_server("coverage_data", data_callable=lambda: coverage_results.get().data if coverage_results.get() else None,
                       filename="coverage_estimates.csv", height="400px")

    @render.ui
    def coverage_plot_wrapper():
        if input.coverage_plot_type() == PlotType.CUMULATIVE_COVERAGE.value:
            if current_formulation.get() is None:
                return ui.div("Design a formulation to view cumulative coverage.", class_="text-center mt-5 text-muted fs-4")
        elif coverage_results.get() is None:
            return ui.div("Calculate coverage to view stratified plots.", class_="text-center mt-5 text-muted fs-4")
        return safe_plot_ui("coverage_plot")

    safe_plot_server("coverage_plot", data_reactive=coverage_plot_router, plot_type=input.coverage_plot_type)

    @render.download(filename=lambda: f"coverage_plot.{input.cov_plot_format()}")
    def btn_dl_cov_plot():
        data = coverage_plot_router()
        if data is None:
            ui.notification_show("No plot data available to export.", type="warning")
            return None
            
        fig = render_plot(data, input.coverage_plot_type())
        def save_fig(p: Path):
            fig.write_image(p, format=input.cov_plot_format(), width=input.cov_plot_width(), height=input.cov_plot_height())
            
        return generate_temp_download(save_fig, f".{input.cov_plot_format()}", "Plot Export Error")