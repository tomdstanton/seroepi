from asyncio import sleep, to_thread
from dataclasses import fields
from pathlib import Path

from plotly.graph_objs import Figure
from shiny import module, reactive, render, ui
from shinywidgets import output_widget, render_widget

from seroepi import estimators
from seroepi.estimators.incidence import ReproductionNumberEstimator
from seroepi.constants import AggregationType, PlotType, TemporalResolution
from seroepi.domains import render_plot

from .icons import icon_lucide
from .utils import (
    EstimatorIntrospector,
    build_grouped_choices,
    dt_download_server,
    dt_download_ui,
    export_settings_ui,
    format_metadata_ui,
    generate_temp_download,
    ui_task,
)


@module.ui
def forecasting_ui():
    """UI layout for deep-diving into specific traits for clinical trials."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Incidence Aggregation",
                    ui.p(
                        "Aggregate historical incidence data for your evaluated trait.", class_="small text-muted mb-2"
                    ),
                    ui.tooltip(
                        ui.input_selectize("forecasting_stratify", "Stratify By (Optional)", choices=[], multiple=True),
                        "Variables to group the data by before calculating incidence. "
                        "A temporal column will be automatically included.",
                    ),
                    ui.tooltip(
                        ui.input_select(
                            "forecasting_freq",
                            "Time Frequency",
                            choices={x.value: x.value.replace('-', ' ').replace('_', ' ').title() for x in TemporalResolution},
                            selected=TemporalResolution.MONTH.value,
                        ),
                        "The time interval to bin the incidence data.",
                    ),
                    ui.tooltip(
                        ui.input_checkbox("forecasting_pad_zeros", "Pad Zeroes (Zero-fill missing)", value=True),
                        "Essential for BSTS models to maintain an unbroken time series. Fills missing time bins with 0 counts.",
                    ),
                    ui.input_action_button(
                        "btn_aggregate_incidence",
                        "Aggregate Data",
                        icon=icon_lucide("calculator"),
                        class_="btn-primary w-100 mt-3",
                    ),
                    icon=icon_lucide("calculator"),
                ),
                ui.accordion_panel(
                    "Incidence Estimation",
                    ui.input_select(
                        "longevity_estimator",
                        "Estimator Model",
                        choices={"bayesian": "Bayesian (BSTS)", "glm": "Frequentist (GLM)"},
                    ),
                    ui.output_ui("longevity_estimator_params_ui"),
                    ui.output_ui("longevity_model_io_ui"),
                    ui.input_action_button(
                        "btn_run_longevity",
                        "Forecast Longevity",
                        icon=icon_lucide("rocket"),
                        class_="btn-primary w-100 mt-2",
                    ),
                    icon=icon_lucide("sparkles"),
                ),
                ui.accordion_panel(
                    "Transmission Dynamics",
                    ui.p("Calculate time-varying effective reproduction number R_e(t).", class_="small text-muted mb-2"),
                    ui.tooltip(
                        ui.input_numeric("transmission_gt_mean", "Generation Time Mean", value=5.0),
                        "Mean generation time of the pathogen."
                    ),
                    ui.tooltip(
                        ui.input_numeric("transmission_gt_std", "Generation Time Std Dev", value=1.9),
                        "Standard deviation of the generation time."
                    ),
                    ui.input_action_button(
                        "btn_run_transmission",
                        "Calculate Re(t)",
                        icon=icon_lucide("activity"),
                        class_="btn-primary w-100 mt-2",
                    ),
                    icon=icon_lucide("activity"),
                ),
                id="forecasting_accordion",
                open=["Incidence Aggregation"],
                multiple=True,
            ),
            width=350,
        ),
        # Main Dashboard Array
        ui.navset_card_tab(
            ui.nav_panel(
                "Aggregates",
                ui.output_ui("forecasting_agg_content"),
                value="tab_forecasting_agg",
                icon=icon_lucide("calculator"),
            ),
            ui.nav_panel(
                "Estimates",
                ui.output_ui("forecasting_est_content"),
                value="tab_forecasting_est",
                icon=icon_lucide("trending-up"),
            ),
            ui.nav_panel(
                "Transmission",
                ui.output_ui("forecasting_trans_content"),
                value="tab_forecasting_trans",
                icon=icon_lucide("activity"),
            ),
            ui.nav_panel(
                "Diagnostics",
                ui.output_ui("forecasting_diag_content"),
                value="tab_forecasting_diag",
                icon=icon_lucide("stethoscope"),
            ),
            ui.nav_panel(
                "Plots",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.tooltip(
                            ui.input_select(
                                "forecasting_plot_type",
                                "Plot Type",
                                choices={
                                    PlotType.LONGEVITY.value: "Longevity Forecast",
                                    PlotType.EPICURVE.value: "Historical Coverage",
                                },
                            ),
                            "Visualization style for forecasting and coverage.",
                        ),
                        ui.hr(),
                        export_settings_ui("forecasting"),
                        ui.download_button(
                            "btn_download_forecasting_plot", "Download Plot", class_="btn-outline-primary w-100"
                        ),
                        width=280,
                    ),
                    ui.output_ui("forecasting_plot_content"),
                ),
                value="tab_forecasting_plots",
                icon=icon_lucide("chart-bar"),
            ),
            id="forecasting_tabs",
        ),
    )


@module.server
def forecasting_server(input, output, session, app_state: dict):
    """Server logic for clinical trial site selection and spatial density analysis."""
    shared_df = app_state["shared_df"]
    prev_results = app_state["prev_results"]
    current_formulation = app_state["current_formulation"]

    # Module-specific state
    shared_agg_inc_df = app_state.setdefault("shared_agg_inc_df", reactive.Value(None))
    shared_forecast = app_state.setdefault("shared_forecast", reactive.Value(None))
    fitted_longevity_estimator = app_state.setdefault("fitted_longevity_estimator", reactive.Value(None))
    shared_trans_results = app_state.setdefault("shared_trans_results", reactive.Value(None))
    ESTIMATOR_MAP = {"bayesian": "BayesianIncidenceEstimator", "glm": "GLMIncidenceEstimator"}

    @reactive.Effect
    def update_forecasting_options():
        df = shared_df.get()
        if df is not None:
            strat_choices = build_grouped_choices(df.epi.stratify_cols, "Other Variables")
            ui.update_selectize("forecasting_stratify", choices=strat_choices)

            if df.epi.has_temporal:
                default_freq = getattr(df.epi, "date_res", TemporalResolution.MONTH.value)
                ui.update_select("forecasting_freq", selected=default_freq)

            plot_choices = {
                PlotType.LONGEVITY.value: "Longevity Forecast",
                PlotType.EPICURVE.value: "Historical Coverage",
            }

            ui.update_select("forecasting_plot_type", choices=plot_choices)

    @reactive.Effect
    def manage_accordion_state():
        if shared_df.get() is None or prev_results.get() is None:
            ui.update_accordion("forecasting_accordion", show="Incidence Aggregation")

    @reactive.Effect
    @reactive.event(input.btn_aggregate_incidence)
    async def aggregate_incidence():
        df = shared_df.get()
        res = prev_results.get()
        if df is None or res is None:
            ui.notification_show("Please ensure data is loaded and a burden run is active.", type="warning")
            return

        if not df.epi.has_temporal:
            ui.notification_show("Incidence aggregation requires temporal metadata.", type="error")
            return

        stratify = list(input.forecasting_stratify())
        freq = input.forecasting_freq()
        pad_zeros = input.forecasting_pad_zeros()

        async with ui_task("Aggregation Error") as p:
            p.set(message="Aggregating incidence...", value=20)
            await sleep(0)

            def run_agg():
                if res.aggregation_type == AggregationType.COMPOSITIONAL:
                    # Ensure the trait is at the end of the strata list for compositional incidence
                    strat = stratify.copy()
                    if res.trait not in strat:
                        strat.append(res.trait)
                    return df.epi.aggregate_incidence(stratify_by=strat, trait_col=None, freq=freq, pad_zeros=pad_zeros)
                else:
                    # Standard binary trait incidence
                    return df.epi.aggregate_incidence(
                        stratify_by=stratify, trait_col=res.trait, freq=freq, pad_zeros=pad_zeros
                    )

            agg_df = await to_thread(run_agg)
            shared_agg_inc_df.set(agg_df)
            p.set(message="Done!", value=100)
            ui.notification_show("Incidence aggregated successfully!", type="message")
            ui.update_accordion("forecasting_accordion", show="Incidence Estimation")
            ui.update_navset("forecasting_tabs", selected="tab_forecasting_agg")

    @render.ui
    def longevity_estimator_params_ui():
        est_key = input.longevity_estimator()

        if "longevity_model_upload" in input and input.longevity_model_upload():
            return ui.div()

        estimator_class_name = ESTIMATOR_MAP.get(est_key)
        if not estimator_class_name:
            return ui.div()

        EstimatorClass = getattr(estimators, estimator_class_name, None)
        if not EstimatorClass:
            return ui.div()

        est = fitted_longevity_estimator.get()
        if est and type(est).__name__ == estimator_class_name and getattr(est, "is_fitted_", False):
            return ui.div(
                ui.hr(),
                ui.p(
                    "A fitted model of this type is currently in memory. "
                    "Forecasting will use this model's learned weights. "
                    "To train a new model, click 'Clear Fitted Model' below.",
                    class_="text-info small mb-1",
                ),
            )

        return EstimatorIntrospector(EstimatorClass).build_ui(
            prefix="longevity_param_", exclude=["self"], default_overrides={"use_relative_incidence": False}
        )

    @render.ui
    def longevity_model_io_ui():
        est_key = input.longevity_estimator()
        estimator_class_name = ESTIMATOR_MAP.get(est_key)
        EstimatorClass = getattr(estimators, estimator_class_name, None)

        if not EstimatorClass or not hasattr(EstimatorClass, "load_model"):
            return ui.div()

        elements = [
            ui.hr(),
            ui.p("Model Weights (Optional)", class_="text-muted small mb-1"),
            ui.input_file("longevity_model_upload", "Load Fitted Model (.pkl)", accept=[".pkl"]),
        ]

        est = fitted_longevity_estimator.get()
        if est and type(est).__name__ == EstimatorClass.__name__ and getattr(est, "is_fitted_", False):
            elements.append(
                ui.download_button(
                    "longevity_model_download", "Download Fitted Model", class_="btn-outline-primary w-100 mb-2"
                )
            )
            elements.append(
                ui.input_action_button(
                    "btn_clear_longevity_model", "Clear Fitted Model", class_="btn-outline-danger w-100 mb-3"
                )
            )

        return ui.div(*elements)

    @reactive.Effect
    @reactive.event(input.btn_clear_longevity_model)
    def clear_longevity_model():
        fitted_longevity_estimator.set(None)
        ui.notification_show("Fitted longevity model cleared from memory.", type="message")

    @render.download_button(filename=lambda: f"fitted_{input.longevity_estimator()}_incidence_model.pkl")
    def longevity_model_download():
        est = fitted_longevity_estimator.get()
        if est and getattr(est, "is_fitted_", False):
            return generate_temp_download(est.save_model, ".pkl", "Model Export Error")

    @reactive.Effect
    @reactive.event(input.btn_run_longevity)
    async def run_longevity():
        inc_df = shared_agg_inc_df.get()
        if inc_df is None:
            ui.notification_show("Please aggregate incidence data first.", type="warning")
            return

        async with ui_task("Longevity Forecasting Error") as p:
            p.set(message="Instantiating estimator...", value=40)
            await sleep(0)

            est_key = input.longevity_estimator()
            estimator_class_name = ESTIMATOR_MAP.get(est_key)
            EstimatorClass = getattr(estimators, estimator_class_name)

            in_memory_est = fitted_longevity_estimator.get()
            can_reuse_memory = (
                in_memory_est is not None
                and type(in_memory_est).__name__ == EstimatorClass.__name__
                and getattr(in_memory_est, "is_fitted_", False)
            )

            file_info = input.longevity_model_upload() if "longevity_model_upload" in input else None

            if hasattr(EstimatorClass, "load_model") and file_info:
                p.set(message="Loading model...", value=50)
                await sleep(0)
                try:
                    est = EstimatorClass.load_model(Path(file_info[0]["datapath"]))
                except Exception as e:
                    ui.notification_show(f"Failed to load model: {str(e)}", type="error", duration=10)
                    return

                p.set(message="Forecasting longevity...", value=70)
                await sleep(0)
                forecast_res = await to_thread(est.predict, inc_df)

            elif can_reuse_memory:
                p.set(message="Using in-memory fitted model...", value=50)
                await sleep(0)
                est = in_memory_est
                forecast_res = await to_thread(est.predict, inc_df)

            else:
                kwargs = EstimatorIntrospector(EstimatorClass).extract_kwargs(
                    input, prefix="longevity_param_", exclude=["self"]
                )

                est = EstimatorClass(**kwargs)
                p.set(message="Fitting model and forecasting...", value=60)
                await sleep(0)
                forecast_res = await to_thread(est.calculate, inc_df)

            fitted_longevity_estimator.set(est)
            shared_forecast.set(forecast_res)

            p.set(message="Done!", value=100)
            ui.notification_show("Longevity forecast complete!", type="message")
            ui.update_navset("forecasting_tabs", selected="tab_forecasting_plots")
            await sleep(0)

    @reactive.Effect
    @reactive.event(input.btn_run_transmission)
    async def run_transmission():
        inc_df = shared_agg_inc_df.get()
        if inc_df is None:
            ui.notification_show("Please aggregate incidence data first.", type="warning")
            return

        async with ui_task("Transmission Error") as p:
            p.set(message="Instantiating estimator...", value=40)
            await sleep(0)

            gt_mean = input.transmission_gt_mean()
            gt_std = input.transmission_gt_std()

            est = ReproductionNumberEstimator(
                time_column="date_bin",
                case_column="variant_count",
                generation_time_mean=gt_mean,
                generation_time_std=gt_std,
            )

            p.set(message="Running Bayesian Inference...", value=70)
            
            def fit_and_predict():
                return est.fit(inc_df).predict(inc_df)

            trans_est = await to_thread(fit_and_predict)
            shared_trans_results.set(trans_est)

            app_df = shared_df.get()
            app_df.epi.update_history(
                est_name=trans_est.method,
                params={"generation_time_mean": gt_mean, "generation_time_std": gt_std},
                metrics={"num_samples": len(trans_est.data)}
            )
            shared_df.set(app_df)
            
            p.set(message="Done!", value=100)
            ui.notification_show("Transmission dynamics calculated!", type="message")
            ui.update_navset("forecasting_tabs", selected="tab_forecasting_trans")

    # --- Tab Renderers ---
    @render.ui
    def forecasting_agg_content():
        if (agg_df := shared_agg_inc_df.get()) is None:
            return ui.div("Please aggregate incidence data in the sidebar.", class_="text-center mt-5 text-muted fs-4")

        meta_dict = (
            getattr(agg_df, "metadata", getattr(agg_df, "attrs", {})).get("metric_meta", {})
            if isinstance(getattr(agg_df, "metadata", getattr(agg_df, "attrs", {})), dict)
            else {}
        )
        meta_ui = format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Incidence Aggregates Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("forecasting_agg_data", "Aggregated Incidence"),
        )

    dt_download_server(
        "forecasting_agg_data", data_callable=lambda: shared_agg_inc_df.get(), filename="aggregated_incidence.csv"
    )

    @render.ui
    def forecasting_est_content():
        if (res := shared_forecast.get()) is None:
            return ui.div(
                "Calculate incidence estimates to view the results data.", class_="text-center mt-5 text-muted fs-4"
            )

        meta_dict = {f.name: getattr(res, f.name) for f in fields(res) if f.name not in ["data", "model_results"]}
        meta_ui = format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Incidence Estimates Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("forecasting_est_data", "Incidence Estimates"),
            ui.card(
                ui.card_header("Model Summary (IRR)"),
                dt_download_ui("forecasting_est_model_results", "Model Summary"),
                class_="mt-3",
            )
            if res.model_results is not None and not res.model_results.is_empty()
            else ui.div(),
        )

    dt_download_server(
        "forecasting_est_data",
        data_callable=lambda: shared_forecast.get().data if shared_forecast.get() else None,
        filename="incidence_estimates.csv",
        height="400px",
    )
    dt_download_server(
        "forecasting_est_model_results",
        data_callable=lambda: shared_forecast.get().model_results if shared_forecast.get() else None,
        filename="incidence_model_summary.csv",
        height="200px",
    )

    @render.ui
    def forecasting_trans_content():
        res = shared_trans_results.get()
        if res is None:
            return ui.p("Calculate transmission dynamics in the sidebar first.", class_="text-muted mt-3")

        cols = res.data.columns
        return ui.div(
            ui.h3("Reproduction Numbers (Re)"),
            format_metadata_ui(res),
            ui.div(
                ui.output_data_frame("forecasting_trans_dt"),
                class_="mt-3",
            ),
        )

    @render.data_frame
    def forecasting_trans_dt():
        res = shared_trans_results.get()
        if res is not None:
            return render.DataGrid(res.data.to_pandas(), width="100%", height="500px")
        return None

    @render.ui
    def forecasting_diag_content():
        if (est := fitted_longevity_estimator.get()) is None:
            return ui.div("Estimate incidence to view model diagnostics.", class_="text-center mt-5 text-muted fs-4")

        if not hasattr(est, "diagnostics"):
            return ui.div(
                ui.p(f"Diagnostics are not applicable for {type(est).__name__}."),
                class_="text-center mt-5 text-muted fs-4",
            )

        try:
            _ = est.diagnostics()
            return dt_download_ui("forecasting_model_diagnostics", "NumPyro Posterior Diagnostics")
        except Exception as e:
            return ui.div(f"Diagnostics Error: {str(e)}", class_="text-danger text-center mt-5 fs-5")

    def get_longevity_diagnostics():
        if (est := fitted_longevity_estimator.get()) is not None and hasattr(est, "diagnostics"):
            try:
                return est.diagnostics()
            except Exception:
                pass
        return None

    dt_download_server(
        "forecasting_model_diagnostics",
        data_callable=get_longevity_diagnostics,
        filename="longevity_diagnostics.csv",
        height="500px",
    )

    @render.ui
    def forecasting_plot_content():
        if input.forecasting_plot_type() == PlotType.LONGEVITY.value and current_formulation.get() is None:
            return ui.div(
                "Please design a vaccine formulation in Tab 3 to view longevity.",
                class_="text-center mt-5 text-muted fs-4",
            )
        if shared_forecast.get() is None:
            return ui.div("Run forecasting to view plots.", class_="text-center mt-5 text-muted fs-4")
        return output_widget("forecasting_plot")

    @render_widget
    def forecasting_plot():
        vac = current_formulation.get()
        forecast = shared_forecast.get()
        p_type = input.forecasting_plot_type()
        empty_theme = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8"))

        try:
            if p_type == PlotType.LONGEVITY.value:
                if forecast is None or vac is None:
                    return Figure().update_layout(title="Run forecasting and design formulation first.", **empty_theme)
                return render_plot(vac, PlotType.LONGEVITY, forecast=forecast)

            elif p_type == PlotType.EPICURVE.value:
                if forecast is None:
                    return Figure().update_layout(title="Run forecasting first.", **empty_theme)
                return render_plot(forecast, PlotType.EPICURVE)
        except Exception as e:
            ui.notification_show(f"Plotting Error: {str(e)}", type="error", duration=10)
            return Figure().update_layout(title=f"Error: {str(e)}", **empty_theme)

    @render.download_button(filename=lambda: f"forecasting_plot.{input.forecasting_plot_format()}")
    def btn_download_forecasting_plot():
        vac = current_formulation.get()
        forecast = shared_forecast.get()
        p_type = input.forecasting_plot_type()

        try:
            if p_type == PlotType.LONGEVITY.value:
                if forecast is None or vac is None:
                    return None
                fig = render_plot(vac, PlotType.LONGEVITY, forecast=forecast)
            elif p_type == PlotType.EPICURVE.value:
                if forecast is None:
                    return None
                fig = render_plot(forecast, PlotType.EPICURVE)
            else:
                return None

            def save_fig(p: Path):
                fig.write_image(
                    p,
                    format=input.forecasting_plot_format(),
                    width=input.forecasting_plot_width(),
                    height=input.forecasting_plot_height(),
                )

            return generate_temp_download(save_fig, f".{input.forecasting_plot_format()}", "Plot Export Error")

        except Exception as e:
            ui.notification_show(f"Export Error: {str(e)}", type="error")
