from asyncio import sleep, to_thread
from dataclasses import fields
from pathlib import Path

from shiny import module, reactive, render, ui

from seroepi import estimators
from seroepi.constants import AggregationType, EstimatorType, PlotType
from seroepi.domains import render_plot

from .icons import icon_lucide
from .utils import (
    EstimatorIntrospector,
    _clean_ui_label,
    build_grouped_choices,
    dt_download_server,
    dt_download_ui,
    export_settings_ui,
    format_metadata_ui,
    generate_temp_download,
    safe_plot_server,
    safe_plot_ui,
    ui_task,
    update_registry,
)


@module.ui
def burden_ui():
    """UI layout for the global epidemiology dashboard."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Prevalence Aggregation",
                    ui.tooltip(
                        ui.input_selectize("prev_trait", "Trait Column (Required)", choices=[]),
                        "The specific genetic trait to measure. This field is required to aggregate your data.",
                    ),
                    ui.tooltip(
                        ui.input_radio_buttons(
                            "prev_agg_type",
                            "Aggregation Mode",
                            choices=AggregationType.ui_labels(),
                            selected=AggregationType.COMPOSITIONAL.value,
                        ),
                        "Compositional calculates the proportion of variants within a group. "
                        "Trait calculates the simple presence/absence of a specific marker.",
                    ),
                    ui.tooltip(
                        ui.input_selectize(
                            "prev_stratify",
                            "Stratify By",
                            choices=[],
                            multiple=True,
                            options={"placeholder": "Optional, none selected..."},
                        ),
                        "Variables to group the data by (e.g., Spatial, Date) before calculating burden.",
                    ),
                    ui.tooltip(
                        ui.input_selectize(
                            "prev_cluster",
                            "Adjustment Column (Optional)",
                            choices=[],
                            options={"placeholder": "Optional, none selected..."},
                        ),
                        "A cluster variable (e.g., Transmission Cluster, Hospital) to adjust the "
                        "burden estimates for sampling bias/outbreaks.",
                    ),
                    ui.tooltip(
                        ui.input_text("prev_negative", "Negative Indicator", value="-"),
                        "The string or character in your data that indicates a trait is absent (commonly '-' or '0').",
                    ),
                    ui.tooltip(
                        ui.input_checkbox("prev_pad_zeros", "Pad Zeroes (Zero-fill missing)", value=False),
                        "Pads missing combinations of strata with zero counts. Essential for Spatial and Hierarchical Bayesian models to map empty regions correctly.",
                    ),
                    ui.input_action_button(
                        "btn_aggregate_prev",
                        "Aggregate Data",
                        icon=icon_lucide("calculator"),
                        class_="btn-primary w-100 mt-3",
                    ),
                    icon=icon_lucide("calculator"),
                ),
                ui.accordion_panel(
                    "Prevalence Estimation",
                    ui.tooltip(
                        ui.input_select("prev_estimator", "Estimator", choices=EstimatorType.ui_labels()),
                        "The statistical model used to calculate burden and confidence intervals.",
                    ),
                    ui.output_ui("estimator_params_ui"),
                    ui.output_ui("model_io_ui"),
                    ui.input_action_button(
                        "btn_estimate_prev", "Estimate Burden", icon=icon_lucide("rocket"), class_="btn-primary w-100"
                    ),
                    icon=icon_lucide("trending-up"),
                ),
                id="burden_accordion",
                open=["Prevalence Aggregation"],
                multiple=True,
            ),
            width=350,
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "Aggregates",
                ui.output_ui("agg_data_content"),
                value="tab_aggregated_data",
                icon=icon_lucide("calculator"),
            ),
            ui.nav_panel(
                "Estimates",
                ui.output_ui("prev_summary_content"),
                value="tab_prevalence_data",
                icon=icon_lucide("trending-up"),
            ),
            ui.nav_panel(
                "Diagnostics",
                ui.output_ui("model_diagnostics_content"),
                value="tab_model_diagnostics",
                icon=icon_lucide("activity"),
            ),
            ui.nav_panel(
                "Plots",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.tooltip(
                            ui.input_select(
                                "prev_plot_type",
                                "Plot Type",
                                choices={
                                    PlotType.FOREST.value: "Forest Plot",
                                    PlotType.COMPOSITION_BAR.value: "Composition Bar",
                                    PlotType.COMPOSITION_HEATMAP.value: "Composition Heatmap",
                                    PlotType.LONGITUDINAL_PREVALENCE: "Longitudinal Prevalence",
                                },
                            ),
                            "The visualization style for the estimated burden data.",
                        ),
                        ui.hr(),
                        export_settings_ui("prev"),
                        ui.download_button("btn_download_plot", "Download Plot", class_="btn-outline-primary w-100"),
                        width=280,
                    ),
                    ui.output_ui("prev_plot_content"),
                ),
                value="tab_prevalence_plot",
                icon=icon_lucide("chart-bar"),
            ),
            id="main_dashboard_tabs",
        ),
    )


@module.server
def burden_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    shared_agg_df = app_state["shared_agg_df"]
    prev_results = app_state["prev_results"]
    fitted_estimator = app_state["fitted_estimator"]
    results_registry = app_state["results_registry"]

    @reactive.Effect
    def manage_accordion_state():
        if shared_df.get() is None:
            ui.update_accordion("burden_accordion", show="Prevalence Aggregation")

    # --- STAGE 3: burden ANALYSIS WORKFLOW ---
    @reactive.Effect
    def update_prev_dropdowns():
        try:
            if (df := shared_df.get()) is not None:
                trait_choices = build_grouped_choices(df.epi.genotypes, "Other Traits", include_empty=True)

                stratify_choices = {}
                if df.epi.has_spatial:
                    stratify_choices["Spatial Coordinates"] = {"latitude": "Latitude", "longitude": "Longitude"}
                stratify_choices.update(build_grouped_choices(df.epi.stratify_cols, "Other Variables"))

                all_cluster_cols = list(dict.fromkeys(df.epi.cluster_cols + df.epi.genotypes))
                cluster_choices = build_grouped_choices(all_cluster_cols, "Clusters", include_empty=True)

                ui.update_selectize("prev_trait", choices=trait_choices, selected="")
                ui.update_selectize("prev_stratify", choices=stratify_choices)
                ui.update_selectize("prev_cluster", choices=cluster_choices, selected="")
        except Exception as e:
            print(f"Error updating prev dropdowns: {e}")

    @render.ui
    def estimator_params_ui():
        est_type = input.prev_estimator()

        # Hide if a model has been uploaded since we won't be compiling it
        if "model_upload" in input and input.model_upload():
            return ui.div()

        try:
            estimator_class_name = EstimatorType(est_type).class_name
            EstimatorClass = getattr(estimators, estimator_class_name, None)
        except Exception:
            return ui.div()

        if not EstimatorClass:
            return ui.div()

        # Check for in-memory model
        est = fitted_estimator.get()
        if est and type(est).__name__ == estimator_class_name and getattr(est, "is_fitted_", False):
            return ui.div(
                ui.hr(),
                ui.p(
                    "A fitted model of this type is currently in memory. "
                    "Prevalence will be predicted using this model's learned weights. "
                    "To train a new model, click 'Clear Fitted Model' below.",
                    class_="text-info small mb-1",
                ),
            )

        return EstimatorIntrospector(EstimatorClass).build_ui(
            prefix="est_param_", exclude=["self", "target_event", "target_n", "lat_col", "lon_col"]
        )

    @render.ui
    def model_io_ui():
        est_type = input.prev_estimator()
        try:
            estimator_class_name = EstimatorType(est_type).class_name
            EstimatorClass = getattr(estimators, estimator_class_name, None)
        except Exception:
            return ui.div()

        # Safely hide the UI if the estimator (like Frequentist) doesn't support model loading
        if not EstimatorClass or not hasattr(EstimatorClass, "load_model"):
            return ui.div()

        elements = [
            ui.hr(),
            ui.p("Model Weights (Optional)", class_="text-muted small mb-1"),
            ui.input_file("model_upload", "Load Fitted Model (.pkl)", accept=[".pkl"]),
        ]

        # If a model of the CURRENT type is fitted in memory, provide the download button
        est = fitted_estimator.get()
        if est and type(est).__name__ == EstimatorClass.__name__ and getattr(est, "is_fitted_", False):
            elements.append(
                ui.download_button("model_download", "Download Fitted Model", class_="btn-outline-primary w-100 mb-2")
            )
            elements.append(
                ui.input_action_button("btn_clear_model", "Clear Fitted Model", class_="btn-outline-danger w-100 mb-3")
            )

        return ui.div(*elements)

    @reactive.Effect
    @reactive.event(input.btn_clear_model)
    def clear_model():
        fitted_estimator.set(None)
        ui.notification_show("Fitted model cleared from memory.", type="message")

    @render.download_button(filename=lambda: f"fitted_{EstimatorType(input.prev_estimator()).name.lower()}_model.pkl")
    def model_download():
        est = fitted_estimator.get()
        if est and getattr(est, "is_fitted_", False):
            return generate_temp_download(est.save_model, ".pkl", "Model Export Error")

    @reactive.Effect
    @reactive.event(input.btn_aggregate_prev)
    async def aggregate_prevalence():
        if (df := shared_df.get()) is None:
            ui.notification_show("Please upload and process data first.", type="warning")
            return

        trait = input.prev_trait() or None
        cluster = input.prev_cluster() or None
        agg_mode = input.prev_agg_type()

        if not trait:
            ui.notification_show("Please select a Trait Column before aggregating.", type="warning")
            return

        stratify = list(input.prev_stratify())

        if agg_mode == AggregationType.TRAIT and not stratify:
            ui.notification_show("Trait burden requires at least one stratification column.", type="error")
            return

        # INTENT ROUTING:
        # If the user wants a compositional breakdown, the trait column acts as the primary grouping variable!
        if agg_mode == AggregationType.COMPOSITIONAL and trait:
            if trait not in stratify:
                stratify.append(trait)
            trait_col = None  # Leave None so the accessor groups compositionally by the last strata
        else:
            trait_col = trait

        async with ui_task("Aggregation Error") as p:
            p.set(message="Aggregating data...", value=20)
            await sleep(0)

            def run_aggregation():
                return df.epi.aggregate_prevalence(
                    stratify_by=stratify,
                    trait_col=trait_col,
                    cluster_col=cluster,
                    negative_indicator=input.prev_negative(),
                    pad_zeros=input.prev_pad_zeros(),
                )

            agg_df = await to_thread(run_aggregation)

            shared_agg_df.set(agg_df)
            p.set(message="Done!", value=100)
            ui.notification_show("Data aggregated successfully!", type="message")
            ui.update_accordion("burden_accordion", show="Prevalence Estimation")
            ui.update_navset("main_dashboard_tabs", selected="tab_aggregated_data")

    @reactive.Effect
    @reactive.event(input.btn_estimate_prev)
    async def estimate_burden():
        if (agg_df := shared_agg_df.get()) is None:
            ui.notification_show("Please aggregate data first.", type="warning")
            return

        est_type = input.prev_estimator()

        async with ui_task("Calculation Error") as p:
            p.set(message="Instantiating estimator...", value=20)
            await sleep(0)

            estimator_class_name = EstimatorType(est_type).class_name

            if not hasattr(estimators, estimator_class_name):
                ui.notification_show(
                    f"{estimator_class_name} is not available. Did you install seroepi[models]?", type="error"
                )
                return

            EstimatorClass = getattr(estimators, estimator_class_name)

            # Check if we have a model already compiled and fitted in memory
            in_memory_est = fitted_estimator.get()
            can_reuse_memory = (
                in_memory_est is not None
                and type(in_memory_est).__name__ == EstimatorClass.__name__
                and getattr(in_memory_est, "is_fitted_", False)
            )

            # Intercept the model upload input if it exists
            file_info = input.model_upload() if "model_upload" in input else None

            if hasattr(EstimatorClass, "load_model") and file_info:
                p.set(message="Loading model...", value=40)
                await sleep(0)
                try:
                    est = EstimatorClass.load_model(Path(file_info[0]["datapath"]))
                except Exception as e:
                    ui.notification_show(f"Failed to load model: {str(e)}", type="error", duration=10)
                    return

                p.set(message="Predicting prevalence...", value=60)
                await sleep(0)
                res = await to_thread(est.predict, agg_df)
            elif can_reuse_memory:
                p.set(message="Using in-memory fitted model...", value=60)
                await sleep(0)
                est = in_memory_est
                res = await to_thread(est.predict, agg_df)
            else:
                # Dynamically extract and assign kwargs for the selected estimator class
                kwargs = EstimatorIntrospector(EstimatorClass).extract_kwargs(
                    input, prefix="est_param_", exclude=["self", "target_event", "target_n", "lat_col", "lon_col"]
                )

                # Ensure SpatialburdenEstimator maps correctly to Pandas generated columns
                if estimator_class_name == "SpatialburdenEstimator":
                    kwargs["lat_col"] = "latitude"
                    kwargs["lon_col"] = "longitude"

                est = EstimatorClass(**kwargs)

                p.set(message="Fitting model and calculating...", value=60)
                await sleep(0)
                res = await to_thread(est.calculate, agg_df)

            fitted_estimator.set(est)
            prev_results.set(res)

            # Cache the run dynamically so it can be picked up by the Formulation tab!
            trait_clean = _clean_ui_label(res.trait)
            strata_str = ", ".join([_clean_ui_label(s) for s in res.stratified_by]) if res.stratified_by else "Global"
            current_df = shared_df.get()
            ds_name = app_state.get("active_dataset_name").get() if app_state.get("active_dataset_name") else "Unknown Dataset"
            if not ds_name:
                ds_name = "Unknown Dataset"
            run_name = f"[{ds_name}] {trait_clean} by {strata_str} ({EstimatorType(est_type).name.title()})"
            update_registry(results_registry, run_name, {"res": res, "est": est, "agg_df": agg_df})
            app_state["active_run_name"].set(run_name)

            ui.notification_show("Prevalence calculated successfully!", type="message")
            p.set(message="Done", value=100)
            ui.update_navset("main_dashboard_tabs", selected="tab_prevalence_data")
            await sleep(0)

    # --- STAGE 4: RENDERING THE DASHBOARDS ---
    @render.ui
    def agg_data_content():
        if (agg_df := shared_agg_df.get()) is None:
            return ui.div(
                "Please aggregate data in the sidebar to view the aggregated dataset.",
                class_="text-center mt-5 text-muted fs-4",
            )

        meta_dict = (
            getattr(agg_df, "metadata", getattr(agg_df, "attrs", {})).get("metric_meta", {})
            if isinstance(getattr(agg_df, "metadata", getattr(agg_df, "attrs", {})), dict)
            else {}
        )
        meta_ui = format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Aggregates Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("agg_data", "Aggregates Data"),
        )

    dt_download_server("agg_data", data_callable=lambda: shared_agg_df.get(), filename="aggregated_data.csv")

    @render.ui
    def prev_summary_content():
        if (res := prev_results.get()) is None:
            return ui.div("Calculate burden to view the results data.", class_="text-center mt-5 text-muted fs-4")

        # Dynamically extract instance attributes into a dictionary
        meta_dict = {f.name: getattr(res, f.name) for f in fields(res) if f.name not in ["data", "model_results"]}
        meta_ui = format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Estimates Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("prev_summary", "Estimates Data"),
        )

    dt_download_server(
        "prev_summary",
        data_callable=lambda: prev_results.get().data if prev_results.get() else None,
        filename="estimates_data.csv",
        height="400px",
    )

    @render.ui
    def model_diagnostics_content():
        if (est := fitted_estimator.get()) is None:
            return ui.div("Estimate burden to view model diagnostics.", class_="text-center mt-5 text-muted fs-4")

        if not hasattr(est, "diagnostics"):
            return ui.div(
                ui.p(f"Diagnostics are not applicable for {type(est).__name__}."),
                class_="text-center mt-5 text-muted fs-4",
            )

        try:
            # Execute a dummy call to gracefully catch TypeErrors (like attempting diagnostics on SVI)
            _ = est.diagnostics()
            return dt_download_ui("model_diagnostics", "NumPyro Posterior Diagnostics")
        except Exception as e:
            return ui.div(f"Diagnostics Error: {str(e)}", class_="text-danger text-center mt-5 fs-5")

    def get_model_diagnostics():
        if (est := fitted_estimator.get()) is not None and hasattr(est, "diagnostics"):
            try:
                return est.diagnostics()
            except Exception:
                pass
        return None

    dt_download_server(
        "model_diagnostics", data_callable=get_model_diagnostics, filename="model_diagnostics.csv", height="500px"
    )

    @render.ui
    def prev_plot_content():
        if prev_results.get() is None:
            return ui.div("Calculate burden to view the plot.", class_="text-center mt-5 text-muted fs-4")
        return safe_plot_ui("prev_plot")

    safe_plot_server("prev_plot", data_reactive=prev_results, plot_type=input.prev_plot_type)

    @render.download_button(filename=lambda: f"burden_plot.{input.prev_plot_format()}")
    def btn_download_plot():
        res = prev_results.get()
        if res is None:
            ui.notification_show("No plot data available to export.", type="warning")
            return

        # Generate the raw Plotly Figure (go.Figure) from the router
        fig = render_plot(res, input.prev_plot_type())

        # Kaleido naturally intercepts write_image locally
        def save_fig(p: Path):
            fig.write_image(
                p, format=input.prev_plot_format(), width=input.prev_plot_width(), height=input.prev_plot_height()
            )

        return generate_temp_download(save_fig, f".{input.prev_plot_format()}", "Plot Export Error")
