from asyncio import sleep, to_thread
from shiny import module, reactive, render, ui

from seroepi.estimators.serology import SerocatalyticEstimator, TiterClassificationEstimator
from .icons import icon_lucide
from .utils import ui_task, format_metadata_ui


@module.ui
def serology_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Data Enrichment",
                    ui.p("Fetch dynamic Life Expectancy metadata from the World Bank API.", class_="small text-muted mb-2"),
                    ui.input_selectize("serology_country_col", "Country Column", choices=[]),
                    ui.input_action_button(
                        "btn_fetch_wb",
                        "Fetch Life Expectancy",
                        icon=icon_lucide("globe"),
                        class_="btn-outline-primary w-100 mt-2",
                    ),
                    icon=icon_lucide("globe"),
                ),
                ui.accordion_panel(
                    "Titer Classification",
                    ui.p("Classify continuous assay data (e.g. ELISA) into positive/negative using a Gaussian Mixture Model.", class_="small text-muted mb-2"),
                    ui.input_selectize("serology_titer_col", "Continuous Titer Column", choices=[]),
                    ui.input_action_button(
                        "btn_run_gmm",
                        "Fit GMM & Classify",
                        icon=icon_lucide("flask-conical"),
                        class_="btn-outline-primary w-100 mt-2",
                    ),
                    icon=icon_lucide("flask-conical"),
                ),
                ui.accordion_panel(
                    "Force of Infection",
                    ui.p("Fit an age-structured catalytic model to seropositivity data to estimate Force of Infection (\u03bb).", class_="small text-muted mb-2"),
                    ui.input_selectize("serology_age_col", "Age Column", choices=[]),
                    ui.input_selectize("serology_status_col", "Serostatus Column", choices=[]),
                    ui.input_select(
                        "serology_model_type",
                        "Catalytic Model Type",
                        choices={"irreversible": "Irreversible (SI)", "reversible": "Reversible (SIS)"}
                    ),
                    ui.input_action_button(
                        "btn_run_foi",
                        "Calculate FOI, R0 & HIT",
                        icon=icon_lucide("activity"),
                        class_="btn-primary w-100 mt-3",
                    ),
                    icon=icon_lucide("activity"),
                ),
                id="serology_accordion",
                open=["Data Enrichment", "Titer Classification", "Force of Infection"],
                multiple=True,
            ),
            width=350,
        ),
        ui.navset_card_tab(
            ui.nav_panel(
                "FOI & Transmission",
                ui.output_ui("serology_foi_content"),
                value="tab_serology_foi",
                icon=icon_lucide("activity"),
            ),
            ui.nav_panel(
                "Titer Classification",
                ui.output_ui("serology_gmm_content"),
                value="tab_serology_gmm",
                icon=icon_lucide("flask-conical"),
            ),
            id="serology_tabs"
        )
    )

@module.server
def serology_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    
    shared_foi_results = reactive.value(None)
    shared_gmm_results = reactive.value(None)
    
    @reactive.Effect
    def update_options():
        df = shared_df.get()
        if df is not None:
            cols = df.columns
            # Update columns
            ui.update_selectize("serology_country_col", choices=cols)
            ui.update_selectize("serology_titer_col", choices=cols)
            ui.update_selectize("serology_age_col", choices=cols)
            
            # If we already have is_seropositive from GMM, select it
            if "is_seropositive" in cols:
                ui.update_selectize("serology_status_col", choices=cols, selected="is_seropositive")
            else:
                ui.update_selectize("serology_status_col", choices=cols)

    @reactive.Effect
    @reactive.event(input.btn_fetch_wb)
    async def fetch_worldbank():
        df = shared_df.get()
        if df is None:
            ui.notification_show("Please load a dataset first.", type="warning")
            return
            
        country_col = input.serology_country_col()
        if not country_col:
            ui.notification_show("Please select a country column.", type="warning")
            return
            
        async with ui_task("WorldBank API Error") as p:
            p.set(message="Fetching Life Expectancy...", value=40)
            await sleep(0)
            
            def do_fetch():
                return df.geo.with_metadata("SP.DYN.LE00.IN", country_col=country_col)
                
            new_df = await to_thread(do_fetch)
            shared_df.set(new_df)
            
            p.set(message="Done!", value=100)
            ui.notification_show("Life Expectancy added to dataset!", type="message")


    @reactive.Effect
    @reactive.event(input.btn_run_gmm)
    async def run_gmm():
        df = shared_df.get()
        if df is None:
            return
            
        titer_col = input.serology_titer_col()
        
        async with ui_task("GMM Error") as p:
            p.set(message="Fitting Gaussian Mixture Model...", value=40)
            await sleep(0)
            
            def fit_gmm():
                est = TiterClassificationEstimator(titer_column=titer_col, n_components=2)
                est.fit(df)
                return est, est.predict(df)
                
            est, res_df = await to_thread(fit_gmm)
            # res_df is just the pl.DataFrame with `is_seropositive` appended
            # we need to set shared_df to this updated dataframe
            
            # Let's clone df and update its data
            new_dataset = df.clone()
            new_dataset._df = res_df
            
            shared_df.set(new_dataset)
            
            # Also store some metrics to show
            shared_gmm_results.set({
                "cutoff": est.cutoff_,
                "neg_mean": est.neg_mean_,
                "pos_mean": est.pos_mean_
            })
            
            p.set(message="Done!", value=100)
            ui.notification_show("Titer classification complete!", type="message")
            ui.update_navset("serology_tabs", selected="tab_serology_gmm")


    @reactive.Effect
    @reactive.event(input.btn_run_foi)
    async def run_foi():
        df = shared_df.get()
        if df is None:
            return
            
        age_col = input.serology_age_col()
        status_col = input.serology_status_col()
        model_type = input.serology_model_type()
        
        # Check for life_expectancy
        le_col = "SP.DYN.LE00.IN"
        life_exp = 70.0 # fallback
        if le_col in df.data.columns:
            le_series = df.data[le_col].drop_nulls()
            if len(le_series) > 0:
                life_exp = le_series.mean()
        
        async with ui_task("Serocatalytic Model Error") as p:
            p.set(message="Fitting Age-Structured Model...", value=50)
            await sleep(0)
            
            def fit_foi():
                est = SerocatalyticEstimator(age_column=age_col, status_column=status_col, model_type=model_type)
                # The FOI predict returns a ForceOfInfectionEstimates object?
                # Wait, SerocatalyticEstimator.predict() returns a DataFrame with lambda_foi and rho_recovery
                # Actually, wait. Let's look at `_generate_predictions` in SerocatalyticEstimator.
                est.fit(df)
                return est.predict(df)
                
            foi_res = await to_thread(fit_foi)
            # FOI estimates result
            
            # calculate R0 and HIT
            r0 = foi_res.calculate_r0(life_expectancy=life_exp)
            hit = foi_res.calculate_hit(life_expectancy=life_exp)
            
            shared_foi_results.set({
                "foi": foi_res,
                "r0": r0,
                "hit": hit,
                "life_expectancy": life_exp
            })
            
            p.set(message="Done!", value=100)
            ui.notification_show("Force of Infection and R0 calculated!", type="message")
            ui.update_navset("serology_tabs", selected="tab_serology_foi")
            
    
    @render.ui
    def serology_gmm_content():
        res = shared_gmm_results.get()
        if not res:
            return ui.p("Fit a GMM first.", class_="text-muted mt-3")
            
        return ui.div(
            ui.h3("GMM Titer Classification Results"),
            ui.p(f"Negative Component Mean: {res['neg_mean']:.3f}"),
            ui.p(f"Positive Component Mean: {res['pos_mean']:.3f}"),
            ui.p(f"Calculated Seropositivity Cutoff: {res['cutoff']:.3f}", class_="fw-bold text-primary"),
        )
        
    @render.ui
    def serology_foi_content():
        res = shared_foi_results.get()
        if not res:
            return ui.p("Calculate FOI first.", class_="text-muted mt-3")
            
        foi_est = res["foi"]
        r0 = res["r0"]
        hit = res["hit"]
        le = res["life_expectancy"]
        
        # foi_est is ForceOfInfectionEstimates. The data dataframe contains lambda_foi and rho_recovery
        lam = foi_est.data["lambda_foi"][0]
        rho = foi_est.data["rho_recovery"][0] if "rho_recovery" in foi_est.data.columns else None
        
        rho_ui = ui.p(f"Recovery Rate (\u03c1): {rho:.4f}") if rho is not None else ui.div()
        
        return ui.div(
            ui.h3("Serocatalytic Model Results"),
            ui.p(f"Force of Infection (\u03bb): {lam:.4f}", class_="fw-bold"),
            rho_ui,
            ui.hr(),
            ui.h3("Transmission Metrics"),
            ui.p(f"Used Life Expectancy: {le:.1f} years", class_="small text-muted"),
            ui.p(f"Basic Reproduction Number (R0): {r0:.2f}", class_="fw-bold text-danger"),
            ui.p(f"Herd Immunity Threshold (HIT): {hit:.1%}", class_="fw-bold text-success"),
        )
