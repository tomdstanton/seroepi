# SeroEpi App
*From genomes to a robust vaccine strategy in 5 minutes*

The SeroEpi App is a world-class, fully featured Shiny web dashboard. You can upload data, train models, and
download your results without writing a single line of code.

---

## Installation

You can install the `seroepi` app directly from PyPI. We highly recommend using uv for lightning-fast installations,
but standard `pip` works perfectly too.

    # Using uv (Recommended)
    uv pip install seroepi[app]
    
    # Using standard pip
    pip install seroepi[app]

## Running the Interactive Dashboard

To launch the app locally from your command line:

    # Using uv (Recommended)
    uv run shiny run app/app.py
    
    # Or simply
    shiny run app/app.py

---

## Tutorial: Cross-Population Vaccine Design & Forecasting

**The Scenario:** You are a computational epidemiologist tasked with designing a novel *Klebsiella pneumoniae*
capsule (K-locus) vaccine. Your primary target population is neonates suffering from sepsis in Low- and Middle-Income
Countries (LMICs). However, to secure funding, you must also demonstrate whether this LMIC-optimized vaccine will
provide long-term, stable coverage for adult bloodstream infections (BSIs) in High-Income Countries (HICs).

### Phase 1: Training the Model (LMIC Neonatal Data)

#### Step 1: Load the Training Dataset
First, we need to load our LMIC neonatal sepsis dataset.

1. Navigate to the **Dataset** tab on the sidebar.
2. Under **Local Files**, upload your Kleborate output CSV to the **Genotype** file input.
3. Set the **Genotype Format** to `Pathogenwatch-Kleborate`.
4. Upload your country data to the **Metadata (Optional)** input.
5. A dynamic mapping menu will appear. Find the **Spatial** dropdown and select your country column (e.g., `Country`),
   and set the **Resolution** to `Country`.
6. Click **Load Files**. The app will parse, validate, and merge your genomic and spatial data.

#### Step 2: Aggregate the CPS burden
We need to figure out the raw proportions of each K-locus (Capsular Polysaccharide) in our dataset.

1. Navigate to the **Burden** tab in the sidebar and open the **Prevalence Aggregation** accordion.
2. **Trait Column**: Select `K Locus`.
3. **Aggregation Mode**: Select **Compositional** (since we want to know the proportion of all K-loci relative to
   one another, not just the presence/absence of a single gene).
4. **Stratify By**: Select your spatial country column (e.g., `Spatial Country`).
5. **Pad Zeroes**: Check this box! This is mathematically crucial for Bayesian models to understand which countries had
   zero observations of a specific K-locus.
6. Click **Aggregate Data**.

#### Step 3: Estimate Bayesian burden
Now we use a Markov Chain Monte Carlo (MCMC) model to estimate the true population burden, drawing power (partial
pooling) across our different countries to adjust for sampling biases.

1. Open the **Prevalence Estimation** accordion.
2. **Estimator**: Select `Bayesian Hierarchical`.
3. Expand the **Hyperparameters** section. By default, the **Inference Method** is set to `MCMC`. You can leave the
   default samples (1500) and chains (4) as they are.
4. Click **Estimate Burden**. *Note: MCMC is mathematically rigorous and computationally intensive. It may take
   a minute to converge depending on the number of countries and loci.*

#### Step 4: Formulate the Vaccine & Inspect Stability
With our Bayesian estimates calculated, we can design the optimal multi-valent vaccine.
1. Navigate to the **Formulation** tab in the sidebar.
2. **Trait Valency**: Use the slider to select your desired number of targets (e.g., `6` for a hexavalent vaccine).
3. **Cross-Validation Stratum**: Select your country column.
4. **Designer Type**: Select `Cross-Validated (Rigorous)`.
5. Click **Generate Formulation**. The app will now perform Leave-One-Out (LOO) cross-validation, completely retraining the MCMC model multiple times, leaving one country out each time to see how the optimal formulation changes.

**Inspecting the Results:**
* Go to the **Stability Metrics** tab in the main window. Look at the **Rank Variance** and **Probability in Top N**. A good vaccine target will have a low rank variance (meaning its importance doesn't drastically change depending on which country you exclude).
* Go to the **Plots** tab to see your Cross-Validation Stability Matrix (a bump chart showing the rank of each K-locus across different geographic holdouts).

### Phase 2: The "Hot-Swap" Evaluation (HIC Adult Data)
We now have a rigorously designed 6-valent vaccine cached in the application's memory. We are going to seamlessly swap our underlying dataset to evaluate this exact formulation on a new population.

#### Step 5: Load the Testing Dataset (Pathogenwatch)
1. **Crucial Step:** Do NOT click "Clear All Data".
2. Navigate back to the **Dataset** tab.
3. Open the **Pathogenwatch** tab.
4. Enter your API key and click **Fetch Collections**.
5. Select your pre-compiled collection of HIC Adult BSI *Klebsiella* genomes.
6. Click **Load Collection**.

Because you didn't clear the memory, the app has safely overwritten the active dataset with the new HIC genomes, but preserved your LMIC-designed vaccine formulation in its reactive state.

#### Step 6: Forecast Vaccine Longevity
The Pathogenwatch loader automatically parses temporal metadata (Collection Date). We will use a Bayesian Structural Time Series (BSTS) to forecast how well our LMIC vaccine will cover the HIC population over time.
1. Navigate to the **Forecasting** tab in the sidebar.
2. Open the **Longevity Forecasting** accordion.
3. **Estimator Model**: Select `Bayesian (BSTS)`.
4. In the hyperparameters, ensure your **Forecast Horizon** is set to your desired future projection (e.g., 12 months or years, depending on your temporal resolution).
5. Click **Forecast Longevity**.

#### Step 7: Analyze the Results
1. In the main window, click on the **Plots** tab.
2. You will see a beautiful dual-axis plot:
    * **The Bars** represent the total absolute case burden of Adult BSIs over time.
    * **The Line** (with confidence intervals) represents the percentage of those cases that are covered by your 6-valent LMIC vaccine.
    * **The Dashed Vertical Line** represents the present day, with the area to the right showing the Bayesian forecast of the evolutionary trajectory.

**Conclusion:** By looking at this plot, you can instantly tell stakeholders: *"Our vaccine, optimized for neonatal sepsis in LMICs, historically covers X% of adult BSIs in HICs, and our BSTS model projects this coverage will remain stable at Y% over the next 5 years."*