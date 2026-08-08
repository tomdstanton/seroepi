# 🦠 Shiny Web Application Guide (`app/app.py`)

`seroepi` provides a powerful, decoupled, production-grade **Shiny for Python** web application located in the `/app` directory. The application delivers an interactive 5-stage bioinformatic and epidemiological dashboard designed for rapid data exploration, statistical burden modeling, Leave-One-Out vaccine formulation, and long-term population coverage forecasting — complete with an embedded context-aware **Gemini AI Assistant**.

---

## ✨ Overview & Architecture

The application is built around a modular 5-stage analytical pipeline:

```
[1. Data Ingestion & QC] ➔ [2. Burden Estimation] ➔ [3. Vaccine Formulation] ➔ [4. Population Coverage] ➔ [5. Longevity Forecasting]
                                                                                                                   │
                                                                                                         🤖 [Ask AI Assistant]
```

### Core Analytical UI Tabs

1. 💽 **1. Dataset (Ingestion & Quality Control)**: Parse genotype files (e.g. Kleborate CSVs), upload distance matrices or trees, link metadata with interactive column mapping (`ColMapper`), query Pathogenwatch collections via API, filter assembly quality (N50, contig count, species), and compute spatiotemporal transmission networks.
2. 🦠 **2. Burden (Prevalence Burden Estimation)**: Aggregate binary or compositional target prevalence across demographic and spatial strata, fit Frequentist GLMs or Bayesian MCMC/SVI models, inspect MCMC diagnostic plots (R-hat, Effective Sample Size), and export fitted model artifacts.
3. 💉 **3. Formulation (Algorithmic Vaccine Design)**: Algorithmically design multivalent vaccine formulations using Post-Hoc or Leave-One-Out Cross-Validation (LOO-CV) strategies, evaluate rank stability bump charts, and configure custom target overrides.
4. 🛡️ **4. Coverage (Multi-Population Coverage Evaluation)**: Evaluate single-population or stratified multivalent vaccine coverage, generate Pareto cumulative coverage curves, and map regional coverage on interactive spatial choropleth maps.
5. 🔮 **5. Forecasting (Incidence & Longevity Forecasting)**: Bin temporal incidence into monthly, weekly, or yearly intervals with zero-padding, fit Bayesian Structural Time Series (BSTS) or GLM incidence models, and project long-term vaccine longevity over dual-axis time horizons.

### Global Navbar Utilities

- 💾 **Workspace Management**: Export and import complete analytical sessions as compressed `.sero` state archives powered by `joblib`.
- 📚 **Active Context Selectors**: Seamlessly route active datasets, burden runs, and vaccine formulations across pipeline tabs without re-computing.
- 🤖 **Ask AI Assistant**: Open an offcanvas slide-out AI assistant powered by Google Gemini (`gemini-2.5-flash`), pre-loaded with your active dataset summary and statistical estimates to answer natural language questions.
- 🌙 **Dark Mode Toggle**: Toggle between crisp modern dark-mode (`shinyswatch.theme.pulse()`) and standard light mode.

---

## 📦 Prerequisites & Installation

### Option A: Install via `uv` (Recommended)

Install `seroepi` with the optional `[app]` dependency group, which includes `shiny`, `shinywidgets`, `shinyswatch`, and `google-genai`:

```bash
uv pip install seroepi[app]
```

### Option B: Standard `pip` Installation

```bash
pip install seroepi[app]
```

### 🔑 Environment Variables (Optional AI Assistant)

To enable the interactive **Ask AI Assistant 🤖** drawer in the web app, set your Google Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

*(Note: If `GEMINI_API_KEY` is omitted, the web dashboard functions completely without crashing; only the AI drawer will display an API key prompt).*

---

## 💻 Local Execution

Launch the Shiny server locally using `uv` or `just`:

### Using `uv`

```bash
uv run shiny run app/app.py --reload --host 127.0.0.1 --port 8000
```

### Using `just`

```bash
just serve
```

Once running, navigate to **`http://127.0.0.1:8000`** in your web browser.

---

## 📖 End-to-End Workflow Tutorial

### Step 1: Data Ingestion, Quality Control & Clustering 💽

1. Navigate to the **1. Dataset** tab.
2. Choose your input source:
   - **Local Upload**: Select a genotype CSV (e.g., Kleborate output), an optional SNP distance matrix or Newick tree, and a metadata CSV.
   - **Pathogenwatch API**: Enter your Pathogenwatch API key and collection ID to fetch surveillance data directly.
   - **Pre-packaged Example**: Click **Load Neonatal Sepsis Dataset** to pre-fill the workspace with pre-packaged *Klebsiella pneumoniae* surveillance data.
3. Use the **Column Mapper** modal to map key columns (`Sample ID`, `Temporal Date`, `Spatial Country/Region`, `Latitude`, `Longitude`).
4. Apply **Quality Control** filters: set minimum N50 (e.g. `10,000 bp`), maximum contig count (e.g. `500`), and target species name.
5. In the **Clustering & Transmission** drawer, set a maximum SNP threshold (e.g. `20 SNPs`) or spatial-temporal thresholds (e.g. `25 km`, `14 days`) to generate transmission clusters and render interactive 2D MDS network graphs.

---

### Step 2: Prevalence Burden Estimation 🦠

1. Switch to the **2. Burden** tab.
2. Configure **Aggregation Options**:
   - Select **Mode**: *Compositional* (relative target proportions summing to 1.0) or *Trait* (binary locus presence/absence).
   - Select **Trait Column** (e.g., `geno_K_locus`).
   - Add **Stratification Variables** (e.g., `spatial_Country`, `clinical_syndrome`).
   - Enable **Zero-Padding** to include unobserved targets across strata.
3. Choose a **Statistical Estimator**:
   - **Frequentist GLM**: Binomial logit link GLM with confidence intervals.
   - **Bayesian Hierarchical**: PyMC/NumPyro MCMC or SVI hierarchical model. Set MCMC chains, sampling steps, and warmup iterations.
   - **Spatial GP**: Spatial Gaussian Process surface model over coordinates.
4. Click **Fit Estimator**. Instantly inspect the fitted prevalence estimates table, MCMC posterior diagnostics (R-hat, ESS), Forest plots, Composition Bar charts, and Composition Heatmaps.
5. Download fitted model weights as `.pkl` artifacts for external reporting.

---

### Step 3: Vaccine Formulation Design 💉

1. Navigate to the **3. Formulation** tab.
2. Select your active **Burden Run** from the context dropdown.
3. Set the desired vaccine **Valency** (e.g. `6-valent` or `10-valent`).
4. Select a **Design Strategy**:
   - **Post-Hoc Designer**: Rapidly selects top N targets based on population-wide marginal prevalence.
   - **Cross-Validated (LOO-CV)**: Performs Leave-One-Out cross-validation across holdout strata (e.g. leaving each country out in turn) to compute rank stability, selection probabilities, and overall coverage variance.
5. Click **Design Formulation**. Explore the resulting target priority table, Leave-One-Out stability metrics, target permutation history, and rank stability bump charts.
6. Use **Custom Overrides** to manually pin or swap specific targets, generating custom formulation objects.

---

### Step 4: Multi-Population Coverage Evaluation 🛡️

1. Switch to the **4. Coverage** tab.
2. Select an active **Vaccine Formulation** and target surveillance population.
3. Choose evaluation mode: **Binary Coverage** (percentage of isolates carrying at least one vaccine target) or **Compositional Breakdown**.
4. Slice coverage across demographic or clinical sub-populations (e.g. AMR status, age group, hospital ward).
5. Render publication-grade **Pareto Cumulative Coverage Curves**, **Stratified Forest Plots**, and interactive **Spatial Choropleth Maps**.

---

### Step 5: Temporal Incidence & Longevity Forecasting 🔮

1. Open the **5. Forecasting** tab.
2. Set temporal binning resolution (**Month**, **Week**, or **Year**).
3. Fit a **Bayesian Structural Time Series (BSTS)** or **GLM Incidence** model to project incidence trends into future temporal windows.
4. Render dual-axis **Longevity Forecast Plots** comparing projected disease incidence against expected vaccine coverage decay over time.

---

## 💾 Workspace Management & Serialization

### Session State Archives (`.sero`)

Save your complete analytical session at any point by clicking **Export Workspace** in the top navbar. `SeroEpi` serializes all active datasets, clean DataFrames, fitted estimators, and formulations into a single compressed `.sero` archive using `joblib`. 

To resume work later or share your analysis with collaborators, simply click **Import Workspace** and upload the `.sero` file.

### Model & Formulation Artifact Exports (`.pkl`)

Individual fitted estimator objects (`PrevalenceEstimates`, `IncidenceEstimates`) and formulation objects (`Formulation`) can be exported as standalone pickle files (`.pkl`) directly from their respective module tabs for integration into custom Python scripts or downstream pipelines.

---

## 🐳 Docker Container Deployment

For cloud environments, HPC clusters, or enterprise server deployments, `SeroEpi` includes a production-ready container definition.

### 1. Build the Docker Image

Use `just` or Docker directly to build the container image:

```bash
# Using just
just docker-build

# Using Docker directly
docker build -t seroepi-app:latest .
```

### 2. Run the Docker Container

Run the container on port `8000`, optionally injecting your `GEMINI_API_KEY`:

```bash
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY="your-api-key-here" \
  --name seroepi-dashboard \
  seroepi-app:latest
```

Access the containerized application at **`http://localhost:8000`**.
