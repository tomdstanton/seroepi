# 🦠💉🌍 `SeroEpi`

`SeroEpi` is a comprehensive, production-grade Python library and interactive Shiny web application for bacterial sero-epidemiology, genomic surveillance, and algorithmic multivalent vaccine formulation design.

Built seamlessly on top of **Polars** and **Patito**, `SeroEpi` is engineered to ingest outputs directly from bioinformatic pipelines (such as **Kleborate** and **Pathogenwatch**), validate genomic metadata, estimate prevalence burdens with rigorous statistical models, map spatiotemporal transmission networks, and computationally design optimal vaccine formulations.

---

## ✨ Key Features & Capabilities

- 📦 **Robust Data Stewardship**: Validate genomic inputs seamlessly against our `SampleModel` schema to enforce standardized column naming, data types, and spatial-temporal integrity across large surveillance cohorts.
- 🗺️ **Domain-Driven Mixins**: Process complex datasets effortlessly using our domain architecture (`GeoMixin`, `EpiMixin`, `GenoMixin`, `QcMixin`), enabling automatic spatial coordinate imputation, assembly quality filtering, and genomic queries.
- 📊 **Advanced Statistical Estimators**: Calculate global, regional, and stratified serotype prevalence using Frequentist Wilson CIs, Generalized Linear Models (GLMs), Bayesian MCMC/SVI hierarchical models, and Gaussian Process (GP) spatial surfaces.
- 💉 **Algorithmic Vaccine Formulation**: Algorithmically select optimal target combinations (e.g. K-loci or O-loci) for multivalent vaccines using Leave-One-Out (LOO) cross-validation to maximize population coverage and rank stability across diverse geographic regions.
- 🕸️ **Spatiotemporal Transmission Networks**: Instantly generate transmission clusters and interactive MDS network graphs by combining pairwise SNP distance matrices with geographic distance and temporal sampling windows.
- 💻 **Decoupled Shiny Dashboard**: Launch an intuitive, dark-mode interactive web dashboard with embedded Gemini AI assistance for zero-code data exploration, model training, and publication-ready visualization export.

---

## 📖 Nomenclature Guide: Trait vs. Target

To maintain strict statistical precision across epidemiological modeling and vaccine formulation pipelines, `SeroEpi` formalizes the distinction between **Traits** and **Targets**:

| Concept | Definition | Code Example |
| :--- | :--- | :--- |
| **Trait** 🧬 | The overarching genomic locus, serotype system, or epidemiological variable being modeled. | `geno_K_locus`, `geno_O_locus`, `amr_KPC` |
| **Target** 🎯 | The specific allele, variant, or level within a Trait system. | `K1`, `K2`, `K64`, `O1`, `O2` |

When aggregating surveillance data (e.g., via `ds.aggregate_prevalence(trait_col="geno_K_locus")`), `SeroEpi` automatically standardizes individual locus variants into a unified `target` column. This standardization ensures that statistical estimators and formulation engines seamlessly process any biological trait system without structural code changes.

---

## 💻 API Quickstart

### 1. Data Ingestion, Patito Validation & Spatial Imputation 📦

Read raw genomic pipeline outputs and metadata, validate inputs against `SampleModel`, and impute missing geographical coordinates automatically based on country gazetteer lookups.

```python
import polars as pl
from seroepi import SeroEpiDataset
from seroepi.io import PathogenwatchKleborateParser

# Ingest Kleborate output and metadata CSVs into a standardized Polars DataFrame
df = PathogenwatchKleborateParser.parse(
    genotype_df=pl.read_csv("kleborate_results.csv"),
    meta_df=pl.read_csv("metadata.csv"),
    meta_kwargs={"id_col": "sample_id", "spatial_col": "Country", "date_col": "collection_date"},
)

# Instantiate the slotted, frozen SeroEpiDataset container
ds = SeroEpiDataset(data=df, name="Neonatal Sepsis Cohort")

# Automatically impute missing lat/lon coordinates from gazetteer lookups
ds_clean = ds.standardize_and_impute()
print(f"Loaded {len(ds_clean)} isolates across {len(ds_clean.columns)} variables.")
```

---

### 2. Prevalence Burden Estimation & Forest Plot Rendering 📊

Aggregate serotype prevalence across geographic strata and fit statistical confidence intervals, then render publication-ready Plotly visualizations instantly.

```python
from seroepi.estimators import UnpooledPrevalenceEstimator
from seroepi.domains import render_plot
from seroepi.constants import PlotType

# Aggregate prevalence of K-locus targets stratified by country
agg_df = ds_clean.aggregate_prevalence(
    stratify_by=["spatial_Country"],
    trait_col="geno_K_locus",
)

# Calculate Wilson score binomial confidence intervals
estimator = UnpooledPrevalenceEstimator(method="wilson")
estimates = estimator.calculate(agg_df)

# Render an interactive publication-ready Forest Plot
fig = render_plot(estimates, PlotType.FOREST)
fig.show()
```

---

### 3. Leave-One-Out Cross-Validated Vaccine Formulation 💉

Algorithmically design a multivalent vaccine formulation by selecting target antigens that maximize population coverage while evaluating rank stability across holdout populations.

```python
from seroepi.formulation import CVFormulationDesigner
from seroepi.domains import render_plot
from seroepi.constants import PlotType

# Fit a 6-valent vaccine formulation using Leave-One-Out CV across country strata
designer = CVFormulationDesigner(valency=6, n_jobs=-1)
designer.fit(estimator, agg_df, loo_col="spatial_Country")

# Extract the optimal vaccine target combination
formulation = designer.formulation_
print("Selected Vaccine Targets:", formulation.get_formulation())

# Visualize target rank stability across geographic holdout iterations
bump_fig = render_plot(formulation, PlotType.STABILITY_BUMP)
bump_fig.show()
```

---

### 4. Spatiotemporal Transmission Networks & Graph Rendering 🕸️

Delineate outbreak clusters and transmission chains by combining genomic clone markers, geographic distances (in kilometers), and temporal sampling intervals (in days).

```python
from seroepi.domains import render_plot
from seroepi.constants import PlotType

# Build spatiotemporal transmission network within 25 km and 14 days
network = ds_clean.transmission_network(
    clone_col="geno_ST",
    spatial_threshold_km=25.0,
    temporal_threshold_days=14,
)

# Extract spatial-temporal outbreak clusters
clusters = network.get_clusters()
print(f"Identified {len(clusters)} distinct transmission clusters.")

# Render 2D MDS network graph with interactive cluster overlays
net_fig = render_plot(network, PlotType.NETWORK)
net_fig.show()
```