# 🦠💉🌍 `seroepi`

`seroepi` is a comprehensive Python toolkit and interactive Shiny dashboard for the epidemiological, geospatial, and genotypic analysis of pathogen isolates.

Built seamlessly on top of `pandas`, it is specifically designed to ingest output from genomic pipelines (like **Kleborate** and **Pathogenwatch**), calculate statistical burdens, map spatial transmissions, and computationally design optimal vaccine formulations.

---

## ✨ What You Can Do

- **Robust Data Stewardship**: Validate inputs seamlessly against our `UnifiedIsolateSchema` (powered by Pandera) to ensure completely standardized genotypic, spatial, and temporal datasets.
- **Interactive Dashboarding**: Launch a beautifully designed, dark-mode Shiny app to explore your data, train models, and generate publication-ready plots instantly.
- **Smart Pandas Accessors**: Clean coordinates, query AMR genes, and generate epidemic curves natively using `df.geo`, `df.geno`, `df.epi`, and `df.qc`.
- **Robust Statistical Modeling**: Estimate global and regional prevalence using Frequentist, Bayesian MCMC/SVI, and Gaussian Process (GP) Spatial models.
- **Vaccine Formulation Engine**: Use rigorous Leave-One-Out (LOO) cross-validation to algorithmically identify the most stable and high-coverage target antigens (e.g., K-loci) for vaccine design.
- **Outbreak & Transmission Clustering**: Instantly generate transmission networks and spatial cliques using SNP distance matrices and spatiotemporal thresholds.

---

## 📦 Installation

You can install `seroepi` directly from PyPI. We highly recommend using uv for lightning-fast installations,
but standard `pip` works perfectly too.

```bash
# Using uv (Recommended)
uv pip install seroepi

# Using standard pip
pip install seroepi
```

### Optional Dependencies
To unlock the advanced Bayesian models, Gaussian Processes, and Plotly visualizations, install the optional dependencies:

```bash
uv pip install seroepi[models,plot]
```

---

## 📖 Nomenclature: Traits vs. Targets

To ensure clarity across epidemiological modeling and vaccine design, `seroepi` strictly defines:
* **Trait**: The overarching epidemiological variable or genomic locus being modeled (e.g., `K_locus`, `O_locus`, `amr_KPC`).
* **Target**: The specific variant or level within that trait (e.g., `K1`, `K2`, `K3`).

When aggregating data, you specify the overarching *Trait* column, and the resulting DataFrames will standardize the individual variants into a `target` column for unified downstream processing.
---

## 💻 API Quickstart

If you prefer working in Jupyter Notebooks or Python scripts, `seroepi` extends standard Pandas DataFrames to make
bioinformatics workflows effortless.

### 1. Data Ingestion, Validation & Spatial Cleaning
```python
import pandas as pd
from seroepi.io import PathogenwatchKleborateParser
import seroepi.accessors  # Magically registers .epi, .geo, .geno, .qc

# Load your Kleborate output and optional Metadata
# Parse, merge, and strictly validate against the UnifiedIsolateSchema
df = PathogenwatchKleborateParser.parse(
    pd.read_csv("kleborate_results.csv"),
    meta_df=pd.read_csv("metadata.csv"),
    meta_kwargs={"id_col": "sample_id", "country_col": "country"}
)

# Automatically impute missing coordinates based on Country names!
df = df.geo.standardize_and_impute()
```

### 2. Genomic Clustering
```python
from seroepi.dist import Distances

# Load a pairwise SNP matrix (e.g., from Pathogenwatch)
dist = Distances.from_pathogenwatch("distances.csv")

# Find isolates separated by ≤ 20 SNPs
clusters = dist.connected_components(threshold=20)

# Merge straight back into your dataframe
df = df.join(clusters, on='sample_id')
```

### 3. Prevalence Estimation & Plotting
```python
from seroepi.estimators import FrequentistPrevalenceEstimator

# Aggregate the data to find the prevalence of K-loci across different countries
agg_df = df.epi.aggregate_prevalence(stratify_by=['country'], trait_col='K_locus')

# Fit the estimator
estimator = FrequentistPrevalenceEstimator(method='wilson')
results = estimator.calculate(agg_df)

# Generate a publication-ready Plotly Forest Plot
fig = results.plot('forest')
fig.show()
```

### 4. Algorithmic Vaccine Formulation
```python
from seroepi.formulation import CVFormulationDesigner

# Design a 6-valent vaccine, using 'country' as the cross-validation holdout
designer = CVFormulationDesigner(valency=6, n_jobs=-1)
designer.fit(estimator, agg_df, loo_col='country')

# View the most stable optimal targets
optimal_vaccine = designer.formulation_
print(optimal_vaccine.get_formulation())
```

---

## 📚 Documentation

For a complete deep-dive into the available methods, classes, and architectural concepts, please refer to the fully
documented **API Reference** (Automatically generated).

---

## People

- [Dr. Tom Stanton](https://wyreslab.com/)
- [A/Prof. Kelly L. Wyres](https://wyreslab.com/)
- [Prof. Kathryn E. Holt](https://holtlab.net)
- [Dr. Ryan R. Wick](https://rrwick.github.io/)

[Contact Kelly and Tom](mailto:kaptive.typing@gmail.com) for help with Kaptive,
or to report bugs or request features.

---

## References

[^1]: Stanton TD, Keegan SP, Abdulahi JA, Amulele AV, Bates M, et al. (2026) Distribution of capsule and O types in
    *Klebsiella pneumoniae* causing neonatal sepsis in Africa and South Asia: A meta-analysis of genome-predicted
    serotype prevalence to inform potential vaccine coverage. PLOS Medicine 23(1): e1004879.
    <https://doi.org/10.1371/journal.pmed.1004879>
