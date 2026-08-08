---
title: Release Notes
author: Tom Stanton
comments: true
tags: [markdown, documentation, web]
icon: lucide/rocket
categories:
  - Development
---

# Release Notes

## v1.0.0
*Published on 2026-08-05*

### What's Changed
* Initial release of **SeroEpi** — Bacterial sero-epidemiology Python library for prevalence estimation, spatial mapping, and vaccine formulation.
* Implemented core Polars-based dataset container `SeroEpiDataset` with domain mixins (`GeoMixin`, `EpiMixin`, `GenoMixin`, `QcMixin`).
* Added statistical prevalence and diversity estimators (Frequentist, Bayesian, Gaussian Process spatial).
* Added Leave-One-Out (LOO) cross-validated vaccine formulation engine (`CVFormulationDesigner`).
* Integrated interactive 5-tab Shiny for Python web application in `/app`.
* Complete documentation overhaul with `zensical` static site builder.
