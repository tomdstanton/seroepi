"""seroepi: A Python library for bacterial sero-epidemiology.

This library provides tools for interacting with the Pathogenwatch API,
standardizing isolate datasets, calculating prevalence and diversity,
and designing vaccine formulations.
"""

import seroepi.domains as _
from seroepi.client import WorldBankClient
from seroepi.dataset import SeroEpiDataset
from seroepi.domains import (
    BasePlotter,
    EpiMixin,
    GenoMixin,
    GeoMixin,
    QcMixin,
    render_plot,
)

__all__ = [
    "SeroEpiDataset",
    "WorldBankClient",
    "GeoMixin",
    "EpiMixin",
    "GenoMixin",
    "QcMixin",
    "BasePlotter",
    "render_plot",
    "_",
]
