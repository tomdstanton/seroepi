# 🗺️📈🧬🛡️ Domain Mixins & Plotting Engine

Domain mixins provide specialized domain-driven accessors and computational methods for `SeroEpiDataset` and Polars DataFrames across geography, epidemiology, genomics, and quality control.

```python
from seroepi.domains.base import BaseDomainMixin, BasePlotter, render_plot
from seroepi.domains.geo import GeoMixin
from seroepi.domains.epi import EpiMixin
from seroepi.domains.geno import GenoMixin
from seroepi.domains.qc import QcMixin
```

---

## 🏛️ Base Domain Mixin

::: seroepi.domains.base.BaseDomainMixin

---

## 🗺️ Geographical Domain (`GeoMixin`)

::: seroepi.domains.geo.GeoMixin

---

## 📈 Epidemiological Domain (`EpiMixin`)

::: seroepi.domains.epi.EpiMixin

---

## 🧬 Genotypic & Phenotypic Domain (`GenoMixin`)

::: seroepi.domains.geno.GenoMixin

---

## 🛡️ Quality Control Domain (`QcMixin`)

::: seroepi.domains.qc.QcMixin

---

## 🎨 Plotting Engine & Router

::: seroepi.domains.base.BasePlotter

::: seroepi.domains.base.render_plot
