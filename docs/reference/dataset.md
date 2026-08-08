# 💽 `SeroEpiDataset` Container

The `SeroEpiDataset` is the central slotted, frozen container in `SeroEpi`. Built on top of **Polars** and validated with **Patito**, it inherits functionality from all four domain mixins (`GeoMixin`, `EpiMixin`, `GenoMixin`, `QcMixin`).

```python
from seroepi.dataset import SeroEpiDataset
```

---

::: seroepi.dataset.SeroEpiDataset
