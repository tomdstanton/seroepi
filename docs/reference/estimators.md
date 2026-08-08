# 📊 Statistical Estimators

`SeroEpi` provides a suite of statistical estimators for calculating pathogen serotype prevalence, diversity indices, GLM regression trends, Bayesian MCMC/SVI posteriors, and Gaussian Process spatial prevalence surfaces.

```python
from seroepi.estimators.base import (
    BaseEstimator,
    PrevalenceEstimates,
    IncidenceEstimates,
    AlphaDiversityEstimates,
    BetaDiversityEstimates,
)
from seroepi.estimators.core import (
    UnpooledPrevalenceEstimator,
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
)
from seroepi.estimators.modelled import (
    GLMPrevalenceEstimator,
    GLMIncidenceEstimator,
    BayesianPrevalenceEstimator,
    SpatialPrevalenceEstimator,
)
```

---

## 🏛️ Estimator Base Class

::: seroepi.estimators.base.BaseEstimator

---

## 📦 Core Estimators

::: seroepi.estimators.core.UnpooledPrevalenceEstimator

::: seroepi.estimators.core.AlphaDiversityEstimator

::: seroepi.estimators.core.BetaDiversityEstimator

---

## 🔮 Advanced Modelled Estimators

::: seroepi.estimators.modelled.GLMPrevalenceEstimator

::: seroepi.estimators.modelled.GLMIncidenceEstimator

::: seroepi.estimators.modelled.BayesianPrevalenceEstimator

::: seroepi.estimators.modelled.SpatialPrevalenceEstimator

---

## 💾 Estimate Result Containers

::: seroepi.estimators.base.PrevalenceEstimates

::: seroepi.estimators.base.IncidenceEstimates

::: seroepi.estimators.base.AlphaDiversityEstimates

::: seroepi.estimators.base.BetaDiversityEstimates
