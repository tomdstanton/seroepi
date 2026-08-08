"""
Deep API Signatures & Public Export Challenger Test
"""

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

failures = []


def assert_true(condition, msg):
    if not condition:
        failures.append(msg)


print("Starting deep signature & public export checks...", flush=True)

# 1. Check top-level seroepi exports
import seroepi  # noqa: E402

for sym in seroepi.__all__:
    assert_true(hasattr(seroepi, sym), f"seroepi.__all__ has '{sym}' but attribute missing on seroepi module")
    val = getattr(seroepi, sym)
    assert_true(val is not None, f"seroepi.{sym} is None")

# 2. Check seroepi.domains exports
import seroepi.domains as domains  # noqa: E402

for sym in domains.__all__:
    assert_true(
        hasattr(domains, sym), f"seroepi.domains.__all__ has '{sym}' but attribute missing on seroepi.domains module"
    )
    val = getattr(domains, sym)
    assert_true(val is not None, f"seroepi.domains.{sym} is None")

# 3. Test `from seroepi import *` behavior
namespace = {}
exec("from seroepi import *", namespace)
for sym in seroepi.__all__:
    assert_true(sym in namespace, f"from seroepi import * failed to import '{sym}'")

# 4. Test `from seroepi.domains import *` behavior
namespace_domains = {}
exec("from seroepi.domains import *", namespace_domains)
for sym in domains.__all__:
    assert_true(sym in namespace_domains, f"from seroepi.domains import * failed to import '{sym}'")

# 5. Check SeroEpiDataset methods & signatures
from seroepi.dataset import SeroEpiDataset  # noqa: E402
from seroepi.domains.epi import EpiMixin  # noqa: E402
from seroepi.domains.geno import GenoMixin  # noqa: E402
from seroepi.domains.geo import GeoMixin  # noqa: E402
from seroepi.domains.qc import QcMixin  # noqa: E402

df = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "lat": [10.0, 11.0, 12.0],
        "lon": [20.0, 21.0, 22.0],
        "year": [2020, 2021, 2022],
        "serotype": ["19F", "6B", "19F"],
        "qc_score": [0.9, 0.95, 0.88],
    }
)
dataset = SeroEpiDataset(data=df, name="test_dataset")

# Verify Mixin methods are present on dataset
for mixin in [GeoMixin, EpiMixin, GenoMixin, QcMixin]:
    for attr_name in dir(mixin):
        if not attr_name.startswith("_"):
            attr = getattr(SeroEpiDataset, attr_name)
            assert_true(
                callable(attr) or isinstance(attr, property),
                f"Mixin method {attr_name} missing or invalid on SeroEpiDataset",
            )

# Test namespace properties on dataset instance
assert_true(dataset.geo is dataset, "dataset.geo is not dataset")
assert_true(dataset.epi is dataset, "dataset.epi is not dataset")
assert_true(dataset.geno is dataset, "dataset.geno is not dataset")
assert_true(dataset.qc is dataset, "dataset.qc is not dataset")

print("Deep signature verification finished successfully.", flush=True)
if failures:
    print("FAILURES DETECTED:", flush=True)
    for f in failures:
        print(" -", f, flush=True)
    sys.exit(1)
else:
    print("ALL DEEP SIGNATURE CHECKS PASSED (0 failures)", flush=True)
    sys.exit(0)
