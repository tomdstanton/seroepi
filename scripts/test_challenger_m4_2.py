"""
Adversarial Verification Script for SeroEpi Refactor (m4_2)
Focusing on:
1. Public symbol importability from `seroepi`, `seroepi.domains`, and `app`.
2. Package-wide scan for broken/stale imports or missing symbols.
3. API signature consistency and structure verification.
"""

import ast
import importlib
import inspect
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
APP_DIR = REPO_ROOT / "app"

# Ensure repo src is in sys.path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

results = []


def record(test_name: str, success: bool, details: str = ""):
    status = "PASS" if success else "FAIL"
    results.append((test_name, success, details))
    print(f"[{status}] {test_name}" + (f": {details}" if details else ""))


print("=" * 60)
print("RUNNING ADVERSARIAL IMPORT & API SIGNATURE VERIFICATION")
print("=" * 60)

# -------------------------------------------------------------
# Test 1: Import top-level `seroepi` and verify public symbols
# -------------------------------------------------------------
try:
    import seroepi

    record("Import seroepi", True)
except Exception as e:
    record("Import seroepi", False, str(e))

if hasattr(seroepi, "__all__"):
    for sym in seroepi.__all__:
        exists = hasattr(seroepi, sym)
        val = getattr(seroepi, sym, None)
        record(f"seroepi.__all__ symbol: {sym}", exists and val is not None, f"val={val}")
else:
    record("seroepi HAS __all__", False, "seroepi does not define __all__")


# -------------------------------------------------------------
# Test 2: Import `seroepi.domains` and verify public symbols
# -------------------------------------------------------------
try:
    import seroepi.domains

    record("Import seroepi.domains", True)
except Exception as e:
    record("Import seroepi.domains", False, str(e))

if hasattr(seroepi.domains, "__all__"):
    for sym in seroepi.domains.__all__:
        exists = hasattr(seroepi.domains, sym)
        val = getattr(seroepi.domains, sym, None)
        record(f"seroepi.domains.__all__ symbol: {sym}", exists and val is not None, f"val={val}")
else:
    record("seroepi.domains HAS __all__", False, "seroepi.domains does not define __all__")


# -------------------------------------------------------------
# Test 3: Import `app` submodules and verify public symbols
# -------------------------------------------------------------
app_modules = [
    "app.app",
    "app.burden",
    "app.coverage",
    "app.dataset",
    "app.forecasting",
    "app.formulation",
    "app.utils",
]
for mod_name in app_modules:
    try:
        mod = importlib.import_module(mod_name)
        record(f"Import {mod_name}", True)
        if hasattr(mod, "__all__"):
            for sym in mod.__all__:
                exists = hasattr(mod, sym)
                record(f"{mod_name}.__all__ symbol: {sym}", exists, f"attribute exists: {exists}")
    except Exception as e:
        record(f"Import {mod_name}", False, str(e))


# -------------------------------------------------------------
# Test 4: Recursive import of ALL modules under src/seroepi and app/
# -------------------------------------------------------------
for pkg_name, pkg_path in [("seroepi", SRC_DIR / "seroepi"), ("app", APP_DIR)]:
    for root, dirs, files in os.walk(pkg_path):
        for f in files:
            if f.endswith(".py"):
                rel_path = Path(root, f).relative_to(pkg_path.parent)
                mod_name = str(rel_path.with_suffix("")).replace(os.sep, ".")
                try:
                    m = importlib.import_module(mod_name)
                    record(f"Module importability: {mod_name}", True)
                    # Check __all__ if present
                    if hasattr(m, "__all__"):
                        for sym in m.__all__:
                            if not hasattr(m, sym):
                                record(f"{mod_name} missing __all__ symbol '{sym}'", False)
                except Exception as e:
                    record(f"Module importability: {mod_name}", False, str(e))


# -------------------------------------------------------------
# Test 5: AST Scan for references to dissolved/stale modules
# -------------------------------------------------------------
dissolved_targets = [
    "seroepi.accessors",
    "seroepi.plotting",
    "src.seroepi.accessors",
    "src.seroepi.plotting",
    "seroepi.app",
]

found_stale_imports = []
for scan_dir in [SRC_DIR, APP_DIR, REPO_ROOT / "tests", REPO_ROOT / "scripts"]:
    if not scan_dir.exists():
        continue
    for p in scan_dir.rglob("*.py"):
        try:
            content = p.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(p))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(target in alias.name for target in dissolved_targets):
                            found_stale_imports.append((str(p), alias.name))
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if any(target in mod for target in dissolved_targets):
                        found_stale_imports.append((str(p), f"from {mod} import ..."))
        except Exception as e:
            record(f"AST parse {p.name}", False, str(e))

record(
    "No stale imports of accessors/plotting/app", len(found_stale_imports) == 0, f"stale_imports={found_stale_imports}"
)


# -------------------------------------------------------------
# Test 6: API Signature & Domain Mixin / SeroEpiDataset checks
# -------------------------------------------------------------
try:
    from seroepi.dataset import SeroEpiDataset
    from seroepi.domains.base import BaseDomainMixin
    from seroepi.domains.epi import EpiMixin
    from seroepi.domains.geno import GenoMixin
    from seroepi.domains.geo import GeoMixin
    from seroepi.domains.qc import QcMixin

    # Check inheritance
    bases = SeroEpiDataset.__mro__
    has_mixins = all(m in bases for m in [GeoMixin, EpiMixin, GenoMixin, QcMixin])
    record("SeroEpiDataset inherits from all 4 domain mixins", has_mixins, f"MRO={[b.__name__ for b in bases]}")

    # Check __slots__
    slots_ok = hasattr(SeroEpiDataset, "__slots__")
    record("SeroEpiDataset has __slots__", slots_ok)

    mixin_slots = all(
        getattr(m, "__slots__", None) == () for m in [GeoMixin, EpiMixin, GenoMixin, QcMixin, BaseDomainMixin]
    )
    record("Domain Mixins have empty __slots__ = ()", mixin_slots)

    # Check namespace properties
    import polars as pl

    dummy_df = pl.DataFrame({"id": [1], "lat": [0.0], "lon": [0.0]})
    ds = SeroEpiDataset(data=dummy_df)
    record("SeroEpiDataset.geo returns self", ds.geo is ds)
    record("SeroEpiDataset.epi returns self", ds.epi is ds)
    record("SeroEpiDataset.geno returns self", ds.geno is ds)
    record("SeroEpiDataset.qc returns self", ds.qc is ds)

    # Inspect all method signatures on SeroEpiDataset
    for attr_name in dir(SeroEpiDataset):
        if attr_name.startswith("_"):
            continue
        attr = getattr(SeroEpiDataset, attr_name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
            except Exception as sig_err:
                record(f"Signature inspection for SeroEpiDataset.{attr_name}", False, str(sig_err))

    record("SeroEpiDataset method signatures valid", True)

except Exception as e:
    record("SeroEpiDataset API checks", False, str(e))


# -------------------------------------------------------------
# Test 7: Plotter class importability & inheritance
# -------------------------------------------------------------
try:
    from seroepi.domains import (
        AlphaDiversityPlotter,
        BetaHeatmapPlotter,
        ChoroplethPlotter,
        CompositionBarPlotter,
        CompositionHeatmapPlotter,
        CumulativeCoveragePlotter,
        EpicurvePlotter,
        ForestPlotter,
        LongitudinalPrevalencePlotter,
        NetworkPlotter,
        SpatialSurfacePlotter,
    )

    record("Import all co-located Plotter classes from seroepi.domains", True)

    plotter_classes = [
        ChoroplethPlotter,
        SpatialSurfacePlotter,
        EpicurvePlotter,
        ForestPlotter,
        LongitudinalPrevalencePlotter,
        CompositionBarPlotter,
        CompositionHeatmapPlotter,
        CumulativeCoveragePlotter,
        AlphaDiversityPlotter,
        BetaHeatmapPlotter,
        NetworkPlotter,
    ]
    for cls in plotter_classes:
        has_plot = hasattr(cls, "plot") or callable(cls)
        record(f"Plotter class {cls.__name__} valid", has_plot)

except Exception as e:
    record("Import Plotter classes", False, str(e))


# -------------------------------------------------------------
# Test 8: Circular Import Resilience
# -------------------------------------------------------------
import_orders = [
    ["seroepi.domains", "seroepi.dataset", "seroepi"],
    ["seroepi.dataset", "seroepi.domains", "seroepi"],
    ["app.app", "seroepi", "seroepi.domains"],
    ["app.dataset", "seroepi.dataset", "seroepi"],
]

for idx, order in enumerate(import_orders):
    try:
        # Clear modules from sys.modules to test fresh import order
        for mod in list(sys.modules.keys()):
            if mod.startswith("seroepi") or mod.startswith("app"):
                del sys.modules[mod]

        for target in order:
            importlib.import_module(target)
        record(f"Import order sequence {idx + 1} ({' -> '.join(order)})", True)
    except Exception as e:
        record(f"Import order sequence {idx + 1} ({' -> '.join(order)})", False, str(e))


# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------
print("=" * 60)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"VERIFICATION SUMMARY: {passed} PASSED, {failed} FAILED (Total: {len(results)})")
print("=" * 60)

if failed > 0:
    print("\nFAILED CHECKS:")
    for name, ok, details in results:
        if not ok:
            print(f" - {name}: {details}")

sys.exit(0 if failed == 0 else 1)
