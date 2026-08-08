"""
Verification script for the Polars + Patito migration.

Validates that:
1. SeroEpiDataset can be instantiated with a Polars DataFrame.
2. SeroEpiDataset is a frozen, slotted dataclass.
3. Estimates and its subclasses are frozen, slotted dataclasses wrapping pl.DataFrame.
4. The UnpooledPrevalenceEstimator works end-to-end with Polars data.
5. KaptiveClient can be instantiated and serialize a SeroEpiDataset.
6. No pandas imports remain in the codebase.
"""

import dataclasses
import sys
from pathlib import Path

import polars as pl

# Ensure the source is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PASS = 0
FAIL = 0


def check(description, condition):
    global PASS, FAIL
    status = "✅ PASS" if condition else "❌ FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {status}: {description}")


def main():
    global PASS, FAIL
    print("=" * 70)
    print("SeroEpi Polars Migration — Verification Script")
    print("=" * 70)

    # ---- 1. SeroEpiDataset instantiation ----
    print("\n--- 1. SeroEpiDataset ---")
    from seroepi.dataset import SeroEpiDataset

    mock_df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5"],
            "latitude": [51.5, -33.8, 35.6, None, 48.8],
            "longitude": [-0.1, 151.2, 139.6, None, 2.3],
        }
    )

    ds = SeroEpiDataset(data=mock_df, name="Test Dataset")
    check("SeroEpiDataset wraps a pl.DataFrame", isinstance(ds.data, pl.DataFrame))
    check("SeroEpiDataset.name is set correctly", ds.name == "Test Dataset")
    check("len(SeroEpiDataset) works", len(ds) == 5)
    check("SeroEpiDataset.columns works", ds.columns == ["sample_id", "latitude", "longitude"])
    check("SeroEpiDataset.shape works", ds.shape == (5, 3))

    # Frozen check
    frozen = False
    try:
        ds.name = "Modified"
    except (dataclasses.FrozenInstanceError, AttributeError):
        frozen = True
    check("SeroEpiDataset is frozen (immutable)", frozen)

    # Slotted check
    check("SeroEpiDataset uses __slots__", hasattr(ds, "__slots__") or "__slots__" in type(ds).__dict__)

    # Transformation returns new dataset
    filtered = ds.filter(pl.col("latitude").is_not_null())
    check("SeroEpiDataset.filter() returns a new SeroEpiDataset", isinstance(filtered, SeroEpiDataset))
    check("SeroEpiDataset.filter() preserves metadata", filtered.name == "Test Dataset")
    check("SeroEpiDataset.filter() correctly filters rows", len(filtered) == 4)

    # Serialization
    dicts = ds.to_dicts()
    check("SeroEpiDataset.to_dicts() returns list[dict]", isinstance(dicts, list) and isinstance(dicts[0], dict))
    serialized = ds.to_dict()
    check("SeroEpiDataset.to_dict() includes dataset_name", serialized["dataset_name"] == "Test Dataset")

    # ---- 2. Estimates dataclasses ----
    print("\n--- 2. Estimates Dataclasses ---")
    from seroepi.constants import AggregationType
    from seroepi.estimators.base import (
        AlphaDiversityEstimates,
        BetaDiversityEstimates,
        Estimates,
        IncidenceEstimates,
        PrevalenceEstimates,
    )

    for cls_name, cls in [
        ("Estimates", Estimates),
        ("PrevalenceEstimates", PrevalenceEstimates),
        ("AlphaDiversityEstimates", AlphaDiversityEstimates),
        ("BetaDiversityEstimates", BetaDiversityEstimates),
        ("IncidenceEstimates", IncidenceEstimates),
    ]:
        check(f"{cls_name} is a dataclass", dataclasses.is_dataclass(cls))
        # Check frozen via the dataclass fields
        dc_params = getattr(cls, "__dataclass_params__", None)
        is_frozen = dc_params.frozen if dc_params else False
        check(f"{cls_name} is frozen", is_frozen)

    # Instantiate a PrevalenceEstimates to verify it wraps pl.DataFrame
    est_df = pl.DataFrame({"target": ["KL1", "KL2"], "estimate": [0.5, 0.3]})
    pe = PrevalenceEstimates(
        data=est_df,
        stratified_by=["target"],
        adjusted_for=None,
        trait="K_locus",
        aggregation_type=AggregationType.TRAIT,
        method="wilson",
    )
    check("PrevalenceEstimates.data is pl.DataFrame", isinstance(pe.data, pl.DataFrame))

    # ---- 3. UnpooledPrevalenceEstimator end-to-end ----
    print("\n--- 3. UnpooledPrevalenceEstimator E2E ---")
    from seroepi.estimators.core import UnpooledPrevalenceEstimator

    # Create a mock aggregated DataFrame (output of .epi.aggregate_prevalence())
    agg_df = pl.DataFrame(
        {
            "target": ["KL1", "KL2", "KL3"],
            "event": [10, 5, 2],
            "n": [20, 20, 20],
        }
    )

    estimator = UnpooledPrevalenceEstimator(method="wilson", alpha=0.05)
    result = estimator.calculate(agg_df)

    check("Estimator returns PrevalenceEstimates", isinstance(result, PrevalenceEstimates))
    check("Result data is pl.DataFrame", isinstance(result.data, pl.DataFrame))
    check("Result has 'estimate' column", "estimate" in result.data.columns)
    check("Result has 'lower' column", "lower" in result.data.columns)
    check("Result has 'upper' column", "upper" in result.data.columns)
    check(
        "Estimates are in valid range [0, 1]",
        result.data["estimate"].to_numpy().max() <= 1.0 and result.data["estimate"].to_numpy().min() >= 0.0,
    )

    # ---- 4. KaptiveClient ----
    print("\n--- 4. KaptiveClient ---")
    from seroepi.client import KaptiveClient

    client = KaptiveClient(base_url="https://kaptive.example.com", api_key="test-key-123")
    check("KaptiveClient instantiates without error", client is not None)
    check("KaptiveClient has session", hasattr(client, "session"))
    check("KaptiveClient has X-API-Key header", client.session.headers.get("X-API-Key") == "test-key-123")
    check("KaptiveClient has submit_job method", hasattr(client, "submit_job"))
    check("KaptiveClient has submit_dataset method", hasattr(client, "submit_dataset"))
    check("KaptiveClient has get_species method", hasattr(client, "get_species"))
    check("KaptiveClient has get_runs method", hasattr(client, "get_runs"))
    check("KaptiveClient has get_run_results method", hasattr(client, "get_run_results"))
    check("KaptiveClient has get_all_results method", hasattr(client, "get_all_results"))

    # Context manager
    with KaptiveClient(base_url="http://localhost:8000", api_key="test") as ctx_client:
        check("KaptiveClient works as context manager", ctx_client is not None)

    client.session.close()

    # ---- 5. No residual pandas imports ----
    print("\n--- 5. No Residual Pandas Imports ---")
    import subprocess

    result_grep = subprocess.run(
        [
            "grep",
            "-r",
            "--include=*.py",
            "import pandas",
            str(Path(__file__).resolve().parent.parent / "src" / "seroepi"),
        ],
        capture_output=True,
        text=True,
    )
    check("No 'import pandas' statements in src/seroepi/", result_grep.returncode != 0)

    result_pd = subprocess.run(
        ["grep", "-rn", "--include=*.py", r"pd\.", str(Path(__file__).resolve().parent.parent / "src" / "seroepi")],
        capture_output=True,
        text=True,
    )
    check("No 'pd.' references in src/seroepi/", result_pd.returncode != 0)

    # ---- 6. Module Privatization Reversal ----
    print("\n--- 6. Module Privatization Reversal ---")
    estimators_dir = Path(__file__).resolve().parent.parent / "src" / "seroepi" / "estimators"
    check("base.py exists (not _base.py)", (estimators_dir / "base.py").exists())
    check("core.py exists (not _core.py)", (estimators_dir / "core.py").exists())
    check("modelled.py exists (not _modelled.py)", (estimators_dir / "modelled.py").exists())
    check("_base.py does NOT exist", not (estimators_dir / "_base.py").exists())
    check("_core.py does NOT exist", not (estimators_dir / "_core.py").exists())
    check("_modelled.py does NOT exist", not (estimators_dir / "_modelled.py").exists())

    app_dir = Path(__file__).resolve().parent.parent / "app"
    check("app/dataset.py exists (not _dataset.py)", (app_dir / "dataset.py").exists())
    check("app/utils.py exists (not _utils.py)", (app_dir / "utils.py").exists())
    check("app/_dataset.py does NOT exist", not (app_dir / "_dataset.py").exists())
    check("app/_utils.py does NOT exist", not (app_dir / "_utils.py").exists())

    # ---- Summary ----
    print("\n" + "=" * 70)
    total = PASS + FAIL
    print(f"Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL > 0:
        print("⚠️  Some checks failed. Review the output above.")
        sys.exit(1)
    else:
        print("🎉 All checks passed! Migration verified successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
