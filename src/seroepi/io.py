"""Module for genotype file I/O and parsing using Polars and Patito."""

import datetime
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional

import patito as pt
import polars as pl
from pydantic import ConfigDict, create_model

from seroepi.constants import Domain, GenotypeFlavour, SpatialResolution, TemporalResolution


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    """Helper function to recursively flatten nested dictionaries."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _pl_dtype_to_python(dtype: pl.DataType) -> type:
    """Helper mapping Polars DataType to standard Python type for Patito models."""
    if dtype in (pl.Boolean,):
        return bool
    elif dtype in (pl.Float32, pl.Float64):
        return float
    elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return int
    elif dtype in (pl.Datetime,):
        return datetime.datetime
    elif dtype in (pl.Date,):
        return datetime.date
    elif dtype in (pl.Categorical, pl.Utf8, pl.String):
        return str
    elif isinstance(dtype, pl.List) or dtype == pl.List or (isinstance(dtype, type) and issubclass(dtype, pl.List)):
        inner = getattr(dtype, "inner", None)
        if inner is not None:
            return list[_pl_dtype_to_python(inner)]  # type: ignore
        return list[Any]
    elif dtype == pl.Null:
        return Any
    elif (
        isinstance(dtype, pl.Struct) or dtype == pl.Struct or (isinstance(dtype, type) and issubclass(dtype, pl.Struct))
    ):
        return dict[str, Any]
    elif dtype == pl.Object:
        return Any
    else:
        return str


# Schema & Data Stewardship --------------------------------------------------------------------------------------------
class SampleModel(pt.Model):
    """Patito model for validating and standardizing isolate datasets in Polars.

    This model ensures that all input data, whether from Pathogenwatch or user
    uploads, conforms to a unified structure for downstream analysis.

    Attributes:
        sample_id: Unique identifier for each isolate.
        latitude: Latitude coordinate (-90.0 to 90.0).
        longitude: Longitude coordinate (-180.0 to 180.0).
    """

    model_config = ConfigDict(extra="allow")

    sample_id: str = pt.Field(unique=True)
    latitude: float | None = pt.Field(ge=-90.0, le=90.0, default=None)
    longitude: float | None = pt.Field(ge=-180.0, le=180.0, default=None)

    @classmethod
    def get_valid_prefixes(cls) -> tuple[str, ...]:
        """Returns tuple of allowed column prefixes for dynamic domain validation."""
        return (
            f"{Domain.QC.value}_",
            f"{Domain.GENOTYPE.value}_",
            f"{Domain.PHENOTYPE.value}_",
            f"{Domain.AMR.value}_",
            f"{Domain.VIRULENCE.value}_",
            f"{Domain.TEMPORAL.value}_",
            f"{Domain.TEMPORAL_RES.value}_",
            f"{Domain.SPATIAL.value}_",
            f"{Domain.SPATIAL_RES.value}_",
            "meta_",
        )

    @classmethod
    def clean_and_coerce(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Enforces domain types and standardizes present core/domain columns without dropping custom columns."""
        core_cols = {"sample_id", "latitude", "longitude"}
        filtered_df = df

        # 1. Guarantee GPS columns are Float64 if present
        exprs = []
        if "latitude" in filtered_df.columns:
            exprs.append(pl.col("latitude").cast(pl.Float64, strict=False))

        if "longitude" in filtered_df.columns:
            exprs.append(pl.col("longitude").cast(pl.Float64, strict=False))

        # 2. Coerce sample_id to Utf8 / String if present
        if "sample_id" in filtered_df.columns:
            exprs.append(pl.col("sample_id").cast(pl.Utf8))

        # 3. Coerce domain-specific columns and validate resolution enums
        for col in filtered_df.columns:
            if col in core_cols:
                continue
            dtype = filtered_df.schema[col]
            if col.startswith(f"{Domain.TEMPORAL_RES.value}_"):
                non_null_vals = filtered_df.select(pl.col(col).drop_nulls().unique()).get_column(col).to_list()
                valid_choices = {v.value for v in TemporalResolution}
                invalid = [v for v in non_null_vals if str(v) not in valid_choices]
                if invalid:
                    raise ValueError(f"Invalid TemporalResolution values in '{col}': {invalid}")
                exprs.append(pl.col(col).cast(pl.Utf8))
            elif col.startswith(f"{Domain.SPATIAL_RES.value}_"):
                non_null_vals = filtered_df.select(pl.col(col).drop_nulls().unique()).get_column(col).to_list()
                valid_choices = {v.value for v in SpatialResolution}
                invalid = [v for v in non_null_vals if str(v) not in valid_choices]
                if invalid:
                    raise ValueError(f"Invalid SpatialResolution values in '{col}': {invalid}")
                exprs.append(pl.col(col).cast(pl.Utf8))
            elif col.startswith(f"{Domain.TEMPORAL.value}_"):
                if dtype in (pl.Datetime, pl.Date):
                    exprs.append(pl.col(col).cast(pl.Datetime))
                else:
                    s_col = pl.col(col).cast(pl.Utf8).str.strip_chars()
                    s_len = s_col.str.len_bytes()
                    date_str_expr = (
                        pl.when(s_len == 4)
                        .then(s_col + "-07-02")
                        .when((s_len >= 6) & (s_len <= 7) & s_col.str.contains(r"-|/"))
                        .then(s_col + "-15")
                        .otherwise(s_col)
                    )
                    exprs.append(date_str_expr.str.to_datetime(strict=False).alias(col))
            elif col.startswith(f"{Domain.SPATIAL.value}_"):
                exprs.append(pl.col(col).cast(pl.Utf8))
            elif dtype == pl.Categorical:
                exprs.append(pl.col(col).cast(pl.Utf8))

        if exprs:
            return filtered_df.with_columns(exprs)
        return filtered_df

    @classmethod
    def validate(cls, dataframe: pl.DataFrame, **kwargs: Any) -> pl.DataFrame:  # type: ignore # noqa: ANN401, D102
        """Pre-cleans the DataFrame and executes Patito validation."""
        if not isinstance(dataframe, pl.DataFrame):
            raise TypeError(f"Expected pl.DataFrame, got {type(dataframe).__name__}")

        cleaned_df = cls.clean_and_coerce(dataframe)

        field_definitions: dict[str, tuple[Any, Any]] = {}
        if "sample_id" in cleaned_df.columns:
            field_definitions["sample_id"] = (str, pt.Field(unique=True))
        if "latitude" in cleaned_df.columns:
            field_definitions["latitude"] = (Optional[float], pt.Field(ge=-90.0, le=90.0, default=None))  # noqa: UP045
        if "longitude" in cleaned_df.columns:
            field_definitions["longitude"] = (Optional[float], pt.Field(ge=-180.0, le=180.0, default=None))  # noqa: UP045

        for col in cleaned_df.columns:
            if col in field_definitions:
                continue
            dtype = cleaned_df.schema[col]
            py_type = _pl_dtype_to_python(dtype)
            if (
                isinstance(dtype, pl.Struct)
                or dtype == pl.Struct
                or (isinstance(dtype, type) and issubclass(dtype, pl.Struct))
            ):
                fields = getattr(dtype, "fields", [])
                if fields:
                    sub_defs = {f.name: (Optional[Any], pt.Field(default=None, dtype=f.dtype)) for f in fields}  # noqa: UP045
                    SubModel = create_model(f"DynamicStruct_{col}", __base__=pt.Model, **sub_defs)  # type: ignore
                    field_definitions[col] = (Optional[SubModel], pt.Field(default=None, dtype=dtype))  # noqa: UP045
                else:
                    field_definitions[col] = (py_type | None, pt.Field(default=None))
            else:
                field_definitions[col] = (py_type | None, pt.Field(default=None, dtype=dtype))

        DynamicModel: type[pt.Model] = create_model(  # type: ignore
            "DynamicSampleModel",
            __base__=pt.Model,
            **field_definitions,
        )

        validated_df = DynamicModel.validate(cleaned_df)
        return pl.DataFrame(validated_df)


# Parsers --------------------------------------------------------------------------------------------------------------
class BaseGenotypeParser:
    """Base class for standardizing external datasets using Polars.

    Subclasses must define column mappings and category definitions for specific
    input formats (e.g., Kleborate output).
    """

    column_map: dict[str, str] = {}
    qc_cols: list[str] = []
    vir_cols: list[str] = []
    amr_cols: list[str] = []
    geno_cols: list[str] = []
    pheno_cols: list[str] = []

    @classmethod
    def get_parser(cls, flavour: str | GenotypeFlavour):  # noqa: ANN206, D102
        flavour_val = flavour.value if isinstance(flavour, GenotypeFlavour) else flavour
        if flavour_val == "pathogenwatch-kleborate":
            return PathogenwatchKleborateParser
        return cls

    @staticmethod
    def _clean_mixed_dates(df: pl.DataFrame, date_col: str, res_col: str) -> pl.DataFrame:
        """Standardizes mixed-format dates (YYYY, YYYY-MM, YYYY-MM-DD) in Polars.

        Args:
            df: The Polars DataFrame to clean.
            date_col: The target prefixed temporal column name.
            res_col: The target prefixed resolution column name.

        Returns:
            The DataFrame with a standardized temporal column and resolution parallel.
        """
        if date_col not in df.columns:
            return df

        s_col = pl.col(date_col).cast(pl.Utf8).str.strip_chars()
        s_len = s_col.str.len_bytes()

        res_expr = (
            pl.when(s_len == 4)
            .then(pl.lit(TemporalResolution.YEAR.value))
            .when((s_len >= 6) & (s_len <= 7) & s_col.str.contains(r"-|/"))
            .then(pl.lit(TemporalResolution.MONTH.value))
            .when(s_len >= 8)
            .then(pl.lit(TemporalResolution.DAY.value))
            .otherwise(pl.lit(TemporalResolution.UNKNOWN.value))
        )

        date_str_expr = (
            pl.when(res_expr == TemporalResolution.YEAR.value)
            .then(s_col + "-07-02")
            .when(res_expr == TemporalResolution.MONTH.value)
            .then(s_col + "-15")
            .otherwise(s_col)
        )

        parsed_date = date_str_expr.str.to_datetime(strict=False)
        final_res = pl.when(parsed_date.is_null()).then(pl.lit(TemporalResolution.UNKNOWN.value)).otherwise(res_expr)

        return df.with_columns(
            [
                parsed_date.alias(date_col),
                final_res.cast(pl.Categorical).alias(res_col),
            ]
        )

    @classmethod
    def _ingest_user_metadata(
        cls,
        meta_df: pl.DataFrame,
        id_col: str,
        date_col: str | None = None,
        date_res: str | None = None,
        spatial_col: str | None = None,
        spatial_res: str | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> pl.DataFrame:
        """Standardizes user-uploaded metadata using Polars.

        Args:
            meta_df: The metadata Polars DataFrame.
            id_col: Column name for sample IDs.
            date_col: Column name for isolation dates.
            date_res: User-specified temporal resolution.
            spatial_col: Column name for the primary geographic level.
            spatial_res: User-specified spatial resolution.
            lat_col: Column name for latitudes.
            lon_col: Column name for longitudes.

        Returns:
            A cleaned metadata DataFrame with prefixed user columns.
        """
        df = meta_df.clone()
        rename_map = {id_col: "sample_id"}
        if lat_col:
            rename_map[lat_col] = "latitude"
        if lon_col:
            rename_map[lon_col] = "longitude"

        existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(existing_renames)

        exprs = []
        if "latitude" in df.columns:
            exprs.append(pl.col("latitude").cast(pl.Float64, strict=False))
        if "longitude" in df.columns:
            exprs.append(pl.col("longitude").cast(pl.Float64, strict=False))
        if exprs:
            df = df.with_columns(exprs)

        if date_col and date_col in df.columns:
            new_date_col = f"{Domain.TEMPORAL.value}_{date_col}"
            res_col = f"{Domain.TEMPORAL_RES.value}_{date_col}"
            df = df.rename({date_col: new_date_col})
            df = cls._clean_mixed_dates(df, date_col=new_date_col, res_col=res_col)
            if date_res and date_res != TemporalResolution.UNKNOWN.value:
                df = df.with_columns(pl.lit(date_res).cast(pl.Categorical).alias(res_col))

        if spatial_col and spatial_col in df.columns:
            new_spatial_col = f"{Domain.SPATIAL.value}_{spatial_col}"
            res_col = f"{Domain.SPATIAL_RES.value}_{spatial_col}"
            df = df.rename({spatial_col: new_spatial_col})
            s_res = (
                spatial_res
                if (spatial_res and spatial_res != SpatialResolution.UNKNOWN.value)
                else SpatialResolution.UNKNOWN.value
            )
            df = df.with_columns(pl.lit(s_res).cast(pl.Categorical).alias(res_col))

        core_prefixes = (
            f"{Domain.TEMPORAL.value}_",
            f"{Domain.TEMPORAL_RES.value}_",
            f"{Domain.SPATIAL.value}_",
            f"{Domain.SPATIAL_RES.value}_",
        )
        new_names = {
            col: f"meta_{col}"
            for col in df.columns
            if col not in ["sample_id", "latitude", "longitude"]
            and not col.startswith(core_prefixes)
            and not col.startswith("meta_")
        }
        return df.rename(new_names)

    @staticmethod
    def _optimize_categorical_dtypes(df: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
        """Converts string columns to Categorical dtype if cardinality is low.

        Args:
            df: The Polars DataFrame to optimize.
            threshold: The ratio of unique values to total rows below which
                a column is converted to categorical. Defaults to 0.5.

        Returns:
            The optimized Polars DataFrame.
        """
        total_rows = len(df)
        if total_rows == 0:
            return df

        exprs = []
        for col, dtype in df.schema.items():
            if dtype in (pl.Utf8, pl.String):
                num_unique = df.select(pl.col(col).n_unique()).item()
                if (num_unique / total_rows) < threshold and num_unique < total_rows:
                    exprs.append(pl.col(col).cast(pl.Categorical))
        if exprs:
            return df.with_columns(exprs)
        return df

    @staticmethod
    def _optimize_binary_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        """Converts numeric columns containing only 0 and 1 to Int8.

        Args:
            df: The Polars DataFrame to optimize.

        Returns:
            The optimized Polars DataFrame.
        """
        exprs = []
        for col, dtype in df.schema.items():
            if dtype in (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            ):
                unique_vals = df.select(pl.col(col).drop_nulls().unique()).get_column(col).to_list()
                if unique_vals and set(unique_vals).issubset({0, 1}):
                    exprs.append(pl.col(col).cast(pl.Int8))
        if exprs:
            return df.with_columns(exprs)
        return df

    @classmethod
    def from_files(
        cls,
        genotype_path: str | Path,
        meta_path: str | Path | None = None,
        meta_kwargs: dict[str, Any] | None = None,
        dataset_name: str = "Unknown Dataset",
    ) -> pl.DataFrame:
        """Convenience factory to read CSV files with Polars and parse them."""
        genotype_df = pl.read_csv(genotype_path, infer_schema_length=10000)
        meta_df = None
        if meta_path is not None:
            meta_df = pl.read_csv(meta_path, infer_schema_length=10000)

        return cls.parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        meta_df: pl.DataFrame | None = None,
        meta_kwargs: dict[str, Any] | None = None,
        sep: str = "/",
        dataset_name: str = "Unknown Dataset",
    ) -> pl.DataFrame:
        """Convenience factory to read nested dictionaries and parse them with Polars."""
        flattened = [_flatten_dict(r, sep=sep) for r in records]
        genotype_df = pl.from_dicts(flattened)
        return cls.parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)

    @classmethod
    def parse(
        cls,
        genotype_df: pl.DataFrame,
        meta_df: pl.DataFrame | None = None,
        meta_kwargs: dict[str, Any] | None = None,
        dataset_name: str = "Unknown Dataset",
    ) -> pl.DataFrame:
        """Parses and validates a genotype dataset, optionally merging with metadata.

        Args:
            genotype_df: The raw genotype Polars DataFrame.
            meta_df: Optional metadata Polars DataFrame to merge.
            meta_kwargs: Arguments for metadata ingestion.
            dataset_name: Name to tag the resulting dataset with.

        Returns:
            A validated Polars DataFrame conforming to SampleModel.
        """
        df = genotype_df.clone()
        rename_map = {k: v for k, v in cls.column_map.items() if k in df.columns}
        df = df.rename(rename_map)

        new_names = {}
        for col in df.columns:
            if col in cls.qc_cols and not col.startswith(f"{Domain.QC.value}_"):
                new_names[col] = f"{Domain.QC.value}_{col}"
            elif col in cls.vir_cols and not col.startswith(f"{Domain.VIRULENCE.value}_"):
                new_names[col] = f"{Domain.VIRULENCE.value}_{col}"
            elif col in cls.amr_cols and not col.startswith(f"{Domain.AMR.value}_"):
                new_names[col] = f"{Domain.AMR.value}_{col}"
            elif col in cls.geno_cols and not col.startswith(f"{Domain.GENOTYPE.value}_"):
                new_names[col] = f"{Domain.GENOTYPE.value}_{col}"
            elif col in cls.pheno_cols and not col.startswith(f"{Domain.PHENOTYPE.value}_"):
                new_names[col] = f"{Domain.PHENOTYPE.value}_{col}"

        if new_names:
            df = df.rename(new_names)

        if meta_df is not None:
            kwargs = meta_kwargs or {}
            clean_meta = cls._ingest_user_metadata(meta_df, **kwargs)
            overlap_cols = [c for c in clean_meta.columns if c in df.columns and c != "sample_id"]
            if overlap_cols:
                clean_meta = clean_meta.drop(overlap_cols)
            df = df.join(clean_meta, on="sample_id", how="left")
        # Ensure latitude and longitude columns exist for raw isolate parser
        gps_exprs = []
        if "latitude" not in df.columns:
            gps_exprs.append(pl.lit(None).cast(pl.Float64).alias("latitude"))
        if "longitude" not in df.columns:
            gps_exprs.append(pl.lit(None).cast(pl.Float64).alias("longitude"))
        if gps_exprs:
            df = df.with_columns(gps_exprs)

        # 1. Patito validation and cleaning
        valid_df = SampleModel.validate(df)

        # AUTOMATIC REVERSE GEOCODING
        from seroepi.dataset import SeroEpiDataset

        spatial_cols = [
            c
            for c in valid_df.columns
            if c.startswith(f"{Domain.SPATIAL.value}_") and not c.startswith(f"{Domain.SPATIAL_RES.value}_")
        ]
        has_coords = False
        if "latitude" in valid_df.columns and "longitude" in valid_df.columns:
            has_coords = valid_df.select(
                pl.col("latitude").is_not_null().any() & pl.col("longitude").is_not_null().any()
            ).item()

        ds = SeroEpiDataset(data=valid_df)
        if not spatial_cols and has_coords:
            ds = ds.geo.reverse_geocode()

        # 2. CUSTOM OPTIMIZATIONS
        ds = ds.geo.standardize_and_impute()
        valid_df = ds.data
        valid_df = cls._optimize_binary_dtypes(valid_df)
        valid_df = cls._optimize_categorical_dtypes(valid_df, threshold=0.4)

        return valid_df


class PathogenwatchKleborateParser(BaseGenotypeParser):
    """Adapter for Kleborate files downloaded from Pathogenwatch."""

    column_map = {
        "Genome Name": "sample_id",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Country": f"{Domain.SPATIAL.value}_Country",
        "Region": f"{Domain.SPATIAL.value}_Region",
        "Continent": f"{Domain.SPATIAL.value}_Continent",
        "Collection Date": f"{Domain.TEMPORAL.value}_Collection_Date",
        "Year": f"{Domain.TEMPORAL.value}_Year",
        "Month": f"{Domain.TEMPORAL.value}_Month",
        "Day": f"{Domain.TEMPORAL.value}_Day",
    }
    geno_cols = ["ST", "K_locus", "O_locus"]
    pheno_cols = ["K_type", "O_type"]
    qc_cols = [
        "species",
        "species_match",
        "contig_count",
        "N50",
        "largest_contig",
        "total_size",
        "ambiguous_bases",
        "QC_warnings",
        "K_locus_confidence",
        "O_locus_confidence",
    ]
    vir_cols = [
        "YbST",
        "Yersiniabactin",
        "CbST",
        "Colibactin",
        "AbST",
        "Aerobactin",
        "SmST",
        "Salmochelin",
        "RmST",
        "RmpADC",
        "virulence_score",
        "rmpA2",
    ]
    amr_cols = [
        "AGly_acquired",
        "Col_acquired",
        "Fcyn_acquired",
        "Flq_acquired",
        "Gly_acquired",
        "MLS_acquired",
        "Phe_acquired",
        "Rif_acquired",
        "Sul_acquired",
        "Tet_acquired",
        "Tgc_acquired",
        "Tmt_acquired",
        "Bla_acquired",
        "Bla_inhR_acquired",
        "Bla_ESBL_acquired",
        "Bla_ESBL_inhR_acquired",
        "Bla_Carb_acquired",
        "Bla_chr",
        "SHV_mutations",
        "Omp_mutations",
        "Col_mutations",
        "Flq_mutations",
        "resistance_score",
        "num_resistance_classes",
        "num_resistance_genes",
        "Ciprofloxacin_prediction",
        "Ciprofloxacin_profile",
        "Ciprofloxacin_MIC_prediction",
    ]


class PathogenwatchGenomeParser(BaseGenotypeParser):
    """Adapter for JSON genome records from Pathogenwatch."""

    column_map = {
        "name": "sample_id",
        "Capsule_type": "K_type",
        "latitude": "latitude",
        "longitude": "longitude",
        "country": f"{Domain.SPATIAL.value}_Country",
        "region": f"{Domain.SPATIAL.value}_Region",
        "continent": f"{Domain.SPATIAL.value}_Continent",
        "collectionDate": f"{Domain.TEMPORAL.value}_Collection_Date",
        "year": f"{Domain.TEMPORAL.value}_Year",
        "month": f"{Domain.TEMPORAL.value}_Month",
        "day": f"{Domain.TEMPORAL.value}_Day",
    }
    geno_cols = ["ST", "LIN_code", "Inc_Types", "K_locus", "OC_locus"]
    pheno_cols = ["K_type"]
    qc_cols = [
        "QC",
        "Genome_length",
        "No._contigs",
        "Largest_contig",
        "Average_contig_length",
        "N50",
        "N's_per_100_kbp",
        "GC_content",
        "Confidence",
    ]
    vir_cols = [
        "Virulence_score",
        "Aerobactin_(AbST)",
        "Colibactin_(CbST)",
        "Salmochelin_(SmST)",
        "Yersiniabactin_(YbST)",
        "RmpADC",
        "rmpA2",
    ]
    amr_cols = [
        "Aminoglycosides",
        "Carbapenems",
        "Cephalosporins_(3rd_gen.)",
        "Cephalosporins_(3rd_gen.)_+_β-lactamase_inhibitors",
        "Colistin",
        "Fluoroquinolones",
        "Fosfomycin",
        "Glycopeptides",
        "Macrolides",
        "OmpK36",
        "Penicillins",
        "Penicillins_+_β-lactamase_inhibitors",
        "Phenicols",
        "Rifampicin",
        "SHV_variants",
        "Sulfonamides",
        "Tetracycline",
        "Tigecycline",
        "Trimethoprim",
    ]

    @classmethod
    def parse(  # noqa: D102
        cls,
        genotype_df: pl.DataFrame,
        meta_df: pl.DataFrame | None = None,
        meta_kwargs: dict[str, Any] | None = None,
        dataset_name: str = "Unknown Dataset",
    ) -> pl.DataFrame:
        rename_map = {c: str(c).split("/")[-1].replace(" ", "_") for c in genotype_df.columns}
        genotype_df = genotype_df.rename(rename_map)

        unique_cols = []
        seen = set()
        for c in genotype_df.columns:
            if c not in seen:
                seen.add(c)
                unique_cols.append(c)
        genotype_df = genotype_df.select(unique_cols)

        return super().parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)


class KaptiveWebJSONParser(BaseGenotypeParser):
    """Adapter for Kaptive Web JSON run results."""

    @classmethod
    def from_run_results(
        cls,
        run_results: dict[str, Any],
        meta_df: pl.DataFrame | None = None,
        meta_kwargs: dict[str, Any] | None = None,
        dataset_name: str = "Unknown Dataset",
    ) -> pl.DataFrame:
        """Parses Kaptive-Web JSON responses into a Polars DataFrame."""
        records = []
        results_dict = run_results.get("results", {})

        for genome_id, dbs in results_dict.items():
            row = {"sample_id": genome_id}
            for db_key, db_result in dbs.items():
                if not isinstance(db_result, dict):
                    continue

                is_k_locus = db_key.endswith("_k_locus")
                is_o_locus = db_key.endswith("_o_locus")

                # Flatten nested problems/hits arrays to strings if needed
                flat_result = _flatten_dict(db_result, sep="_")

                for k, v in flat_result.items():
                    if k == "genome":
                        continue
                    if k == "phenotype":
                        if is_k_locus:
                            row["K_type"] = v
                        elif is_o_locus:
                            row["O_type"] = v
                        else:
                            row[f"{db_key}_{k}"] = v
                    elif k == "best_locus_name":
                        if is_k_locus:
                            row["K_locus"] = v
                        elif is_o_locus:
                            row["O_locus"] = v
                        else:
                            row[f"{db_key}_{k}"] = v
                    else:
                        row[f"{db_key}_{k}"] = str(v) if isinstance(v, (list, dict)) else v
            records.append(row)

        genotype_df = pl.from_dicts(records) if records else pl.DataFrame({"sample_id": []})
        return cls.parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)


class KaptiveNativeParser(BaseGenotypeParser):
    """Adapter for native Kaptive SerotypingResult objects."""

    @classmethod
    def from_results(
        cls,
        results: Iterable[Any],
        meta_df: pl.DataFrame | None = None,
        meta_kwargs: dict[str, Any] | None = None,
        dataset_name: str = "Unknown Dataset",
    ) -> pl.DataFrame:
        """Parses native Kaptive SerotypingResult objects into a Polars DataFrame."""
        import re

        grouped_results: dict[str, dict[str, dict[str, Any]]] = {}
        for result in results:
            genome_id = result.genome
            if genome_id not in grouped_results:
                grouped_results[genome_id] = {}

            db_name = result.database_name
            db_key = re.sub(r"[\s\-]+", "_", db_name.strip()).lower()
            grouped_results[genome_id][db_key] = result.to_dict()

        records = []
        for genome_id, dbs in grouped_results.items():
            row = {"sample_id": genome_id}
            for db_key, db_result in dbs.items():
                if not isinstance(db_result, dict):
                    continue

                is_k_locus = db_key.endswith("_k_locus")
                is_o_locus = db_key.endswith("_o_locus")

                flat_result = _flatten_dict(db_result, sep="_")

                for k, v in flat_result.items():
                    if k in ("genome", "database_name"):
                        continue
                    if k == "phenotype":
                        if is_k_locus:
                            row["K_type"] = v
                        elif is_o_locus:
                            row["O_type"] = v
                        else:
                            row[f"{db_key}_{k}"] = v
                    elif k == "best_locus_name":
                        if is_k_locus:
                            row["K_locus"] = v
                        elif is_o_locus:
                            row["O_locus"] = v
                        else:
                            row[f"{db_key}_{k}"] = v
                    else:
                        row[f"{db_key}_{k}"] = str(v) if isinstance(v, (list, dict)) else v
            records.append(row)

        genotype_df = pl.from_dicts(records) if records else pl.DataFrame({"sample_id": []})
        return cls.parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)
