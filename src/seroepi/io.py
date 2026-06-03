"""
Module for genotype file I/O and parsing.
"""
from pathlib import Path
from typing import Optional, Union
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series
from seroepi.constants import TemporalResolution, SpatialResolution, Domain, GenotypeFlavour


# Schema & Data Stewardship --------------------------------------------------------------------------------------------
class UnifiedIsolateSchema(pa.DataFrameModel):
    """
    Pandera schema for validating and standardizing isolate datasets.

    This schema ensures that all input data, whether from Pathogenwatch or user
    uploads, conforms to a unified structure for downstream analysis.

    Attributes:
        sample_id: Unique identifier for each isolate.
        latitude: Latitude coordinate (-90 to 90).
        longitude: Longitude coordinate (-180 to 180).
        qc_metrics: Dynamic columns for quality control (prefixed with 'qc_').
        geno_traits: Dynamic columns for genotypes/alleles (prefixed with 'geno_').
        pheno_traits: Dynamic columns for phenotypic traits (prefixed with 'pheno_').
        amr_traits: Dynamic columns for AMR markers (prefixed with 'amr_').
        vir_traits: Dynamic columns for virulence markers (prefixed with 'vir_').
        temporal_cols: Dynamic columns for temporal data (prefixed with 'temporal_').
        temporal_res_cols: Dynamic columns for temporal resolution (prefixed with 'temporal_res_').
        spatial_cols: Dynamic columns for spatial data (prefixed with 'spatial_').
        spatial_res_cols: Dynamic columns for spatial resolution (prefixed with 'spatial_res_').
        user_metadata: Dynamic columns for user metadata (prefixed with 'meta_').

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'sample_id': ['S1'], 'K_locus': ['KL1']})
        >>> validated_df = UnifiedIsolateSchema.validate(df)
    """
    # Core
    sample_id: Series["string"] = pa.Field(unique=True, coerce=True)

    # Raw GPS Fields
    latitude: Optional[Series["Float64"]] = pa.Field(ge=-90, le=90, nullable=True, coerce=True)
    longitude: Optional[Series["Float64"]] = pa.Field(ge=-180, le=180, nullable=True, coerce=True)

    class Config:
        strict = "filter"

    @classmethod
    def to_schema(cls) -> pa.DataFrameSchema:
        schema = super().to_schema()
        # Dynamically attach regex fields using the Object-based API to bypass the DataFrameModel bug
        # where Optional[] is ignored for regex fields.
        return schema.add_columns({
            f"^{Domain.QC.value}_.*$": pa.Column(regex=True, required=False, nullable=True),
            f"^{Domain.GENOTYPE.value}_.*$": pa.Column(regex=True, required=False, nullable=True),
            f"^{Domain.PHENOTYPE.value}_.*$": pa.Column(regex=True, required=False, nullable=True),
            f"^{Domain.AMR.value}_.*$": pa.Column(regex=True, required=False, nullable=True),
            f"^{Domain.VIRULENCE.value}_.*$": pa.Column(regex=True, required=False, nullable=True),
            f"^{Domain.TEMPORAL.value}_(?!res_).*$": pa.Column("datetime64[ns]", regex=True, required=False, nullable=True, coerce=True),
            f"^{Domain.TEMPORAL_RES.value}_.*$": pa.Column("category", regex=True, required=False, nullable=True, coerce=True, checks=pa.Check.isin(TemporalResolution.choices())),
            f"^{Domain.SPATIAL.value}_(?!res_).*$": pa.Column("string", regex=True, required=False, nullable=True, coerce=True),
            f"^{Domain.SPATIAL_RES.value}_.*$": pa.Column("category", regex=True, required=False, nullable=True, coerce=True, checks=pa.Check.isin(SpatialResolution.choices())),
            "^meta_.*$": pa.Column(regex=True, required=False, nullable=True)
        })


# Parsers --------------------------------------------------------------------------------------------------------------
class BaseGenotypeParser:
    """
    Base class for standardizing external datasets.

    Subclasses must define column mappings and category definitions for specific
    input formats (e.g., Kleborate output).
    """

    # Subclasses define how to map raw columns to UnifiedIsolateSchema columns
    column_map: dict[str, str] = {}
    qc_cols: list[str] = []
    vir_cols: list[str] = []
    amr_cols: list[str] = []
    geno_cols: list[str] = []
    pheno_cols: list[str] = []

    @classmethod
    def get_parser(cls, flavour: Union[str, GenotypeFlavour]):
        flavour_val = flavour.value if isinstance(flavour, GenotypeFlavour) else flavour
        if flavour_val == "pathogenwatch-kleborate":
            return PathogenwatchKleborateParser
        return cls

    @staticmethod
    def _clean_mixed_dates(df: pd.DataFrame, date_col: str, res_col: str) -> pd.DataFrame:
        """
        Standardizes mixed-format dates (YYYY, YYYY-MM, YYYY-MM-DD).

        Args:
            df: The DataFrame to clean.
            date_col: The target prefixed temporal column name.
            res_col: The target prefixed resolution column name.

        Returns:
            The DataFrame with a standardized temporal column and resolution parallel.
        """
        if date_col not in df.columns:
            return df
        s = df[date_col].copy()
        s = s.astype(str).str.strip()
        s = s.replace(['nan', '<NA>', 'None', ''], pd.NA)
        lengths = s.str.len()

        df[res_col] = TemporalResolution.UNKNOWN.value
        df.loc[lengths == 4, res_col] = TemporalResolution.YEAR.value
        df.loc[(lengths >= 6) & (lengths <= 7) & s.str.contains(r'-|/', na=False), res_col] = TemporalResolution.MONTH.value
        df.loc[lengths >= 8, res_col] = TemporalResolution.DAY.value

        is_year = df[res_col] == TemporalResolution.YEAR.value
        is_month = df[res_col] == TemporalResolution.MONTH.value

        s.loc[is_year] = s.loc[is_year] + '-07-02'
        s.loc[is_month] = s.loc[is_month] + '-15'

        df[date_col] = pd.to_datetime(s, errors='coerce', format='mixed')
        df.loc[df[date_col].isna(), res_col] = TemporalResolution.UNKNOWN.value
        return df

    @classmethod
    def _ingest_user_metadata(cls, meta_df: pd.DataFrame, id_col: str,
                              date_col: str = None, date_res: str = None, spatial_col: str = None,
                              spatial_res: str = None, lat_col: str = None, lon_col: str = None) -> pd.DataFrame:
        """
        Standardizes user-uploaded metadata.

        Args:
            meta_df: The metadata DataFrame.
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
        df = meta_df.copy()
        rename_map = {id_col: 'sample_id'}
        if lat_col: rename_map[lat_col] = 'latitude'
        if lon_col: rename_map[lon_col] = 'longitude'

        df = df.rename(columns=rename_map)

        if 'latitude' in df.columns:
            df['latitude'] = df['latitude'].astype("Float64")
        if 'longitude' in df.columns:
            df['longitude'] = df['longitude'].astype("Float64")

        # Safely prefix the temporal data while keeping the user's name intact
        if date_col and date_col in df.columns:
            new_date_col = f"{Domain.TEMPORAL.value}_{date_col}"
            res_col = f"{Domain.TEMPORAL_RES.value}_{date_col}"
            df = df.rename(columns={date_col: new_date_col})
            df = cls._clean_mixed_dates(df, date_col=new_date_col, res_col=res_col)
            if date_res and date_res != TemporalResolution.UNKNOWN.value:
                df[res_col] = date_res

        # Safely prefix the spatial data while keeping the user's name intact
        if spatial_col and spatial_col in df.columns:
            new_spatial_col = f"{Domain.SPATIAL.value}_{spatial_col}"
            res_col = f"{Domain.SPATIAL_RES.value}_{spatial_col}"
            df = df.rename(columns={spatial_col: new_spatial_col})
            if spatial_res and spatial_res != SpatialResolution.UNKNOWN.value:
                df[res_col] = spatial_res
            else:
                df[res_col] = SpatialResolution.UNKNOWN.value

        core_prefixes = (f"{Domain.TEMPORAL.value}_", f"{Domain.TEMPORAL_RES.value}_", 
                         f"{Domain.SPATIAL.value}_", f"{Domain.SPATIAL_RES.value}_")
        new_names = {col: f"meta_{col}" for col in df.columns if col not in ['sample_id', 'latitude', 'longitude'] and not col.startswith(core_prefixes)}
        return df.rename(columns=new_names)

    @staticmethod
    def _optimize_categorical_dtypes(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Converts string columns to categorical dtypes if cardinality is low.

        Args:
            df: The DataFrame to optimize.
            threshold: The ratio of unique values to total rows below which
                a column is converted to categorical. Defaults to 0.5.

        Returns:
            The optimized DataFrame.
        """
        df_opt = df.copy()
        total_rows = len(df_opt)
        if total_rows == 0:
            return df_opt

        string_cols = df_opt.select_dtypes(include=['object', 'string', 'category']).columns
        for col in string_cols:
            is_cat = isinstance(df_opt[col].dtype, pd.CategoricalDtype)
            if is_cat:
                if not df_opt[col].cat.ordered:
                    df_opt[col] = df_opt[col].cat.as_ordered()
            else:
                num_unique = df_opt[col].nunique(dropna=False)
                if (num_unique / total_rows) < threshold and num_unique < total_rows:
                    df_opt[col] = df_opt[col].astype('category').cat.as_ordered()
        return df_opt

    @staticmethod
    def _optimize_binary_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts numeric columns containing only 0 and 1 to Int8.

        Args:
            df: The DataFrame to optimize.

        Returns:
            The optimized DataFrame.
        """
        df_opt = df.copy()
        for col in df_opt.columns:
            if pd.api.types.is_numeric_dtype(df_opt[col]):
                unique_vals = df_opt[col].dropna().unique()
                if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    df_opt[col] = df_opt[col].astype('Int8')
        return df_opt

    @classmethod
    def from_files(
            cls,
            genotype_path: Union[str, Path],
            meta_path: Optional[Union[str, Path]] = None,
            meta_kwargs: dict = None,
            dataset_name: str = "Unknown Dataset"
    ) -> pd.DataFrame:
        """
        Convenience factory to read CSV files and parse them.
        
        Args:
            genotype_path: Path to the raw genotype CSV.
            meta_path: Optional path to the metadata CSV.
            meta_kwargs: Arguments for metadata ingestion.
            dataset_name: Name to tag the resulting dataset with.
        """
        genotype_df = pd.read_csv(genotype_path, engine="pyarrow")
        meta_df = None
        
        if meta_path is not None:
            meta_df = pd.read_csv(meta_path, engine="pyarrow")
            
        return cls.parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)

    @classmethod
    def from_records(
            cls,
            records: list[dict],
            meta_df: Optional[pd.DataFrame] = None,
            meta_kwargs: dict = None,
            sep: str = '/',
            dataset_name: str = "Unknown Dataset"
    ) -> pd.DataFrame:
        """
        Convenience factory to read a list of nested dictionaries (e.g., from an API) and parse them.
        
        Args:
            records: List of nested dictionaries.
            meta_df: Optional metadata DataFrame to merge.
            meta_kwargs: Arguments for metadata ingestion.
            sep: Separator for flattening nested JSON keys. Defaults to '/'.
            dataset_name: Name to tag the resulting dataset with.
        """
        genotype_df = pd.json_normalize(records, sep=sep)
        return cls.parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)

    @classmethod
    def parse(
            cls,
            genotype_df: pd.DataFrame,
            meta_df: Optional[pd.DataFrame] = None,
            meta_kwargs: dict = None,
            dataset_name: str = "Unknown Dataset"
    ) -> pd.DataFrame:
        """
        Parses and validates a genotype dataset, optionally merging with metadata.

        Args:
            genotype_df: The raw genotype DataFrame.
            meta_df: Optional metadata DataFrame to merge.
            meta_kwargs: Arguments for metadata ingestion (e.g., column names).
            dataset_name: Name to tag the resulting dataset with.

        Returns:
            A validated DataFrame conforming to UnifiedIsolateSchema.
        """
        attrs = genotype_df.attrs
        df = genotype_df.copy()
        df = df.rename(columns=cls.column_map)
        new_names = {}
        for col in df.columns:
            if col in cls.qc_cols:
                new_names[col] = f"{Domain.QC.value}_{col}"
            elif col in cls.vir_cols:
                new_names[col] = f"{Domain.VIRULENCE.value}_{col}"
            elif col in cls.amr_cols:
                new_names[col] = f"{Domain.AMR.value}_{col}"
            elif col in cls.geno_cols:
                new_names[col] = f"{Domain.GENOTYPE.value}_{col}"
            elif col in cls.pheno_cols:
                new_names[col] = f"{Domain.PHENOTYPE.value}_{col}"

        df = df.rename(columns=new_names)


        if meta_df is not None:
            kwargs = meta_kwargs or {}
            # Use the namespaced schema method
            clean_meta = cls._ingest_user_metadata(meta_df, **kwargs)
            overlap_cols = [col for col in clean_meta.columns
                            if col in df.columns and col != 'sample_id']
            df = pd.merge(df.drop(columns=overlap_cols), clean_meta,
                          on='sample_id', how='left')

        # 1. PANDERA VALIDATION FIRST
        # This ensures schema coercions (like Float64, datetime) are safely applied
        # before we aggressively downcast memory with custom categories.
        valid_df = UnifiedIsolateSchema.validate(df)

        # Guarantee GPS columns exist post-validation so accessors can safely impute
        # (This bypasses the pd.NA evaluation crash during Pandera's ge/le checks)
        if 'latitude' not in valid_df.columns:
            valid_df['latitude'] = pd.NA
        valid_df['latitude'] = valid_df['latitude'].astype("Float64")
        
        if 'longitude' not in valid_df.columns:
            valid_df['longitude'] = pd.NA
        valid_df['longitude'] = valid_df['longitude'].astype("Float64")

        # AUTOMATIC REVERSE GEOCODING
        # If the dataset has coordinates but lacks spatial regions, infer them!
        spatial_cols = valid_df.filter(regex=f"^{Domain.SPATIAL.value}_(?!res_)").columns.tolist()
        has_coords = valid_df['latitude'].notna().any() and valid_df['longitude'].notna().any()
        
        if not spatial_cols and has_coords:
            valid_df = valid_df.geo.reverse_geocode()

        # 2. CUSTOM OPTIMIZATIONS
        valid_df = valid_df.geo.standardize_and_impute()
        valid_df = cls._optimize_binary_dtypes(valid_df)
        valid_df = cls._optimize_categorical_dtypes(valid_df, threshold=0.4)
        
        # Pandera strips attrs on validation/copying, so attach them right at the end
        valid_df.attrs = attrs
        valid_df.attrs["dataset_name"] = dataset_name

        return valid_df


class PathogenwatchKleborateParser(BaseGenotypeParser):
    """
    Adapter for Kleborate files downloaded from Pathogenwatch.

    This parser maps Kleborate's specific column names and categories to the
    UnifiedIsolateSchema.
    """
    column_map = {
        'Genome Name': 'sample_id',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'Country': f'{Domain.SPATIAL.value}_Country',
        'Region': f'{Domain.SPATIAL.value}_Region',
        'Continent': f'{Domain.SPATIAL.value}_Continent',
        'Collection Date': f'{Domain.TEMPORAL.value}_Collection_Date',
        'Year': f'{Domain.TEMPORAL.value}_Year',
        'Month': f'{Domain.TEMPORAL.value}_Month',
        'Day': f'{Domain.TEMPORAL.value}_Day'
    }
    geno_cols = ['ST', 'K_locus', 'O_locus']
    pheno_cols = ['K_type', 'O_type']
    qc_cols = ['species', 'species_match', 'contig_count', 'N50', 'largest_contig', 'total_size', 'ambiguous_bases',
               'QC_warnings', 'K_locus_confidence', 'O_locus_confidence']
    vir_cols = ['YbST', 'Yersiniabactin', 'CbST', 'Colibactin', 'AbST', 'Aerobactin', 'SmST', 'Salmochelin', 'RmST',
                'RmpADC', 'virulence_score', 'rmpA2']
    amr_cols = ['AGly_acquired', 'Col_acquired', 'Fcyn_acquired', 'Flq_acquired', 'Gly_acquired', 'MLS_acquired',
                'Phe_acquired', 'Rif_acquired', 'Sul_acquired', 'Tet_acquired', 'Tgc_acquired', 'Tmt_acquired',
                'Bla_acquired', 'Bla_inhR_acquired', 'Bla_ESBL_acquired', 'Bla_ESBL_inhR_acquired', 'Bla_Carb_acquired',
                'Bla_chr', 'SHV_mutations', 'Omp_mutations', 'Col_mutations', 'Flq_mutations', 'resistance_score',
                'num_resistance_classes', 'num_resistance_genes', 'Ciprofloxacin_prediction', 'Ciprofloxacin_profile',
                'Ciprofloxacin_MIC_prediction']


class PathogenwatchGenomeParser(BaseGenotypeParser):
    column_map = {
        'name': 'sample_id',
        'Capsule_type': 'K_type',
        'latitude': 'latitude',
        'longitude': 'longitude',
        'country': f'{Domain.SPATIAL.value}_Country',
        'region': f'{Domain.SPATIAL.value}_Region',
        'continent': f'{Domain.SPATIAL.value}_Continent',
        'collectionDate': f'{Domain.TEMPORAL.value}_Collection_Date',
        'year': f'{Domain.TEMPORAL.value}_Year',
        'month': f'{Domain.TEMPORAL.value}_Month',
        'day': f'{Domain.TEMPORAL.value}_Day'
    }
    geno_cols = ['ST', 'LIN_code', 'Inc_Types', 'K_locus', 'OC_locus']
    pheno_cols = ['K_type']
    
    qc_cols = [
        'QC',
        'Genome_length',
        'No._contigs',
        'Largest_contig',
        'Average_contig_length',
        'N50',
        "N's_per_100_kbp",
        'GC_content',
        'Confidence'
    ]
    vir_cols = [
        'Virulence_score',
        'Aerobactin_(AbST)',
        'Colibactin_(CbST)',
        'Salmochelin_(SmST)',
        'Yersiniabactin_(YbST)',
        'RmpADC',
        'rmpA2'
    ]
    amr_cols = [
        'Aminoglycosides',
        'Carbapenems',
        'Cephalosporins_(3rd_gen.)',
        'Cephalosporins_(3rd_gen.)_+_β-lactamase_inhibitors',
        'Colistin',
        'Fluoroquinolones',
        'Fosfomycin',
        'Glycopeptides',
        'Macrolides',
        'OmpK36',
        'Penicillins',
        'Penicillins_+_β-lactamase_inhibitors',
        'Phenicols',
        'Rifampicin',
        'SHV_variants',
        'Sulfonamides',
        'Tetracycline',
        'Tigecycline',
        'Trimethoprim'
    ]

    @classmethod
    def parse(
            cls,
            genotype_df: pd.DataFrame,
            meta_df: Optional[pd.DataFrame] = None,
            meta_kwargs: dict = None,
            dataset_name: str = "Unknown Dataset"
    ) -> pd.DataFrame:
        # Clean column names to remove JSON paths and replace spaces
        genotype_df = genotype_df.rename(columns=lambda c: str(c).split('/')[-1].replace(' ', '_'))
        
        # Flattening nested paths often creates duplicate column names (e.g. multiple 'Confidence' columns).
        # We drop the duplicates so Pandas doesn't return DataFrames when we select a single column string.
        genotype_df = genotype_df.loc[:, ~genotype_df.columns.duplicated()]
        
        # Identify mapped/known columns to avoid prefixing them
        # known_cols = set(cls.column_map.keys()).union(
        #     cls.geno_cols, cls.pheno_cols, cls.qc_cols, cls.vir_cols, cls.amr_cols
        # )
        
        # # Assign all unknown columns as metadata so the schema doesn't discard them
        # meta_renames = {col: f"meta_{col}" for col in genotype_df.columns if col not in known_cols}
        # genotype_df = genotype_df.rename(columns=meta_renames)

        return super().parse(genotype_df, meta_df=meta_df, meta_kwargs=meta_kwargs, dataset_name=dataset_name)