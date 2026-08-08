"""Centralized trait definitions and UI choice helpers for SeroEpi."""

from enum import StrEnum
from typing import Any, Self

__all__ = [
    "ChoiceEnum",
    "CoreTrait",
    "KpAmrTrait",
    "KpSeroTrait",
    "KpVirulenceTrait",
]


class ChoiceEnum(StrEnum):
    """Base class for string enums providing UI choice dictionary and label list helpers."""

    @classmethod
    def ui_labels(cls) -> dict[Any, str]:
        """Returns a dictionary mapping enum members to human-readable UI labels."""
        labels: dict[Self, str] = {}
        domain_prefixes = (
            "geno_",
            "pheno_",
            "amr_",
            "virulence_",
            "vir_",
            "spatial_",
            "temporal_",
            "qc_",
            "meta_",
        )
        for e in cls:
            val = e.value
            for prefix in domain_prefixes:
                if val.startswith(prefix):
                    val = val[len(prefix) :]
                    break
            labels[e] = val.replace("_", " ").title()
        return labels

    @classmethod
    def choices(cls) -> dict[str, str]:
        """Returns a dictionary mapping string values to human-readable UI labels for dropdowns."""
        return {str(k.value if hasattr(k, "value") else k): v for k, v in cls.ui_labels().items()}

    @classmethod
    def labels(cls) -> list[str]:
        """Returns a list of human-readable UI labels."""
        return list(cls.ui_labels().values())


class CoreTrait(ChoiceEnum):
    """Core epidemiological and spatio-temporal traits matching Polars DataFrame columns."""

    SAMPLE_ID = "sample_id"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    COUNTRY = "spatial_Country"
    REGION = "spatial_Region"
    CONTINENT = "spatial_Continent"
    COLLECTION_DATE = "temporal_Collection_Date"
    YEAR = "temporal_Year"
    MONTH = "temporal_Month"
    DAY = "temporal_Day"


class KpSeroTrait(ChoiceEnum):
    """Klebsiella pneumoniae serotype and genotype traits."""

    ST = "geno_ST"
    K_LOCUS = "geno_K_locus"
    O_LOCUS = "geno_O_locus"
    K_TYPE = "pheno_K_type"
    O_TYPE = "pheno_O_type"


class KpAmrTrait(ChoiceEnum):
    """Klebsiella pneumoniae AMR determinants and resistance scores."""

    RESISTANCE_SCORE = "amr_resistance_score"
    NUM_RESISTANCE_CLASSES = "amr_num_resistance_classes"
    NUM_RESISTANCE_GENES = "amr_num_resistance_genes"
    BLA_ACQUIRED = "amr_Bla_acquired"
    BLA_ESBL_ACQUIRED = "amr_Bla_ESBL_acquired"
    BLA_CARB_ACQUIRED = "amr_Bla_Carb_acquired"
    AGLY_ACQUIRED = "amr_AGly_acquired"
    COL_ACQUIRED = "amr_Col_acquired"
    FLQ_ACQUIRED = "amr_Flq_acquired"
    GLY_ACQUIRED = "amr_Gly_acquired"
    MLS_ACQUIRED = "amr_MLS_acquired"
    PHE_ACQUIRED = "amr_Phe_acquired"
    RIF_ACQUIRED = "amr_Rif_acquired"
    SUL_ACQUIRED = "amr_Sul_acquired"
    TET_ACQUIRED = "amr_Tet_acquired"
    TGC_ACQUIRED = "amr_Tgc_acquired"
    TMT_ACQUIRED = "amr_Tmt_acquired"


class KpVirulenceTrait(ChoiceEnum):
    """Klebsiella pneumoniae virulence markers and scores."""

    VIRULENCE_SCORE = "virulence_virulence_score"
    YERSINIABACTIN = "virulence_Yersiniabactin"
    YBST = "virulence_YbST"
    AEROBACTIN = "virulence_Aerobactin"
    ABST = "virulence_AbST"
    COLIBACTIN = "virulence_Colibactin"
    CBST = "virulence_CbST"
    SALMOCHELIN = "virulence_Salmochelin"
    SMST = "virulence_SmST"
    RMPADC = "virulence_RmpADC"
    RMST = "virulence_RmST"
    RMPA2 = "virulence_rmpA2"
