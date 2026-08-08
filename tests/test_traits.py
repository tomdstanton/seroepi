"""Unit tests for ChoiceEnum, trait definitions, and constants backward compatibility."""

from seroepi.traits import ChoiceEnum, CoreTrait, KpAmrTrait, KpSeroTrait, KpVirulenceTrait


class DummyChoiceEnum(ChoiceEnum):
    """Dummy choice enum for unit testing base functionality."""

    DUMMY_GENO = "geno_sample_col"
    DUMMY_AMR = "amr_resistance_gene"
    DUMMY_PLAIN = "plain_trait"


class CustomLabelEnum(ChoiceEnum):
    """Dummy enum overriding ui_labels."""

    FOO = "foo"
    BAR = "bar"

    @classmethod
    def ui_labels(cls) -> dict[str, str]:
        return {
            cls.FOO.value: "Custom Foo",
            cls.BAR.value: "Custom Bar",
        }


def test_choice_enum_methods():
    """Test ChoiceEnum.ui_labels(), choices(), and labels() methods."""
    ui_labels = DummyChoiceEnum.ui_labels()
    assert isinstance(ui_labels, dict)
    assert ui_labels[DummyChoiceEnum.DUMMY_GENO] == "Sample Col"
    assert ui_labels[DummyChoiceEnum.DUMMY_AMR] == "Resistance Gene"
    assert ui_labels[DummyChoiceEnum.DUMMY_PLAIN] == "Plain Trait"

    choices = DummyChoiceEnum.choices()
    assert isinstance(choices, dict)
    assert choices["geno_sample_col"] == "Sample Col"
    assert choices["amr_resistance_gene"] == "Resistance Gene"
    assert choices["plain_trait"] == "Plain Trait"

    labels = DummyChoiceEnum.labels()
    assert isinstance(labels, list)
    assert labels == ["Sample Col", "Resistance Gene", "Plain Trait"]


def test_custom_ui_labels_override():
    """Test custom ui_labels() override support on ChoiceEnum subclasses."""
    assert CustomLabelEnum.choices() == {"foo": "Custom Foo", "bar": "Custom Bar"}
    assert CustomLabelEnum.labels() == ["Custom Foo", "Custom Bar"]


def test_core_trait_values_and_methods():
    """Test CoreTrait enum values match expected column names and helpers work."""
    assert CoreTrait.SAMPLE_ID == "sample_id"
    assert CoreTrait.LATITUDE == "latitude"
    assert CoreTrait.LONGITUDE == "longitude"
    assert CoreTrait.COUNTRY == "spatial_Country"
    assert CoreTrait.REGION == "spatial_Region"
    assert CoreTrait.CONTINENT == "spatial_Continent"
    assert CoreTrait.COLLECTION_DATE == "temporal_Collection_Date"
    assert CoreTrait.YEAR == "temporal_Year"
    assert CoreTrait.MONTH == "temporal_Month"
    assert CoreTrait.DAY == "temporal_Day"

    choices = CoreTrait.choices()
    assert choices["sample_id"] == "Sample Id"
    assert choices["spatial_Country"] == "Country"
    assert choices["temporal_Collection_Date"] == "Collection Date"


def test_kp_sero_trait_values_and_methods():
    """Test KpSeroTrait enum values match Polars DataFrame column names."""
    assert KpSeroTrait.ST == "geno_ST"
    assert KpSeroTrait.K_LOCUS == "geno_K_locus"
    assert KpSeroTrait.O_LOCUS == "geno_O_locus"
    assert KpSeroTrait.K_TYPE == "pheno_K_type"
    assert KpSeroTrait.O_TYPE == "pheno_O_type"

    choices = KpSeroTrait.choices()
    assert choices["geno_K_locus"] == "K Locus"
    assert choices["pheno_K_type"] == "K Type"
    assert "K Locus" in KpSeroTrait.labels()


def test_kp_amr_trait_values_and_methods():
    """Test KpAmrTrait enum values match Polars DataFrame column names."""
    assert KpAmrTrait.RESISTANCE_SCORE == "amr_resistance_score"
    assert KpAmrTrait.NUM_RESISTANCE_CLASSES == "amr_num_resistance_classes"
    assert KpAmrTrait.NUM_RESISTANCE_GENES == "amr_num_resistance_genes"
    assert KpAmrTrait.BLA_ACQUIRED == "amr_Bla_acquired"
    assert KpAmrTrait.BLA_ESBL_ACQUIRED == "amr_Bla_ESBL_acquired"
    assert KpAmrTrait.BLA_CARB_ACQUIRED == "amr_Bla_Carb_acquired"

    choices = KpAmrTrait.choices()
    assert choices["amr_resistance_score"] == "Resistance Score"
    assert choices["amr_Bla_Carb_acquired"] == "Bla Carb Acquired"


def test_kp_virulence_trait_values_and_methods():
    """Test KpVirulenceTrait enum values match Polars DataFrame column names."""
    assert KpVirulenceTrait.VIRULENCE_SCORE == "virulence_virulence_score"
    assert KpVirulenceTrait.YERSINIABACTIN == "virulence_Yersiniabactin"
    assert KpVirulenceTrait.COLIBACTIN == "virulence_Colibactin"
    assert KpVirulenceTrait.AEROBACTIN == "virulence_Aerobactin"
    assert KpVirulenceTrait.SALMOCHELIN == "virulence_Salmochelin"
    assert KpVirulenceTrait.RMPA2 == "virulence_rmpA2"

    choices = KpVirulenceTrait.choices()
    assert choices["virulence_virulence_score"] == "Virulence Score"
    assert choices["virulence_Yersiniabactin"] == "Yersiniabactin"


def test_str_enum_nature():
    """Test that trait enums inherit from StrEnum and act as native strings."""
    assert isinstance(KpSeroTrait.K_LOCUS, str)
    assert KpSeroTrait.K_LOCUS == "geno_K_locus"
    assert f"prefix_{KpSeroTrait.ST}" == "prefix_geno_ST"


