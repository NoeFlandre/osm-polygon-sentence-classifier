from dataclasses import is_dataclass

import pytest

from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
    DatasetContract,
    DatasetContractError,
    DatasetProvenance,
)

REQUIRED_COLUMNS = (
    "sentence_id",
    "polygon_id",
    "wikidata",
    "document_id",
    "article_id",
    "source",
    "language",
    "site",
    "page_title",
    "section_id",
    "section_index",
    "section_path",
    "sentence_index",
    "sentence_text_raw",
    "sentence_text_normalized",
    "previous_sentence",
    "next_sentence",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "document_content_hash",
    "section_content_hash",
    "sentence_content_hash",
    "duplicate_occurrence_count",
    "duplicate_sources",
    "polygon_name",
    "osm_primary_tag",
    "osm_tags",
    "region",
    "lat",
    "lon",
    "input_dataset_revision",
    "pipeline_version",
    "landuse_relevance",
    "polygon_relevance",
    "landuse_reason",
    "polygon_reason",
    "label_evidence",
)

EXPECTED_REPOSITORY_REVISION = "07e421a3020127ced2c19304645a6f63e6735966"
EXPECTED_PARQUET_SHA256 = (
    "4e9f3300a3f93d5485ada9f950ed77f05b448f9ffba09500ca25b33930b15eb0"
)
WORLDWIDE_REQUIRED_COLUMNS = (
    "sentence_id",
    "polygon_id",
    "wikidata",
    "document_id",
    "article_id",
    "source",
    "language",
    "site",
    "page_title",
    "section_id",
    "section_index",
    "section_path",
    "sentence_index",
    "sentence_text_raw",
    "sentence_text_normalized",
    "previous_sentence",
    "next_sentence",
    "url",
    "page_id",
    "revision_id",
    "revision_timestamp",
    "document_content_hash",
    "section_content_hash",
    "sentence_content_hash",
    "duplicate_occurrence_count",
    "duplicate_sources",
    "polygon_name",
    "osm_primary_tag",
    "osm_tags",
    "region",
    "lat",
    "lon",
    "input_dataset_revision",
    "pipeline_version",
    "area_km2",
    "area_bucket",
    "place_relevance",
    "yes_logprob",
    "no_logprob",
    "logit_margin",
    "two_class_probability",
    "geometry",
)
WORLDWIDE_DATASET_REVISION = "4d0d2b5d53630c24acfb280e9d8159bf6ed0d3fa"
WORLDWIDE_PARQUET_SHA256 = (
    "0f76f64f9d2a13081ad26cacb27b18f48c204d9150a494dafb340375e82eb270"
)
REORDERED_COLUMNS = (
    REQUIRED_COLUMNS[:2]
    + (
        REQUIRED_COLUMNS[3],
        REQUIRED_COLUMNS[2],
    )
    + REQUIRED_COLUMNS[4:]
)


def _valid_row() -> dict[str, object]:
    row = dict.fromkeys(REQUIRED_COLUMNS)
    row.update(
        {
            "region": "afghanistan",
            "sentence_text_normalized": "A sentence about this landuse.",
            "landuse_relevance": "yes",
        }
    )
    return row


def test_landuse_contract_identifies_the_training_boundary() -> None:
    contract = LANDUSE_DATASET_CONTRACT

    assert isinstance(contract, DatasetContract)
    assert contract.dataset_id == ("NoeFlandre/osm-polygon-wikidata-sentence-relevance")
    assert contract.config == "default"
    assert contract.split == "train"
    assert contract.region == "afghanistan"
    assert contract.label_column == "landuse_relevance"
    assert contract.supported_label_values == ("no", "yes", "uncertain")
    assert contract.training_label_values == ("no", "yes")
    assert contract.model_input_columns == ("sentence_text_normalized",)
    assert {
        "landuse_relevance",
        "polygon_relevance",
        "landuse_reason",
        "polygon_reason",
        "label_evidence",
    }.issubset(contract.forbidden_model_input_columns)
    assert "uncertain" not in contract.training_label_values
    assert set(contract.model_input_columns).isdisjoint(
        contract.forbidden_model_input_columns
    )


def test_contract_and_provenance_are_immutable_slotted_dataclasses() -> None:
    contract = LANDUSE_DATASET_CONTRACT
    provenance = contract.provenance

    assert is_dataclass(contract)
    assert is_dataclass(provenance)
    assert not hasattr(contract, "__dict__")
    assert not hasattr(provenance, "__dict__")

    with pytest.raises(AttributeError):
        contract.region = "iran"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        provenance.repository_revision = "0" * 40  # type: ignore[misc]


def test_contract_exposes_pinned_dataset_provenance() -> None:
    provenance = LANDUSE_DATASET_CONTRACT.provenance

    assert isinstance(provenance, DatasetProvenance)
    assert provenance.repository_revision == EXPECTED_REPOSITORY_REVISION
    assert provenance.parquet_sha256 == EXPECTED_PARQUET_SHA256


@pytest.mark.parametrize(
    ("repository_revision", "parquet_sha256"),
    [
        ("short", EXPECTED_PARQUET_SHA256),
        ("z" * 40, EXPECTED_PARQUET_SHA256),
        (EXPECTED_REPOSITORY_REVISION, "0" * 63),
        (EXPECTED_REPOSITORY_REVISION, "G" * 64),
        (EXPECTED_REPOSITORY_REVISION, "A" * 64),
    ],
)
def test_provenance_rejects_invalid_hash_values(
    repository_revision: str,
    parquet_sha256: str,
) -> None:
    with pytest.raises(ValueError, match="must be exactly"):
        DatasetProvenance(
            repository_revision=repository_revision,
            parquet_sha256=parquet_sha256,
        )


def test_required_columns_match_the_live_schema_exactly() -> None:
    assert LANDUSE_DATASET_CONTRACT.required_columns == REQUIRED_COLUMNS


def test_validate_columns_accepts_the_required_schema_in_order() -> None:
    assert LANDUSE_DATASET_CONTRACT.validate_columns(REQUIRED_COLUMNS) is None


@pytest.mark.parametrize(
    ("columns", "expected_message"),
    [
        (REQUIRED_COLUMNS[:-1], "missing required columns"),
        (REQUIRED_COLUMNS + ("unexpected",), "unexpected columns"),
        (REORDERED_COLUMNS, "required column order"),
        (
            REQUIRED_COLUMNS[:-1] + (REQUIRED_COLUMNS[-2], REQUIRED_COLUMNS[-1]),
            "duplicate columns",
        ),
    ],
)
def test_validate_columns_rejects_schema_mismatches(
    columns: tuple[str, ...], expected_message: str
) -> None:
    with pytest.raises(DatasetContractError, match=expected_message):
        LANDUSE_DATASET_CONTRACT.validate_columns(columns)


@pytest.mark.parametrize("label", ["no", "yes", "uncertain"])
def test_validate_row_accepts_supported_labels(label: str) -> None:
    row = _valid_row()
    row["landuse_relevance"] = label

    assert LANDUSE_DATASET_CONTRACT.validate_row(row) is None


def test_validate_row_requires_the_contract_keys() -> None:
    row = _valid_row()
    del row["sentence_text_raw"]

    with pytest.raises(DatasetContractError, match="sentence_text_raw"):
        LANDUSE_DATASET_CONTRACT.validate_row(row)


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("region", "iran", "region"),
        ("sentence_text_normalized", "", "sentence_text_normalized"),
        ("sentence_text_normalized", "   ", "sentence_text_normalized"),
        ("sentence_text_normalized", None, "sentence_text_normalized"),
        ("landuse_relevance", "maybe", "landuse_relevance"),
    ],
)
def test_validate_row_rejects_values_outside_the_contract(
    field: str, value: object, expected_message: str
) -> None:
    row = _valid_row()
    row[field] = value

    with pytest.raises(DatasetContractError, match=expected_message):
        LANDUSE_DATASET_CONTRACT.validate_row(row)


def test_worldwide_v2_contract_identifies_a_separate_binary_task() -> None:
    contract = WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT

    assert contract.dataset_id == LANDUSE_DATASET_CONTRACT.dataset_id
    assert contract.config == "v2-worldwide"
    assert contract.split == "train"
    assert contract.region is None
    assert contract.label_column == "place_relevance"
    assert contract.supported_label_values == ("no", "yes")
    assert contract.training_label_values == ("no", "yes")
    assert contract.model_input_columns == ("sentence_text_normalized",)
    assert {
        "place_relevance",
        "yes_logprob",
        "no_logprob",
        "logit_margin",
        "two_class_probability",
        "geometry",
    }.issubset(contract.forbidden_model_input_columns)
    assert contract.provenance.repository_revision == WORLDWIDE_DATASET_REVISION
    assert contract.provenance.parquet_sha256 == WORLDWIDE_PARQUET_SHA256


def test_worldwide_v2_contract_matches_the_published_schema() -> None:
    assert (
        WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.required_columns
        == WORLDWIDE_REQUIRED_COLUMNS
    )


def test_worldwide_v2_contract_accepts_any_non_blank_region() -> None:
    row = dict.fromkeys(WORLDWIDE_REQUIRED_COLUMNS)
    row.update(
        {
            "region": "jp-hokkaido",
            "sentence_text_normalized": "A physical description.",
            "place_relevance": "yes",
        }
    )

    assert WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.validate_row(row) is None
