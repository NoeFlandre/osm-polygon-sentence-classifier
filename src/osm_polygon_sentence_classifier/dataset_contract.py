"""Immutable dataset and training-boundary contracts for landuse classification."""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "DatasetContract",
    "DatasetContractError",
    "DatasetProvenance",
    "LANDUSE_DATASET_CONTRACT",
    "WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT",
]


class DatasetContractError(ValueError):
    """Raised when dataset columns or values violate the contract."""


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    """Pinned source revision and published Parquet artifact digest."""

    repository_revision: str
    parquet_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repository_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.repository_revision) is None
        ):
            raise ValueError(
                "repository_revision must be exactly 40 lowercase hexadecimal "
                "characters"
            )
        if (
            not isinstance(self.parquet_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.parquet_sha256) is None
        ):
            raise ValueError(
                "parquet_sha256 must be exactly 64 lowercase hexadecimal characters"
            )

    @property
    def published_parquet_sha256(self) -> str:
        """Return the digest of the published Parquet artifact."""

        return self.parquet_sha256


@dataclass(frozen=True, slots=True)
class DatasetContract:
    """Immutable schema, label, input, and provenance rules for one dataset."""

    dataset_id: str
    config: str
    split: str
    region: str | None
    label_column: str
    supported_label_values: tuple[str, ...]
    training_label_values: tuple[str, ...]
    model_input_columns: tuple[str, ...]
    forbidden_model_input_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    provenance: DatasetProvenance

    @property
    def supported_labels(self) -> tuple[str, ...]:
        """Return all labels permitted by the published dataset contract."""

        return self.supported_label_values

    @property
    def leakage_annotation_columns(self) -> tuple[str, ...]:
        """Return target and annotation fields forbidden as model inputs."""

        return self.forbidden_model_input_columns

    def validate_columns(self, columns: Iterable[str]) -> None:
        """Require an exact, ordered, non-duplicated column sequence."""

        actual = tuple(columns)
        seen: set[str] = set()
        duplicates: list[str] = []
        for column in actual:
            if column in seen and column not in duplicates:
                duplicates.append(column)
            seen.add(column)
        if duplicates:
            raise DatasetContractError("duplicate columns: " + ", ".join(duplicates))

        missing = tuple(
            column for column in self.required_columns if column not in actual
        )
        if missing:
            raise DatasetContractError(
                "missing required columns: " + ", ".join(missing)
            )

        extra = tuple(
            column for column in actual if column not in self.required_columns
        )
        if extra:
            raise DatasetContractError("unexpected columns: " + ", ".join(extra))

        if actual != self.required_columns:
            raise DatasetContractError(
                "required column order does not match the dataset contract"
            )

    def validate_row(self, row: Mapping[str, object]) -> None:
        """Validate required row keys and the contract's classification fields."""

        missing = tuple(column for column in self.required_columns if column not in row)
        if missing:
            raise DatasetContractError(
                "missing required row keys: " + ", ".join(missing)
            )

        if self.region is not None and row["region"] != self.region:
            raise DatasetContractError(
                f"region must be {self.region!r}, got {row['region']!r}"
            )
        if self.region is None and (
            not isinstance(row["region"], str) or not row["region"].strip()
        ):
            raise DatasetContractError("region must be a non-empty string")

        normalized_sentence = row["sentence_text_normalized"]
        if not isinstance(normalized_sentence, str) or not normalized_sentence.strip():
            raise DatasetContractError(
                "sentence_text_normalized must be a non-empty string"
            )

        label = row[self.label_column]
        if label not in self.supported_label_values:
            raise DatasetContractError(
                f"{self.label_column} must be one of "
                f"{self.supported_label_values!r}, got {label!r}"
            )


LANDUSE_DATASET_CONTRACT = DatasetContract(
    dataset_id="NoeFlandre/osm-polygon-wikidata-sentence-relevance",
    config="default",
    split="train",
    region="afghanistan",
    label_column="landuse_relevance",
    supported_label_values=("no", "yes", "uncertain"),
    training_label_values=("no", "yes"),
    model_input_columns=("sentence_text_normalized",),
    forbidden_model_input_columns=(
        "landuse_relevance",
        "polygon_relevance",
        "landuse_reason",
        "polygon_reason",
        "label_evidence",
    ),
    required_columns=(
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
    ),
    provenance=DatasetProvenance(
        repository_revision="07e421a3020127ced2c19304645a6f63e6735966",
        parquet_sha256=(
            "4e9f3300a3f93d5485ada9f950ed77f05b448f9ffba09500ca25b33930b15eb0"
        ),
    ),
)


WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT = DatasetContract(
    dataset_id="NoeFlandre/osm-polygon-wikidata-sentence-relevance",
    config="v2-worldwide",
    split="train",
    region=None,
    label_column="place_relevance",
    supported_label_values=("no", "yes"),
    training_label_values=("no", "yes"),
    model_input_columns=("sentence_text_normalized",),
    forbidden_model_input_columns=(
        "place_relevance",
        "yes_logprob",
        "no_logprob",
        "logit_margin",
        "two_class_probability",
        "geometry",
        "polygon_id",
        "region",
        "language",
        "source",
        "page_title",
        "previous_sentence",
        "next_sentence",
    ),
    required_columns=(
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
    ),
    provenance=DatasetProvenance(
        repository_revision="4d0d2b5d53630c24acfb280e9d8159bf6ed0d3fa",
        parquet_sha256=(
            "0f76f64f9d2a13081ad26cacb27b18f48c204d9150a494dafb340375e82eb270"
        ),
    ),
)
