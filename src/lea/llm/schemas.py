from enum import Enum
from typing import Literal, Optional, List, Union, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetAvailabilityStatus(str, Enum):
    PUBLICLY_AVAILABLE = "publicly_available"
    RESTRICTED = "restricted"
    NOT_AVAILABLE = "not_available"
    NOT_REPORTED = "not_reported"
    UNCLEAR = "unclear"


class PaperAvailabilityStatus(str, Enum):
    PUBLICLY_AVAILABLE = "publicly_available"
    RESTRICTED = "restricted"
    NOT_AVAILABLE = "not_available"
    NOT_REPORTED = "not_reported"
    UNCLEAR = "unclear"
    MIXED = "mixed"


class DatasetRole(str, Enum):
    PRIMARY = "primary"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"
    DERIVED = "derived"
    SUPPLEMENTARY = "supplementary"
    UNKNOWN = "unknown"


class DatasetOwnership(str, Enum):
    PUBLIC = "public"
    ACADEMIC = "academic"
    GOVERNMENT = "government"
    COMMERCIAL = "commercial"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class VerificationStatus(str, Enum):
    NOT_CHECKED = "not_checked"
    PARTIALLY_VERIFIED = "partially_verified"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"


class AvailabilityEvidence(BaseModel):
    source_chunk_id: Union[UUID, str]
    quote: str = Field(min_length=1)
    section_title: Optional[str] = None
    page_number: Optional[int] = None

    @field_validator("source_chunk_id", mode="before")
    @classmethod
    def stringify_id(cls, v: Any) -> str:
        if isinstance(v, UUID):
            return str(v)
        return str(v)


class DatasetAvailability(BaseModel):
    dataset_name: Optional[str] = None
    role: DatasetRole = DatasetRole.UNKNOWN
    status: DatasetAvailabilityStatus
    repository: Optional[str] = None
    accession: Optional[str] = None
    url: Optional[str] = None
    access_conditions: Optional[str] = None
    ownership: DatasetOwnership = DatasetOwnership.UNKNOWN
    required_for_reproduction: Optional[bool] = None
    evidence: List[AvailabilityEvidence] = Field(default_factory=list)

    @field_validator("ownership", mode="before")
    @classmethod
    def validate_ownership(cls, v: Any) -> DatasetOwnership:
        if v is None:
            return DatasetOwnership.UNKNOWN
        if isinstance(v, DatasetOwnership):
            return v
        if isinstance(v, str):
            val = v.strip().lower()
            for item in DatasetOwnership:
                if item.value == val:
                    return item
        return DatasetOwnership.UNKNOWN

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v: Any) -> DatasetRole:
        if v is None:
            return DatasetRole.UNKNOWN
        if isinstance(v, DatasetRole):
            return v
        if isinstance(v, str):
            val = v.strip().lower()
            for item in DatasetRole:
                if item.value == val:
                    return item
        return DatasetRole.UNKNOWN

    @model_validator(mode="after")
    def validate_dataset_rules(self) -> "DatasetAvailability":
        # 1. Non-not_reported datasets should ordinarily contain evidence
        if self.status != DatasetAvailabilityStatus.NOT_REPORTED and not self.evidence:
            raise ValueError(f"Dataset with status '{self.status.value}' must contain at least one evidence item.")

        # 2. Check for public repository vs code repository
        # Code repos (e.g., GitHub) alone do not constitute study data public availability unless data is explicitly indicated.
        if self.status == DatasetAvailabilityStatus.PUBLICLY_AVAILABLE:
            has_url = bool(self.url and self.url.strip())
            has_accession = bool(self.accession and self.accession.strip())
            has_repo = bool(self.repository and self.repository.strip())
            
            url_str = (self.url or "").lower()
            repo_str = (self.repository or "").lower()
            is_code_host = any(h in url_str or h in repo_str for h in ["github", "gitlab", "bitbucket"])

            evidence_text = " ".join([e.quote for e in self.evidence]).lower()
            has_data_mention = any(k in evidence_text for k in ["dataset", "raw data", "study data", "counts", "matrices", "table", "supplement"])

            if is_code_host and not has_data_mention:
                raise ValueError(
                    "Code or software repository (e.g., GitHub) alone does not constitute study data public availability."
                )

            in_article_or_supp = any(
                phrase in evidence_text
                for phrase in [
                    "article", "supplementary", "supplement", "additional file",
                    "included in", "table", "attached", "available in the paper"
                ]
            )

            if not (has_url or has_accession or (has_repo and not is_code_host) or in_article_or_supp):
                raise ValueError(
                    "publicly_available dataset must include a URL, accession, a named non-code data repository, "
                    "or explicit evidence that data are included in the article or supplementary files."
                )

        # 3. not_available requires evidence discussing non-availability
        if self.status == DatasetAvailabilityStatus.NOT_AVAILABLE:
            combined_text = (self.access_conditions or "") + " " + " ".join([e.quote for e in self.evidence])
            combined_lower = combined_text.lower()
            neg_keywords = ["cannot", "not available", "will not", "unable", "privacy", "restricted", "private", "no public", "withheld"]
            if not any(k in combined_lower for k in neg_keywords):
                raise ValueError("not_available status requires affirmative evidence explaining why data cannot be shared.")

        # 4. not_reported must not have explicit access quotes claiming data sharing
        if self.status == DatasetAvailabilityStatus.NOT_REPORTED and self.evidence:
            evidence_text = " ".join([e.quote for e in self.evidence]).lower()
            if any(k in evidence_text for k in ["available", "deposited", "accession", "gse", "sra", "download", "request"]):
                raise ValueError("not_reported status must not be used when evidence explicitly discusses data access.")

        return self


class DataAvailabilityAssessment(BaseModel):
    overall_status: PaperAvailabilityStatus
    datasets: List[DatasetAvailability] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    verification_status: VerificationStatus = VerificationStatus.NOT_CHECKED

    @model_validator(mode="after")
    def validate_assessment(self) -> "DataAvailabilityAssessment":
        # Check mixed status rule
        if self.overall_status == PaperAvailabilityStatus.MIXED:
            statuses = {d.status for d in self.datasets}
            # Remove NOT_REPORTED if present alongside substantive statuses
            substantive_statuses = {s for s in statuses if s != DatasetAvailabilityStatus.NOT_REPORTED}
            if len(substantive_statuses) < 2 and len(statuses) < 2:
                raise ValueError("Paper-level status 'mixed' requires at least two datasets with materially different statuses.")
        return self


class SelfCritiqueAssessment(BaseModel):
    is_relevant_to_seed_topic: bool = Field(
        description="True if retrieved text chunks directly address the seed paper's core problem formulation."
    )
    relevance_score: float = Field(
        ge=0.0, le=10.0,
        description="Score from 0.0 to 10.0 measuring relevance of candidate paper content to seed topic."
    )
    factual_grounding_score: float = Field(
        ge=0.0, le=10.0,
        description="Score from 0.0 to 10.0 measuring how strictly summary claims are supported by verbatim text chunks."
    )
    critique_rationale: str = Field(
        description="1-2 sentence justification for the relevance and grounding assessment."
    )
    verdict: Literal["accepted", "marginal", "rejected"] = Field(
        description="Decision: 'accepted' (score >= 6.5), 'marginal' (5.0-6.4), 'rejected' (< 5.0 or off-topic)."
    )


class TechnicalSummary(BaseModel):
    problem_formulation: str = Field(
        min_length=1,
        description="Mathematical, statistical, or scientific problem statement",
    )
    methodological_novelty: str = Field(
        min_length=1,
        description="Core algorithmic, empirical, or theoretical novelty",
    )
    empirical_findings: str = Field(
        min_length=1,
        description="Quantifiable validation, or an explicit statement that it was not reported",
    )
    paragraph_summary: str = Field(
        min_length=1,
        description="Single-paragraph technical synthesis of at most 300 words",
    )
    relationship_to_target: str = Field(
        default="Related literature candidate.",
        min_length=1,
        description="Description of how this paper relates to the target input paper",
    )
    data_availability: PaperAvailabilityStatus = Field(
        description="Data availability status: publicly_available, restricted, not_available, not_reported, unclear, or mixed",
    )
    data_location: Optional[str] = Field(
        default=None,
        description="Dataset location, URL, accession number, repository name, or access conditions",
    )
    critique: Optional[SelfCritiqueAssessment] = Field(
        default=None,
        description="Self-critique assessment evaluating topic relevance and factual grounding"
    )

    @field_validator("data_availability", mode="before")
    @classmethod
    def validate_data_availability_enum(cls, value: Any) -> PaperAvailabilityStatus:
        if isinstance(value, PaperAvailabilityStatus):
            return value
        if isinstance(value, str):
            val = value.strip().lower()
            for status in PaperAvailabilityStatus:
                if status.value == val:
                    return status
        raise ValueError(f"Invalid data_availability status '{value}'. Must be a valid PaperAvailabilityStatus enum.")

    @field_validator("paragraph_summary")
    @classmethod
    def validate_paragraph_summary(cls, value: str) -> str:
        words = value.strip().split()
        if len(words) > 300:
            raise ValueError(f"paragraph_summary must contain at most 300 words (got {len(words)})")
        if "\n" in value:
            raise ValueError("paragraph_summary must be a single paragraph")
        return value.strip()
