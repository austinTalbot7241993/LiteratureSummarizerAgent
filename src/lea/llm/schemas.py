from pydantic import BaseModel, Field, field_validator

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

    @field_validator("paragraph_summary")
    @classmethod
    def validate_paragraph_summary(cls, value: str) -> str:
        words = value.strip().split()
        if len(words) > 300:
            raise ValueError(f"paragraph_summary must contain at most 300 words (got {len(words)})")
        if "\n" in value:
            raise ValueError("paragraph_summary must be a single paragraph")
        return value.strip()
