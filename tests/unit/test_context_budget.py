import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from lea.db.models import Base
from lea.db.repository import LEARepository
from lea.rag.availability_retrieval import retrieve_data_availability_context, format_availability_context
from lea.llm.schemas import (
    DataAvailabilityAssessment,
    DatasetAvailability,
    DatasetAvailabilityStatus,
    PaperAvailabilityStatus,
    AvailabilityEvidence,
    DatasetRole
)
from lea.llm.backends import validate_evidence_quotes, MockLLMBackend
from lea.exceptions import SummaryValidationError


def test_case18_missing_json_fields_raises_validation_error():
    backend = MockLLMBackend(preset="malformed")
    with pytest.raises(SummaryValidationError):
        backend.generate_data_availability("system prompt", "user prompt")


def test_case19_fabricated_quote_or_unknown_chunk_id_rejected():
    retrieved_chunks = [
        {"id": "chunk-100", "content": "Data are available at https://example.org/data."}
    ]

    # 1. Unknown chunk ID
    assessment_bad_id = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        datasets=[
            DatasetAvailability(
                dataset_name="Test Data",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
                url="https://example.org/data",
                evidence=[AvailabilityEvidence(source_chunk_id="chunk-999", quote="Data are available at https://example.org/data.")]
            )
        ],
        rationale="Unknown chunk ID test"
    )
    with pytest.raises(ValueError, match="unknown source_chunk_id"):
        validate_evidence_quotes(assessment_bad_id, retrieved_chunks)

    # 2. Fabricated quote not in source chunk
    assessment_bad_quote = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        datasets=[
            DatasetAvailability(
                dataset_name="Test Data",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
                url="https://example.org/data",
                evidence=[AvailabilityEvidence(source_chunk_id="chunk-100", quote="This quote was fabricated and does not exist in the source.")]
            )
        ],
        rationale="Fabricated quote test"
    )
    with pytest.raises(ValueError, match="was not found in source chunk"):
        validate_evidence_quotes(assessment_bad_quote, retrieved_chunks)


def test_case20_late_paper_chunk_survives_context_selection():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    repo = LEARepository(session)

    p_input = repo.create_paper(sha256_hash="input", title="Input Paper")
    run = repo.create_discovery_run(input_paper_id=p_input.id)
    paper = repo.create_paper(sha256_hash="target", title="Target Paper")

    # Add 25 noise chunks followed by 1 late chunk containing the data availability statement
    for i in range(25):
        repo.add_chunk(
            paper_id=paper.id,
            run_id=run.id,
            chunk_type="child",
            content=f"Methodology details paragraph {i} with model descriptions.",
            chunk_index=i,
            token_count=50
        )

    late_chunk = repo.add_chunk(
        paper_id=paper.id,
        run_id=run.id,
        chunk_type="child",
        content="Data Availability Statement: Raw sequencing reads were deposited in GEO under accession GSE999888.",
        chunk_index=25,
        token_count=30,
        section_title="Data Availability"
    )

    retrieved = retrieve_data_availability_context(repo, run.id, paper.id, max_tokens=1000)
    retrieved_ids = [c["id"] for c in retrieved]
    assert str(late_chunk.id) in retrieved_ids, "Late chunk with Data Availability section title must be retrieved!"


def test_case21_restricted_data_retains_location_and_access_conditions():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.RESTRICTED,
        datasets=[
            DatasetAvailability(
                dataset_name="UK Biobank Cohort",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.RESTRICTED,
                repository="UK Biobank",
                accession="Application 12345",
                url="https://www.ukbiobank.ac.uk",
                access_conditions="Requires application approval and signed data-use agreement.",
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Dataset available via UK Biobank application approval.")]
            )
        ],
        rationale="UK Biobank application required."
    )
    d = assessment.datasets[0]
    assert d.repository == "UK Biobank"
    assert d.accession == "Application 12345"
    assert d.url == "https://www.ukbiobank.ac.uk"
    assert d.access_conditions == "Requires application approval and signed data-use agreement."
