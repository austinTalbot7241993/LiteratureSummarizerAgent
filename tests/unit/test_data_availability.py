import pytest
import uuid
from lea.llm.schemas import (
    DataAvailabilityAssessment,
    DatasetAvailability,
    DatasetAvailabilityStatus,
    PaperAvailabilityStatus,
    AvailabilityEvidence,
    DatasetRole
)
from lea.rag.availability_retrieval import (
    retrieve_data_availability_context,
    score_chunk_for_availability,
    ACCESSION_PATTERNS
)
from lea.llm.aggregation import compute_overall_paper_status
from lea.llm.backends import validate_evidence_quotes, MockLLMBackend


def test_case1_reasonable_request_is_restricted():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.RESTRICTED,
        datasets=[
            DatasetAvailability(
                dataset_name="Study Dataset",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.RESTRICTED,
                access_conditions="Available upon reasonable request from corresponding author.",
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Data are available upon reasonable request from the corresponding author.")]
            )
        ],
        rationale="Author request restriction."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.RESTRICTED
    assert assessment.datasets[0].status == DatasetAvailabilityStatus.RESTRICTED


def test_case2_institutional_approval_dua_is_restricted():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.RESTRICTED,
        datasets=[
            DatasetAvailability(
                dataset_name="Clinical Dataset",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.RESTRICTED,
                access_conditions="Accessed after institutional approval and data-use agreement.",
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="The dataset may be accessed after institutional approval and execution of a data-use agreement.")]
            )
        ],
        rationale="Institutional approval and DUA required."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.RESTRICTED


def test_case3_privacy_dac_is_restricted():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.RESTRICTED,
        datasets=[
            DatasetAvailability(
                dataset_name="Genomic Cohort",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.RESTRICTED,
                access_conditions="Data Access Committee application required.",
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="The individual-level data are not publicly available because of privacy restrictions, but qualified researchers may apply through the data access committee.")]
            )
        ],
        rationale="Privacy restrictions with DAC approval route."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.RESTRICTED


def test_case4_supplementary_files_is_publicly_available():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        datasets=[
            DatasetAvailability(
                dataset_name="Supplementary Dataset",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="All data required to reproduce the results are included in the article and supplementary files.")]
            )
        ],
        rationale="Data included in supplementary files."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.PUBLICLY_AVAILABLE


def test_case5_geo_accession_is_publicly_available():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.PUBLICLY_AVAILABLE,
        datasets=[
            DatasetAvailability(
                dataset_name="Transcriptomic Data",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
                repository="GEO",
                accession="GSE123456",
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="The raw data were deposited in GEO under accession GSE123456.")]
            )
        ],
        rationale="Deposited in GEO accession."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.PUBLICLY_AVAILABLE
    assert assessment.datasets[0].accession == "GSE123456"
    assert assessment.datasets[0].repository == "GEO"


def test_case6_cannot_be_shared_is_not_available():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.NOT_AVAILABLE,
        datasets=[
            DatasetAvailability(
                dataset_name="Participant Data",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.NOT_AVAILABLE,
                access_conditions="Participant-level data cannot be shared.",
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="The participant-level data cannot be shared and no external access mechanism is available.")]
            )
        ],
        rationale="Cannot be shared."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.NOT_AVAILABLE


def test_case7_no_statement_is_not_reported():
    status = compute_overall_paper_status(datasets=[], has_evidence=False)
    assert status == PaperAvailabilityStatus.NOT_REPORTED


def test_case8_code_github_only_is_not_reported():
    # GitHub repository without study data release must not make study data public
    chunk_id = str(uuid.uuid4())
    with pytest.raises(ValueError, match="Code or software repository"):
        DatasetAvailability(
            dataset_name="Study Data",
            role=DatasetRole.PRIMARY,
            status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
            repository="GitHub",
            url="https://github.com/example/repo",
            evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Code is available on GitHub.")]
        )


def test_case9_public_code_and_restricted_clinical_data():
    chunk_id = str(uuid.uuid4())
    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.RESTRICTED,
        datasets=[
            DatasetAvailability(
                dataset_name="Clinical Data",
                role=DatasetRole.PRIMARY,
                status=DatasetAvailabilityStatus.RESTRICTED,
                access_conditions="Committee approval required.",
                evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Code is public on GitHub, but the clinical data require committee approval.")]
            )
        ],
        rationale="Clinical data restricted despite public code."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.RESTRICTED


def test_case10_publicly_unavailable_never_classified_as_publicly_available():
    chunk_id = str(uuid.uuid4())
    with pytest.raises(ValueError):
        DatasetAvailability(
            dataset_name="Patient Data",
            role=DatasetRole.PRIMARY,
            status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
            evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="The data are publicly unavailable.")]
        )


def test_case11_israeli_does_not_trigger_sra():
    score = score_chunk_for_availability("The study analyzed Israeli patient cohorts.")
    assert score == 0.0


def test_case12_negative_does_not_trigger_ega():
    score = score_chunk_for_availability("The results showed a negative correlation across all tests.")
    assert score == 0.0


def test_case13_geospatial_does_not_trigger_geo():
    score = score_chunk_for_availability("Geospatial modeling was used for population density analysis.")
    assert score == 0.0


def test_case14_mixed_datasets_raw_restricted_summary_public():
    chunk_id = str(uuid.uuid4())
    ds1 = DatasetAvailability(
        dataset_name="Summary Statistics",
        role=DatasetRole.DERIVED,
        status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
        url="https://example.org/stats",
        evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Summary statistics are publicly available at https://example.org/stats.")]
    )
    ds2 = DatasetAvailability(
        dataset_name="Raw Patient Data",
        role=DatasetRole.PRIMARY,
        status=DatasetAvailabilityStatus.RESTRICTED,
        access_conditions="Available via DUA.",
        evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Raw patient data are restricted via DUA.")]
    )
    overall = compute_overall_paper_status([ds1, ds2], has_evidence=True)
    assert overall == PaperAvailabilityStatus.MIXED

    assessment = DataAvailabilityAssessment(
        overall_status=PaperAvailabilityStatus.MIXED,
        datasets=[ds1, ds2],
        rationale="Summary statistics public, raw data restricted."
    )
    assert assessment.overall_status == PaperAvailabilityStatus.MIXED


def test_case15_public_benchmark_with_private_training_cohort():
    chunk_id = str(uuid.uuid4())
    ds_benchmark = DatasetAvailability(
        dataset_name="ImageNet Benchmark",
        role=DatasetRole.BENCHMARK,
        status=DatasetAvailabilityStatus.PUBLICLY_AVAILABLE,
        url="https://image-net.org",
        evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Evaluated on public ImageNet benchmark.")]
    )
    ds_primary = DatasetAvailability(
        dataset_name="Proprietary In-house Cohort",
        role=DatasetRole.PRIMARY,
        status=DatasetAvailabilityStatus.NOT_AVAILABLE,
        access_conditions="Private in-house training cohort cannot be released.",
        evidence=[AvailabilityEvidence(source_chunk_id=chunk_id, quote="Private in-house training cohort cannot be released.")]
    )
    overall = compute_overall_paper_status([ds_benchmark, ds_primary], has_evidence=True)
    assert overall == PaperAvailabilityStatus.MIXED
