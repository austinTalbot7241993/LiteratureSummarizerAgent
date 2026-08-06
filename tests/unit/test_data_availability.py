import pytest
from lea.llm.schemas import TechnicalSummary
from lea.llm.backends import parse_data_availability_statement

def test_data_availability_defaulting():
    ts = TechnicalSummary(
        problem_formulation="Problem formulation.",
        methodological_novelty="Methodological novelty.",
        empirical_findings="Empirical findings.",
        paragraph_summary="Paragraph summary."
    )
    assert ts.data_availability == "proprietary"
    assert ts.data_location is None

def test_data_availability_normalization():
    ts1 = TechnicalSummary(
        problem_formulation="P",
        methodological_novelty="M",
        empirical_findings="E",
        paragraph_summary="S",
        data_availability="Publicly Available",
        data_location="https://www.internationalgenome.org"
    )
    assert ts1.data_availability == "publicly_available"
    assert ts1.data_location == "https://www.internationalgenome.org"

    ts2 = TechnicalSummary(
        problem_formulation="P",
        methodological_novelty="M",
        empirical_findings="E",
        paragraph_summary="S",
        data_availability="restricted access",
    )
    assert ts2.data_availability == "restricted"

    ts3 = TechnicalSummary(
        problem_formulation="P",
        methodological_novelty="M",
        empirical_findings="E",
        paragraph_summary="S",
        data_availability="unknown status",
    )
    assert ts3.data_availability == "proprietary"

def test_negated_public_text_parses_as_proprietary():
    # Example 1 reported by user: "preventing public access..."
    text1 = "Data availability is classified as proprietary. The study involves clinical patient data, preventing public access to raw data."
    status, location = parse_data_availability_statement(text1)
    assert status == "proprietary"
    assert location is None

    # Example 3 reported by user: "...are not publicly available"
    text3 = "Proprietary (Data from the 1000 Genomes Project and low-coverage ONT data used for polishing are not publicly available)."
    status, location = parse_data_availability_statement(text3)
    assert status == "proprietary"
    assert location is None

def test_uk_biobank_parses_as_restricted():
    # Example 4 reported by user: UK Biobank URL/text
    text = "Data available via UK Biobank application: https://www.ukbiobank.ac.uk/"
    status, location = parse_data_availability_statement(text)
    assert status == "restricted"
    assert location is None  # Restricted access datasets do not provide open download URLs

def test_undetermined_cohort_parses_as_proprietary():
    # Example 2 reported by user: DDD-Africa cohort dataset without public repository
    text = "Data from the Deciphering Developmental Disorders in Africa study (DDD-Africa) dataset."
    status, location = parse_data_availability_statement(text)
    assert status == "proprietary"
    assert location is None

def test_publicly_available_repository_extraction():
    text = "1000 Genomes Project data downloaded from https://www.internationalgenome.org"
    status, location = parse_data_availability_statement(text)
    assert status == "publicly_available"
    assert location == "https://www.internationalgenome.org"

    text_geo = "Gene Expression Omnibus accession GSE123456"
    status_geo, loc_geo = parse_data_availability_statement(text_geo)
    assert status_geo == "publicly_available"
    assert loc_geo == "GSE123456"
