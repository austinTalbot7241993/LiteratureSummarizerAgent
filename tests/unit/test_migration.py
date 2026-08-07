import sqlite3
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from lea.db.session import create_tables
from lea.db.models import TechnicalSummaryModel, Base
from lea.db.repository import LEARepository
from lea.llm.schemas import TechnicalSummary, PaperAvailabilityStatus


def test_case22_legacy_proprietary_db_rows_migrate_to_unclear(tmp_path):
    db_path = tmp_path / "legacy_test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create legacy table schema with data_availability column containing 'proprietary'
    cursor.execute("""
        CREATE TABLE technical_summaries (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            candidate_paper_id TEXT NOT NULL,
            problem_formulation TEXT NOT NULL,
            methodological_novelty TEXT NOT NULL,
            empirical_findings TEXT NOT NULL,
            paragraph_summary TEXT NOT NULL,
            data_availability TEXT DEFAULT 'proprietary',
            data_location TEXT,
            model_name TEXT NOT NULL,
            created_at TEXT
        );
    """)
    cursor.execute("""
        INSERT INTO technical_summaries (id, run_id, candidate_paper_id, problem_formulation, methodological_novelty, empirical_findings, paragraph_summary, data_availability, model_name)
        VALUES ('sum-1', 'run-1', 'cand-1', 'P', 'M', 'E', 'S', 'proprietary', 'test-model');
    """)
    cursor.execute("""
        INSERT INTO technical_summaries (id, run_id, candidate_paper_id, problem_formulation, methodological_novelty, empirical_findings, paragraph_summary, data_availability, model_name)
        VALUES ('sum-2', 'run-1', 'cand-2', 'P', 'M', 'E', 'S', 'publicly_available', 'test-model');
    """)
    conn.commit()
    conn.close()

    # Run create_tables on legacy DB engine
    engine = create_engine(f"sqlite:///{db_path}")
    create_tables(engine)

    with engine.connect() as check_conn:
        res1 = check_conn.execute(text("SELECT data_availability FROM technical_summaries WHERE id = 'sum-1';")).scalar()
        res2 = check_conn.execute(text("SELECT data_availability FROM technical_summaries WHERE id = 'sum-2';")).scalar()

        # Legacy 'proprietary' migrated to 'unclear'
        assert res1 == "unclear"
        # Legacy 'publicly_available' preserved
        assert res2 == "publicly_available"


def test_case23_no_substantive_proprietary_defaults():
    # TechnicalSummary schema requires explicit status
    with pytest.raises(ValueError):
        TechnicalSummary(
            problem_formulation="P",
            methodological_novelty="M",
            empirical_findings="E",
            paragraph_summary="S"
        )

    # Database model TechnicalSummaryModel column has no default='proprietary'
    col_default = TechnicalSummaryModel.data_availability.property.columns[0].default
    assert col_default is None, "data_availability Column must not have a default='proprietary'"
