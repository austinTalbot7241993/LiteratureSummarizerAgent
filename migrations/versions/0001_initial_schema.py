"""initial schema

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-08-01 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

    op.create_table(
        'papers',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('sha256_hash', sa.String(64), unique=True, nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('authors', JSONB(), server_default='[]'),
        sa.Column('doi', sa.String(255), nullable=True),
        sa.Column('arxiv_id', sa.String(255), nullable=True),
        sa.Column('openalex_id', sa.String(255), nullable=True),
        sa.Column('s2_id', sa.String(255), nullable=True),
        sa.Column('publication_year', sa.Integer(), nullable=True),
        sa.Column('venue', sa.Text(), nullable=True),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('pdf_path', sa.Text(), nullable=True),
        sa.Column('is_open_access', sa.Boolean(), server_default='false'),
        sa.Column('oa_pdf_url', sa.Text(), nullable=True),
        sa.Column('raw_bibtex', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'paper_references',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('source_paper_id', UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('raw_citation', sa.Text(), nullable=True),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('authors', JSONB(), server_default='[]'),
        sa.Column('doi', sa.String(255), nullable=True),
        sa.Column('arxiv_id', sa.String(255), nullable=True),
        sa.Column('openalex_id', sa.String(255), nullable=True),
        sa.Column('s2_id', sa.String(255), nullable=True),
        sa.Column('year', sa.Integer(), nullable=True),
        sa.Column('target_paper_id', UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='SET NULL'), nullable=True),
        sa.Column('extraction_method', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'discovery_runs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('input_paper_id', UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_status', sa.String(50), nullable=False, server_default='initialized'),
        sa.Column('exclusion_status', sa.String(50), nullable=False, server_default='complete'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'candidate_papers',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('run_id', UUID(as_uuid=True), sa.ForeignKey('discovery_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('paper_id', UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('score', sa.Float(), server_default='0.0'),
        sa.Column('rrf_rank', sa.Integer(), nullable=True),
        sa.Column('source_apis', JSONB(), server_default='[]'),
        sa.Column('open_access_url', sa.Text(), nullable=True),
        sa.Column('pdf_path', sa.Text(), nullable=True),
        sa.Column('is_downloaded', sa.Boolean(), server_default='false'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'text_chunks',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('paper_id', UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', UUID(as_uuid=True), sa.ForeignKey('discovery_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('chunk_type', sa.String(20), nullable=False),
        sa.Column('parent_id', UUID(as_uuid=True), sa.ForeignKey('text_chunks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('embedding', Vector(1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )

    op.create_table(
        'technical_summaries',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('run_id', UUID(as_uuid=True), sa.ForeignKey('discovery_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('candidate_paper_id', UUID(as_uuid=True), sa.ForeignKey('candidate_papers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('problem_formulation', sa.Text(), nullable=False),
        sa.Column('methodological_novelty', sa.Text(), nullable=False),
        sa.Column('empirical_findings', sa.Text(), nullable=False),
        sa.Column('paragraph_summary', sa.Text(), nullable=False),
        sa.Column('model_name', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now())
    )


def downgrade() -> None:
    op.drop_table('technical_summaries')
    op.drop_table('text_chunks')
    op.drop_table('candidate_papers')
    op.drop_table('discovery_runs')
    op.drop_table('paper_references')
    op.drop_table('papers')
