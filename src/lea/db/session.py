import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from lea.db.models import Base

_engine = None
_SessionLocal = None

def init_db(database_url: str):
    global _engine, _SessionLocal
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)

    is_sqlite = database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    _engine = create_engine(database_url, connect_args=connect_args)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    create_tables(_engine)
    return _engine

def get_engine():
    global _engine
    if _engine is None:
        from lea.config import load_config
        cfg = load_config()
        init_db(cfg.services.database_url)
    return _engine

def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        get_engine()
    return _SessionLocal

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def create_tables(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(bind=engine)

    from lea.logging import logger

    with engine.connect() as conn:
        is_sqlite = engine.dialect.name == "sqlite"
        summary_columns = [
            ("data_availability", "VARCHAR(50)"),
            ("data_location", "TEXT"),
            ("data_availability_assessment", "JSONB" if not is_sqlite else "JSON")
        ]
        chunk_columns = [
            ("section_title", "TEXT"),
            ("page_number", "INTEGER")
        ]

        if is_sqlite:
            tech_info = [r[1] for r in conn.execute(text("PRAGMA table_info(technical_summaries);")).fetchall()]
            for col_name, col_type in summary_columns:
                if col_name not in tech_info:
                    try:
                        conn.execute(text(f"ALTER TABLE technical_summaries ADD COLUMN {col_name} {col_type};"))
                        conn.commit()
                    except Exception as exc:
                        logger.error(f"Failed to add column {col_name} to technical_summaries: {exc}")
                        conn.rollback()
                        raise

            chunk_info = [r[1] for r in conn.execute(text("PRAGMA table_info(text_chunks);")).fetchall()]
            for col_name, col_type in chunk_columns:
                if col_name not in chunk_info:
                    try:
                        conn.execute(text(f"ALTER TABLE text_chunks ADD COLUMN {col_name} {col_type};"))
                        conn.commit()
                    except Exception as exc:
                        logger.error(f"Failed to add column {col_name} to text_chunks: {exc}")
                        conn.rollback()
                        raise
        else:
            for col_name, col_type in summary_columns:
                try:
                    conn.execute(text(f"ALTER TABLE technical_summaries ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                    conn.commit()
                except Exception as exc:
                    logger.error(f"Failed to add column {col_name} to technical_summaries: {exc}")
                    conn.rollback()
                    raise

            for col_name, col_type in chunk_columns:
                try:
                    conn.execute(text(f"ALTER TABLE text_chunks ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                    conn.commit()
                except Exception as exc:
                    logger.error(f"Failed to add column {col_name} to text_chunks: {exc}")
                    conn.rollback()
                    raise

        # Migration decision: Map legacy 'proprietary' values to 'unclear'
        # Old implementation conflated private data, missing evidence, parser failure, and undetermined status.
        try:
            conn.execute(text("UPDATE technical_summaries SET data_availability = 'unclear' WHERE LOWER(data_availability) = 'proprietary';"))
            conn.commit()
        except Exception as exc:
            logger.error(f"Failed to migrate legacy proprietary rows to unclear: {exc}")
            conn.rollback()
            raise

