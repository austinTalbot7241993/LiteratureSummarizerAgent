import os
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from lea.db.models import Base

_engine = None
_SessionLocal = None

def init_db(database_url: str):
    global _engine, _SessionLocal
    # Handle SQLite for testing if database_url is sqlite
    is_sqlite = database_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    _engine = create_engine(database_url, connect_args=connect_args)
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine

def get_engine():
    global _engine
    if _engine is None:
        db_url = os.getenv("LEA_DATABASE_URL", "postgresql://lea_user:lea_pass@localhost:5432/lea_db")
        init_db(db_url)
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
