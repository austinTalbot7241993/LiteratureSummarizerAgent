class LEAError(Exception):
    """Base exception for LEA application errors."""
    pass

class IngestError(LEAError):
    """Raised when paper ingestion or parsing fails."""
    pass

class BibliographyExtractionError(LEAError):
    """Raised when bibliography extraction fails or is incomplete under strict settings."""
    pass

class ExclusionViolationError(LEAError):
    """Raised when candidate discovery violates the citation exclusion invariant."""
    pass

class AcquisitionError(LEAError):
    """Raised when open-access PDF downloading or validation fails."""
    pass

class SummaryValidationError(LEAError):
    """Raised when structured LLM summary validation fails."""
    pass

class DatabaseError(LEAError):
    """Raised when database operations encounter fatal errors."""
    pass
