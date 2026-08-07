from typing import List
from lea.llm.schemas import DatasetAvailability, DatasetAvailabilityStatus, PaperAvailabilityStatus


def compute_overall_paper_status(datasets: List[DatasetAvailability], has_evidence: bool) -> PaperAvailabilityStatus:
    if not datasets:
        return PaperAvailabilityStatus.NOT_REPORTED if not has_evidence else PaperAvailabilityStatus.UNCLEAR

    # Collect distinct dataset statuses
    statuses = {d.status for d in datasets}

    # Filter out NOT_REPORTED if substantive statuses exist
    substantive = {s for s in statuses if s != DatasetAvailabilityStatus.NOT_REPORTED}

    if not substantive:
        # Only NOT_REPORTED
        return PaperAvailabilityStatus.NOT_REPORTED

    if len(substantive) > 1:
        # Multiple materially different statuses across datasets -> MIXED
        return PaperAvailabilityStatus.MIXED

    single_status = list(substantive)[0]
    if single_status == DatasetAvailabilityStatus.PUBLICLY_AVAILABLE:
        return PaperAvailabilityStatus.PUBLICLY_AVAILABLE
    elif single_status == DatasetAvailabilityStatus.RESTRICTED:
        return PaperAvailabilityStatus.RESTRICTED
    elif single_status == DatasetAvailabilityStatus.NOT_AVAILABLE:
        return PaperAvailabilityStatus.NOT_AVAILABLE
    elif single_status == DatasetAvailabilityStatus.UNCLEAR:
        return PaperAvailabilityStatus.UNCLEAR

    return PaperAvailabilityStatus.UNCLEAR
