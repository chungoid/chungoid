from enum import Enum

"""Common enumerations used across the chungoid system."""


class StageStatus(Enum):
    """Represents the completion status of a stage in a workflow."""

    SUCCESS = "PASS"
    FAILURE = "FAIL"
    RUNNING = "RUNNING" # Added for orchestrator to indicate active stage
    # Could add PENDING, SKIPPED, etc. in the future 