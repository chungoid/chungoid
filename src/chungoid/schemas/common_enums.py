from enum import Enum

"""Common enumerations used across the chungoid system."""


class StageStatus(Enum):
    """Represents the completion status of a stage in a workflow."""

    SUCCESS = "PASS"
    FAILURE = "FAIL"
    RUNNING = "RUNNING" # Added for orchestrator to indicate active stage
    PENDING = "PENDING" # Added for initializing stage status
    # Could add SKIPPED, etc. in the future 


class FlowPauseStatus(Enum):
    """Represents the reason or status of a paused flow."""

    # General Pause Reasons
    PAUSED_FOR_INTERVENTION = "PAUSED_FOR_INTERVENTION" # Generic human intervention needed
    PAUSED_FOR_CHECKPOINT = "PAUSED_FOR_CHECKPOINT"   # Paused at a predefined review/checkpoint

    # Autonomous Flow - Failures
    PAUSED_AUTONOMOUS_FAILURE_AGENT_REPORTED = "PAUSED_AUTONOMOUS_FAILURE_AGENT_REPORTED"
    PAUSED_AUTONOMOUS_FAILURE_UNHANDLED_EXCEPTION = "PAUSED_AUTONOMOUS_FAILURE_UNHANDLED_EXCEPTION"
    PAUSED_AUTONOMOUS_FAILURE_CRITERIA = "PAUSED_AUTONOMOUS_FAILURE_CRITERIA" # Success criteria not met

    # Autonomous Flow - Clarifications (P2.5)
    PAUSED_CLARIFICATION_NEEDED_BY_ORCHESTRATOR = "PAUSED_CLARIFICATION_NEEDED_BY_ORCHESTRATOR"
    PAUSED_CLARIFICATION_NEEDED_BY_AGENT = "PAUSED_CLARIFICATION_NEEDED_BY_AGENT"
    PAUSED_CLARIFICATION_NEEDED_AT_DSL_CHECKPOINT = "PAUSED_CLARIFICATION_NEEDED_AT_DSL_CHECKPOINT" # New for explicit DSL checkpoints

    # Master Flow Specific (from blueprint, potentially could be merged or kept distinct)
    PAUSED_FOR_MASTER_CHECKPOINT_BEFORE = "PAUSED_FOR_MASTER_CHECKPOINT_BEFORE" # Master flow specific checkpoint
    PAUSED_FOR_AGENT_FAILURE_IN_MASTER = "PAUSED_FOR_AGENT_FAILURE_IN_MASTER" # Agent failure within a master flow stage

    # User Escalation (aligning with test_orchestrator_reviewer_integration.py)
    USER_INTERVENTION_REQUIRED = "USER_INTERVENTION_REQUIRED" # Explicitly for ESCALATE_TO_USER action

    # Fallback/Unknown
    PAUSED_UNKNOWN = "PAUSED_UNKNOWN" 

class OnFailureAction(str, Enum):
    """Defines actions to take when a master stage encounters an error."""
    FAIL_MASTER_FLOW = "FAIL_MASTER_FLOW"
    GOTO_MASTER_STAGE = "GOTO_MASTER_STAGE"
    PAUSE_FOR_INTERVENTION = "PAUSE_FOR_INTERVENTION"
    INVOKE_REVIEWER = "INVOKE_REVIEWER"  # This is used by orchestrator as default
    RETRY_STAGE = "RETRY_STAGE"          # This is also used by orchestrator 