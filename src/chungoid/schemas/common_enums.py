from enum import Enum

"""Common enumerations used across the chungoid system."""


class StageStatus(Enum):
    """Represents the completion status of a stage in a workflow."""

    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"
    COMPLETED_FAILURE = "COMPLETED_FAILURE"
    SUCCESS = "COMPLETED_SUCCESS"  # Was "PASS"
    FAILURE = "COMPLETED_FAILURE"  # Was "FAIL"
    RUNNING = "RUNNING" # Added for orchestrator to indicate active stage
    PENDING = "PENDING" # Added for initializing stage status
    ERROR = "ERROR"   # Added for fatal errors during execution
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS" # ADDED for PROCEED_AS_IS
    # Could add SKIPPED, etc. in the future 


class FlowPauseStatus(Enum):
    """Represents the reason or status of a paused flow."""

    NOT_PAUSED = "NOT_PAUSED" # ADDED: Represents a non-paused state

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

    CRITICAL_ERROR_REQUIRES_MANUAL_INTERVENTION = "CRITICAL_ERROR_REQUIRES_MANUAL_INTERVENTION" # ADDED

class OnFailureAction(str, Enum):
    """Defines actions to take when a master stage encounters an error."""
    FAIL_MASTER_FLOW = "FAIL_MASTER_FLOW"
    GOTO_MASTER_STAGE = "GOTO_MASTER_STAGE"
    PAUSE_FOR_INTERVENTION = "PAUSE_FOR_INTERVENTION"
    INVOKE_REVIEWER = "INVOKE_REVIEWER"  # This is used by orchestrator as default
    RETRY_STAGE = "RETRY_STAGE"          # This is also used by orchestrator 

class HumanReviewDecision(str, Enum):
    """Defines the possible decisions a human can make during a project review."""
    PROCEED_TO_NEXT_AUTONOMOUS_PHASE = "PROCEED_TO_NEXT_AUTONOMOUS_PHASE"
    INITIATE_REFINEMENT_CYCLE = "INITIATE_REFINEMENT_CYCLE"
    MODIFY_PROJECT_GOAL = "MODIFY_PROJECT_GOAL"
    PAUSE_PROJECT = "PAUSE_PROJECT"
    ARCHIVE_PROJECT_SUCCESS = "ARCHIVE_PROJECT_SUCCESS"
    ARCHIVE_PROJECT_FAILURE = "ARCHIVE_PROJECT_FAILURE"

class FlowStatus(Enum):
    """Represents the overall status of a flow execution."""
    RUNNING = "RUNNING"
    COMPLETED_SUCCESS = "COMPLETED_SUCCESS"
    COMPLETED_FAILURE = "COMPLETED_FAILURE"
    PAUSED = "PAUSED"
    PAUSED_CRITICAL = "PAUSED_CRITICAL" # For pauses that require intervention and might not be resumable without changes

class ResumeActionType(str, Enum):
    """Defines the actions that can be taken when resuming a paused flow."""
    RETRY_STAGE_AS_IS = "RETRY_STAGE_AS_IS"
    RETRY_STAGE_WITH_CHANGES = "RETRY_STAGE_WITH_CHANGES"
    SKIP_STAGE_AND_PROCEED = "SKIP_STAGE_AND_PROCEED"
    BRANCH_TO_STAGE = "BRANCH_TO_STAGE"
    ABORT_FLOW = "ABORT_FLOW"
    PROVIDE_CLARIFICATION = "PROVIDE_CLARIFICATION" # User provides data for a clarification checkpoint
    # Could add more specific actions like MODIFY_SHARED_CONTEXT_AND_RETRY, etc.

class ReviewerActionType(str, Enum):
    """Actions a reviewer agent can suggest after a stage failure or checkpoint."""
    RETRY_STAGE_AS_IS = "RETRY_STAGE_AS_IS"
    RETRY_STAGE_WITH_CHANGES = "RETRY_STAGE_WITH_CHANGES"
    PROCEED_AS_IS = "PROCEED_AS_IS" # Mark current stage as completed with warning, and proceed
    PROCEED_TO_NEXT_STAGE = "PROCEED_TO_NEXT_STAGE" # Equivalent to skipping the problematic part of current, and moving to its designated next
    BRANCH_TO_NEW_STAGE = "BRANCH_TO_NEW_STAGE"
    FAIL_THE_FLOW = "FAIL_THE_FLOW"
    ESCALATE_TO_USER = "ESCALATE_TO_USER"
    REQUEST_CLARIFICATION_FROM_USER = "REQUEST_CLARIFICATION_FROM_USER" # Suggests pausing for user input
    ADD_CLARIFICATION_STAGE = "ADD_CLARIFICATION_STAGE" # ADDED MISSING MEMBER
    MODIFY_MASTER_PLAN = "MODIFY_MASTER_PLAN" # Suggests a new master plan
    NO_ACTION_SUGGESTED = "NO_ACTION_SUGGESTED" # ADDED MISSING MEMBER

__all__ = [
    "StageStatus", 
    "FlowPauseStatus", 
    "OnFailureAction",
    "HumanReviewDecision", # Add new enum to __all__
    "FlowStatus", # Add FlowStatus to __all__
    "ResumeActionType",
    "ReviewerActionType"
] 