from enum import Enum

class AgentCategory(str, Enum):
    TESTING_MOCK = "TESTING_MOCK"
    SYSTEM_ORCHESTRATION = "SYSTEM_ORCHESTRATION"
    CODE_GENERATION = "CODE_GENERATION" # Placeholder
    DATA_ANALYSIS = "DATA_ANALYSIS"   # Placeholder
    # Add more categories as needed

class AgentVisibility(str, Enum):
    INTERNAL = "INTERNAL" # For system agents not typically user-selectable
    PUBLIC = "PUBLIC"   # For agents that can be listed or selected by users
    # Add more visibility levels as needed

# AgentCard itself is defined in chungoid.utils.agent_registry
# This file is for associated metadata types like enums, or if AgentCard
# were to become very complex and split. 

# System Agents
SYSTEM_MASTER_PLANNER_AGENT_V1 = "SystemMasterPlannerAgent_v1"
SYSTEM_MASTER_PLANNER_REVIEWER_AGENT_V1 = "SystemMasterPlannerReviewerAgent_v1"
CORE_STAGE_EXECUTOR_AGENT_V1 = "CoreStageExecutorAgent_v1"

# Core Functional Agents
CORE_CODE_GENERATOR_AGENT_V1 = "CoreCodeGeneratorAgent_v1"
CORE_TEST_GENERATOR_AGENT_V1 = "CoreTestGeneratorAgent_v1"

# Mock Agents (primarily for testing and MVP)
MOCK_HUMAN_INPUT_AGENT_V1 = "MockHumanInputAgent_v1" 