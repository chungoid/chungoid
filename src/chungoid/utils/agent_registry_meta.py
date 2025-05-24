from enum import Enum

class AgentCategory(str, Enum):
    TESTING_MOCK = "TESTING_MOCK"
    SYSTEM_ORCHESTRATION = "SYSTEM_ORCHESTRATION"
    AUTONOMOUS_COORDINATION = "AUTONOMOUS_COORDINATION" # For agents like ARCA
    REQUIREMENTS_ANALYSIS = "REQUIREMENTS_ANALYSIS" # For ProductAnalystAgent, RTA (partial)
    PLANNING_AND_DESIGN = "PLANNING_AND_DESIGN" # For ArchitectAgent, BlueprintToFlowAgent
    CODE_GENERATION = "CODE_GENERATION"
    CODE_INTEGRATION = "CODE_INTEGRATION" # For SmartCodeIntegrationAgent
    CODE_EDITING = "CODE_EDITING"     # General code modification
    CODE_REMEDIATION = "CODE_REMEDIATION" # For CodeDebuggingAgent
    TEST_GENERATION = "TEST_GENERATION"
    TEST_EXECUTION = "TEST_EXECUTION"   # For SystemTestRunnerAgent
    QUALITY_ASSURANCE = "QUALITY_ASSURANCE" # For BlueprintReviewer, RTA (partial), PRAA (partial)
    RISK_ASSESSMENT = "RISK_ASSESSMENT" # For ProactiveRiskAssessorAgent
    DOCUMENTATION_GENERATION = "DOCUMENTATION_GENERATION" # For ProjectDocumentationAgent
    DATA_ANALYSIS = "DATA_ANALYSIS"
    FILE_MANAGEMENT = "FILE_MANAGEMENT" # For SystemFileSystemAgent
    AUTONOMOUS_PROJECT_ENGINE = "AUTONOMOUS_PROJECT_ENGINE" # For overarching project build/execution agents
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
SYSTEM_MASTER_PLANNER_REVIEWER_AGENT_V1 = "system.master_planner_reviewer_agent_v1"
CORE_STAGE_EXECUTOR_AGENT_V1 = "CoreStageExecutorAgent_v1"

# Core Functional Agents
CORE_CODE_GENERATOR_AGENT_V1 = "CoreCodeGeneratorAgent_v1"
CORE_TEST_GENERATOR_AGENT_V1 = "CoreTestGeneratorAgent_v1"

# Mock Agents (primarily for testing and MVP)
MOCK_HUMAN_INPUT_AGENT_V1 = "MockHumanInputAgent_v1" 