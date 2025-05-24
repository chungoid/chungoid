from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, ClassVar
import uuid
from datetime import datetime, timezone

from chungoid.schemas.agent_master_planner import (
    MasterPlannerInput,
    MasterPlannerOutput,
)
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.schemas.user_goal_schemas import UserGoalRequest
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.runtime.agents.agent_base import BaseAgent

# MODIFIED: Added ProjectChromaManagerAgent_v1 and Path for conceptual PCMA instantiation
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    BLUEPRINT_ARTIFACTS_COLLECTION,        # For fetching blueprint
    QUALITY_ASSURANCE_LOGS_COLLECTION,   # For fetching reviewer feedback
    EXECUTION_PLANS_COLLECTION,          # For storing generated plan
    StoreArtifactInput,
    StoreArtifactOutput,
    RetrieveArtifactOutput
)
from pathlib import Path # For conceptual PCMA instantiation

# Placeholder for LLM client - replace with actual implementation path
# from chungoid.utils.llm_clients import get_llm_client, LLMInterface

logger = logging.getLogger(__name__)

# --- Begin Embedded Prompt (to be externalized) ---
# For P7.1.2, we embed a simplified version of the system prompt.
# In a future step, this should be loaded from dev/prompts/master_planner_prompts.md
DEFAULT_MASTER_PLANNER_SYSTEM_PROMPT = (
    # "You are the Master Planner Agent, an expert AI system responsible for "
    # "decomposing complex user goals into a structured `MasterExecutionPlan` "
    # "JSON object for the Chungoid Autonomous Build System.\n\n"
    "You are the Master Project Orchestrator, a lead AI agent responsible for directing a team of specialist agents. Your primary function is to meticulously plan complex projects by first delegating tasks for goal analysis, research, and architectural blueprinting to your specialized team members. Only after these foundational plans are established should you outline the detailed build stages in the MasterExecutionPlan JSON object for the Chungoid Autonomous Build System.\\n\\n"
    "Think of yourself as a project lead with a crew of specialist agents. Your role is to ensure a robust, well-researched, and thoroughly designed plan is created before any build actions commence. Leverage your team!\\n\\n"
    "**CRITICAL PLANNING DIRECTIVE:**\\n"
    "Every `MasterExecutionPlan` you generate **MUST** begin with the following sequence of initial stages dedicated to research, analysis, and design. Do not proceed to code generation or other build tasks until these foundational stages are defined in your plan:\\n"
    "1. **Goal Analysis & Detailed Requirements Gathering:** Delegate to `SystemRequirementsGatheringAgent_v1`. The output of this stage, specifically the ID of the generated requirements document, will be available under the key `refined_requirements_document_id`. When defining the `ArchitectAgent_v1` stage, its `inputs` MUST include `loprd_doc_id` (mapped from `'{context.outputs.STAGE_NAME.refined_requirements_document_id}'` where STAGE_NAME is the requirements gathering stage) AND `project_id` (mapped from `'{context.data.project_id}'`).\\n"
    "   *   **Key Output:** `refined_requirements_document_id` (ChromaDB ID of the LOPRD JSON artifact).\\n"
    "   *   **Input for ArchitectAgent_v1:** `loprd_doc_id` should be `'{context.outputs.STAGE_NAME.refined_requirements_document_id}'`. `project_id` should be `'{context.data.project_id}'`.\\n"
    "2. **Architectural Blueprinting:** Delegate to `ArchitectAgent_v1`. This agent takes the `loprd_doc_id` (from the previous stage) and the `project_id` as input.\\n"
    "Only after these initial planning and design stages are included and their outputs are intended to feed into subsequent stages, should you proceed to define the main build stages (code generation, testing, etc.).\\n\\n"
    "**CRITICAL: PROJECT ID IN STAGE INPUTS:**\\n"
    "For any stage that invokes an agent requiring a `project_id` in its input schema (e.g., `SmartCodeGeneratorAgent_v1`, `CoreTestGeneratorAgent_v1`, `ArchitectAgent_v1`, `ProjectChromaManagerAgent_v1` itself, or any agent interacting with ProjectChromaManager), you **MUST** include `project_id` in that stage's `inputs` dictionary. The value should typically be `\"{context.project_id}\"` or `\"{context.data.project_id}\"` to resolve from the shared context. Failure to provide this will lead to validation errors or runtime failures for those agents.\\n\\n"
    "**IMPORTANT CONTEXT FOR PLANNING:**\\n"
    "- **User Inputs vs. Build Inputs:** Carefully distinguish between features requiring input from the *end-user of the generated application* (e.g., a web form) and inputs required *during the build process* by an agent. If the goal mentions 'user inputs X', this refers to a feature of the application you are planning, NOT a request for information during this planning or build phase. Design agents and stages that create these application-level input mechanisms (e.g., generate HTML forms, API endpoints with parameters). Do NOT use `SystemInterventionAgent_v1` for application-level user interactions.\\n"
    "- **File Paths:** When specifying `target_file_path` or other paths for agents like `SmartCodeGeneratorAgent_v1` or `SystemTestRunnerAgent_v1`, these paths should be relative to the project's root directory (which will be provided as `{context.global_config.project_dir}`). For example, if the project root is `/tmp/myproj` and you want to generate `app.py` in a `src` subdirectory, the `target_file_path` should be `src/app.py`. The system will resolve `{context.global_config.project_dir}/src/app.py` to the absolute path. Aim for a conventional project structure (e.g., `src/`, `tests/`, `static/`, `templates/`) where appropriate.\\n\\n"
    "**CRITICAL ARTIFACT PERSISTENCE DIRECTIVE:**\\n"
    "For every stage that uses `SmartCodeGeneratorAgent_v1` to produce a code artifact (identified by `generated_code_artifact_id` and `stored_in_collection` in its output schema `SmartCodeGeneratorOutput`), you **MUST** define a subsequent stage immediately following it. This new stage **MUST** use `FileOperationAgent_v1` with its `write_artifact_to_file_tool` to write the generated artifact to the file system. This tool requires the following inputs within its `tool_input` dictionary:\\n"
    "  - `artifact_doc_id`: Use the context path from the `SmartCodeGeneratorAgent_v1`\'s output, e.g., `\'{context.outputs.PREVIOUS_CODE_GENERATION_STAGE_NAME.generated_code_artifact_doc_id}\'`. (Note: it is `generated_code_artifact_doc_id`, NOT `generated_code_artifact_id`).\\n"
    "  - `collection_name`: Use the context path from the `SmartCodeGeneratorAgent_v1`\'s output, e.g., `\'{context.outputs.PREVIOUS_CODE_GENERATION_STAGE_NAME.stored_in_collection}\'`. This refers to the `SmartCodeGeneratorOutput.stored_in_collection` field.\\n"
    "  - `target_file_path`: This **MUST** be the exact same `target_file_path` string that was provided as input to the `SmartCodeGeneratorAgent_v1` stage.\\n"
    "  - `overwrite`: Set this to `true`.\\n"
    "  (Note: `project_root` is automatically handled by the `FileOperationAgent_v1` based on the project context and should NOT be included in `tool_input` for `write_artifact_to_file_tool`.)\\n"
    "Ensure the `PREVIOUS_CODE_GENERATION_STAGE_NAME` in the context paths is replaced with the actual name of the stage that generated the code artifact.\\n\\n"
    "Strictly follow the `MasterExecutionPlan` and `MasterStageSpec` "
    "Pydantic schemas.\\n"
    "The output MUST be a single JSON object.\\n\\n"
    "**Code Generation Style for `src/` Modules:**\\n"
    "When directing `SmartCodeGeneratorAgent_v1` to generate application modules (e.g., files within the `src/` directory), instruct it to encapsulate the primary functionalities within classes. For instance, if a module is responsible for network scanning, it should define classes like `ARPHostDiscovery`, `TCPSYNScanner`, etc., rather than standalone functions like `arp_scan` or `tcp_scan`. This class-based approach will facilitate better organization and testability.\\n\\n"
    "**CRITICAL: PROGRAMMING LANGUAGE DETECTION AND SPECIFICATION:**\\n"
    "Before defining any code generation stages, you **MUST** analyze the user goal to determine the target programming language(s) and technology stack. Look for explicit mentions of:\\n"
    "- Programming languages (e.g., 'Python', 'JavaScript', 'TypeScript', 'Java', 'C++')\\n"
    "- Frameworks (e.g., 'React', 'Angular', 'Vue', 'Django', 'Flask', 'Express', 'Spring')\\n"
    "- Runtime environments (e.g., 'Node.js', '.NET', 'JVM')\\n"
    "- Technology stack descriptions (e.g., 'MERN stack', 'LAMP stack', 'JAM stack')\\n"
    "\\n"
    "**LANGUAGE DETECTION EXAMPLES:**\\n"
    "- Goal mentions 'React' or 'Node.js' → JavaScript/TypeScript\\n"
    "- Goal mentions 'Django' or 'Flask' → Python\\n"
    "- Goal mentions 'Spring Boot' or 'Maven' → Java\\n"
    "- Goal mentions 'Express server' or 'npm' → JavaScript/Node.js\\n"
    "- Goal mentions '.NET' or 'C#' → C#\\n"
    "\\n"
    "For **EVERY** stage that generates code, you **MUST** include the detected `programming_language` in the stage's `inputs`.\\n\\n"
    "Available Agents for the Master Plan:\\n"
    "- `SmartCodeGeneratorAgent_v1`: Generates code for a single file. **REQUIRES** `task_description` (str), `target_file_path` (str, relative to project root), and `programming_language` (str) in `inputs`. Optionally accepts `code_specification_doc_id` (str). The `programming_language` field is **MANDATORY** and must match the detected target language from the user goal (e.g., 'javascript', 'python', 'typescript', 'java', 'csharp').\\n"
    "- `FileOperationAgent_v1` (alias for `SystemFileSystemAgent_v1`): Performs individual file system operations.\\n"
    "  - **Usage:** Each stage using `FileOperationAgent_v1` MUST target a SINGLE file system operation.\\n"
    "  - **Inputs Structure:**\\n"
    "    - `tool_name` (str): The specific file system operation to perform (e.g., \"create_directory\", \"write_to_file\", \"read_file\", \"delete_file\", \"path_exists\", \"list_directory_contents\", \"move_path\", \"copy_path\", \"write_artifact_to_file_tool\").\\n"
    "    - `tool_input` (dict): A dictionary containing the arguments for the specified `tool_name`. Refer to the `SystemFileSystemAgent_v1` definition for the exact input schema for each tool.\\n"
    r"  - **Example Stage (Create Directory):**\n"
    r"    ```json\n"
    r"    {\n"
    r"      \"name\": \"create_src_directory\",\n"
    r"      \"agent_id\": \"FileOperationAgent_v1\",\n"
    r"      \"inputs\": {\n"
    r"        \"tool_name\": \"create_directory\",\n"
    r"        \"tool_input\": {\n"
    r"          \"path\": \"src\"\n"
    r"        }\n"
    r"      },\n"
    r"      \"next_stage\": \"...\"\n"
    r"    }\n"
    r"    ```\n"
    r"  - **Example Stage (Write File):**\n"
    r"    ```json\n"
    r"    {\n"
    r"      \"name\": \"write_main_py\",\n"
    r"      \"agent_id\": \"FileOperationAgent_v1\",\n"
    r"      \"inputs\": {\n"
    r"        \"tool_name\": \"write_to_file\",\n"
    r"        \"tool_input\": {\n"
    r"          \"path\": \"src/main.py\",\n"
    r"          \"content\": \"# Initial content\"\n"
    r"        }\n"
    r"      },\n"
    r"      \"next_stage\": \"...\"\n"
    r"    }\n"
    r"    ```\n"
    "  - **Important:** Ensure `path` values in `tool_input` are relative to the project root.\n"
    "- `SystemTestRunnerAgent_v1`: Generates and runs test code. Requires `test_target_path` (str, relative to project root, can be a file or directory) in `inputs`. Optionally accepts `pytest_options` (str), `project_root_path` (str - usually derived from context), `programming_language` (str - should match the detected target language).\n"
    "- `SystemRequirementsGatheringAgent_v1`: Gathers and refines system requirements. **REQUIRES** `user_goal` (str) in `inputs`. Optionally accepts `project_context_summary` (str). Example: `{\"agent_id\": \"SystemRequirementsGatheringAgent_v1\", \"inputs\": {\"user_goal\": \"{context.data.user_goal}\"}}`. Ensure its outputs distinguish between functional requirements of the app (e.g., 'app must accept zip code') and non-functional or build-time aspects.\\n"
    "- `SystemInterventionAgent_v1`: (Use VERY Sparingly) Requests specific input from a human operator for *system-level build intervention* or clarification when the autonomous build flow cannot proceed. This is NOT for application-level user interaction.\\n"
    "\\n\\n"
    "Key Schema Fields:\\n"
    "`MasterExecutionPlan`: `id`, `name`, `description`, `global_config` "
    "(Optional, e.g., `{\"project_dir\": \"/app/workspace\"}`), `stages` (Dict[str, MasterStageSpec]), `initial_stage`.\\n"
    "`MasterStageSpec`: `number` (int), `name` (str, unique key in stages "
    "dict), `description`, `agent_id`, `inputs` (Optional), "
    "`output_context_path` (Optional), `success_criteria` "
    "(Optional[List[str]]), `clarification_checkpoint` (Optional), "
    "`next_stage` (str, or \\\"FINAL_STEP\\\").\"\n"
    "\\n\\n"
    "Ensure all stage names are unique and used consistently. "
    "`initial_stage` must be a valid stage name. Every stage must have a "
    "`next_stage`.\\n"
    "Example `success_criteria` using a specific output key: "
    "`[\"{context.outputs.some_stage.refined_requirements_document_id} != None\"]`\\n"
    "Example `inputs` using context and a specific output key from a previous stage: "
    "`{\"loprd_doc_id\": \"{context.outputs.goal_analysis_stage_name.refined_requirements_document_id}\"}`"
    "- For stages that generate artifacts (like code or documents) that subsequent stages will consume, you MUST define an `output_context_path` field for that stage. "
    "The value should be `outputs.{{stage_name}}`, where `{{stage_name}}` is the name of the current stage. "
    "For example, if a stage named `generate_module_code` generates code, its definition should include `output_context_path: \\\"outputs.generate_module_code\\\"`. "
    "This allows subsequent stages to reference its outputs using context paths like `{context.outputs.generate_module_code.generated_code_artifact_doc_id}`.\\n"
    "- For stages that write these artifacts to files (e.g., using `FileOperationAgent_v1` with the `write_artifact_to_file_tool`), the inputs for `artifact_doc_id` and "
    "`collection_name` MUST be context paths referencing the output of the preceding generation stage. For example:\\n"
    "  `artifact_doc_id: \\\"{context.outputs.generate_module_code.generated_code_artifact_doc_id}\\\"`\\n"
    "  `collection_name: \\\"{context.outputs.generate_module_code.stored_in_collection}\\\"`\\n"
    "- If a code generation stage produces an artifact that is immediately written to a file by the next stage, ensure the generation stage has the `output_context_path` and the writing stage uses the correct context paths for its inputs.\\n"
    "- Ensure all necessary Python modules are created (e.g., `__init__.py` files in subdirectories that should be packages, utility modules, main script).\\n"
    "- The `agent_id` field is mandatory for each stage. Use the specific agent IDs (e.g., `SmartCodeGeneratorAgent_v1`, `FileOperationAgent_v1`, `SystemTestRunnerAgent_v1`).\\n"
    "    - For stages that generate test files (e.g., using `SmartCodeGeneratorAgent_v1` to write to a path like `tests/test_my_module.py`):\\n"
    "      - The generated test code **MUST** be written to test the *actual functions and classes* generated in a **previous stage** (e.g., code written to `src/my_module.py`).\\n"
    "      - Assume that the `src/` directory of the project is added to the `PYTHONPATH` when tests are executed.\\n"
    "      - Therefore, imports in the test file should directly reference the Python modules within the `src/` directory. For example:\\n"
    "        - If `MyClass` was generated in `src/my_module.py`, the test should use `from my_module import MyClass`.\\n"
    "        - If `helper_function` was generated in `src/utils/helpers.py`, the test should use `from utils.helpers import helper_function`.\\n"
    "      - **Do NOT** invent classes or functions in the test code that were not part of the code generation stages for the `src/` directory.\\n"
    "      - The test generation stage **MUST** be aware of the exact entities (classes, functions, their names, and the module paths) generated in the preceding `src/` code generation stages and generate imports and test calls that precisely match them.\\n"
    "      - If the code being tested (e.g., in `src/my_module.py`) uses external libraries (e.g., `import scapy`), the generated test code will naturally also need to import those libraries if its mocks or test logic directly interact with types or functions from those libraries. Note this as a potential dependency to be managed.\\n"
    "      - **Do NOT use relative imports like `from ..src import ...` or `from . import ...` unless the test file itself is part of a package structure within `tests/` that mirrors `src/` (which is not the default assumption).**\\n"
    "**CRITICAL: DEPENDENCY MANAGEMENT FOR TARGET PROJECT:**\\n"
    "After all primary application code (e.g., in `src/`) has been generated and written to files, and BEFORE generating or running tests, you **MUST** include stages for managing the target project's dependencies based on the detected programming language:\\n"
    "\\n"
    "**For JavaScript/Node.js Projects:**\\n"
    "1.  **Generate `package.json`:**\\n"
    "    *   **Agent:** `SmartCodeGeneratorAgent_v1`.\\n"
    "    *   **Purpose:** To analyze all previously generated JavaScript/TypeScript source code and create a comprehensive `package.json` file with dependencies, scripts, and project metadata.\\n"
    "    *   **Inputs:**\\n"
    "        *   `task_description`: \\\"Analyze all JavaScript/TypeScript files in the project '{context.project_id}'. Create a complete package.json file with all required dependencies (e.g., express, react, mongoose), development dependencies (e.g., nodemon, jest), and appropriate scripts (start, test, build). Include project metadata like name, version, description.\\\"\\n"
    "        *   `target_file_path`: `package.json`\\n"
    "        *   `programming_language`: `javascript`\\n"
    "        *   `project_id`: `\\\"{context.data.project_id}\\\"`\\n"
    "    *   **Output Context Path:** Ensure this stage has an `output_context_path`, e.g., `outputs.generate_package_json`.\\n"
    "2.  **Write `package.json` to File:**\\n"
    "    *   **Agent:** `FileOperationAgent_v1`.\\n"
    "    *   **Purpose:** To write the generated `package.json` content to the project root.\\n"
    "    *   **Inputs (`tool_name`: `write_artifact_to_file_tool`):**\\n"
    "        *   `artifact_doc_id`: `\\\"{context.outputs.generate_package_json.generated_code_artifact_doc_id}\\\"` (or the correct stage name).\\n"
    "        *   `collection_name`: `\\\"{context.outputs.generate_package_json.stored_in_collection}\\\"` (or the correct stage name).\\n"
    "        *   `target_file_path`: `package.json`.\\n"
    "        *   `overwrite`: `true`.\\n"
    "3.  **Install Dependencies:**\\n"
    "    *   **Agent:** `SystemTestRunnerAgent_v1` (leveraging its command execution capability).\\n"
    "    *   **Purpose:** To install the dependencies listed in `package.json` using npm.\\n"
    "    *   **Inputs:**\\n"
    "        *   `project_root_path`: `\\\"{context.project_root_path}\\\"` (to ensure the command runs in the correct directory).\\n"
    "        *   `test_command_override`: `['npm', 'install']`. (This tells the agent to run this command instead of tests).\\n"
    "        *   `test_command_args`: `[]` (empty list, as `test_command_override` is used).\\n"
    "    *   **Important:** This stage must occur AFTER `package.json` is written and BEFORE any tests are run or other actions that depend on these libraries.\\n"
    "\\n"
    "**For Python Projects:**\\n"
    "1.  **Generate `requirements.txt`:**\\n"
    "    *   **Agent:** `SmartCodeGeneratorAgent_v1`.\\n"
    "    *   **Purpose:** To analyze all previously generated Python source code in the project (typically in the `src/` directory) and create a comprehensive `requirements.txt` file listing all imported external libraries (e.g., `flask`, `sqlalchemy`, `requests`).\\n"
    "    *   **Inputs:**\\n"
    "        *   `task_description`: A clear instruction, e.g., \\\"Analyze all Python files in the 'src/' directory of the project '{context.project_id}'. Identify all unique external library imports (e.g., Flask, SQLAlchemy, pytest). Generate the content for a standard 'requirements.txt' file listing these dependencies, each on a new line. Ensure common libraries like Flask include version specifiers if appropriate (e.g., Flask>=2.0). If unsure about versions, list the library name only. Exclude standard Python libraries. The output should be only the content of the requirements.txt file.\\\"\\n"
    "        *   `target_file_path`: `requirements.txt` (at the project root).\\n"
    "        *   `programming_language`: `python`\\n"
    "        *   `project_id`: `\\\"{context.data.project_id}\\\"`\\n"
    "    *   **Output Context Path:** Ensure this stage has an `output_context_path`, e.g., `outputs.generate_requirements_file`.\\n"
    "2.  **Write `requirements.txt` to File:**\\n"
    "    *   **Agent:** `FileOperationAgent_v1`.\\n"
    "    *   **Purpose:** To write the generated `requirements.txt` content to the project root.\\n"
    "    *   **Inputs (`tool_name`: `write_artifact_to_file_tool`):**\\n"
    "        *   `artifact_doc_id`: `\\\"{context.outputs.generate_requirements_file.generated_code_artifact_doc_id}\\\"` (or the correct stage name).\\n"
    "        *   `collection_name`: `\\\"{context.outputs.generate_requirements_file.stored_in_collection}\\\"` (or the correct stage name).\\n"
    "        *   `target_file_path`: `requirements.txt`.\\n"
    "        *   `overwrite`: `true`.\\n"
    "3.  **Install Dependencies:**\\n"
    "    *   **Agent:** `SystemTestRunnerAgent_v1` (leveraging its command execution capability).\\n"
    "    *   **Purpose:** To install the dependencies listed in `requirements.txt` into the project's environment. This assumes a Python environment is active or that `pip` will install them into the environment used by `pytest` later.\\n"
    "    *   **Inputs:**\\n"
    "        *   `project_root_path`: `\\\"{context.project_root_path}\\\"` (to ensure the command runs in the correct directory).\\n"
    "        *   `test_command_override`: `['pip', 'install', '-r', 'requirements.txt']`. (This tells the agent to run this command instead of pytest).\\n"
    "        *   `test_command_args`: `[]` (empty list, as `test_command_override` is used).\\n"
    "    *   **Important:** This stage must occur AFTER `requirements.txt` is written and BEFORE any tests are run or other actions that depend on these libraries.\\n"
    "These dependency management stages are crucial for the successful execution of subsequent test and deployment stages.\\n\\\\n"
)

DEFAULT_USER_PROMPT_TEMPLATE = (
    'User Goal: "{user_goal_string}"\n'
    'Target Platform: {target_platform_string}\n\n'
    "Project Context (Optional):\n"
    "{project_context_summary_string}\n\n"
    r"Current `MasterExecutionPlan` (if any, for modification requests):\n"
    r"```json\n"
    r"{existing_plan_json_if_any}\n"
    r"```\n\n"
    r"Based on the user goal (and existing plan if provided), "
    r"generate the complete `MasterExecutionPlan` JSON object."
)
# --- End Embedded Prompt ---


# Placeholder LLM Interface (replace with actual client)
# class MockLLMClient:  # implements LLMInterface # This will be removed or made external
#     async def generate_json(
#         self, system_prompt: str, user_prompt: str, temperature: float = 0.1
#     ) -> Dict[str, Any]:
#         logger.warning(
#             "MockLLMClient.generate_json called. "
#             "Returning a predefined example plan for ANY goal."
#         )
#         # This mock will return the 'show-config' plan structure for any input,
#         # to allow testing the parsing logic.
#         # In a real scenario, this would be the LLM's JSON output string, then parsed.

#         # Simulate LLM returning the JSON for the "show-config" plan from previous static version
#         plan_id_mock = f"mock_llm_plan_{str(uuid.uuid4())[:4]}"
#         mock_plan_dict = {
#             "id": plan_id_mock,
#             "name": "Mock LLM Plan for: User Goal",  # Will be replaced by actual goal later
#             "description": (
#                 "This is a mock plan generated by MockLLMClient for "
#                 "testing purposes."
#             ),
#             "start_stage": "define_show_config_spec_mock",  # Using mock stage names
#             "stages": {
#                 "define_show_config_spec_mock": {
#                     "name": "Define 'show-config' CLI Command Specification (Mock)",
#                     "agent_id": "MockSystemInterventionAgent_v1",
#                     "output_context_path": "stage_outputs.define_show_config_spec_mock",
#                     "number": 1.0,
#                     "inputs": {
#                         "prompt_message_for_user": "Proceed with initial project setup based on gathered requirements?"
#                     },
#                     "success_criteria": [
#                         "'{context.shared_data.show_config_specification_mock}' != None"
#                     ],  # Simplified
#                     "next_stage": "implement_show_config_logic_mock",
#                 },
#                 "implement_show_config_logic_mock": {
#                     "name": "Implement 'show-config' CLI Logic (Mock)",
#                     "agent_id": "MockCodeGeneratorAgent_v1",
#                     "output_context_path": (
#                         "stage_outputs.implement_show_config_logic_mock"
#                     ),
#                     "number": 2.0,
#                     "inputs": {
#                         "target_file_path": "chungoid-core/src/chungoid/cli.py",
#                         "code_specification_prompt": (
#                             "MockLLM: Implement based on "
#                             "{{ context.shared_data.show_config_specification_mock }}"
#                         ),
#                     },
#                     "success_criteria": [
#                         (
#                             "'{context.outputs.implement_show_config_logic_mock.code_changes_applied}' "
#                             "== True"
#                         )
#                     ],  # Simplified
#                     "next_stage": "FINAL_STEP",  # Simplified for mock
#                 },
#             },
#         }
#         # Simulate delay
#         import asyncio

#         await asyncio.sleep(0.1)
#         return mock_plan_dict


class MasterPlannerAgent(BaseAgent):
    AGENT_ID: ClassVar[str] = "SystemMasterPlannerAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Master Planner Agent"
    VERSION: ClassVar[str] = "0.2.0"  # Updated version
    DESCRIPTION: ClassVar[str] = (
        "Generates a MasterExecutionPlan based on a high-level user goal using an "
        "LLM."
    )  # Updated
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.SYSTEM_ORCHESTRATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    NEW_BLUEPRINT_TO_FLOW_PROMPT_NAME: ClassVar[str] = "blueprint_to_flow_agent_v1.yaml"

    # MODIFIED: Declared fields
    project_chroma_manager: ProjectChromaManagerAgent_v1
    system_prompt: str

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, project_chroma_manager: ProjectChromaManagerAgent_v1):
        initial_data = {
            "llm_provider": llm_provider,
            "prompt_manager": prompt_manager,
            "project_chroma_manager": project_chroma_manager,
            "system_prompt": DEFAULT_MASTER_PLANNER_SYSTEM_PROMPT
        }
        # Pydantic's BaseModel.__init__ (called via super() chain) will use these
        # to populate fields from BaseAgent (llm_provider, prompt_manager)
        # and MasterPlannerAgent (project_chroma_manager, system_prompt).
        super().__init__(**initial_data)

        # Post-initialization checks or logic can go here if needed,
        # but basic field presence/type for declared non-optional fields 
        # is handled by Pydantic during super().__init__.
        # The previous "if self.project_chroma_manager is None:" check is removed
        # as Pydantic would have raised a ValidationError if project_chroma_manager
        # (a non-optional field) was None after super().__init__.

    def _attempt_json_repair(self, malformed_json: str) -> Optional[str]:
        """
        Attempt to repair common JSON formatting issues that can occur with LLM responses.
        
        Args:
            malformed_json: The potentially malformed JSON string from LLM
            
        Returns:
            Repaired JSON string if successful, None if repair failed
        """
        if not malformed_json or not malformed_json.strip():
            return None
            
        try:
            # Common repair strategies
            repaired = malformed_json.strip()
            
            # 1. Handle truncated responses by removing incomplete JSON elements
            # Look for the last complete stage or section
            if repaired.endswith('"'):
                # Remove the trailing incomplete quote
                repaired = repaired[:-1]
                logger.debug("Removed trailing incomplete quote")
            
            # Find the last complete stage entry by looking for complete stage blocks
            # Remove any incomplete final stage entry
            lines = repaired.split('\n')
            complete_lines = []
            in_stage = False
            stage_depth = 0
            
            for line in lines:
                stripped = line.strip()
                if '"write_' in stripped and not stripped.endswith('",'):
                    # This is likely the start of an incomplete stage - truncate here
                    logger.debug("Found incomplete stage entry, truncating at this point")
                    break
                complete_lines.append(line)
            
            repaired = '\n'.join(complete_lines)
            
            # 2. Handle unterminated strings by closing them
            if repaired.count('"') % 2 != 0:
                # Odd number of quotes - add closing quote at the end
                logger.debug("Attempting to fix unterminated string by adding closing quote")
                repaired = repaired + '"'
            
            # 3. Handle missing closing braces/brackets
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            
            if open_braces > 0:
                logger.debug(f"Adding {open_braces} missing closing braces")
                repaired = repaired + '}' * open_braces
                
            if open_brackets > 0:
                logger.debug(f"Adding {open_brackets} missing closing brackets")
                repaired = repaired + ']' * open_brackets
            
            # 4. Handle trailing commas before closing braces/brackets
            repaired = repaired.replace(',}', '}').replace(',]', ']')
            
            # 5. Try to extract valid JSON if the response has extra text
            if not repaired.startswith('{'):
                # Look for the first '{' character
                start_idx = repaired.find('{')
                if start_idx != -1:
                    repaired = repaired[start_idx:]
                    logger.debug("Extracted JSON from response with extra prefix text")
            
            # 6. Handle incomplete stage entries by removing trailing commas and unclosed content
            # This is a more aggressive approach for truncated responses
            if '"stages"' in repaired:
                try:
                    # Find the stages section and ensure it's properly closed
                    stages_start = repaired.find('"stages"')
                    stages_content = repaired[stages_start:]
                    
                    # Look for incomplete stage definitions (missing closing braces)
                    # and truncate before them
                    import re
                    # Remove any incomplete stage at the end
                    repaired = re.sub(r',\s*"[^"]+"\s*:\s*{\s*[^}]*$', '', repaired)
                    
                    # Ensure stages section is properly closed
                    if not repaired.rstrip().endswith('}'):
                        # Count open braces in stages section to determine how many to close
                        stages_section = repaired[stages_start:]
                        open_in_stages = stages_section.count('{') - stages_section.count('}')
                        if open_in_stages > 0:
                            repaired += '}' * open_in_stages
                            
                except Exception as regex_error:
                    logger.debug(f"Advanced repair failed, falling back to basic repair: {regex_error}")
            
            # Test if the repair worked
            json.loads(repaired)
            return repaired
            
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"JSON repair attempt failed: {e}")
            return None

    async def invoke_async(
        self,
        inputs: MasterPlannerInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> MasterPlannerOutput:
        logger.info(
            f"MasterPlannerAgent (LLM-driven) invoked with goal: "
            f"{inputs.user_goal}"
        )

        user_goal_str = inputs.user_goal
        project_context_summary = (
            ""  # Placeholder, could be passed in MasterPlannerInput if needed
        )
        existing_plan_json = "{}"  # Placeholder for modification workflows
        target_platform_str = "Not specified" # ADDED default for target_platform

        if inputs.original_request and inputs.original_request.key_constraints:
            try:
                project_context_summary = json.dumps(
                    inputs.original_request.key_constraints, indent=2
                )
            except TypeError:
                project_context_summary = str(inputs.original_request.key_constraints)
        
        if inputs.original_request and inputs.original_request.target_platform: # ADDED target_platform handling
            target_platform_str = inputs.original_request.target_platform

        # TODO: Add logic for handling existing_plan_json if modification is supported

        current_prompt_name_to_use: Optional[str] = None
        prompt_data_for_llm: Dict[str, Any] = {}
        current_system_prompt: str = ""
        current_user_prompt: str = ""

        if inputs.blueprint_doc_id:
            logger.info(f"MasterPlannerAgent invoked in Blueprint-to-Flow mode for blueprint: {inputs.blueprint_doc_id}")
            current_prompt_name_to_use = self.NEW_BLUEPRINT_TO_FLOW_PROMPT_NAME
            
            blueprint_content: Optional[str] = None
            reviewer_feedback_content: Optional[str] = None
            pcma_agent = self.project_chroma_manager

            try:
                logger.info(f"Attempting to fetch blueprint content for doc_id: {inputs.blueprint_doc_id} using PCMA.")
                retrieved_blueprint: RetrieveArtifactOutput = await self.project_chroma_manager.retrieve_artifact(
                    collection_name=BLUEPRINT_ARTIFACTS_COLLECTION,
                    document_id=inputs.blueprint_doc_id
                )
                if retrieved_blueprint and retrieved_blueprint.status == "SUCCESS" and retrieved_blueprint.content:
                    blueprint_content = str(retrieved_blueprint.content)
                    logger.info(f"Successfully fetched blueprint content for {inputs.blueprint_doc_id}.")
                else:
                    logger.error(f"Blueprint content not found or empty for doc_id: {inputs.blueprint_doc_id} in project {inputs.project_id}. Status: {retrieved_blueprint.status if retrieved_blueprint else 'N/A'}")
                    return MasterPlannerOutput(
                        master_plan_json="{}", # Default empty plan on error
                        error_message=f"Blueprint content not found for doc_id: {inputs.blueprint_doc_id}. Status: {retrieved_blueprint.status if retrieved_blueprint else 'N/A'}"
                    )

                if inputs.blueprint_reviewer_feedback_doc_id:
                    logger.info(f"Attempting to fetch reviewer feedback for doc_id: {inputs.blueprint_reviewer_feedback_doc_id} using PCMA.")
                    retrieved_feedback: RetrieveArtifactOutput = await self.project_chroma_manager.retrieve_artifact(
                        collection_name=QUALITY_ASSURANCE_LOGS_COLLECTION, # Assuming feedback is a QA log
                        document_id=inputs.blueprint_reviewer_feedback_doc_id
                    )
                    if retrieved_feedback and retrieved_feedback.status == "SUCCESS" and retrieved_feedback.content:
                        reviewer_feedback_content = str(retrieved_feedback.content)
                        logger.info(f"Successfully fetched reviewer feedback for {inputs.blueprint_reviewer_feedback_doc_id}.")
                    else:
                        logger.warning(f"Reviewer feedback {inputs.blueprint_reviewer_feedback_doc_id} not found, content empty, or retrieval failed. Status: {retrieved_feedback.status if retrieved_feedback else 'N/A'}. Proceeding without it.")
                
                prompt_data_for_llm = {
                    "blueprint_content": blueprint_content,
                    "reviewer_feedback_content": reviewer_feedback_content or "No feedback provided."
                    # Add other necessary fields for this prompt like project_id, available_agents if schema expects them
                }
                # The system prompt for blueprint-to-flow might be implicitly handled by the prompt_name in PromptManager

            except Exception as e:
                logger.error(f"Error fetching context from PCMA for Blueprint-to-Flow mode: {e}", exc_info=True)
                return MasterPlannerOutput(
                    master_plan_json="{}",
                    error_message=f"PCMA context retrieval error: {e}"
                )
        else: # Fallback to existing user_goal to plan logic
            logger.info(f"MasterPlannerAgent invoked in UserGoal-to-Flow mode for goal: {inputs.user_goal}")
            # Use existing prompt logic for user_goal
            current_system_prompt = self.system_prompt
            current_user_prompt = DEFAULT_USER_PROMPT_TEMPLATE.format(
                user_goal_string=user_goal_str,
                target_platform_string=target_platform_str,
                project_context_summary_string=project_context_summary,
                existing_plan_json_if_any=existing_plan_json,
            )

        logger.debug(
            f"MasterPlannerAgent System Prompt (effective):\n{current_system_prompt}"
        )
        logger.debug(
            f"MasterPlannerAgent User Prompt (effective):\n{current_user_prompt}"
        )

        try:
            # MODIFIED: Use self.llm_provider with higher token limit
            llm_response_str = await self.llm_provider.generate(
                system_prompt=current_system_prompt,
                prompt=current_user_prompt,
                temperature=0.1,       # Consistent temperature
                max_tokens=4000,       # Reduced for gpt-3.5-turbo compatibility (max 4096)
                response_format={"type": "json_object"}
            )
            logger.debug(f"Raw LLM JSON response: {llm_response_str}")

            # Step 2: Parse the string response as JSON with error recovery
            try:
                llm_generated_plan_dict = json.loads(llm_response_str)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Initial JSON parse failed: {json_error}. Attempting to repair JSON...")
                # Try to repair common JSON issues
                repaired_json = self._attempt_json_repair(llm_response_str)
                if repaired_json:
                    llm_generated_plan_dict = json.loads(repaired_json)
                    logger.info("Successfully repaired and parsed JSON response")
                else:
                    raise json_error  # Re-raise original error if repair failed

            # --- ADDED: Ensure 'id' is present, a non-empty string ---
            current_id = llm_generated_plan_dict.get("id")
            if not isinstance(current_id, str) or not current_id.strip():
                new_plan_id = uuid.uuid4().hex
                logger.warning(
                    f"LLM-generated plan was missing an 'id', had an empty 'id', or 'id' was not a string. "
                    f"Original id: '{current_id}'. Assigning a new UUID: {new_plan_id}"
                )
                llm_generated_plan_dict["id"] = new_plan_id
            # --- END 'id' ensuring block ---

            # --- Inject stage IDs from dictionary keys ---
            stages_from_llm = llm_generated_plan_dict.get("stages")
            if isinstance(stages_from_llm, dict):
                for stage_key, stage_spec_dict in stages_from_llm.items():
                    if isinstance(stage_spec_dict, dict):
                        stage_spec_dict["id"] = stage_key
                        
                        # Sanitize clarification_checkpoint: if present and not a dict, set to None
                        if "clarification_checkpoint" in stage_spec_dict and not isinstance(stage_spec_dict["clarification_checkpoint"], dict):
                            logger.warning(
                                f"LLM provided a non-dictionary value for clarification_checkpoint in stage '{stage_key}'. "
                                f"Received: {stage_spec_dict['clarification_checkpoint']}. Setting to None."
                            )
                            stage_spec_dict["clarification_checkpoint"] = None
                            
                llm_generated_plan_dict["stages"] = stages_from_llm
            # --- End stage ID injection and sanitization ---

            # ADDED: Handle common LLM mistake of using 'initial_stage' instead of 'start_stage'
            if "initial_stage" in llm_generated_plan_dict and "start_stage" not in llm_generated_plan_dict:
                logger.info("Found 'initial_stage' in LLM response, converting to 'start_stage'.")
                llm_generated_plan_dict["start_stage"] = llm_generated_plan_dict.pop("initial_stage")

            # Ensure 'original_request' from MasterPlannerInput is added to the plan
            # if the LLM didn't include it (which it likely won't if not explicitly prompted)
            if (
                "original_request" not in llm_generated_plan_dict
                and inputs.original_request
            ):
                llm_generated_plan_dict["original_request"] = (
                    inputs.original_request.model_dump()
                )

            # Update name and description if the mock LLM didn't use the actual goal
            if "Mock LLM Plan for: User Goal" in llm_generated_plan_dict.get(
                "name", ""
            ):
                llm_generated_plan_dict["name"] = f"Plan for: {user_goal_str}"
            if (
                "This is a mock plan generated by MockLLMClient"
                in llm_generated_plan_dict.get("description", "")
            ):
                llm_generated_plan_dict["description"] = (
                    f"Master plan autonomously generated for goal: "
                    f"{user_goal_str}"
                )

            # Ensure project_id from input is added to the plan data if not present
            if inputs.project_id and "project_id" not in llm_generated_plan_dict:
                llm_generated_plan_dict["project_id"] = inputs.project_id

            # Parse the LLM's response (which is already a dict from the mock)
            # If llm_client.generate_json returned a string, we'd do:
            # parsed_llm_json = json.loads(llm_response_json_str)
            # plan = MasterExecutionPlan.model_validate(parsed_llm_json)
            plan = MasterExecutionPlan.model_validate(llm_generated_plan_dict)

            if inputs.original_request:
                plan.original_request = inputs.original_request
            # project_id is now part of the model and should be set from llm_generated_plan_dict by model_validate
            # if inputs.project_id and not plan.project_id: # Add project_id if generating from blueprint and not set by LLM
            #     plan.project_id = inputs.project_id

            logger.info(f"MasterPlannerAgent successfully generated plan: {plan.id} for project {plan.project_id}")

            # --- Store generated plan to PCMA --- 
            generated_plan_artifact_id: Optional[str] = None
            stored_in_collection_name: Optional[str] = None

            if inputs.project_id: # Only store if project_id is available
                try:
                    # Prepare metadata, ensuring no None values are passed directly for string fields
                    plan_metadata = {
                        "artifact_type": "MasterExecutionPlan",
                        "plan_name": plan.name or "", # Convert None to empty string
                        "plan_version": plan.version if hasattr(plan, 'version') else "1.0",
                        "generated_by_agent": self.AGENT_ID,
                        "user_goal": inputs.user_goal or "", # Convert None to empty string (though user_goal is usually str)
                        "source_blueprint_id": inputs.blueprint_doc_id or "", # Convert None to empty string
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    store_input = StoreArtifactInput(
                        base_collection_name=EXECUTION_PLANS_COLLECTION,
                        artifact_content=plan.model_dump_json(indent=2), # Store the JSON string
                        metadata=plan_metadata, # Use the sanitized metadata
                        project_id=inputs.project_id,
                        document_id=plan.id, # Use plan's own ID
                        cycle_id=inputs.current_context.get("cycle_id") if inputs.current_context else None
                    )
                    logger.info(f"Storing MasterExecutionPlan '{plan.id}' in PCMA collection {EXECUTION_PLANS_COLLECTION}.")
                    store_output: StoreArtifactOutput = await self.project_chroma_manager.store_artifact(args=store_input)

                    if store_output and store_output.status == "SUCCESS":
                        generated_plan_artifact_id = store_output.document_id
                        stored_in_collection_name = EXECUTION_PLANS_COLLECTION
                        logger.info(f"MasterExecutionPlan stored successfully in PCMA. Doc ID: {generated_plan_artifact_id}")
                    else:
                        logger.error(f"Failed to store MasterExecutionPlan in PCMA. Status: {store_output.status if store_output else 'N/A'}, Message: {store_output.message if store_output else 'N/A'}")
                
                except Exception as e_store:
                    logger.error(f"Exception during PCMA storage of MasterExecutionPlan: {e_store}", exc_info=True)
            else:
                logger.warning("project_id not provided in MasterPlannerInput, generated plan will not be stored in PCMA.")

            return MasterPlannerOutput(
                master_plan_json=plan.model_dump_json(indent=2),
                confidence_score=0.75,  # Confidence from LLM (mocked)
                planner_notes="Plan generated by LLM (mocked).",
                generated_plan_artifact_id=generated_plan_artifact_id,
                stored_in_collection=stored_in_collection_name
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return MasterPlannerOutput(
                master_plan_json="",
                error_message=f"LLM output was not valid JSON: {e}",
                planner_notes="LLM output parsing failed.",
            )
        except Exception as e:  # Catch Pydantic validation errors and other issues
            logger.error(
                f"Error processing LLM response or validating plan: {e}",
                exc_info=True,
            )
            return MasterPlannerOutput(
                master_plan_json="",
                error_message=f"Error generating or validating plan: {str(e)}",
                planner_notes="Plan generation/validation failed.",
            )


def get_agent_card_static() -> AgentCard:
    """Returns the static AgentCard for the MasterPlannerAgent."""
    return AgentCard(
        agent_id=MasterPlannerAgent.AGENT_ID,
        name=MasterPlannerAgent.AGENT_NAME,
        version=MasterPlannerAgent.VERSION,  # Ensure this matches class version
        description=MasterPlannerAgent.DESCRIPTION,  # Ensure this matches
        category=MasterPlannerAgent.CATEGORY,
        visibility=MasterPlannerAgent.VISIBILITY,
        input_schema=MasterPlannerInput.model_json_schema(),
        output_schema=MasterPlannerOutput.model_json_schema(),
    )


async def main_test():
    logging.basicConfig(level=logging.INFO)
    logger.info("Running MasterPlannerAgent (LLM-driven) test...")
    planner = MasterPlannerAgent()

    # Test 1: Simple goal
    test_goal_1 = UserGoalRequest(
        goal_description="Implement a new feature foo_bar.",
        target_platform="chungoid-mcp"
    )
    test_input_1 = MasterPlannerInput(
        user_goal=test_goal_1.goal_description,
        original_request=test_goal_1
    )
    logger.info(f"--- Test 1: Goal: {test_goal_1.goal_description} ---")
    output_1 = await planner.invoke_async(test_input_1)

    if output_1.error_message:
        print(f"Error: {output_1.error_message}")
    else:
        print("Generated Master Plan JSON (Test 1):")
        print(output_1.master_plan_json)
        try:
            parsed_plan_1 = MasterExecutionPlan.model_validate_json(
                output_1.master_plan_json
            )
            print("\nPlan 1 successfully parsed.")
            print(f"Plan ID: {parsed_plan_1.id}, Name: {parsed_plan_1.name}")
        except Exception as e:
            print("\nError parsing generated plan 1: {e}")

    # Test 2: Another goal to ensure mock is not hardcoded to one specific input text
    test_goal_2 = UserGoalRequest(
        goal_description="Refactor the authentication module.",
        target_platform="chungoid-mcp",
        key_constraints={"details": "auth module is in src/auth"},
    )
    test_input_2 = MasterPlannerInput(
        user_goal=test_goal_2.goal_description,
        original_request=test_goal_2
    )
    logger.info(f"--- Test 2: Goal: {test_goal_2.goal_description} ---")
    output_2 = await planner.invoke_async(test_input_2)

    if output_2.error_message:
        print(f"Error: {output_2.error_message}")
    else:
        print("\nGenerated Master Plan JSON (Test 2):")
        print(output_2.master_plan_json)
        try:
            parsed_plan_2 = MasterExecutionPlan.model_validate_json(
                output_2.master_plan_json
            )
            print("\nPlan 2 successfully parsed.")
            print(f"Plan ID: {parsed_plan_2.id}, Name: {parsed_plan_2.name}")
            if parsed_plan_2.original_request:
                print(
                    f"Original request in plan 2: "
                    f"{parsed_plan_2.original_request.goal_description}"
                )
                print(
                    f"Key constraints in plan 2: "
                    f"{parsed_plan_2.original_request.key_constraints}"
                )

        except Exception as e:
            print("\nError parsing generated plan 2: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_test())
