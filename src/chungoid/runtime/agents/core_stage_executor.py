from typing import Any, Dict
import logging
import yaml
from pathlib import Path
import copy # For deepcopying context

# Attempt to import ChungoidEngine and related components
# This assumes a certain directory structure. Adjust if necessary.
try:
    from ...engine import ChungoidEngine # Corrected relative import
    from ...utils.config_loader import ConfigError
    from ...utils.state_manager import StatusFileError # For engine init errors
    from ...utils.prompt_manager import PromptLoadError # For engine init errors
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import engine components: {e}. Check relative paths.")
    # Define dummy classes if import fails to allow AgentCard definition below
    class ChungoidEngine: pass 
    class ConfigError(Exception): pass
    class StatusFileError(Exception): pass
    class PromptLoadError(Exception): pass

# --- Agent Definition ---
from pydantic import BaseModel, Field, ValidationError
from ...utils.agent_registry import AgentCard 

logger = logging.getLogger(__name__)

# --- Input Schema --- 
class CoreStageExecutorInputs(BaseModel):
    stage_definition_filename: str = Field(..., description="Filename of the stageX.yaml (e.g., 'stage0.yaml') to execute, relative to server_prompts/stages/.")
    current_project_root: str = Field(..., description="The absolute path to the project root for this execution.")
    
    class Config:
        extra = 'allow' # Allow other fields from master flow inputs to pass through

# --- AgentCard Definition --- 
core_stage_executor_card = AgentCard(
    agent_id="CoreStageExecutorAgent",
    name="Core Stage Executor Agent",
    description=(
        "A generic agent that executes the MCP actions defined within a specified Chungoid Stage Definition YAML file. "
        "Requires 'stage_definition_filename' (e.g., 'stage0.yaml') and 'current_project_root' in inputs."
    ),
    version="1.0.0",
    author="Chungoid Core Team",
    inputs_json_schema=CoreStageExecutorInputs.model_json_schema(),
    outputs_json_schema={
        "title": "CoreStageExecutorOutput",
        "type": "object",
        "description": "Output from the executed stageX.yaml. Structure varies.",
        "properties": {
            "status": {"type": "string", "description": "Execution status of the sub-stage."},
            "message": {"type": "string", "description": "A message detailing the outcome."},
            "mcp_results": {"type": "array", "description": "List of results from executed MCP actions."},
            "final_context": {"type": "object", "description": "The context after sub-stage execution."},
            "yaml_content": {"type": "object", "description": "Parsed YAML content if no actions were defined."}
        },
        "required": ["status", "message"],
        "additionalProperties": True, 
    },
    mcp_tool_input_schemas=None
)


# --- Agent Logic ---
async def core_stage_executor_agent(context: Dict[str, Any]) -> Dict[str, Any]:
    """ 
    Executes a specific stageX.yaml file's MCP actions.
    Expects 'stage_definition_filename' and 'current_project_root' in context['inputs'].
    """
    logger.info(f"CoreStageExecutorAgent invoked. Full context keys: {list(context.keys())}")
    
    agent_inputs_data = context.get('inputs', {}) # Inputs provided by MasterStageSpec
    # Merge top-level context (excluding 'outputs' and 'inputs' keys from master flow) 
    # into agent_inputs_data to make them available for validation and sub-stage context.
    for key, val in context.items():
        if key not in ['outputs', 'inputs']:
            if key not in agent_inputs_data: # Don't overwrite inputs specific in MasterStageSpec
                 agent_inputs_data[key] = val

    try:
        # Validate inputs using the Pydantic model
        validated_inputs = CoreStageExecutorInputs(**agent_inputs_data)
        stage_def_filename = validated_inputs.stage_definition_filename
        current_project_root = validated_inputs.current_project_root
    except ValidationError as p_exc:
        logger.error(f"Input validation failed for CoreStageExecutorAgent: {p_exc}. Provided data: {agent_inputs_data}")
        return {"status": "error", "message": f"Input validation failed: {p_exc}", "error_details": p_exc.errors()}
    except Exception as e: # Catch other potential errors during validation
         logger.error(f"Unexpected error during input validation: {e}. Provided data: {agent_inputs_data}")
         return {"status": "error", "message": f"Unexpected validation error: {e}"}

    logger.info(f"Project root for sub-stage execution: {current_project_root}")
    logger.info(f"Executing sub-stage definition file: {stage_def_filename}")

    # <<< FIX: Load sub-stage YAML directly, not via PromptManager >>>
    # Construct the path relative to the provided project root
    stages_dir_path = Path(current_project_root) / "server_prompts" / "stages"
    sub_stage_yaml_path = stages_dir_path / stage_def_filename
    logger.info(f"Attempting to load sub-stage YAML directly from: {sub_stage_yaml_path}")

    if not sub_stage_yaml_path.exists():
        logger.error(f"Sub-stage YAML file not found at expected path: {sub_stage_yaml_path}")
        return {"status": "error", "message": f"Sub-stage YAML file not found: {stage_def_filename}"}

    try:
        with open(sub_stage_yaml_path, 'r', encoding='utf-8') as f:
            sub_stage_yaml_content = yaml.safe_load(f)
        if not isinstance(sub_stage_yaml_content, dict):
            raise PromptLoadError(f"Sub-stage YAML content is not a dictionary: {sub_stage_yaml_path}")
    except FileNotFoundError:
        # This check is slightly redundant due to exists() above, but good practice
        logger.error(f"Sub-stage YAML file not found during open: {sub_stage_yaml_path}")
        return {"status": "error", "message": f"Sub-stage YAML file not found: {stage_def_filename}"}
    except (yaml.YAMLError, PromptLoadError) as e:
        logger.error(f"Error loading/parsing sub-stage YAML {sub_stage_yaml_path}: {e}")
        return {"status": "error", "message": f"Cannot load/parse sub-stage YAML: {e}"}
    except Exception as e:
        logger.exception(f"Unexpected error loading sub-stage YAML {sub_stage_yaml_path}: {e}")
        return {"status": "error", "message": f"Unexpected error loading sub-stage YAML: {e}"}


    # Initialize a temporary ChungoidEngine for executing MCP actions within this sub-stage
    # Note: We load the YAML *before* initializing the engine to ensure the file exists.
    # The engine itself might *also* try to load standard stage files via its PromptManager,
    # but that's okay - we primarily use this engine instance to call execute_mcp_tool.
    try:
        temp_engine = ChungoidEngine(project_directory=current_project_root)
    except NameError: # If ChungoidEngine import failed
        logger.error("ChungoidEngine class not available. Cannot execute sub-stage.")
        return {"status":"error", "message": "Engine components not loaded."}
    except (ConfigError, StatusFileError, PromptLoadError, RuntimeError, ValueError) as engine_init_err:
        logger.error(f"Failed to initialize temporary ChungoidEngine for project {current_project_root}: {engine_init_err}")
        return {"status": "error", "message": f"Failed to initialize engine for sub-stage: {engine_init_err}"}
    
    # --- Prepare Sub-Stage Context --- 
    # Start with a clean slate or copy from master? Let's copy master context for now.
    # This allows sub-stage actions to potentially use outputs from previous master stages.
    sub_stage_context = copy.deepcopy(context)
    # Add the specific inputs validated for this agent call (could include initial_project_brief etc.)
    sub_stage_context.update(validated_inputs.model_dump()) 
    # Ensure outputs dict exists for this sub-stage context
    if 'outputs' not in sub_stage_context: sub_stage_context['outputs'] = {}

    # --- Execute MCP Actions (if any) --- 
    mcp_actions = sub_stage_yaml_content.get('mcp_actions', [])
    executed_mcp_results = []

    if not mcp_actions:
        logger.info(f"Sub-stage {stage_def_filename} has no direct 'mcp_actions'. Returning content.")
        return {
            "status": "no_actions_defined",
            "message": f"Sub-stage {stage_def_filename} loaded. Contains no direct mcp_actions.",
            "yaml_content": sub_stage_yaml_content,
            "final_context": sub_stage_context 
        }

    logger.info(f"Found {len(mcp_actions)} MCP actions in {stage_def_filename}. Attempting execution.")
    action_error = None
    for action_idx, action_config in enumerate(mcp_actions):
        tool_name = action_config.get('tool_name')
        tool_arguments_template = action_config.get('tool_arguments', {})
        if not tool_name:
            logger.warning(f"Skipping action #{action_idx} in {stage_def_filename} due to missing 'tool_name'.")
            continue
        
        # Resolve tool_arguments from sub_stage_context
        resolved_tool_arguments = {}
        try:
            for arg_key, arg_val_template in tool_arguments_template.items():
                if isinstance(arg_val_template, str) and arg_val_template.startswith("context."):
                    path = arg_val_template.split('.')[1:]
                    current_val = sub_stage_context
                    for p_item in path:
                        if isinstance(current_val, list) and p_item.isdigit():
                            current_val = current_val[int(p_item)]
                        elif isinstance(current_val, dict):
                            current_val = current_val[p_item]
                        else: raise KeyError(f"Path element {p_item} not applicable to {type(current_val)}")
                    resolved_tool_arguments[arg_key] = current_val
                else:
                    resolved_tool_arguments[arg_key] = arg_val_template
        except (KeyError, IndexError, TypeError) as e:
             logger.error(f"Context resolution failed for tool '{tool_name}' arg '{arg_key}' ('{arg_val_template}'): {e}")
             action_error = {"status":"error", "message":f"Context resolution failed for {tool_name}: {e}", "failed_action_index": action_idx}
             break # Stop processing actions on resolution error

        if action_error: break # Exit outer loop if inner loop broke

        logger.info(f"Executing MCP Action #{action_idx}: '{tool_name}' with resolved args: {resolved_tool_arguments}")
        try:
            # Call synchronously - blocking the async agent function.
            # TODO: Address sync/async call pattern if needed.
            tool_call_id = f"csea-{Path(stage_def_filename).stem}-action{action_idx}"
            raw_tool_output = temp_engine.execute_mcp_tool(tool_name, resolved_tool_arguments, tool_call_id=tool_call_id)
            executed_mcp_results.append(raw_tool_output) 

            # Simplistic merge of output back into sub_stage_context['outputs'][tool_name_idx]
            output_key = f"{tool_name}_{action_idx}"
            if isinstance(raw_tool_output, dict):
                 sub_stage_context['outputs'][output_key] = raw_tool_output # Store raw output
                 if raw_tool_output.get("isError", False):
                    logger.error(f"MCP tool '{tool_name}' reported an error: {raw_tool_output.get('error')}")
                    action_error = {"status":"error", "message":f"Tool {tool_name} reported error.", "failed_action_index": action_idx, "tool_output": raw_tool_output}
                    break # Stop processing actions
            else: # Handle non-dict output? Store it anyway.
                 sub_stage_context['outputs'][output_key] = {"raw_output": raw_tool_output}

        except Exception as tool_exec_err:
            logger.exception(f"CoreStageExecutorAgent: Unhandled exception executing MCP tool '{tool_name}' for {stage_def_filename}: {tool_exec_err}")
            action_error = {"status":"error", "message":f"Exception executing tool {tool_name}: {tool_exec_err}", "failed_action_index": action_idx}
            break # Stop processing actions
    # --- End Action Loop --- 

    if action_error:
        # Return error state if any action failed
        return {
            **action_error, # Include status, message, index
            "executed_mcp_results": executed_mcp_results,
            "final_context": sub_stage_context 
        }
    else:
        # Success case - all actions completed
        logger.info(f"CoreStageExecutorAgent successfully executed all actions for {stage_def_filename}.")
        return {
            "status": "success",
            "message": f"Sub-stage {stage_def_filename} MCP actions executed successfully.",
            "executed_mcp_results": executed_mcp_results,
            "final_context": sub_stage_context 
        } 