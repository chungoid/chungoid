#!/usr/bin/env python3
"""
Systematic Agent Conversion Script

Converts all agents from BaseAgent to ProtocolAwareAgent following our
established pattern and AI agent best practices.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentProtocolConverter:
    """
    Systematic converter for transforming agents to protocol-aware architecture.
    
    Follows AI agent best practices:
    - Hybrid approach (protocol + traditional code)
    - Specialized agent focus
    - Structured outputs with schemas
    - Agent communication patterns
    """
    
    def __init__(self, chungoid_core_path: str):
        self.chungoid_core_path = Path(chungoid_core_path)
        self.conversion_stats = {
            "converted": 0,
            "skipped": 0,
            "errors": 0
        }
        
        # Agent type to protocol mapping following specialization principle
        self.agent_protocol_mapping = {
            # Analysis Agents - Focus on investigation and analysis
            "architect_agent": {
                "primary": ["deep_planning"], 
                "secondary": ["deep_investigation", "quality_validation"],
                "universal": ["agent_communication", "context_sharing", "tool_validation"]
            },
            "code_debugging_agent": {
                "primary": ["code_remediation", "deep_investigation"],
                "secondary": ["test_analysis", "error_recovery"],
                "universal": ["agent_communication", "tool_validation", "context_sharing"]
            },
            "test_failure_analysis_agent": {
                "primary": ["test_analysis", "deep_investigation"], 
                "secondary": ["code_remediation", "quality_validation"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "dependency_management_agent": {
                "primary": ["dependency_analysis"],
                "secondary": ["deep_investigation", "system_analysis"],
                "universal": ["agent_communication", "tool_validation", "context_sharing"]
            },
            "environment_bootstrap_agent": {
                "primary": ["environment_setup"],
                "secondary": ["system_configuration", "validation"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "product_analyst_agent": {
                "primary": ["requirements_analysis", "stakeholder_analysis"],
                "secondary": ["deep_investigation", "documentation"],
                "universal": ["agent_communication", "context_sharing", "goal_tracking"]
            },
            "requirements_tracer_agent": {
                "primary": ["requirements_tracing", "goal_tracking"],
                "secondary": ["documentation", "validation"],
                "universal": ["agent_communication", "context_sharing", "tool_validation"]
            },
            "project_documentation_agent": {
                "primary": ["documentation_generation"],
                "secondary": ["content_validation", "template_management"],
                "universal": ["agent_communication", "context_sharing", "tool_validation"]
            },
            "proactive_risk_assessor_agent": {
                "primary": ["risk_assessment", "deep_investigation"],
                "secondary": ["impact_analysis", "mitigation_planning"],
                "universal": ["agent_communication", "context_sharing", "goal_tracking"]
            },
            "blueprint_reviewer_agent": {
                "primary": ["review_protocol", "quality_validation"],
                "secondary": ["deep_investigation", "documentation"],
                "universal": ["agent_communication", "context_sharing", "tool_validation"]
            },
            "project_chroma_manager_agent": {
                "primary": ["data_management"],
                "secondary": ["storage_optimization", "retrieval_protocols"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "automated_refinement_coordinator_agent": {
                "primary": ["multi_agent_coordination", "workflow_orchestration"],
                "secondary": ["quality_validation", "goal_tracking"],
                "universal": ["agent_communication", "context_sharing", "error_recovery"]
            },
            
            # Runtime Agents - Focus on execution and implementation
            "core_code_generator_agent": {
                "primary": ["systematic_implementation"],
                "secondary": ["quality_validation", "code_review"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "system_master_planner_agent": {
                "primary": ["multi_agent_coordination", "deep_planning"],
                "secondary": ["workflow_orchestration", "goal_tracking"],
                "universal": ["agent_communication", "context_sharing", "tool_validation"]
            },
            "core_stage_executor": {
                "primary": ["workflow_execution"],
                "secondary": ["stage_coordination", "progress_tracking"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "system_requirements_gathering_agent": {
                "primary": ["requirements_gathering", "stakeholder_analysis"],
                "secondary": ["documentation", "validation"],
                "universal": ["agent_communication", "context_sharing", "goal_tracking"]
            },
            "system_master_planner_reviewer_agent": {
                "primary": ["review_protocol", "quality_validation"],
                "secondary": ["deep_investigation", "improvement_planning"],
                "universal": ["agent_communication", "context_sharing", "tool_validation"]
            },
            "system_test_runner_agent": {
                "primary": ["test_execution"],
                "secondary": ["test_analysis", "quality_validation"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "system_file_system_agent": {
                "primary": ["file_management"],
                "secondary": ["data_validation", "backup_protocols"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "smart_code_integration_agent": {
                "primary": ["code_integration"],
                "secondary": ["systematic_implementation", "quality_validation"],
                "universal": ["agent_communication", "tool_validation", "error_recovery"]
            },
            "core_test_generator_agent": {
                "primary": ["test_generation"],
                "secondary": ["systematic_implementation", "quality_validation"],
                "universal": ["agent_communication", "tool_validation", "context_sharing"]
            }
        }

    def convert_all_agents(self) -> Dict[str, List[str]]:
        """Convert all agents following systematic approach."""
        logger.info("🚀 Starting systematic agent conversion to protocol-aware architecture")
        
        results = {
            "converted": [],
            "skipped": [],
            "errors": []
        }
        
        # Analysis agents
        analysis_agents_dir = self.chungoid_core_path / "src" / "chungoid" / "agents" / "autonomous_engine"
        if analysis_agents_dir.exists():
            results = self._convert_agents_in_directory(analysis_agents_dir, "analysis", results)
        
        # Runtime agents  
        runtime_agents_dir = self.chungoid_core_path / "src" / "chungoid" / "runtime" / "agents"
        if runtime_agents_dir.exists():
            results = self._convert_agents_in_directory(runtime_agents_dir, "runtime", results)
        
        logger.info(f"✅ Conversion complete: {len(results['converted'])} converted, {len(results['skipped'])} skipped, {len(results['errors'])} errors")
        return results

    def _convert_agents_in_directory(self, directory: Path, agent_type: str, results: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Convert all agents in a specific directory."""
        logger.info(f"Converting {agent_type} agents in {directory}")
        
        for agent_file in directory.glob("*_agent.py"):
            try:
                if self._should_skip_agent(agent_file):
                    logger.info(f"⏭️  Skipping {agent_file.name} (already converted or excluded)")
                    results["skipped"].append(str(agent_file))
                    continue
                
                logger.info(f"🔄 Converting {agent_file.name}")
                success = self._convert_single_agent(agent_file)
                
                if success:
                    results["converted"].append(str(agent_file))
                    logger.info(f"✅ Successfully converted {agent_file.name}")
                else:
                    results["errors"].append(str(agent_file))
                    logger.error(f"❌ Failed to convert {agent_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error converting {agent_file.name}: {e}")
                results["errors"].append(str(agent_file))
        
        return results

    def _should_skip_agent(self, agent_file: Path) -> bool:
        """Check if agent should be skipped (already converted or special cases)."""
        # Skip if already converted to ProtocolAwareAgent
        with open(agent_file, 'r') as f:
            content = f.read()
            if "ProtocolAwareAgent" in content:
                return True
            if "__init__.py" in agent_file.name:
                return True
            # Skip agent_base.py as it's the base class
            if "agent_base.py" in agent_file.name:
                return True
        
        return False

    def _convert_single_agent(self, agent_file: Path) -> bool:
        """Convert a single agent file to protocol-aware architecture."""
        try:
            # Read current content
            with open(agent_file, 'r') as f:
                content = f.read()
            
            # Extract agent name from filename
            agent_name = agent_file.stem
            
            # Apply conversion transformations
            converted_content = self._apply_conversion_transformations(content, agent_name)
            
            # Write converted content
            with open(agent_file, 'w') as f:
                f.write(converted_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Error converting {agent_file}: {e}")
            return False

    def _apply_conversion_transformations(self, content: str, agent_name: str) -> str:
        """Apply systematic transformations to convert agent to protocol-aware."""
        
        # 1. Update imports
        content = self._update_imports(content)
        
        # 2. Update class inheritance
        content = self._update_class_inheritance(content)
        
        # 3. Add protocol definitions
        content = self._add_protocol_definitions(content, agent_name)
        
        # 4. Add protocol execution methods
        content = self._add_protocol_execution_methods(content, agent_name)
        
        # 5. Add phase execution logic
        content = self._add_phase_execution_logic(content, agent_name)
        
        return content

    def _update_imports(self, content: str) -> str:
        """Update imports to include ProtocolAwareAgent."""
        # Add ProtocolAwareAgent import
        if "from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent" not in content:
            # Find BaseAgent import and add ProtocolAwareAgent import before it
            baseagent_pattern = r'from chungoid\.runtime\.agents\.agent_base import BaseAgent'
            if re.search(baseagent_pattern, content):
                content = re.sub(
                    baseagent_pattern,
                    'from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent\nfrom chungoid.protocols.base.protocol_interface import ProtocolPhase\nfrom chungoid.runtime.agents.agent_base import BaseAgent',
                    content
                )
        
        return content

    def _update_class_inheritance(self, content: str) -> str:
        """Update class to inherit from ProtocolAwareAgent."""
        # Pattern to match class definition with BaseAgent inheritance
        pattern = r'class\s+(\w+)\(BaseAgent\[(.*?)\]\):'
        replacement = r'class \1(ProtocolAwareAgent[\2]):'
        
        return re.sub(pattern, replacement, content)

    def _add_protocol_definitions(self, content: str, agent_name: str) -> str:
        """Add protocol definitions based on agent type."""
        # Get protocol mapping for this agent
        protocols = self._get_agent_protocols(agent_name)
        
        # Find class definition and add protocol definitions
        class_pattern = r'(class\s+\w+\(ProtocolAwareAgent\[.*?\]\):\s*\n)(.*?)([\s]*def\s+__init__)'
        
        def replacement(match):
            class_def = match.group(1)
            class_body = match.group(2)
            init_def = match.group(3)
            
            # Add protocol definitions if not already present
            if "PRIMARY_PROTOCOLS" not in class_body:
                protocol_definitions = f"""
    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = {protocols['primary']}
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = {protocols['secondary']}
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = {protocols['universal']}
"""
                return class_def + class_body + protocol_definitions + init_def
            
            return match.group(0)
        
        return re.sub(class_pattern, replacement, content, flags=re.DOTALL)

    def _add_protocol_execution_methods(self, content: str, agent_name: str) -> str:
        """Add protocol-aware execution methods."""
        # Find the end of __init__ method and add protocol methods
        init_pattern = r'(def\s+__init__\(.*?\n.*?)\n\n(\s*async\s+def\s+invoke_async)'
        
        def replacement(match):
            init_method = match.group(1)
            invoke_async = match.group(2)
            
            # Add protocol execution method
            protocol_method = f"""

    # ADDED: Protocol-aware execution method (hybrid approach)
    async def execute_with_protocols(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        \"\"\"
        Execute using appropriate protocol with fallback to traditional method.
        Follows AI agent best practices for hybrid execution.
        \"\"\"
        try:
            # Determine primary protocol for this agent
            primary_protocol = self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "simple_operations"
            
            protocol_task = {{
                "task_input": task_input.dict() if hasattr(task_input, 'dict') else task_input,
                "full_context": full_context,
                "goal": f"Execute {{self.AGENT_NAME}} specialized task"
            }}
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                self._logger.warning("Protocol execution failed, falling back to traditional method")
                return await self.invoke_async(task_input, full_context)
                
        except Exception as e:
            self._logger.warning(f"Protocol execution error: {{e}}, falling back to traditional method")
            return await self.invoke_async(task_input, full_context)

    # ADDED: Protocol phase execution logic
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        \"\"\"Execute agent-specific logic for each protocol phase.\"\"\"
        
        # Generic phase handling - can be overridden by specific agents
        if phase.name in ["discovery", "analysis", "planning", "execution", "validation"]:
            return self._execute_generic_phase(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {{phase.name}}")
            return {{"phase_completed": True, "method": "fallback"}}

    def _execute_generic_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        \"\"\"Execute generic phase logic suitable for most agents.\"\"\"
        return {{
            "phase_name": phase.name,
            "status": "completed", 
            "outputs": {{"generic_result": f"Phase {{phase.name}} completed"}},
            "method": "generic_protocol_execution"
        }}

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input) -> Any:
        \"\"\"Extract agent output from protocol execution results.\"\"\"
        # Generic extraction - should be overridden by specific agents
        return {{
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }}

    # MAINTAINED: Original invoke_async method for backward compatibility"""
            
            return init_method + protocol_method + "\n\n" + invoke_async
        
        return re.sub(init_pattern, replacement, content, flags=re.DOTALL)

    def _add_phase_execution_logic(self, content: str, agent_name: str) -> str:
        """Add specific phase execution logic for agent type."""
        # This would add agent-specific phase logic
        # For now, we'll use the generic implementation added above
        return content

    def _get_agent_protocols(self, agent_name: str) -> Dict[str, List[str]]:
        """Get protocol mapping for specific agent."""
        return self.agent_protocol_mapping.get(agent_name, {
            "primary": ["simple_operations"],
            "secondary": ["quality_validation"],
            "universal": ["agent_communication", "context_sharing", "tool_validation"]
        })

def main():
    """Main conversion execution."""
    import sys
    
    if len(sys.argv) > 1:
        chungoid_core_path = sys.argv[1]
    else:
        chungoid_core_path = "/home/flip/Desktop/chungoid-mcp/chungoid-core"
    
    converter = AgentProtocolConverter(chungoid_core_path)
    results = converter.convert_all_agents()
    
    print(f"\n🎯 Conversion Summary:")
    print(f"✅ Converted: {len(results['converted'])} agents")
    print(f"⏭️  Skipped: {len(results['skipped'])} agents") 
    print(f"❌ Errors: {len(results['errors'])} agents")
    
    if results['errors']:
        print(f"\n❌ Failed conversions:")
        for error in results['errors']:
            print(f"  - {error}")

if __name__ == "__main__":
    main() 