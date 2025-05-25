#!/usr/bin/env python3
"""
Pure Protocol Transformation Script

Transforms all agents from hybrid (protocol + fallback) to pure protocol architecture.
Eliminates ALL backward compatibility for clean, maintainable codebase.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PureProtocolTransformer:
    """
    Transforms agents to pure protocol architecture by:
    1. Removing all invoke_async methods
    2. Removing fallback logic from execute_with_protocols
    3. Renaming execute_with_protocols to execute
    4. Simplifying agent inheritance
    5. Cleaning up imports and documentation
    """
    
    def __init__(self, chungoid_core_path: str):
        self.chungoid_core_path = Path(chungoid_core_path)
        self.transformation_stats = {
            "transformed": 0,
            "skipped": 0,
            "errors": 0
        }

    def transform_all_agents(self) -> Dict[str, List[str]]:
        """Transform all agents to pure protocol architecture."""
        logger.info("🚀 Starting PURE PROTOCOL transformation - FULL FORWARD, NO BACKWARD COMPATIBILITY")
        
        results = {
            "transformed": [],
            "skipped": [],
            "errors": []
        }
        
        # Analysis agents
        analysis_agents_dir = self.chungoid_core_path / "src" / "chungoid" / "agents" / "autonomous_engine"
        if analysis_agents_dir.exists():
            results = self._transform_agents_in_directory(analysis_agents_dir, "analysis", results)
        
        # Runtime agents  
        runtime_agents_dir = self.chungoid_core_path / "src" / "chungoid" / "runtime" / "agents"
        if runtime_agents_dir.exists():
            results = self._transform_agents_in_directory(runtime_agents_dir, "runtime", results)
        
        logger.info(f"✅ PURE PROTOCOL transformation complete: {len(results['transformed'])} transformed, {len(results['skipped'])} skipped, {len(results['errors'])} errors")
        return results

    def _transform_agents_in_directory(self, directory: Path, agent_type: str, results: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Transform all agents in a specific directory."""
        logger.info(f"Transforming {agent_type} agents in {directory}")
        
        for agent_file in directory.glob("*_agent.py"):
            try:
                if self._should_skip_agent(agent_file):
                    logger.info(f"⏭️  Skipping {agent_file.name} (not a ProtocolAwareAgent)")
                    results["skipped"].append(str(agent_file))
                    continue
                
                logger.info(f"🔄 Transforming {agent_file.name} to PURE PROTOCOL")
                success = self._transform_single_agent(agent_file)
                
                if success:
                    results["transformed"].append(str(agent_file))
                    logger.info(f"✅ Successfully transformed {agent_file.name} to pure protocol")
                else:
                    results["errors"].append(str(agent_file))
                    logger.error(f"❌ Failed to transform {agent_file.name}")
                    
            except Exception as e:
                logger.error(f"❌ Error transforming {agent_file.name}: {e}")
                results["errors"].append(str(agent_file))
        
        return results

    def _should_skip_agent(self, agent_file: Path) -> bool:
        """Check if agent should be skipped (not a ProtocolAwareAgent)."""
        with open(agent_file, 'r') as f:
            content = f.read()
            # Only transform agents that are already ProtocolAwareAgent
            if "ProtocolAwareAgent[" not in content:
                return True
            if "__init__.py" in agent_file.name:
                return True
            if "agent_base.py" in agent_file.name:
                return True
        
        return False

    def _transform_single_agent(self, agent_file: Path) -> bool:
        """Transform a single agent file to pure protocol architecture."""
        try:
            # Read current content
            with open(agent_file, 'r') as f:
                content = f.read()
            
            # Apply pure protocol transformations
            transformed_content = self._apply_pure_protocol_transformations(content)
            
            # Write transformed content
            with open(agent_file, 'w') as f:
                f.write(transformed_content)
            
            return True
            
        except Exception as e:
            logger.error(f"Error transforming {agent_file}: {e}")
            return False

    def _apply_pure_protocol_transformations(self, content: str) -> str:
        """Apply transformations to create pure protocol architecture."""
        
        # 1. Remove entire invoke_async method and its docstring
        content = self._remove_invoke_async_method(content)
        
        # 2. Transform execute_with_protocols to pure protocol execution
        content = self._transform_execute_with_protocols(content)
        
        # 3. Remove fallback-related helper methods
        content = self._remove_fallback_helpers(content)
        
        # 4. Clean up comments and documentation
        content = self._clean_documentation(content)
        
        # 5. Update class docstrings
        content = self._update_class_docstring(content)
        
        return content

    def _remove_invoke_async_method(self, content: str) -> str:
        """Remove the entire invoke_async method and its docstring."""
        # Pattern to match invoke_async method with its docstring and full implementation
        pattern = r'(\s*# MAINTAINED:.*?method.*?compatibility.*?\n)?\s*async def invoke_async\([\s\S]*?(?=\n\s*(?:async def|def|class|\Z))'
        
        # More aggressive pattern to catch the entire method
        invoke_async_pattern = r'\n\s*# MAINTAINED:.*?invoke_async.*?backward compatibility.*?\n\s*async def invoke_async\([\s\S]*?(?=\n\s*(?:@|async def|def|class|\Z))'
        
        content = re.sub(invoke_async_pattern, '\n', content, flags=re.MULTILINE)
        
        # Fallback: simpler pattern
        simple_pattern = r'\s*async def invoke_async\([^)]*\)[\s\S]*?(?=\n\s*(?:@|async def|def|class|\Z))'
        content = re.sub(simple_pattern, '', content, flags=re.MULTILINE)
        
        return content

    def _transform_execute_with_protocols(self, content: str) -> str:
        """Transform execute_with_protocols to pure protocol execution."""
        
        # Find and replace the execute_with_protocols method
        old_method_pattern = r'(\s*# ADDED:.*?Protocol-aware execution.*?\n)?\s*async def execute_with_protocols\([^)]*\)[\s]*->[\s]*[^:]*:([\s\S]*?)(?=\n\s*def)'
        
        # New pure protocol implementation
        pure_protocol_method = '''
    async def execute(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using pure protocol architecture.
        No fallback - protocol execution only for clean, maintainable code.
        """
        try:
            # Determine primary protocol for this agent
            primary_protocol = self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "simple_operations"
            
            protocol_task = {
                "task_input": task_input.dict() if hasattr(task_input, 'dict') else task_input,
                "full_context": full_context,
                "goal": f"Execute {self.AGENT_NAME} specialized task"
            }
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                # Enhanced error handling instead of fallback
                error_msg = f"Protocol execution failed for {self.AGENT_NAME}: {protocol_result.get('error', 'Unknown error')}"
                self._logger.error(error_msg)
                raise ProtocolExecutionError(error_msg)
                
        except Exception as e:
            error_msg = f"Pure protocol execution failed for {self.AGENT_NAME}: {e}"
            self._logger.error(error_msg)
            raise ProtocolExecutionError(error_msg)

    def'''
        
        # Replace the method
        content = re.sub(old_method_pattern, pure_protocol_method, content, flags=re.DOTALL)
        
        # Also handle cases where method signature might be different
        alt_pattern = r'async def execute_with_protocols\([^)]*\):[^{]*\{[^}]*\}'
        content = re.sub(alt_pattern, pure_protocol_method.strip(), content, flags=re.DOTALL)
        
        return content

    def _remove_fallback_helpers(self, content: str) -> str:
        """Remove fallback-related helper methods and comments."""
        
        # Remove "MAINTAINED:" comments about backward compatibility
        content = re.sub(r'\s*# MAINTAINED:.*?backward compatibility.*?\n', '\n', content, flags=re.IGNORECASE)
        
        # Remove hybrid approach comments
        content = re.sub(r'\s*# ADDED:.*?hybrid approach.*?\n', '\n', content, flags=re.IGNORECASE)
        
        return content

    def _clean_documentation(self, content: str) -> str:
        """Clean up documentation to reflect pure protocol architecture."""
        
        # Update docstrings that mention fallback or hybrid approaches
        content = re.sub(
            r'"""[\s\S]*?hybrid.*?fallback.*?"""',
            '"""Execute using pure protocol architecture for clean, maintainable agent execution."""',
            content,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        return content

    def _update_class_docstring(self, content: str) -> str:
        """Update class docstring to reflect pure protocol architecture."""
        
        # Add pure protocol architecture notice to class docstring
        class_pattern = r'(class\s+\w+.*?:\s*""")([\s\S]*?)(""")'
        
        def update_docstring(match):
            class_def = match.group(1)
            docstring_content = match.group(2)
            closing = match.group(3)
            
            # Add pure protocol notice if not already present
            if "pure protocol architecture" not in docstring_content.lower():
                docstring_content += "\n    \n    ✨ PURE PROTOCOL ARCHITECTURE - No backward compatibility, clean execution paths only."
            
            return class_def + docstring_content + closing
        
        content = re.sub(class_pattern, update_docstring, content, flags=re.DOTALL)
        
        return content

# Custom exception for protocol execution failures
PROTOCOL_EXCEPTION_CODE = '''
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails in pure protocol architecture."""
    pass
'''

def main():
    """Main transformation execution."""
    import sys
    
    if len(sys.argv) > 1:
        chungoid_core_path = sys.argv[1]
    else:
        chungoid_core_path = "/home/flip/Desktop/chungoid-mcp/chungoid-core"
    
    transformer = PureProtocolTransformer(chungoid_core_path)
    results = transformer.transform_all_agents()
    
    print(f"\n🎯 PURE PROTOCOL Transformation Summary:")
    print(f"✅ Transformed: {len(results['transformed'])} agents")
    print(f"⏭️  Skipped: {len(results['skipped'])} agents") 
    print(f"❌ Errors: {len(results['errors'])} agents")
    
    if results['errors']:
        print(f"\n❌ Failed transformations:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\n🚀 PURE PROTOCOL ARCHITECTURE: Clean, maintainable, no technical debt!")

if __name__ == "__main__":
    main() 