#!/usr/bin/env python3
"""
Pure Protocol Cleanup Script

Final cleanup to remove any remaining invoke_async references and polish
the pure protocol architecture.
"""

import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_agent_file(agent_file: Path) -> bool:
    """Clean up remaining invoke_async references in agent file."""
    try:
        with open(agent_file, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Remove any remaining invoke_async calls
        content = re.sub(r'return await self\.invoke_async\([^)]*\)', 
                        'raise ProtocolExecutionError("Pure protocol execution failed")', 
                        content)
        
        # Remove execute_with_protocols method calls that fallback to invoke_async
        content = re.sub(r'self\.invoke_async\([^)]*\)', 
                        'self.execute(task_input, full_context)', 
                        content)
        
        # Remove any remaining "MAINTAINED:" comments
        content = re.sub(r'\s*# MAINTAINED:.*?\n', '\n', content, flags=re.IGNORECASE)
        
        # Fix any broken method signatures from transformation
        content = re.sub(r'def\s*\n\s*def ', 'def ', content)
        
        # Add ProtocolExecutionError import if not present
        if 'ProtocolExecutionError' in content and 'class ProtocolExecutionError' not in content:
            # Add import at top of file
            if 'from typing import' in content:
                content = content.replace(
                    'from typing import',
                    'class ProtocolExecutionError(Exception):\n    """Raised when protocol execution fails."""\n    pass\n\nfrom typing import'
                )
        
        if content != original_content:
            with open(agent_file, 'w') as f:
                f.write(content)
            logger.info(f"✅ Cleaned up {agent_file.name}")
            return True
        else:
            logger.info(f"⏭️  No cleanup needed for {agent_file.name}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error cleaning up {agent_file}: {e}")
        return False

def main():
    """Main cleanup execution."""
    chungoid_core_path = Path("/home/flip/Desktop/chungoid-mcp/chungoid-core")
    
    cleaned_count = 0
    
    # Clean up all agent files
    for agent_dir in [
        chungoid_core_path / "src" / "chungoid" / "agents" / "autonomous_engine",
        chungoid_core_path / "src" / "chungoid" / "runtime" / "agents"
    ]:
        if agent_dir.exists():
            for agent_file in agent_dir.glob("*_agent.py"):
                if cleanup_agent_file(agent_file):
                    cleaned_count += 1
    
    logger.info(f"🎯 Cleanup complete: {cleaned_count} files cleaned")

if __name__ == "__main__":
    main() 