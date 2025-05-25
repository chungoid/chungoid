#!/usr/bin/env python3
"""
Syntax Cleanup Script

Fixes broken syntax from previous transformation scripts, specifically:
- "def specialized task" broken string literals
- Missing closing braces/quotes in protocol task definitions
"""

import re
import os
from pathlib import Path

def fix_specialized_task_syntax(content: str) -> str:
    """Fix broken 'def specialized task' syntax."""
    
    # Pattern 1: Find and fix the broken goal line
    broken_goal_pattern = r'"goal": f"Execute {self\.AGENT_NAME} specialized task"'
    fixed_goal = r'"goal": f"Execute {self.AGENT_NAME} specialized task"'
    content = re.sub(broken_goal_pattern, fixed_goal, content)
    
    # Pattern 2: Fix incomplete protocol_task definitions
    broken_task_pattern = r'(\s*protocol_task = \{[^}]*"goal": f"Execute \{self\.AGENT_NAME\} specialized task"\s*)\n\s*protocol_result'
    def fix_task_replacement(match):
        return match.group(1) + '\n            }\n            \n            protocol_result'
    
    content = re.sub(broken_task_pattern, fix_task_replacement, content, flags=re.MULTILINE)
    
    # Pattern 3: Fix completely broken lines with "def specialized task"
    broken_def_pattern = r'\s*def specialized task".*?\n'
    content = re.sub(broken_def_pattern, '', content, flags=re.MULTILINE)
    
    # Pattern 4: Fix incomplete execute method with missing closing brace
    incomplete_execute_pattern = r'(\s*protocol_task = \{.*?"goal": f"Execute \{self\.AGENT_NAME\} specialized task")\s*\n\s*\n\s*protocol_result'
    def fix_execute_replacement(match):
        return match.group(1) + '\n            }\n            \n            protocol_result'
    
    content = re.sub(incomplete_execute_pattern, fix_execute_replacement, content, flags=re.MULTILINE | re.DOTALL)
    
    return content

def fix_duplicated_protocol_methods(content: str) -> str:
    """Remove duplicated protocol execution methods."""
    
    # Remove duplicate protocol methods (keep the first one)
    patterns_to_remove = [
        r'\n\s*# ADDED: Protocol phase execution logic\s*\n\s*def _execute_phase_logic.*?return \{[^}]*\}\s*\n\s*def _extract_output_from_protocol_result.*?return \{[^}]*\}',
        r'\n\s*def _execute_generic_phase.*?return \{[^}]*\}',
        r'\n\s*def _extract_output_from_protocol_result.*?# Generic extraction.*?return \{[^}]*\}'
    ]
    
    for pattern in patterns_to_remove:
        # Only remove if it's a duplicate (appears after the first occurrence)
        matches = list(re.finditer(pattern, content, flags=re.MULTILINE | re.DOTALL))
        if len(matches) > 1:
            # Remove all but the first occurrence
            for match in reversed(matches[1:]):
                content = content[:match.start()] + content[match.end():]
    
    return content

def fix_blueprint_reviewer_specific_issues(content: str) -> str:
    """Fix specific issues in blueprint_reviewer_agent.py"""
    
    # Fix incomplete execute method in BlueprintReviewerAgent
    broken_execute_pattern = r'(\s*"goal": f"Execute \{self\.AGENT_NAME\} specialized task"\s*)\n\s*\n\s*protocol_result'
    def fix_blueprint_execute(match):
        return match.group(1) + '\n            }\n            \n            protocol_result'
    
    content = re.sub(broken_execute_pattern, fix_blueprint_execute, content)
    
    # Fix incomplete line ending with just "e" 
    incomplete_line_pattern = r'\s*else:\s*\n\s*self\._logger\.warning.*?\n\s*return \{.*?\}\s*\n\s*e\s*$'
    def fix_incomplete_line(match):
        return match.group(0).replace('\n        e', '\n            return {"phase_completed": True, "method": "fallback"}')
    
    content = re.sub(incomplete_line_pattern, fix_incomplete_line, content, flags=re.MULTILINE)
    
    return content

def fix_code_debugging_agent_issues(content: str) -> str:
    """Fix specific issues in code_debugging_agent.py"""
    
    # Fix the duplicate and broken execute method sections
    duplicate_execute_pattern = r'(\s*def specialized task".*?\n.*?protocol_result = self\.execute_with_protocol.*?\n.*?if protocol_result.*?\n.*?else:.*?\n.*?except Exception.*?\n.*?raise ProtocolExecutionError.*?\n)'
    content = re.sub(duplicate_execute_pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    return content

def cleanup_agent_file(file_path: Path) -> bool:
    """Clean up syntax issues in a single agent file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply general fixes
        content = fix_specialized_task_syntax(content)
        content = fix_duplicated_protocol_methods(content)
        
        # Apply specific fixes based on file name
        if 'blueprint_reviewer_agent.py' in str(file_path):
            content = fix_blueprint_reviewer_specific_issues(content)
        elif 'code_debugging_agent.py' in str(file_path):
            content = fix_code_debugging_agent_issues(content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed syntax in {file_path}")
            return True
        else:
            print(f"ℹ️  No changes needed in {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def main():
    """Main cleanup function."""
    print("🔧 Starting syntax cleanup...")
    
    # Find all Python files that might have syntax issues
    search_paths = [
        Path("src/chungoid/agents/autonomous_engine/"),
        Path("src/chungoid/runtime/agents/"),
    ]
    
    files_to_check = []
    for search_path in search_paths:
        if search_path.exists():
            files_to_check.extend(search_path.glob("**/*.py"))
    
    # Also check specific files we know have issues
    specific_files = [
        "src/chungoid/agents/autonomous_engine/code_debugging_agent.py",
        "src/chungoid/agents/autonomous_engine/blueprint_reviewer_agent.py",
    ]
    
    for file_path in specific_files:
        path = Path(file_path)
        if path.exists() and path not in files_to_check:
            files_to_check.append(path)
    
    fixed_count = 0
    for file_path in files_to_check:
        if cleanup_agent_file(file_path):
            fixed_count += 1
    
    print(f"\n✨ Syntax cleanup complete! Fixed {fixed_count} files.")

if __name__ == "__main__":
    main() 