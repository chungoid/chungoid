#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from chungoid.mcp_tools.intelligence.manifest_initialization import _generate_capabilities_from_name

print("=== TESTING CAPABILITY GENERATION ===")

test_tools = [
    'filesystem_read_file',
    'chroma_query_documents', 
    'terminal_execute_command',
    'web_content_extract',
    'adaptive_learning_analyze'
]

for tool_name in test_tools:
    caps = _generate_capabilities_from_name(tool_name)
    print(f'\n{tool_name}:')
    print(f'  Generated capabilities: {len(caps)}')
    for cap in caps:
        print(f'    - {cap.name}: {cap.description}') 