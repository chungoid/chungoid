#!/usr/bin/env python3

import sys
import asyncio
sys.path.insert(0, 'src')

from chungoid.mcp_tools.intelligence.tool_manifest import discover_tools

async def test_discovery():
    print("=== TESTING INTELLIGENT TOOL DISCOVERY ===")
    
    # Test 1: Filesystem operations
    print("\n1. Testing filesystem operations query...")
    result = await discover_tools('filesystem operations')
    print(f"   Success: {result['success']}")
    print(f"   Found: {len(result['discovered_tools'])} tools")
    if result['discovered_tools']:
        print("   Top matches:")
        for tool in result['discovered_tools'][:3]:
            print(f"     - {tool['tool_name']}: {tool['description']}")
    
    # Test 2: Database query
    print("\n2. Testing database query...")
    result = await discover_tools('database query')
    print(f"   Success: {result['success']}")
    print(f"   Found: {len(result['discovered_tools'])} tools")
    if result['discovered_tools']:
        print("   Top matches:")
        for tool in result['discovered_tools'][:3]:
            print(f"     - {tool['tool_name']}: {tool['description']}")
    
    # Test 3: Terminal commands
    print("\n3. Testing terminal commands...")
    result = await discover_tools('execute command')
    print(f"   Success: {result['success']}")
    print(f"   Found: {len(result['discovered_tools'])} tools")
    if result['discovered_tools']:
        print("   Top matches:")
        for tool in result['discovered_tools'][:3]:
            print(f"     - {tool['tool_name']}: {tool['description']}")
    
    print("\n=== DISCOVERY TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_discovery()) 