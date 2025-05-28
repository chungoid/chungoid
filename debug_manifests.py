#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from chungoid.mcp_tools.intelligence.tool_manifest import tool_discovery

print("=== DEBUGGING TOOL MANIFESTS ===")

print(f"Total manifests: {len(tool_discovery.manifests)}")

print("\nSample tools and their capabilities:")
count = 0
for name, manifest in tool_discovery.manifests.items():
    if count >= 10:
        break
    capabilities = [cap.name for cap in manifest.capabilities]
    print(f"  {name}: {capabilities}")
    count += 1

print(f"\nLooking for filesystem tools...")
fs_tools = []
for name, manifest in tool_discovery.manifests.items():
    if "filesystem" in name.lower() or "file" in name.lower():
        fs_tools.append(name)

print(f"Found filesystem tools: {fs_tools}")

print(f"\nTesting capability search...")
matching_tools = tool_discovery.find_tools_by_capability("filesystem")
print(f"Tools matching 'filesystem': {[tool.tool_name for tool in matching_tools]}")

matching_tools = tool_discovery.find_tools_by_capability("file")
print(f"Tools matching 'file': {[tool.tool_name for tool in matching_tools]}")

matching_tools = tool_discovery.find_tools_by_capability("data_retrieval")
print(f"Tools matching 'data_retrieval': {[tool.tool_name for tool in matching_tools]}") 