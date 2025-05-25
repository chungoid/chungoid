#!/usr/bin/env python3

from chungoid.protocols import list_available_protocols, get_protocol

def test_protocol_system():
    print("🔍 Testing Universal Protocol Infrastructure")
    print("=" * 50)
    
    # Test protocol registry
    protocols = list_available_protocols()
    print(f"Available protocols ({len(protocols)}):")
    for p in protocols:
        print(f"  - {p}")
    
    print()
    print("Testing universal protocols:")
    universal_protocols = [
        'agent_communication', 
        'context_sharing', 
        'tool_validation', 
        'error_recovery', 
        'goal_tracking'
    ]
    
    for protocol_name in universal_protocols:
        status = "✅ OK" if protocol_name in protocols else "❌ MISSING"
        print(f"  {protocol_name}: {status}")
    
    print()
    print("Testing protocol instantiation:")
    
    # Test instantiating a universal protocol
    try:
        comm_protocol = get_protocol('agent_communication')
        print(f"✅ agent_communication protocol loaded")
        print(f"   Name: {comm_protocol.name}")
        print(f"   Description: {comm_protocol.description}")
        print(f"   Total time: {comm_protocol.total_estimated_time}h")
        
        # Test setup
        comm_protocol.setup({'test': 'context'})
        print(f"   Phases after setup: {len(comm_protocol.phases)}")
        for i, phase in enumerate(comm_protocol.phases):
            print(f"     Phase {i+1}: {phase.name} ({phase.time_box_hours}h)")
            
    except Exception as e:
        print(f"❌ Error testing agent_communication protocol: {e}")
    
    print()
    print("🎉 Universal Protocol Infrastructure Test Complete!")

if __name__ == "__main__":
    test_protocol_system() 