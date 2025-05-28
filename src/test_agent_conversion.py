#!/usr/bin/env python3

def check_agent_conversion():
    print("🔍 Agent Universal Protocol Conversion Status")
    print("=" * 60)
    
    converted_agents = []
    not_converted_agents = []
    
    agents_to_check = [
        ('BlueprintReviewerAgent_v1', 'chungoid.agents.autonomous_engine.blueprint_reviewer_agent'),
        ('ProjectDocumentationAgent_v1', 'chungoid.agents.autonomous_engine.project_documentation_agent'),
        ('AutomatedRefinementCoordinatorAgent_v1', 'chungoid.agents.autonomous_engine.automated_refinement_coordinator_agent'),
        ('ArchitectAgent_v1', 'chungoid.agents.autonomous_engine.architect_agent'),
        ('ProductAnalystAgent_v1', 'chungoid.agents.autonomous_engine.product_analyst_agent'),
        ('EnhancedArchitectAgent_v1', 'chungoid.agents.autonomous_engine.architect_agent'),
        ('ProactiveRiskAssessorAgent_v1', 'chungoid.agents.autonomous_engine.proactive_risk_assessor_agent'),
        ('RequirementsTracerAgent_v1', 'chungoid.agents.autonomous_engine.requirements_tracer_agent'),
        ('CodeDebuggingAgent_v1', 'chungoid.agents.autonomous_engine.code_debugging_agent'),
        ('EnvironmentBootstrapAgent', 'chungoid.agents.autonomous_engine.environment_bootstrap_agent'),
        ('DependencyManagementAgent_v1', 'chungoid.agents.autonomous_engine.dependency_management_agent')
    ]
    
    for agent_name, module_path in agents_to_check:
        try:
            module = __import__(module_path, fromlist=[agent_name])
            agent_class = getattr(module, agent_name)
            
            has_primary = hasattr(agent_class, 'PRIMARY_PROTOCOLS') and getattr(agent_class, 'PRIMARY_PROTOCOLS', None)
            has_universal = hasattr(agent_class, 'UNIVERSAL_PROTOCOLS') and getattr(agent_class, 'UNIVERSAL_PROTOCOLS', None)
            
            if has_primary and has_universal:
                status = '✅ CONVERTED'
                primary = getattr(agent_class, 'PRIMARY_PROTOCOLS', [])
                universal = getattr(agent_class, 'UNIVERSAL_PROTOCOLS', [])
                converted_agents.append(agent_name)
                print(f"{agent_name:40} {status}")
                print(f"{'':40}   Primary: {primary}")
                print(f"{'':40}   Universal: {universal}")
            else:
                status = '❌ NOT CONVERTED'
                not_converted_agents.append(agent_name)
                print(f"{agent_name:40} {status}")
                if has_primary and not has_universal:
                    print(f"{'':40}   Missing: UNIVERSAL_PROTOCOLS")
                elif not has_primary and has_universal:
                    print(f"{'':40}   Missing: PRIMARY_PROTOCOLS")
                else:
                    print(f"{'':40}   Missing: Both PRIMARY_PROTOCOLS and UNIVERSAL_PROTOCOLS")
        except Exception as e:
            print(f"{agent_name:40} ❌ ERROR: {e}")
            not_converted_agents.append(agent_name)
    
    print()
    print("📊 SUMMARY")
    print("=" * 30)
    print(f"✅ Converted agents: {len(converted_agents)}")
    print(f"❌ Not converted: {len(not_converted_agents)}")
    total_agents = len(converted_agents) + len(not_converted_agents)
    conversion_rate = (len(converted_agents) / total_agents * 100) if total_agents > 0 else 0
    print(f"📈 Conversion rate: {conversion_rate:.1f}%")
    
    if converted_agents:
        print(f"\n✅ CONVERTED AGENTS ({len(converted_agents)}):")
        for agent in converted_agents:
            print(f"   - {agent}")
    
    if not_converted_agents:
        print(f"\n❌ NOT CONVERTED AGENTS ({len(not_converted_agents)}):")
        for agent in not_converted_agents:
            print(f"   - {agent}")

if __name__ == "__main__":
    check_agent_conversion() 