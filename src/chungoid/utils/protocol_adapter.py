"""
Protocol Execution Adapter

Bridges the legacy invoke_async interface to the new execute_with_protocol execution model.
This adapter allows the agent resolver and orchestrator to work with ProtocolAwareAgent instances
without requiring changes to the calling code.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from ..agents.protocol_aware_agent import ProtocolAwareAgent
from ..schemas.agent_outputs import AgentOutput

logger = logging.getLogger(__name__)


class ProtocolExecutionAdapter:
    """
    Adapter to bridge invoke_async interface to execute_with_protocol.
    
    This adapter allows legacy code that expects invoke_async to work seamlessly
    with new ProtocolAwareAgent instances that only support execute_with_protocol.
    """
    
    def __init__(self, agent_instance: ProtocolAwareAgent):
        """Initialize the adapter with a ProtocolAwareAgent instance."""
        if not isinstance(agent_instance, ProtocolAwareAgent):
            raise TypeError(f"Expected ProtocolAwareAgent, got {type(agent_instance)}")
        
        self.agent = agent_instance
        self.logger = logging.getLogger(f"adapter.{agent_instance.agent_id}")
        
        # Validate agent has required protocols
        if not self.agent.PRIMARY_PROTOCOLS:
            raise ValueError(f"Agent {agent_instance.agent_id} has no PRIMARY_PROTOCOLS defined")
    
    async def __call__(self, inputs: Any = None, full_context: Dict[str, Any] = None, input_payload: Any = None, task_input: Any = None, **kwargs) -> Any:
        """
        Adapter callable that mimics invoke_async interface.
        
        Args:
            inputs: Task inputs (can be any type) - new interface
            full_context: Full execution context
            input_payload: Task inputs (can be any type) - legacy interface for backward compatibility
            task_input: Task inputs (can be any type) - orchestrator interface
            **kwargs: Additional keyword arguments for flexibility
            
        Returns:
            Task results in the format expected by legacy code
        """
        try:
            self.logger.info(f"Protocol adapter executing for agent: {self.agent.agent_id}")
            
            # Handle multiple parameter names for backward compatibility
            actual_inputs = inputs if inputs is not None else (input_payload if input_payload is not None else task_input)
            if actual_inputs is None:
                raise ValueError("Either 'inputs', 'input_payload', or 'task_input' must be provided")
            
            # Handle additional context from kwargs
            if full_context is None:
                full_context = {}
            
            # Merge any additional context from kwargs
            for key, value in kwargs.items():
                if key not in ['inputs', 'input_payload', 'task_input'] and key not in full_context:
                    full_context[key] = value
            
            # Check if agent has custom execute_with_protocols method (with 's')
            if hasattr(self.agent, 'execute_with_protocols') and callable(getattr(self.agent, 'execute_with_protocols')):
                self.logger.debug(f"Using agent's custom execute_with_protocols method")
                # Call the custom method directly with inputs and full_context
                result = await self.agent.execute_with_protocols(actual_inputs, full_context)
            else:
                # Fallback to generic protocol execution
                self.logger.debug(f"Using generic execute_with_protocol method")
                
                # Determine primary protocol
                primary_protocol = self.agent.PRIMARY_PROTOCOLS[0]
                self.logger.debug(f"Using primary protocol: {primary_protocol}")
                
                # Create protocol context from legacy inputs
                protocol_context = self._create_protocol_context(actual_inputs, full_context)
                
                # Execute via protocol
                result = await self.agent.execute_with_protocol(primary_protocol, protocol_context)
            
            # Convert protocol result to legacy format
            legacy_result = self._convert_to_legacy_format(result, actual_inputs)
            
            self.logger.info(f"Protocol adapter execution completed successfully")
            return legacy_result
                
        except Exception as e:
            self.logger.error(f"Protocol adapter execution failed: {e}")
            # Re-raise the exception to maintain error handling behavior
            raise
    
    def _create_protocol_context(self, inputs: Any, full_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create protocol context from legacy invoke_async parameters.
        
        Args:
            inputs: Original task inputs
            full_context: Original full context
            
        Returns:
            Protocol-compatible context dictionary
        """
        protocol_context = {
            "inputs": inputs,
            "full_context": full_context or {},
            "execution_mode": "orchestrated",
            "timestamp": datetime.now().isoformat(),
            "adapter_version": "1.0.0"
        }
        
        # If inputs is a Pydantic model, convert to dict
        if hasattr(inputs, 'dict'):
            protocol_context["inputs"] = inputs.dict()
        elif hasattr(inputs, 'model_dump'):
            protocol_context["inputs"] = inputs.model_dump()
        
        # Add agent-specific context if available
        if hasattr(self.agent, '_config'):
            protocol_context["agent_config"] = self.agent._config
        
        return protocol_context
    
    def _convert_to_legacy_format(self, protocol_result: Any, original_inputs: Any) -> Any:
        """
        Convert protocol execution result to legacy invoke_async format.
        
        Args:
            protocol_result: Result from execute_with_protocol (could be AgentOutput or direct result)
            original_inputs: Original inputs for context
            
        Returns:
            Result in the format expected by legacy code
        """
        # If the protocol result is already in the expected format (e.g., MasterPlannerOutput),
        # return it directly without conversion
        if hasattr(protocol_result, 'master_plan_json') or hasattr(protocol_result, 'model_dump'):
            self.logger.debug(f"Protocol result is already in expected format, returning directly")
            return protocol_result
        
        # If it's an AgentOutput, extract the data
        if hasattr(protocol_result, 'success') and hasattr(protocol_result, 'data'):
            # If protocol execution failed, raise an exception
            if not protocol_result.success:
                error_msg = protocol_result.error or "Protocol execution failed"
                raise RuntimeError(f"Agent execution failed: {error_msg}")
            
            # Extract data from AgentOutput
            result_data = protocol_result.data
            
            # If the result data has a specific structure expected by the caller,
            # we may need to transform it. For now, return the data as-is.
            if isinstance(result_data, dict):
                # Add some metadata for debugging/monitoring
                result_data["_protocol_execution_info"] = {
                    "protocol_used": protocol_result.protocol_used,
                    "execution_time": protocol_result.execution_time,
                    "phases_completed": protocol_result.phases_completed,
                    "agent_id": protocol_result.agent_id
                }
                return result_data
            else:
                # If data is not a dict, wrap it in a dict with metadata
                return {
                    "result": result_data,
                    "_protocol_execution_info": {
                        "protocol_used": protocol_result.protocol_used,
                        "execution_time": protocol_result.execution_time,
                        "phases_completed": protocol_result.phases_completed,
                        "agent_id": protocol_result.agent_id
                    }
                }
        
        # If it's a simple dict or other format, return as-is
        if isinstance(protocol_result, dict):
            return protocol_result
        
        # For any other format, wrap it appropriately
        return {"result": protocol_result}


class AdapterFactory:
    """Factory for creating protocol adapters with validation."""
    
    @staticmethod
    def create_adapter(agent_instance: ProtocolAwareAgent) -> ProtocolExecutionAdapter:
        """
        Create a protocol adapter for the given agent instance.
        
        Args:
            agent_instance: ProtocolAwareAgent to wrap
            
        Returns:
            ProtocolExecutionAdapter instance
            
        Raises:
            TypeError: If agent_instance is not a ProtocolAwareAgent
            ValueError: If agent has no protocols defined
        """
        return ProtocolExecutionAdapter(agent_instance)
    
    @staticmethod
    def validate_agent_compatibility(agent_instance: Any) -> bool:
        """
        Validate that an agent instance is compatible with protocol adapters.
        
        Args:
            agent_instance: Agent instance to validate
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            if not isinstance(agent_instance, ProtocolAwareAgent):
                return False
            
            if not hasattr(agent_instance, 'PRIMARY_PROTOCOLS'):
                return False
            
            if not agent_instance.PRIMARY_PROTOCOLS:
                return False
            
            return True
        except Exception:
            return False 