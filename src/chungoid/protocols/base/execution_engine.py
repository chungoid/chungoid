"""
Protocol Execution Engine

Provides the core execution engine for running protocols systematically,
managing phase transitions, and coordinating validation.
"""

from typing import List, Dict, Any, Optional, Callable
import asyncio
import logging
from datetime import datetime, timedelta
from .protocol_interface import ProtocolInterface, ProtocolPhase
from .validation import ProtocolValidator, ValidationResult, ValidationLevel

logger = logging.getLogger(__name__)

class ExecutionContext:
    """Context object passed through protocol execution"""
    
    def __init__(self, protocol_name: str, initial_inputs: Dict[str, Any]):
        self.protocol_name = protocol_name
        self.inputs = initial_inputs
        self.phase_results = {}
        self.accumulated_outputs = {}
        self.start_time = datetime.now()
        self.current_phase = None
        self.validation_results = []
        self.errors = []
        
    def add_phase_result(self, phase_name: str, result: Dict[str, Any]):
        """Add result from a completed phase"""
        self.phase_results[phase_name] = result
        self.accumulated_outputs.update(result)
        
    def get_phase_result(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """Get result from a specific phase"""
        return self.phase_results.get(phase_name)
        
    def add_validation_result(self, result: ValidationResult):
        """Add a validation result"""
        self.validation_results.append(result)
        
    def add_error(self, error: str, phase: str = None):
        """Add an error during execution"""
        error_info = {
            "error": error,
            "phase": phase or self.current_phase,
            "timestamp": datetime.now()
        }
        self.errors.append(error_info)
        
    def has_critical_errors(self) -> bool:
        """Check if there are critical validation errors"""
        return any(
            result.level == ValidationLevel.CRITICAL 
            for result in self.validation_results
            if not result.passed
        )

class ProtocolExecutionEngine:
    """
    Core engine for executing protocols systematically.
    
    Manages phase execution, validation gates, and error handling
    to ensure reliable protocol completion.
    """
    
    def __init__(self, validator: Optional[ProtocolValidator] = None):
        self.validator = validator or ProtocolValidator()
        self.execution_hooks = {}
        self.phase_callbacks = {}
        
    def register_phase_callback(self, phase_name: str, callback: Callable):
        """Register a callback for specific phase completion"""
        if phase_name not in self.phase_callbacks:
            self.phase_callbacks[phase_name] = []
        self.phase_callbacks[phase_name].append(callback)
        
    def register_execution_hook(self, hook_type: str, callback: Callable):
        """Register execution hooks (pre_execution, post_execution, etc.)"""
        if hook_type not in self.execution_hooks:
            self.execution_hooks[hook_type] = []
        self.execution_hooks[hook_type].append(callback)
        
    def execute_protocol(self, protocol: ProtocolInterface, 
                        inputs: Dict[str, Any],
                        agent_executor: Optional[Callable] = None) -> ExecutionContext:
        """
        Execute a protocol completely with validation gates.
        
        Args:
            protocol: Protocol to execute
            inputs: Initial inputs for protocol
            agent_executor: Optional agent executor function
            
        Returns:
            ExecutionContext with results and validation info
        """
        context = ExecutionContext(protocol.name, inputs)
        
        try:
            # Pre-execution hooks
            self._run_hooks("pre_execution", context, protocol)
            
            # Execute each phase sequentially
            for phase in protocol.phases:
                context.current_phase = phase.name
                logger.info(f"Executing phase: {phase.name}")
                
                # Phase execution
                phase_result = self._execute_phase(phase, context, agent_executor)
                context.add_phase_result(phase.name, phase_result)
                
                # Phase validation
                validation_results = self._validate_phase(phase, phase_result, context)
                for result in validation_results:
                    context.add_validation_result(result)
                
                # Check for critical errors
                if context.has_critical_errors():
                    error_msg = f"Critical validation errors in phase {phase.name}"
                    context.add_error(error_msg)
                    logger.error(error_msg)
                    break
                    
                # Phase completion callbacks
                self._run_phase_callbacks(phase.name, context)
                
            # Post-execution hooks
            self._run_hooks("post_execution", context, protocol)
            
        except Exception as e:
            error_msg = f"Protocol execution failed: {str(e)}"
            context.add_error(error_msg)
            logger.error(error_msg, exc_info=True)
            
        return context
        
    async def execute_protocol_async(self, protocol: ProtocolInterface,
                                   inputs: Dict[str, Any],
                                   agent_executor: Optional[Callable] = None) -> ExecutionContext:
        """
        Execute a protocol asynchronously.
        
        Args:
            protocol: Protocol to execute
            inputs: Initial inputs for protocol
            agent_executor: Optional async agent executor function
            
        Returns:
            ExecutionContext with results and validation info
        """
        context = ExecutionContext(protocol.name, inputs)
        
        try:
            # Pre-execution hooks
            await self._run_hooks_async("pre_execution", context, protocol)
            
            # Execute each phase sequentially
            for phase in protocol.phases:
                context.current_phase = phase.name
                logger.info(f"Executing phase: {phase.name}")
                
                # Check timeout
                if self._is_phase_timeout(phase, context):
                    error_msg = f"Phase {phase.name} timeout exceeded"
                    context.add_error(error_msg)
                    logger.warning(error_msg)
                    break
                
                # Phase execution
                if agent_executor and asyncio.iscoroutinefunction(agent_executor):
                    phase_result = await agent_executor(phase, context)
                else:
                    phase_result = self._execute_phase(phase, context, agent_executor)
                    
                context.add_phase_result(phase.name, phase_result)
                
                # Phase validation
                validation_results = self._validate_phase(phase, phase_result, context)
                for result in validation_results:
                    context.add_validation_result(result)
                
                # Check for critical errors
                if context.has_critical_errors():
                    error_msg = f"Critical validation errors in phase {phase.name}"
                    context.add_error(error_msg)
                    logger.error(error_msg)
                    break
                    
                # Phase completion callbacks
                await self._run_phase_callbacks_async(phase.name, context)
                
            # Post-execution hooks
            await self._run_hooks_async("post_execution", context, protocol)
            
        except Exception as e:
            error_msg = f"Protocol execution failed: {str(e)}"
            context.add_error(error_msg)
            logger.error(error_msg, exc_info=True)
            
        return context
        
    def _execute_phase(self, phase: ProtocolPhase, context: ExecutionContext,
                      agent_executor: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute a single protocol phase"""
        if agent_executor:
            return agent_executor(phase, context)
        else:
            # Default phase execution - placeholder
            logger.warning(f"No agent executor provided for phase {phase.name}")
            return {
                "phase_name": phase.name,
                "status": "completed",
                "message": f"Phase {phase.name} executed with default handler"
            }
            
    def _validate_phase(self, phase: ProtocolPhase, phase_result: Dict[str, Any],
                       context: ExecutionContext) -> List[ValidationResult]:
        """Validate phase completion and outputs"""
        return self.validator.validate_protocol_phase(
            phase.name, 
            phase_result, 
            phase.validation_criteria
        )
        
    def _is_phase_timeout(self, phase: ProtocolPhase, context: ExecutionContext) -> bool:
        """Check if phase has exceeded its time box"""
        if not phase.time_box_hours:
            return False
            
        elapsed = datetime.now() - context.start_time
        timeout = timedelta(hours=phase.time_box_hours)
        return elapsed > timeout
        
    def _run_hooks(self, hook_type: str, context: ExecutionContext, 
                  protocol: ProtocolInterface):
        """Run synchronous execution hooks"""
        if hook_type in self.execution_hooks:
            for hook in self.execution_hooks[hook_type]:
                try:
                    hook(context, protocol)
                except Exception as e:
                    logger.error(f"Hook {hook_type} failed: {e}")
                    
    async def _run_hooks_async(self, hook_type: str, context: ExecutionContext,
                              protocol: ProtocolInterface):
        """Run asynchronous execution hooks"""
        if hook_type in self.execution_hooks:
            for hook in self.execution_hooks[hook_type]:
                try:
                    if asyncio.iscoroutinefunction(hook):
                        await hook(context, protocol)
                    else:
                        hook(context, protocol)
                except Exception as e:
                    logger.error(f"Hook {hook_type} failed: {e}")
                    
    def _run_phase_callbacks(self, phase_name: str, context: ExecutionContext):
        """Run synchronous phase callbacks"""
        if phase_name in self.phase_callbacks:
            for callback in self.phase_callbacks[phase_name]:
                try:
                    callback(context)
                except Exception as e:
                    logger.error(f"Phase callback for {phase_name} failed: {e}")
                    
    async def _run_phase_callbacks_async(self, phase_name: str, context: ExecutionContext):
        """Run asynchronous phase callbacks"""
        if phase_name in self.phase_callbacks:
            for callback in self.phase_callbacks[phase_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(context)
                    else:
                        callback(context)
                except Exception as e:
                    logger.error(f"Phase callback for {phase_name} failed: {e}") 