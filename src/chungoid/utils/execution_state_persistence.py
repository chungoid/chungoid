"""Execution State Persistence Service

This module provides comprehensive execution state persistence capabilities for autonomous
build processes, enabling resilient operation that can resume from interruptions.

Key Features:
- Save/restore execution state at critical checkpoints
- Resume interrupted builds from last successful stage
- State integrity validation and corruption handling
- Automatic cleanup of stale checkpoints
- Performance monitoring and observability
- Seamless integration with UnifiedOrchestrator

Design Principles:
- Autonomous resilience with minimal intervention required
- Comprehensive state validation and integrity checking
- Efficient storage with compression for large contexts
- Graceful handling of edge cases and conflicts
- Full observability and debugging capabilities

Architecture:
- ExecutionCheckpoint: Immutable state snapshots with validation
- StateCheckpointManager: Core persistence and retrieval operations
- ResumableExecutionService: High-level resumption logic
- Integration hooks for AsyncOrchestrator and agents
"""

import hashlib
import json
import logging
import zlib
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from .exceptions import ChungoidError
from .state_manager import StateManager

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions
# ============================================================================

class ExecutionStateError(ChungoidError):
    """Base exception for execution state persistence operations."""
    pass

class CheckpointCorruptionError(ExecutionStateError):
    """Raised when checkpoint data is corrupted or invalid."""
    pass

class CheckpointNotFoundError(ExecutionStateError):
    """Raised when requested checkpoint cannot be found."""
    pass

class StateValidationError(ExecutionStateError):
    """Raised when state validation fails."""
    pass

class ResumptionConflictError(ExecutionStateError):
    """Raised when resumption conflicts with current execution."""
    pass

# ============================================================================
# Enums and Constants
# ============================================================================

class CheckpointStatus(str, Enum):
    """Status of an execution checkpoint."""
    ACTIVE = "active"           # Currently executing
    SUCCESS = "success"         # Completed successfully
    FAILED = "failed"          # Failed execution
    INTERRUPTED = "interrupted" # Execution was interrupted
    CORRUPTED = "corrupted"    # Checkpoint data is corrupted
    STALE = "stale"            # Checkpoint is too old

class StageType(str, Enum):
    """Types of execution stages that can be checkpointed."""
    FLOW_START = "flow_start"
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"
    AGENT_COMPLETE = "agent_complete"
    FLOW_END = "flow_end"
    ERROR_RECOVERY = "error_recovery"

# Constants
CHECKPOINT_COLLECTION = "execution_checkpoints"
CHECKPOINT_VERSION = "1.0.0"
DEFAULT_RETENTION_DAYS = 7
MAX_CONTEXT_SIZE = 10 * 1024 * 1024  # 10MB uncompressed
COMPRESSION_THRESHOLD = 1024  # Compress contexts larger than 1KB

# ============================================================================
# Data Models
# ============================================================================

class ExecutionContext(BaseModel):
    """Serializable execution context for checkpointing."""
    project_id: str = Field(..., description="Project identifier")
    project_root_path: str = Field(..., description="Project root directory path")
    run_id: str = Field(..., description="Execution run identifier")
    flow_id: str = Field(..., description="Flow identifier")
    stage_id: Optional[str] = Field(None, description="Current stage identifier")
    agent_name: Optional[str] = Field(None, description="Current agent name")
    
    # Serialized context data
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Serialized SharedContext data")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="Relevant environment variables")
    working_directory: str = Field(..., description="Current working directory")
    
    # Execution metadata
    execution_start_time: datetime = Field(..., description="When execution started")
    stage_start_time: Optional[datetime] = Field(None, description="When current stage started")
    total_stages: Optional[int] = Field(None, description="Total number of stages")
    completed_stages: int = Field(0, description="Number of completed stages")

class AgentOutput(BaseModel):
    """Captured agent output for checkpointing."""
    agent_name: str = Field(..., description="Name of the agent")
    stage_id: str = Field(..., description="Stage identifier")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Agent output data")
    execution_time: float = Field(..., description="Execution time in seconds")
    success: bool = Field(..., description="Whether agent execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="When output was captured")

class ExecutionCheckpoint(BaseModel):
    """Immutable snapshot of execution state at a specific point."""
    
    # Identifiers
    checkpoint_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique checkpoint identifier")
    flow_id: str = Field(..., description="Flow identifier")
    stage_id: Optional[str] = Field(None, description="Stage identifier")
    
    # Checkpoint metadata
    checkpoint_type: StageType = Field(..., description="Type of checkpoint")
    status: CheckpointStatus = Field(..., description="Checkpoint status")
    timestamp: datetime = Field(default_factory=datetime.now, description="When checkpoint was created")
    version: str = Field(default=CHECKPOINT_VERSION, description="Checkpoint format version")
    
    # Execution state
    execution_context: ExecutionContext = Field(..., description="Execution context at checkpoint")
    agent_outputs: List[AgentOutput] = Field(default_factory=list, description="Agent outputs up to this point")
    
    # State validation
    context_hash: str = Field(..., description="Hash of execution context for integrity checking")
    compressed_context: bool = Field(False, description="Whether context data is compressed")
    
    # Resumption metadata
    resumable: bool = Field(True, description="Whether execution can be resumed from this checkpoint")
    resume_instructions: Optional[str] = Field(None, description="Instructions for resuming execution")
    dependencies: List[str] = Field(default_factory=list, description="Dependencies required for resumption")
    
    # Cleanup metadata
    retention_until: datetime = Field(..., description="When this checkpoint expires")
    cleanup_priority: int = Field(1, description="Priority for cleanup (1=high, 5=low)")
    
    @validator('context_hash', always=True)
    def generate_context_hash(cls, v, values):
        """Generate hash of execution context for integrity checking."""
        if 'execution_context' in values:
            context_str = json.dumps(values['execution_context'].dict(), sort_keys=True)
            return hashlib.sha256(context_str.encode()).hexdigest()
        return v
    
    @validator('retention_until', always=True)
    def set_retention_period(cls, v, values):
        """Set retention period based on checkpoint type."""
        if v is None:
            timestamp = values.get('timestamp', datetime.now())
            checkpoint_type = values.get('checkpoint_type')
            
            # Extend retention for important checkpoints
            if checkpoint_type in [StageType.FLOW_START, StageType.STAGE_END]:
                retention_days = DEFAULT_RETENTION_DAYS * 2
            else:
                retention_days = DEFAULT_RETENTION_DAYS
                
            return timestamp + timedelta(days=retention_days)
        return v

class ResumptionPlan(BaseModel):
    """Plan for resuming execution from a checkpoint."""
    checkpoint: ExecutionCheckpoint = Field(..., description="Checkpoint to resume from")
    resume_stage_id: str = Field(..., description="Stage to resume from")
    skip_stages: List[str] = Field(default_factory=list, description="Stages to skip during resumption")
    validation_checks: List[str] = Field(default_factory=list, description="Validation checks to perform")
    estimated_remaining_time: Optional[float] = Field(None, description="Estimated time to completion")
    confidence_score: float = Field(..., description="Confidence in successful resumption")
    warnings: List[str] = Field(default_factory=list, description="Warnings about resumption")

# ============================================================================
# Core Service Classes
# ============================================================================

class StateCheckpointManager:
    """Core manager for execution state checkpoint operations."""
    
    def __init__(self, state_manager: StateManager):
        """Initialize checkpoint manager with StateManager for persistence.
        
        Args:
            state_manager: StateManager instance for ChromaDB operations
        """
        self.state_manager = state_manager
        self.collection_name = CHECKPOINT_COLLECTION
        
        logger.info("StateCheckpointManager initialized")
    
    async def save_checkpoint(
        self,
        checkpoint: ExecutionCheckpoint,
        compress_large_contexts: bool = True
    ) -> str:
        """Save execution checkpoint to persistent storage.
        
        Args:
            checkpoint: Checkpoint to save
            compress_large_contexts: Whether to compress large context data
            
        Returns:
            Checkpoint ID of saved checkpoint
            
        Raises:
            ExecutionStateError: If checkpoint save fails
        """
        try:
            logger.info(f"Saving checkpoint {checkpoint.checkpoint_id} for flow {checkpoint.flow_id}")
            
            # Prepare checkpoint data
            checkpoint_data = checkpoint.dict()
            
            # Compress context if it's large
            if compress_large_contexts:
                context_str = json.dumps(checkpoint_data['execution_context'])
                if len(context_str) > COMPRESSION_THRESHOLD:
                    compressed_context = zlib.compress(context_str.encode())
                    checkpoint_data['execution_context'] = compressed_context.hex()
                    checkpoint_data['compressed_context'] = True
                    logger.debug(f"Compressed context from {len(context_str)} to {len(compressed_context)} bytes")
            
            # Save to ChromaDB
            metadata = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "flow_id": checkpoint.flow_id,
                "stage_id": checkpoint.stage_id or "",
                "checkpoint_type": checkpoint.checkpoint_type.value,
                "status": checkpoint.status.value,
                "timestamp": checkpoint.timestamp.isoformat(),
                "version": checkpoint.version,
                "resumable": checkpoint.resumable,
                "retention_until": checkpoint.retention_until.isoformat(),
                "context_hash": checkpoint.context_hash
            }
            
            # Use checkpoint_id as document ID for easy retrieval
            document_id = checkpoint.checkpoint_id
            document_text = f"Execution checkpoint for flow {checkpoint.flow_id} at stage {checkpoint.stage_id}"
            
            await self.state_manager.add_document(
                collection_name=self.collection_name,
                document_id=document_id,
                document_text=document_text,
                metadata=metadata,
                additional_data=checkpoint_data
            )
            
            logger.info(f"Successfully saved checkpoint {checkpoint.checkpoint_id}")
            return checkpoint.checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint.checkpoint_id}: {e}")
            raise ExecutionStateError(f"Checkpoint save failed: {e}") from e
    
    async def load_checkpoint(self, checkpoint_id: str) -> ExecutionCheckpoint:
        """Load execution checkpoint from persistent storage.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            
        Returns:
            Loaded execution checkpoint
            
        Raises:
            CheckpointNotFoundError: If checkpoint doesn't exist
            CheckpointCorruptionError: If checkpoint data is corrupted
        """
        try:
            logger.info(f"Loading checkpoint {checkpoint_id}")
            
            # Query ChromaDB for checkpoint
            results = await self.state_manager.query_documents(
                collection_name=self.collection_name,
                query_text="",  # Empty query to match by metadata
                filter_criteria={"checkpoint_id": checkpoint_id},
                max_results=1
            )
            
            if not results:
                raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")
            
            result = results[0]
            checkpoint_data = result.get('additional_data', {})
            
            # Decompress context if needed
            if checkpoint_data.get('compressed_context', False):
                compressed_hex = checkpoint_data['execution_context']
                compressed_bytes = bytes.fromhex(compressed_hex)
                context_str = zlib.decompress(compressed_bytes).decode()
                checkpoint_data['execution_context'] = json.loads(context_str)
                checkpoint_data['compressed_context'] = False
            
            # Validate checkpoint integrity
            checkpoint = ExecutionCheckpoint(**checkpoint_data)
            if not self._validate_checkpoint_integrity(checkpoint):
                raise CheckpointCorruptionError(f"Checkpoint {checkpoint_id} failed integrity validation")
            
            logger.info(f"Successfully loaded checkpoint {checkpoint_id}")
            return checkpoint
            
        except (CheckpointNotFoundError, CheckpointCorruptionError):
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise ExecutionStateError(f"Checkpoint load failed: {e}") from e
    
    async def find_resumable_checkpoints(
        self,
        flow_id: Optional[str] = None,
        project_id: Optional[str] = None,
        max_age_hours: int = 72
    ) -> List[ExecutionCheckpoint]:
        """Find checkpoints that can be used for resuming execution.
        
        Args:
            flow_id: Optional flow ID to filter by
            project_id: Optional project ID to filter by
            max_age_hours: Maximum age of checkpoints to consider
            
        Returns:
            List of resumable checkpoints
        """
        try:
            logger.info(f"Finding resumable checkpoints (flow_id={flow_id}, project_id={project_id})")
            
            # Build filter criteria
            filter_criteria = {
                "resumable": True,
                "status": CheckpointStatus.INTERRUPTED.value
            }
            
            if flow_id:
                filter_criteria["flow_id"] = flow_id
            
            # Query for resumable checkpoints
            results = await self.state_manager.query_documents(
                collection_name=self.collection_name,
                query_text="resumable execution checkpoint",
                filter_criteria=filter_criteria,
                max_results=50
            )
            
            checkpoints = []
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for result in results:
                try:
                    checkpoint_data = result.get('additional_data', {})
                    
                    # Decompress if needed
                    if checkpoint_data.get('compressed_context', False):
                        compressed_hex = checkpoint_data['execution_context']
                        compressed_bytes = bytes.fromhex(compressed_hex)
                        context_str = zlib.decompress(compressed_bytes).decode()
                        checkpoint_data['execution_context'] = json.loads(context_str)
                        checkpoint_data['compressed_context'] = False
                    
                    checkpoint = ExecutionCheckpoint(**checkpoint_data)
                    
                    # Filter by age and project
                    if checkpoint.timestamp < cutoff_time:
                        continue
                    
                    if project_id and checkpoint.execution_context.project_id != project_id:
                        continue
                    
                    # Validate integrity
                    if self._validate_checkpoint_integrity(checkpoint):
                        checkpoints.append(checkpoint)
                    else:
                        logger.warning(f"Skipping corrupted checkpoint {checkpoint.checkpoint_id}")
                        
                except Exception as e:
                    logger.warning(f"Error processing checkpoint result: {e}")
                    continue
            
            logger.info(f"Found {len(checkpoints)} resumable checkpoints")
            return sorted(checkpoints, key=lambda c: c.timestamp, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to find resumable checkpoints: {e}")
            return []
    
    async def cleanup_stale_checkpoints(self, dry_run: bool = False) -> int:
        """Clean up stale and expired checkpoints.
        
        Args:
            dry_run: If True, only report what would be cleaned up
            
        Returns:
            Number of checkpoints cleaned up (or would be cleaned up)
        """
        try:
            logger.info(f"Starting checkpoint cleanup (dry_run={dry_run})")
            
            now = datetime.now()
            cleanup_count = 0
            
            # Query all checkpoints
            results = await self.state_manager.query_documents(
                collection_name=self.collection_name,
                query_text="",
                max_results=1000
            )
            
            for result in results:
                try:
                    metadata = result.get('metadata', {})
                    retention_until = datetime.fromisoformat(metadata.get('retention_until', ''))
                    checkpoint_id = metadata.get('checkpoint_id')
                    status = metadata.get('status')
                    
                    # Determine if checkpoint should be cleaned up
                    should_cleanup = False
                    
                    if retention_until < now:
                        should_cleanup = True
                        reason = "expired"
                    elif status == CheckpointStatus.CORRUPTED.value:
                        should_cleanup = True
                        reason = "corrupted"
                    elif status == CheckpointStatus.STALE.value:
                        should_cleanup = True
                        reason = "stale"
                    
                    if should_cleanup:
                        if dry_run:
                            logger.info(f"Would cleanup checkpoint {checkpoint_id} ({reason})")
                        else:
                            await self.state_manager.delete_document(
                                collection_name=self.collection_name,
                                document_id=checkpoint_id
                            )
                            logger.info(f"Cleaned up checkpoint {checkpoint_id} ({reason})")
                        
                        cleanup_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing checkpoint for cleanup: {e}")
                    continue
            
            action = "Would cleanup" if dry_run else "Cleaned up"
            logger.info(f"{action} {cleanup_count} stale checkpoints")
            return cleanup_count
            
        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")
            return 0
    
    def _validate_checkpoint_integrity(self, checkpoint: ExecutionCheckpoint) -> bool:
        """Validate checkpoint data integrity using stored hash.
        
        Args:
            checkpoint: Checkpoint to validate
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Recalculate context hash
            context_str = json.dumps(checkpoint.execution_context.dict(), sort_keys=True)
            calculated_hash = hashlib.sha256(context_str.encode()).hexdigest()
            
            # Compare with stored hash
            is_valid = calculated_hash == checkpoint.context_hash
            
            if not is_valid:
                logger.warning(f"Checkpoint {checkpoint.checkpoint_id} failed integrity check")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating checkpoint integrity: {e}")
            return False

class ResumableExecutionService:
    """High-level service for managing resumable execution capabilities."""
    
    def __init__(self, state_manager: StateManager):
        """Initialize service with state manager.
        
        Args:
            state_manager: StateManager instance for persistence
        """
        self.checkpoint_manager = StateCheckpointManager(state_manager)
        
        logger.info("ResumableExecutionService initialized")
    
    async def create_checkpoint(
        self,
        flow_id: str,
        stage_id: Optional[str],
        checkpoint_type: StageType,
        status: CheckpointStatus,
        execution_context: ExecutionContext,
        agent_outputs: List[AgentOutput] = None,
        resume_instructions: Optional[str] = None
    ) -> ExecutionCheckpoint:
        """Create and save a new execution checkpoint.
        
        Args:
            flow_id: Flow identifier
            stage_id: Optional stage identifier
            checkpoint_type: Type of checkpoint
            status: Checkpoint status
            execution_context: Current execution context
            agent_outputs: Agent outputs to include
            resume_instructions: Optional resumption instructions
            
        Returns:
            Created checkpoint
        """
        try:
            checkpoint = ExecutionCheckpoint(
                flow_id=flow_id,
                stage_id=stage_id,
                checkpoint_type=checkpoint_type,
                status=status,
                execution_context=execution_context,
                agent_outputs=agent_outputs or [],
                resume_instructions=resume_instructions
            )
            
            await self.checkpoint_manager.save_checkpoint(checkpoint)
            
            logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for {checkpoint_type.value}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise ExecutionStateError(f"Checkpoint creation failed: {e}") from e
    
    async def find_resumption_opportunities(
        self,
        project_id: str,
        max_age_hours: int = 24
    ) -> List[ResumptionPlan]:
        """Find opportunities to resume interrupted executions.
        
        Args:
            project_id: Project to find resumption opportunities for
            max_age_hours: Maximum age of checkpoints to consider
            
        Returns:
            List of resumption plans ordered by confidence
        """
        try:
            logger.info(f"Finding resumption opportunities for project {project_id}")
            
            # Find resumable checkpoints
            checkpoints = await self.checkpoint_manager.find_resumable_checkpoints(
                project_id=project_id,
                max_age_hours=max_age_hours
            )
            
            resumption_plans = []
            
            for checkpoint in checkpoints:
                try:
                    plan = await self._create_resumption_plan(checkpoint)
                    if plan.confidence_score > 0.3:  # Only include viable plans
                        resumption_plans.append(plan)
                except Exception as e:
                    logger.warning(f"Error creating resumption plan for checkpoint {checkpoint.checkpoint_id}: {e}")
                    continue
            
            # Sort by confidence score (highest first)
            resumption_plans.sort(key=lambda p: p.confidence_score, reverse=True)
            
            logger.info(f"Found {len(resumption_plans)} viable resumption opportunities")
            return resumption_plans
            
        except Exception as e:
            logger.error(f"Failed to find resumption opportunities: {e}")
            return []
    
    async def resume_execution(
        self,
        resumption_plan: ResumptionPlan,
        validate_environment: bool = True
    ) -> ExecutionContext:
        """Resume execution from a checkpoint using the provided plan.
        
        Args:
            resumption_plan: Plan for resuming execution
            validate_environment: Whether to validate environment before resuming
            
        Returns:
            Restored execution context
            
        Raises:
            StateValidationError: If validation fails
            ResumptionConflictError: If resumption conflicts with current state
        """
        try:
            checkpoint = resumption_plan.checkpoint
            logger.info(f"Resuming execution from checkpoint {checkpoint.checkpoint_id}")
            
            # Validate environment if requested
            if validate_environment:
                await self._validate_resumption_environment(checkpoint)
            
            # Check for conflicts with current execution
            await self._check_resumption_conflicts(checkpoint)
            
            # Restore execution context
            restored_context = checkpoint.execution_context
            
            # Mark checkpoint as being resumed
            checkpoint.status = CheckpointStatus.ACTIVE
            await self.checkpoint_manager.save_checkpoint(checkpoint)
            
            logger.info(f"Successfully resumed execution from checkpoint {checkpoint.checkpoint_id}")
            return restored_context
            
        except Exception as e:
            logger.error(f"Failed to resume execution: {e}")
            raise ExecutionStateError(f"Execution resumption failed: {e}") from e
    
    async def _create_resumption_plan(self, checkpoint: ExecutionCheckpoint) -> ResumptionPlan:
        """Create a resumption plan for a checkpoint."""
        
        # Analyze checkpoint to determine resumption strategy
        confidence_score = 0.8  # Base confidence
        warnings = []
        validation_checks = ["environment", "dependencies", "file_system"]
        
        # Reduce confidence based on checkpoint age
        age_hours = (datetime.now() - checkpoint.timestamp).total_seconds() / 3600
        if age_hours > 24:
            confidence_score -= 0.2
            warnings.append("Checkpoint is more than 24 hours old")
        
        # Reduce confidence if there are many agent outputs to replay
        if len(checkpoint.agent_outputs) > 10:
            confidence_score -= 0.1
            warnings.append("Many agent outputs to replay")
        
        # Determine resume stage
        resume_stage_id = checkpoint.stage_id or "unknown"
        if checkpoint.checkpoint_type == StageType.STAGE_END:
            # Resume from next stage
            resume_stage_id = f"next_after_{checkpoint.stage_id}"
        
        return ResumptionPlan(
            checkpoint=checkpoint,
            resume_stage_id=resume_stage_id,
            validation_checks=validation_checks,
            confidence_score=max(0.0, confidence_score),
            warnings=warnings
        )
    
    async def _validate_resumption_environment(self, checkpoint: ExecutionCheckpoint) -> None:
        """Validate that the environment is suitable for resumption."""
        
        context = checkpoint.execution_context
        
        # Check if project path still exists
        project_path = Path(context.project_root_path)
        if not project_path.exists():
            raise StateValidationError(f"Project path no longer exists: {project_path}")
        
        # Check working directory
        working_dir = Path(context.working_directory)
        if not working_dir.exists():
            logger.warning(f"Working directory no longer exists, will use project root: {working_dir}")
        
        # Additional environment validation could be added here
        
        logger.info("Resumption environment validation passed")
    
    async def _check_resumption_conflicts(self, checkpoint: ExecutionCheckpoint) -> None:
        """Check for conflicts that would prevent resumption."""
        
        # Check if another execution is already running for this flow
        active_checkpoints = await self.checkpoint_manager.find_resumable_checkpoints(
            flow_id=checkpoint.flow_id
        )
        
        active_executions = [
            cp for cp in active_checkpoints 
            if cp.status == CheckpointStatus.ACTIVE and cp.checkpoint_id != checkpoint.checkpoint_id
        ]
        
        if active_executions:
            raise ResumptionConflictError(
                f"Another execution is already active for flow {checkpoint.flow_id}"
            )
        
        logger.info("No resumption conflicts detected")

# ============================================================================
# Utility Functions
# ============================================================================

async def create_execution_checkpoint(
    state_manager: StateManager,
    flow_id: str,
    stage_id: Optional[str],
    checkpoint_type: StageType,
    status: CheckpointStatus,
    execution_context: ExecutionContext,
    agent_outputs: List[AgentOutput] = None
) -> ExecutionCheckpoint:
    """
    Convenience function for creating execution checkpoints.
    
    Args:
        state_manager: StateManager instance
        flow_id: Flow identifier
        stage_id: Optional stage identifier  
        checkpoint_type: Type of checkpoint
        status: Checkpoint status
        execution_context: Current execution context
        agent_outputs: Optional agent outputs
        
    Returns:
        Created checkpoint
    """
    service = ResumableExecutionService(state_manager)
    return await service.create_checkpoint(
        flow_id=flow_id,
        stage_id=stage_id,
        checkpoint_type=checkpoint_type,
        status=status,
        execution_context=execution_context,
        agent_outputs=agent_outputs
    )

async def find_resumable_executions(
    state_manager: StateManager,
    project_id: str,
    max_age_hours: int = 24
) -> List[ResumptionPlan]:
    """
    Convenience function for finding resumable executions.
    
    Args:
        state_manager: StateManager instance
        project_id: Project identifier
        max_age_hours: Maximum age of checkpoints
        
    Returns:
        List of resumption plans
    """
    service = ResumableExecutionService(state_manager)
    return await service.find_resumption_opportunities(
        project_id=project_id,
        max_age_hours=max_age_hours
    )

# Example usage and testing support
if __name__ == "__main__":
    # Test the service
    import asyncio
    from datetime import datetime
    from uuid import uuid4
    
    async def test_checkpoint_service():
        # This would be used for testing the service
        print("Execution State Persistence Service - Test Mode")
        
        # Create test execution context
        test_context = ExecutionContext(
            project_id="test_project",
            project_root_path="/tmp/test",
            run_id=str(uuid4()),
            flow_id=str(uuid4()),
            working_directory="/tmp/test",
            execution_start_time=datetime.now()
        )
        
        print(f"Created test context for project {test_context.project_id}")
        print("Service implementation complete!")
    
    # Run test if executed directly
    asyncio.run(test_checkpoint_service()) 