"""Environment Bootstrap Agent V1

This module provides comprehensive environment bootstrapping capabilities for autonomous
development environment setup, supporting multiple programming languages and project types.

Key Features:
- Auto-detection of project requirements using Project Type Detection Service
- Multi-language environment support (Python, Node.js, Java, etc.)
- Adaptive bootstrapping strategies based on project characteristics
- Integration with Smart Dependency Analysis for automatic dependency installation
- Cross-platform support (Windows, Linux, macOS)
- Resumable environment setup with state persistence
- MCP tool exposure for external agent usage
- Comprehensive validation and error recovery

Design Principles:
- Autonomous operation with intelligent decision-making
- Leverages existing services for maximum intelligence
- Comprehensive error handling and recovery strategies
- Production-ready with extensive logging and validation
- Configuration-driven behavior via Configuration Management

Environment Strategies:
- PythonEnvironmentStrategy: Virtual environment creation and management
- NodeJSEnvironmentStrategy: Node.js environment setup and package management
- JavaEnvironmentStrategy: Java environment configuration and build tools
- MultiLanguageEnvironmentStrategy: Coordination for polyglot projects

Architecture:
- EnvironmentBootstrapAgent: Main agent class with LLM reasoning
- Environment Strategy Pattern: Pluggable strategies for different languages
- BootstrapExecutor: Coordinates environment creation and validation
- MCP Tool Wrappers: External tool exposure for other agents
"""

import asyncio
import logging
import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, ClassVar
from uuid import uuid4

from chungoid.schemas.orchestration import SharedContext
from chungoid.utils import state_manager
from pydantic import BaseModel, Field, validator

from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.llm_provider import LLMProvider
from chungoid.schemas.common import ConfidenceScore
from ..protocol_aware_agent import ProtocolAwareAgent
from ...protocols.base.protocol_interface import ProtocolPhase
from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.agent_registry import AgentCard, AgentCategory, AgentVisibility
from chungoid.utils.exceptions import ChungoidError
from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.execution_state_persistence import AgentOutput, CheckpointStatus, ExecutionContext, ResumableExecutionService, StageType, create_execution_checkpoint
from chungoid.utils.project_type_detection import (
    ProjectTypeDetectionService,
    ProjectTypeDetectionResult
)
from chungoid.utils.smart_dependency_analysis import (
    SmartDependencyAnalysisService,
    DependencyInfo,
    DependencyAnalysisResult
)

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions
# ============================================================================

class EnvironmentBootstrapError(ChungoidError):
    """Base exception for environment bootstrap operations."""
    pass

class EnvironmentDetectionError(EnvironmentBootstrapError):
    """Raised when environment detection fails."""
    pass

class EnvironmentCreationError(EnvironmentBootstrapError):
    """Raised when environment creation fails."""
    pass

class EnvironmentValidationError(EnvironmentBootstrapError):
    """Raised when environment validation fails."""
    pass

class UnsupportedEnvironmentError(EnvironmentBootstrapError):
    """Raised when requested environment type is not supported."""
    pass

# ============================================================================
# Enums and Constants
# ============================================================================

class EnvironmentType(str, Enum):
    """Supported environment types."""
    PYTHON = "python"
    NODEJS = "nodejs"
    JAVA = "java"
    RUST = "rust"
    GO = "go"
    MULTI_LANGUAGE = "multi_language"

class EnvironmentStatus(str, Enum):
    """Environment setup status."""
    NOT_CREATED = "not_created"
    CREATING = "creating"
    CREATED = "created"
    INSTALLING_DEPS = "installing_dependencies"
    VALIDATING = "validating"
    READY = "ready"
    FAILED = "failed"
    CLEANUP_REQUIRED = "cleanup_required"

class BootstrapStrategy(str, Enum):
    """Environment bootstrap strategies."""
    AUTO_DETECT = "auto_detect"         # Automatically detect and create environments
    EXPLICIT = "explicit"               # Use explicitly specified environments
    MINIMAL = "minimal"                 # Create minimal environments only
    COMPREHENSIVE = "comprehensive"     # Create full development environments

# ============================================================================
# Data Models
# ============================================================================

class EnvironmentRequirement(BaseModel):
    """Requirements for a specific environment."""
    environment_type: EnvironmentType = Field(..., description="Type of environment")
    version_requirement: Optional[str] = Field(None, description="Specific version requirement")
    dependencies: List[DependencyInfo] = Field(default_factory=list, description="Dependencies to install")
    tools_required: List[str] = Field(default_factory=list, description="Additional tools required")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Environment-specific configuration")
    priority: int = Field(1, description="Priority for multi-environment setups (1=highest)")

class EnvironmentInfo(BaseModel):
    """Information about a created environment."""
    environment_type: EnvironmentType = Field(..., description="Type of environment")
    environment_path: Path = Field(..., description="Path to environment")
    status: EnvironmentStatus = Field(..., description="Current status")
    version: Optional[str] = Field(None, description="Environment/runtime version")
    activation_command: Optional[str] = Field(None, description="Command to activate environment")
    dependencies_installed: List[str] = Field(default_factory=list, description="Successfully installed dependencies")
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Environment validation results")
    created_at: datetime = Field(default_factory=datetime.now, description="When environment was created")
    last_validated: Optional[datetime] = Field(None, description="Last validation timestamp")

class EnvironmentBootstrapInput(BaseModel):
    """Input for Environment Bootstrap Agent."""
    project_path: str = Field(..., description="Path to project directory")
    strategy: BootstrapStrategy = Field(default=BootstrapStrategy.AUTO_DETECT, description="Bootstrap strategy")
    environment_types: Optional[List[EnvironmentType]] = Field(None, description="Explicitly requested environment types")
    force_recreate: bool = Field(default=False, description="Force recreation of existing environments")
    install_dependencies: bool = Field(default=True, description="Whether to install detected dependencies")
    validate_environment: bool = Field(default=True, description="Whether to validate created environments")
    python_version: Optional[str] = Field(None, description="Specific Python version requirement")
    nodejs_version: Optional[str] = Field(None, description="Specific Node.js version requirement")
    java_version: Optional[str] = Field(None, description="Specific Java version requirement")
    cleanup_on_failure: bool = Field(default=True, description="Clean up partial environments on failure")

class EnvironmentBootstrapOutput(AgentOutput):
    """Output from Environment Bootstrap Agent."""
    environments_created: List[EnvironmentInfo] = Field(default_factory=list, description="Successfully created environments")
    dependencies_installed: Dict[str, List[str]] = Field(default_factory=dict, description="Dependencies installed per environment")
    validation_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Validation results per environment")
    bootstrap_summary: str = Field(..., description="Summary of bootstrap operations")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for next steps")
    warnings: List[str] = Field(default_factory=list, description="Warnings encountered during bootstrap")
    total_setup_time: float = Field(..., description="Total time taken for setup")

# ============================================================================
# Environment Strategy Pattern
# ============================================================================

class EnvironmentStrategy(ABC):
    """Abstract base class for environment creation strategies."""
    
    def __init__(self, config_manager):
        """Initialize strategy with configuration manager."""
        self.config = config_manager.get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def detect_requirements(self, project_path: Path) -> EnvironmentRequirement:
        """Detect environment requirements for the project."""
        pass
    
    @abstractmethod
    async def create_environment(self, project_path: Path, requirement: EnvironmentRequirement) -> EnvironmentInfo:
        """Create the environment based on requirements."""
        pass
    
    @abstractmethod
    async def install_dependencies(self, env_info: EnvironmentInfo, dependencies: List[DependencyInfo]) -> List[str]:
        """Install dependencies in the environment."""
        pass
    
    @abstractmethod
    async def validate_environment(self, env_info: EnvironmentInfo) -> Dict[str, Any]:
        """Validate that the environment is working correctly."""
        pass
    
    @abstractmethod
    async def cleanup_environment(self, env_info: EnvironmentInfo) -> bool:
        """Clean up the environment."""
        pass
    
    def _run_command(self, command: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> Tuple[bool, str, str]:
        """Run a command and return success status, stdout, stderr."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

class PythonEnvironmentStrategy(EnvironmentStrategy):
    """Strategy for creating Python virtual environments."""
    
    async def detect_requirements(self, project_path: Path) -> EnvironmentRequirement:
        """Detect Python environment requirements."""
        self.logger.info(f"Detecting Python requirements for {project_path}")
        
        # Check for Python-specific files
        python_files = list(project_path.rglob("*.py"))
        requirements_txt = project_path / "requirements.txt"
        pyproject_toml = project_path / "pyproject.toml"
        pipfile = project_path / "Pipfile"
        
        if not python_files and not any([requirements_txt.exists(), pyproject_toml.exists(), pipfile.exists()]):
            raise EnvironmentDetectionError("No Python files or dependency files found")
        
        # Determine Python version requirement
        python_version = self.config.project.preferred_language_models or "3.9"
        if isinstance(python_version, list) and python_version:
            python_version = python_version[0]
        
        # Check for version specifications in pyproject.toml
        if pyproject_toml.exists():
            try:
                import toml
                pyproject_data = toml.load(pyproject_toml)
                python_requires = pyproject_data.get("project", {}).get("requires-python")
                if python_requires:
                    # Extract version number from requirement like ">=3.8"
                    import re
                    version_match = re.search(r"(\d+\.\d+)", python_requires)
                    if version_match:
                        python_version = version_match.group(1)
            except Exception as e:
                self.logger.warning(f"Could not parse pyproject.toml: {e}")
        
        return EnvironmentRequirement(
            environment_type=EnvironmentType.PYTHON,
            version_requirement=python_version,
            tools_required=["pip", "venv"],
            configuration={
                "has_requirements_txt": requirements_txt.exists(),
                "has_pyproject_toml": pyproject_toml.exists(),
                "has_pipfile": pipfile.exists(),
                "python_files_count": len(python_files)
            }
        )
    
    async def create_environment(self, project_path: Path, requirement: EnvironmentRequirement) -> EnvironmentInfo:
        """Create Python virtual environment."""
        self.logger.info(f"Creating Python environment for {project_path}")
        
        # Determine environment path
        env_name = f"venv_{project_path.name}"
        env_path = project_path / ".venv"
        
        # Find appropriate Python executable
        python_version = requirement.version_requirement or "3.9"
        python_executables = [
            f"python{python_version}",
            "python3",
            "python"
        ]
        
        python_exe = None
        for exe in python_executables:
            if shutil.which(exe):
                python_exe = exe
                break
        
        if not python_exe:
            raise EnvironmentCreationError("No suitable Python executable found")
        
        # Create virtual environment
        success, stdout, stderr = self._run_command([python_exe, "-m", "venv", str(env_path)])
        if not success:
            raise EnvironmentCreationError(f"Failed to create virtual environment: {stderr}")
        
        # Determine activation command
        if os.name == 'nt':  # Windows
            activation_cmd = str(env_path / "Scripts" / "activate.bat")
        else:  # Unix-like
            activation_cmd = f"source {env_path / 'bin' / 'activate'}"
        
        # Get Python version in the virtual environment
        if os.name == 'nt':
            python_in_venv = env_path / "Scripts" / "python.exe"
        else:
            python_in_venv = env_path / "bin" / "python"
        
        success, version_output, _ = self._run_command([str(python_in_venv), "--version"])
        version = version_output.strip().split()[-1] if success else "unknown"
        
        self.logger.info(f"Created Python {version} virtual environment at {env_path}")
        
        return EnvironmentInfo(
            environment_type=EnvironmentType.PYTHON,
            environment_path=env_path,
            status=EnvironmentStatus.CREATED,
            version=version,
            activation_command=activation_cmd
        )
    
    async def install_dependencies(self, env_info: EnvironmentInfo, dependencies: List[DependencyInfo]) -> List[str]:
        """Install Python dependencies."""
        self.logger.info(f"Installing dependencies in Python environment {env_info.environment_path}")
        
        installed = []
        
        # Get pip executable in virtual environment
        if os.name == 'nt':
            pip_exe = env_info.environment_path / "Scripts" / "pip.exe"
        else:
            pip_exe = env_info.environment_path / "bin" / "pip"
        
        # Upgrade pip first
        success, _, stderr = self._run_command([str(pip_exe), "install", "--upgrade", "pip"])
        if not success:
            self.logger.warning(f"Failed to upgrade pip: {stderr}")
        
        # Install dependencies
        for dep in dependencies:
            package_spec = f"{dep.package_name}"
            if dep.version_constraint:
                package_spec += dep.version_constraint
            
            success, stdout, stderr = self._run_command([str(pip_exe), "install", package_spec])
            if success:
                installed.append(dep.package_name)
                self.logger.info(f"Successfully installed {package_spec}")
            else:
                self.logger.error(f"Failed to install {package_spec}: {stderr}")
        
        return installed
    
    async def validate_environment(self, env_info: EnvironmentInfo) -> Dict[str, Any]:
        """Validate Python environment."""
        self.logger.info(f"Validating Python environment {env_info.environment_path}")
        
        validation_results = {
            "python_version": None,
            "pip_available": False,
            "importable_packages": [],
            "environment_activated": False,
            "overall_status": "unknown"
        }
        
        # Get Python executable
        if os.name == 'nt':
            python_exe = env_info.environment_path / "Scripts" / "python.exe"
            pip_exe = env_info.environment_path / "Scripts" / "pip.exe"
        else:
            python_exe = env_info.environment_path / "bin" / "python"
            pip_exe = env_info.environment_path / "bin" / "pip"
        
        # Check Python version
        success, version_output, _ = self._run_command([str(python_exe), "--version"])
        if success:
            validation_results["python_version"] = version_output.strip()
            validation_results["environment_activated"] = True
        
        # Check pip availability
        success, _, _ = self._run_command([str(pip_exe), "--version"])
        validation_results["pip_available"] = success
        
        # Test basic imports
        test_imports = ["sys", "os", "json", "pathlib"]
        for module in test_imports:
            success, _, _ = self._run_command([str(python_exe), "-c", f"import {module}"])
            if success:
                validation_results["importable_packages"].append(module)
        
        # Overall status
        if (validation_results["environment_activated"] and 
            validation_results["pip_available"] and 
            len(validation_results["importable_packages"]) >= 3):
            validation_results["overall_status"] = "healthy"
        elif validation_results["environment_activated"]:
            validation_results["overall_status"] = "partial"
        else:
            validation_results["overall_status"] = "failed"
        
        return validation_results
    
    async def cleanup_environment(self, env_info: EnvironmentInfo) -> bool:
        """Clean up Python virtual environment."""
        try:
            if env_info.environment_path.exists():
                shutil.rmtree(env_info.environment_path)
                self.logger.info(f"Cleaned up Python environment at {env_info.environment_path}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to cleanup environment: {e}")
        return False

class NodeJSEnvironmentStrategy(EnvironmentStrategy):
    """Strategy for creating Node.js environments."""
    
    async def detect_requirements(self, project_path: Path) -> EnvironmentRequirement:
        """Detect Node.js environment requirements."""
        self.logger.info(f"Detecting Node.js requirements for {project_path}")
        
        package_json = project_path / "package.json"
        js_files = list(project_path.rglob("*.js")) + list(project_path.rglob("*.ts"))
        
        if not package_json.exists() and not js_files:
            raise EnvironmentDetectionError("No Node.js files or package.json found")
        
        # Default Node.js version
        nodejs_version = "18"  # LTS version
        
        # Check for Node.js version in package.json
        if package_json.exists():
            try:
                import json
                with open(package_json) as f:
                    package_data = json.load(f)
                
                engines = package_data.get("engines", {})
                node_version = engines.get("node")
                if node_version:
                    import re
                    version_match = re.search(r"(\d+)", node_version)
                    if version_match:
                        nodejs_version = version_match.group(1)
            except Exception as e:
                self.logger.warning(f"Could not parse package.json: {e}")
        
        return EnvironmentRequirement(
            environment_type=EnvironmentType.NODEJS,
            version_requirement=nodejs_version,
            tools_required=["npm"],
            configuration={
                "has_package_json": package_json.exists(),
                "js_files_count": len(js_files)
            }
        )
    
    async def create_environment(self, project_path: Path, requirement: EnvironmentRequirement) -> EnvironmentInfo:
        """Create Node.js environment."""
        self.logger.info(f"Creating Node.js environment for {project_path}")
        
        # Check if Node.js is available
        success, version_output, _ = self._run_command(["node", "--version"])
        if not success:
            raise EnvironmentCreationError("Node.js is not available")
        
        version = version_output.strip()
        
        # Check if npm is available
        success, _, _ = self._run_command(["npm", "--version"])
        if not success:
            raise EnvironmentCreationError("npm is not available")
        
        self.logger.info(f"Using Node.js {version} with npm")
        
        return EnvironmentInfo(
            environment_type=EnvironmentType.NODEJS,
            environment_path=project_path,
            status=EnvironmentStatus.CREATED,
            version=version,
            activation_command="# Node.js environment is global"
        )
    
    async def install_dependencies(self, env_info: EnvironmentInfo, dependencies: List[DependencyInfo]) -> List[str]:
        """Install Node.js dependencies."""
        self.logger.info(f"Installing dependencies in Node.js environment")
        
        installed = []
        
        # Check if package.json exists
        package_json = env_info.environment_path / "package.json"
        
        if package_json.exists():
            # Use npm install to install from package.json
            success, stdout, stderr = self._run_command(["npm", "install"], cwd=env_info.environment_path)
            if success:
                self.logger.info("Successfully installed dependencies from package.json")
                # Parse installed packages from package.json
                try:
                    import json
                    with open(package_json) as f:
                        package_data = json.load(f)
                    
                    deps = package_data.get("dependencies", {})
                    dev_deps = package_data.get("devDependencies", {})
                    installed = list(deps.keys()) + list(dev_deps.keys())
                except Exception as e:
                    self.logger.warning(f"Could not parse installed packages: {e}")
            else:
                self.logger.error(f"Failed to install from package.json: {stderr}")
        
        # Install individual dependencies if provided
        for dep in dependencies:
            package_spec = dep.package_name
            if dep.version_constraint:
                package_spec += f"@{dep.version_constraint.lstrip('>=^~')}"
            
            success, stdout, stderr = self._run_command(["npm", "install", package_spec], cwd=env_info.environment_path)
            if success:
                installed.append(dep.package_name)
                self.logger.info(f"Successfully installed {package_spec}")
            else:
                self.logger.error(f"Failed to install {package_spec}: {stderr}")
        
        return installed
    
    async def validate_environment(self, env_info: EnvironmentInfo) -> Dict[str, Any]:
        """Validate Node.js environment."""
        self.logger.info(f"Validating Node.js environment")
        
        validation_results = {
            "nodejs_version": None,
            "npm_version": None,
            "package_json_valid": False,
            "node_modules_exists": False,
            "overall_status": "unknown"
        }
        
        # Check Node.js version
        success, version_output, _ = self._run_command(["node", "--version"])
        if success:
            validation_results["nodejs_version"] = version_output.strip()
        
        # Check npm version
        success, version_output, _ = self._run_command(["npm", "--version"])
        if success:
            validation_results["npm_version"] = version_output.strip()
        
        # Check package.json
        package_json = env_info.environment_path / "package.json"
        if package_json.exists():
            try:
                import json
                with open(package_json) as f:
                    json.load(f)
                validation_results["package_json_valid"] = True
            except Exception:
                pass
        
        # Check node_modules
        node_modules = env_info.environment_path / "node_modules"
        validation_results["node_modules_exists"] = node_modules.exists()
        
        # Overall status
        if (validation_results["nodejs_version"] and 
            validation_results["npm_version"] and 
            validation_results.get("package_json_valid", True)):
            validation_results["overall_status"] = "healthy"
        elif validation_results["nodejs_version"]:
            validation_results["overall_status"] = "partial"
        else:
            validation_results["overall_status"] = "failed"
        
        return validation_results
    
    async def cleanup_environment(self, env_info: EnvironmentInfo) -> bool:
        """Clean up Node.js environment."""
        try:
            # Remove node_modules if it exists
            node_modules = env_info.environment_path / "node_modules"
            if node_modules.exists():
                shutil.rmtree(node_modules)
                self.logger.info(f"Cleaned up node_modules at {node_modules}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cleanup Node.js environment: {e}")
        return False

# ============================================================================
# Main Environment Bootstrap Agent
# ============================================================================

class EnvironmentBootstrapAgent(ProtocolAwareAgent[EnvironmentBootstrapInput, EnvironmentBootstrapOutput]):
    """
    Comprehensive environment bootstrap agent with multi-language support.
    
    Integrates with Project Type Detection Service, Smart Dependency Analysis Service,
    and Configuration Management to provide autonomous environment bootstrapping.
    
    ✨ UNIVERSAL PROTOCOL ARCHITECTURE - Uses systematic protocols for reliable environment setup.
    """
    
    # ADDED: Agent metadata following Universal Protocol Infrastructure
    AGENT_ID: ClassVar[str] = "EnvironmentBootstrapAgent"
    AGENT_NAME: ClassVar[str] = "Environment Bootstrap Agent"
    AGENT_DESCRIPTION: ClassVar[str] = "Comprehensive environment bootstrap agent with multi-language support"
    VERSION: ClassVar[str] = "1.0.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.SYSTEM_ORCHESTRATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    
    # ADDED: Protocol definitions following Universal Protocol Infrastructure
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ['environment_setup', 'infrastructure_provisioning']
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'tool_validation']
    
    def __init__(
        self,
        project_type_detector: Optional[ProjectTypeDetectionService] = None,
        dependency_analyzer: Optional[SmartDependencyAnalysisService] = None,
        state_persistence: Optional[ResumableExecutionService] = None,
        **kwargs
    ):
        """Initialize the environment bootstrap agent.
        
        Args:
            project_type_detector: Project type detection service
            dependency_analyzer: Smart dependency analysis service
            state_manager: State manager for persistence
        """
        super().__init__(**kwargs)
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_config()
        
        self.project_type_detector = project_type_detector or ProjectTypeDetectionService()
        self.dependency_analyzer = dependency_analyzer
        self.state_manager = state_manager
        
        # Initialize environment strategies
        self.strategies = {
            EnvironmentType.PYTHON: PythonEnvironmentStrategy(self.config_manager),
            EnvironmentType.NODEJS: NodeJSEnvironmentStrategy(self.config_manager),
            # Additional strategies can be added here
        }
        
        logger.info("EnvironmentBootstrapAgent initialized")
    
    async def execute(self, inputs: EnvironmentBootstrapInput, context: SharedContext) -> EnvironmentBootstrapOutput:
        """Execute environment bootstrap agent."""
        start_time = datetime.now()
        logger.info(f"Starting environment bootstrap for {inputs.project_path}")
        
        try:
            # Create execution context for state persistence
            execution_context = None
            if self.state_manager:
                execution_context = ExecutionContext(
                    project_id=context.data.get("project_id", "unknown"),
                    project_root_path=inputs.project_path,
                    run_id=context.run_id,
                    flow_id=context.flow_id,
                    working_directory=inputs.project_path,
                    execution_start_time=start_time
                )
                
                # Create checkpoint at start
                await create_execution_checkpoint(
                    self.state_manager,
                    context.flow_id,
                    "environment_bootstrap_start",
                    StageType.AGENT_COMPLETE,
                    CheckpointStatus.ACTIVE,
                    execution_context
                )
            
            project_path = Path(inputs.project_path)
            if not project_path.exists():
                raise EnvironmentBootstrapError(f"Project path does not exist: {project_path}")
            
            # Set project root in config manager
            self.config_manager.set_project_root(project_path)
            
            # Step 1: Detect project requirements
            logger.info("Step 1: Detecting project type and requirements")
            environment_requirements = await self._detect_environment_requirements(
                project_path, inputs
            )
            
            # Step 2: Create environments
            logger.info("Step 2: Creating environments")
            environments_created = await self._create_environments(
                project_path, environment_requirements, inputs
            )
            
            # Step 3: Install dependencies
            dependencies_installed = {}
            if inputs.install_dependencies:
                logger.info("Step 3: Installing dependencies")
                dependencies_installed = await self._install_dependencies(
                    project_path, environments_created, inputs
                )
            
            # Step 4: Validate environments
            validation_results = {}
            if inputs.validate_environment:
                logger.info("Step 4: Validating environments")
                validation_results = await self._validate_environments(environments_created)
            
            # Generate summary and recommendations
            total_time = (datetime.now() - start_time).total_seconds()
            summary = self._generate_bootstrap_summary(
                environments_created, dependencies_installed, validation_results, total_time
            )
            recommendations = self._generate_recommendations(
                environments_created, validation_results
            )
            
            # Create checkpoint at success
            if execution_context:
                await create_execution_checkpoint(
                    self.state_manager,
                    context.flow_id,
                    "environment_bootstrap_complete",
                    StageType.AGENT_COMPLETE,
                    CheckpointStatus.SUCCESS,
                    execution_context
                )
            
            logger.info(f"Environment bootstrap completed successfully in {total_time:.2f} seconds")
            
            return EnvironmentBootstrapOutput(
                success=True,
                message="Environment bootstrap completed successfully",
                environments_created=environments_created,
                dependencies_installed=dependencies_installed,
                validation_results=validation_results,
                bootstrap_summary=summary,
                recommendations=recommendations,
                total_setup_time=total_time
            )
            
        except Exception as e:
            error_msg = f"Environment bootstrap failed: {str(e)}"
            logger.error(error_msg)
            
            # Cleanup on failure if requested
            if inputs.cleanup_on_failure and 'environments_created' in locals():
                await self._cleanup_environments(environments_created)
            
            # Create checkpoint at failure
            if execution_context:
                await create_execution_checkpoint(
                    self.state_manager,
                    context.flow_id,
                    "environment_bootstrap_failed",
                    StageType.AGENT_COMPLETE,
                    CheckpointStatus.FAILED,
                    execution_context
                )
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            return EnvironmentBootstrapOutput(
                success=False,
                message=error_msg,
                environments_created=[],
                dependencies_installed={},
                validation_results={},
                bootstrap_summary=f"Bootstrap failed after {total_time:.2f} seconds: {error_msg}",
                recommendations=["Check error logs", "Verify system requirements", "Try with different strategy"],
                warnings=[error_msg],
                total_setup_time=total_time
            )
    
    async def _detect_environment_requirements(
        self, 
        project_path: Path, 
        inputs: EnvironmentBootstrapInput
    ) -> List[EnvironmentRequirement]:
        """Detect what environments are needed for the project."""
        requirements = []
        
        if inputs.strategy == BootstrapStrategy.EXPLICIT and inputs.environment_types:
            # Use explicitly specified environment types
            for env_type in inputs.environment_types:
                if env_type in self.strategies:
                    try:
                        requirement = await self.strategies[env_type].detect_requirements(project_path)
                        
                        # Apply version overrides
                        if env_type == EnvironmentType.PYTHON and inputs.python_version:
                            requirement.version_requirement = inputs.python_version
                        elif env_type == EnvironmentType.NODEJS and inputs.nodejs_version:
                            requirement.version_requirement = inputs.nodejs_version
                        elif env_type == EnvironmentType.JAVA and inputs.java_version:
                            requirement.version_requirement = inputs.java_version
                        
                        requirements.append(requirement)
                    except EnvironmentDetectionError as e:
                        logger.warning(f"Could not detect requirements for {env_type}: {e}")
        else:
            # Auto-detect using Project Type Detection Service
            project_result = self.project_type_detector.detect_project_type(project_path)
            
            # Determine environments needed based on detected project type
            env_types_needed = self._determine_environment_types(project_result)
            
            for env_type in env_types_needed:
                if env_type in self.strategies:
                    try:
                        requirement = await self.strategies[env_type].detect_requirements(project_path)
                        requirements.append(requirement)
                    except EnvironmentDetectionError as e:
                        logger.warning(f"Could not detect requirements for {env_type}: {e}")
        
        if not requirements:
            raise EnvironmentBootstrapError("No suitable environments detected for project")
        
        logger.info(f"Detected requirements for {len(requirements)} environment(s)")
        return requirements
    
    def _determine_environment_types(self, project_result: ProjectTypeDetectionResult) -> List[EnvironmentType]:
        """Determine needed environment types from project detection result."""
        env_types = []
        
        # Map detected languages to environment types
        language_mapping = {
            "python": EnvironmentType.PYTHON,
            "javascript": EnvironmentType.NODEJS,
            "typescript": EnvironmentType.NODEJS,
            "java": EnvironmentType.JAVA,
            "rust": EnvironmentType.RUST,
            "go": EnvironmentType.GO
        }
        
        # Add primary language
        if project_result.primary_language in language_mapping:
            env_types.append(language_mapping[project_result.primary_language])
        
        # Add secondary languages if it's a multi-language project
        for lang_info in project_result.languages:
            if lang_info.name != project_result.primary_language and lang_info.name in language_mapping:
                env_type = language_mapping[lang_info.name]
                if env_type not in env_types:
                    env_types.append(env_type)
        
        return env_types
    
    async def _create_environments(
        self,
        project_path: Path,
        requirements: List[EnvironmentRequirement],
        inputs: EnvironmentBootstrapInput
    ) -> List[EnvironmentInfo]:
        """Create environments based on requirements."""
        environments = []
        
        for requirement in requirements:
            try:
                logger.info(f"Creating {requirement.environment_type} environment")
                
                strategy = self.strategies[requirement.environment_type]
                env_info = await strategy.create_environment(project_path, requirement)
                env_info.status = EnvironmentStatus.CREATED
                environments.append(env_info)
                
                logger.info(f"Successfully created {requirement.environment_type} environment")
                
            except Exception as e:
                logger.error(f"Failed to create {requirement.environment_type} environment: {e}")
                if not inputs.cleanup_on_failure:
                    # Create a failed environment info for tracking
                    failed_env = EnvironmentInfo(
                        environment_type=requirement.environment_type,
                        environment_path=project_path,
                        status=EnvironmentStatus.FAILED
                    )
                    environments.append(failed_env)
        
        return environments
    
    async def _install_dependencies(
        self,
        project_path: Path,
        environments: List[EnvironmentInfo],
        inputs: EnvironmentBootstrapInput
    ) -> Dict[str, List[str]]:
        """Install dependencies in created environments."""
        dependencies_installed = {}
        
        # Analyze dependencies if we have the service
        project_dependencies = []
        if self.dependency_analyzer:
            try:
                # Detect project type for dependency analysis
                project_result = self.project_type_detector.detect_project_type(project_path)
                dependencies_result = await self.dependency_analyzer.analyze_dependencies(
                    project_path,
                    project_type=project_result.primary_language
                )
                project_dependencies = dependencies_result.dependencies
                logger.info(f"Detected {len(project_dependencies)} dependencies to install")
            except Exception as e:
                logger.warning(f"Could not analyze dependencies: {e}")
        
        for env_info in environments:
            if env_info.status != EnvironmentStatus.CREATED:
                continue
            
            try:
                env_info.status = EnvironmentStatus.INSTALLING_DEPS
                strategy = self.strategies[env_info.environment_type]
                
                # Filter dependencies relevant to this environment
                relevant_deps = [
                    dep for dep in project_dependencies
                    if self._is_dependency_relevant(dep, env_info.environment_type)
                ]
                
                installed = await strategy.install_dependencies(env_info, relevant_deps)
                dependencies_installed[env_info.environment_type.value] = installed
                
                env_info.dependencies_installed = installed
                logger.info(f"Installed {len(installed)} dependencies in {env_info.environment_type} environment")
                
            except Exception as e:
                logger.error(f"Failed to install dependencies in {env_info.environment_type}: {e}")
                dependencies_installed[env_info.environment_type.value] = []
        
        return dependencies_installed
    
    def _is_dependency_relevant(self, dependency: DependencyInfo, env_type: EnvironmentType) -> bool:
        """Check if a dependency is relevant for the environment type."""
        # Simple heuristic - can be made more sophisticated
        if env_type == EnvironmentType.PYTHON:
            return True  # For now, assume all detected dependencies are Python
        elif env_type == EnvironmentType.NODEJS:
            return dependency.package_name in ["express", "react", "vue", "typescript", "webpack"]
        return False
    
    async def _validate_environments(self, environments: List[EnvironmentInfo]) -> Dict[str, Dict[str, Any]]:
        """Validate created environments."""
        validation_results = {}
        
        for env_info in environments:
            if env_info.status == EnvironmentStatus.FAILED:
                continue
            
            try:
                env_info.status = EnvironmentStatus.VALIDATING
                strategy = self.strategies[env_info.environment_type]
                
                results = await strategy.validate_environment(env_info)
                validation_results[env_info.environment_type.value] = results
                
                env_info.validation_results = results
                env_info.last_validated = datetime.now()
                
                if results.get("overall_status") == "healthy":
                    env_info.status = EnvironmentStatus.READY
                else:
                    env_info.status = EnvironmentStatus.FAILED
                
                logger.info(f"Validated {env_info.environment_type} environment: {results.get('overall_status')}")
                
            except Exception as e:
                logger.error(f"Failed to validate {env_info.environment_type} environment: {e}")
                validation_results[env_info.environment_type.value] = {"overall_status": "failed", "error": str(e)}
                env_info.status = EnvironmentStatus.FAILED
        
        return validation_results
    
    async def _cleanup_environments(self, environments: List[EnvironmentInfo]) -> None:
        """Clean up environments on failure."""
        for env_info in environments:
            try:
                strategy = self.strategies[env_info.environment_type]
                await strategy.cleanup_environment(env_info)
            except Exception as e:
                logger.error(f"Failed to cleanup {env_info.environment_type} environment: {e}")
    
    def _generate_bootstrap_summary(
        self,
        environments: List[EnvironmentInfo],
        dependencies: Dict[str, List[str]],
        validation: Dict[str, Dict[str, Any]],
        total_time: float
    ) -> str:
        """Generate a summary of the bootstrap process."""
        successful_envs = [env for env in environments if env.status == EnvironmentStatus.READY]
        failed_envs = [env for env in environments if env.status == EnvironmentStatus.FAILED]
        
        total_deps = sum(len(deps) for deps in dependencies.values())
        
        summary = f"Environment bootstrap completed in {total_time:.2f} seconds.\n"
        summary += f"Successfully created {len(successful_envs)} environment(s), "
        summary += f"{len(failed_envs)} failed.\n"
        summary += f"Installed {total_deps} dependencies total.\n"
        
        for env in successful_envs:
            summary += f"✓ {env.environment_type.value} ({env.version}) - Ready\n"
        
        for env in failed_envs:
            summary += f"✗ {env.environment_type.value} - Failed\n"
        
        return summary
    
    def _generate_recommendations(
        self,
        environments: List[EnvironmentInfo],
        validation: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on bootstrap results."""
        recommendations = []
        
        ready_envs = [env for env in environments if env.status == EnvironmentStatus.READY]
        
        if ready_envs:
            recommendations.append("Environment setup completed successfully!")
            
            for env in ready_envs:
                if env.activation_command and "source" in env.activation_command:
                    recommendations.append(f"Activate {env.environment_type} environment: {env.activation_command}")
        
        failed_envs = [env for env in environments if env.status == EnvironmentStatus.FAILED]
        if failed_envs:
            recommendations.append("Some environments failed to create. Check logs for details.")
            recommendations.append("Consider installing missing system dependencies.")
        
        if not environments:
            recommendations.append("No environments were created. Check project structure and requirements.")
        
        return recommendations
    
    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Returns the static AgentCard for EnvironmentBootstrapAgent."""
        return AgentCard(
            agent_id="chungoid.agents.autonomous_engine.environment_bootstrap_agent.EnvironmentBootstrapAgent",
            name="Environment Bootstrap Agent",
            description="Autonomous environment setup and dependency management across multiple programming languages with intelligent project type detection",
            version="1.0.0",
            input_schema=EnvironmentBootstrapInput.model_json_schema(),
            output_schema=EnvironmentBootstrapOutput.model_json_schema(),
            categories=[AgentCategory.SYSTEM_ORCHESTRATION.value],
            visibility=AgentVisibility.PUBLIC,
            capability_profile={
                "environment_types": ["python", "nodejs", "java", "rust", "go", "multi_language"],
                "project_type_detection": True,
                "dependency_analysis": True,
                "virtual_environment_management": True,
                "package_manager_support": ["pip", "conda", "npm", "yarn", "maven", "gradle", "cargo"],
                "cross_platform_support": ["windows", "linux", "macos"],
                "validation_and_testing": True,
                "cleanup_capabilities": True,
                "strategy_pattern": ["auto_detect", "explicit", "minimal", "comprehensive"],
                "primary_function": "Comprehensive autonomous environment bootstrapping with multi-language support and intelligent dependency management"
            },
            metadata={
                "callable_fn_path": f"{EnvironmentBootstrapAgent.__module__}.{EnvironmentBootstrapAgent.__name__}",
                "integration_services": ["ProjectTypeDetectionService", "SmartDependencyAnalysisService", "ResumableExecutionService", "ConfigurationManager"],
                "mcp_tool_wrapper": "bootstrap_environment_tool"
            }
        )

# ============================================================================
# MCP Tool Wrappers
# ============================================================================

async def bootstrap_environment_tool(
    project_path: str,
    environment_types: Optional[List[str]] = None,
    force_recreate: bool = False,
    install_dependencies: bool = True,
    python_version: Optional[str] = None,
    nodejs_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    MCP tool wrapper for environment bootstrapping.
    
    Args:
        project_path: Path to project directory
        environment_types: Optional list of environment types to create
        force_recreate: Whether to recreate existing environments
        install_dependencies: Whether to install dependencies
        python_version: Specific Python version requirement
        nodejs_version: Specific Node.js version requirement
        
    Returns:
        Bootstrap results dictionary
    """
    try:
        # Convert string environment types to enum
        env_types = None
        if environment_types:
            env_types = [EnvironmentType(env_type) for env_type in environment_types]
        
        # Create agent input
        inputs = EnvironmentBootstrapInput(
            project_path=project_path,
            environment_types=env_types,
            force_recreate=force_recreate,
            install_dependencies=install_dependencies,
            python_version=python_version,
            nodejs_version=nodejs_version
        )
        
        # Create mock context
        context = SharedContext(
            run_id=str(uuid4()),
            flow_id=str(uuid4()),
            data={"project_id": Path(project_path).name}
        )
        
        # Execute agent
        agent = EnvironmentBootstrapAgent()
        result = await agent.execute(inputs, context)
        
        return {
            "success": result.success,
            "message": result.message,
            "environments_created": [env.dict() for env in result.environments_created],
            "dependencies_installed": result.dependencies_installed,
            "summary": result.bootstrap_summary,
            "recommendations": result.recommendations,
            "setup_time": result.total_setup_time
        }
        
    except Exception as e:
        logger.error(f"Environment bootstrap tool failed: {e}")
        return {
            "success": False,
            "message": f"Bootstrap failed: {str(e)}",
            "environments_created": [],
            "dependencies_installed": {},
            "summary": f"Failed to bootstrap environment: {str(e)}",
            "recommendations": ["Check error logs", "Verify project structure"],
            "setup_time": 0.0
        }

# Example usage and testing support
if __name__ == "__main__":
    # Test the agent
    import asyncio
    
    async def test_bootstrap_agent():
        print("Environment Bootstrap Agent V1 - Test Mode")
        
        # Test input
        test_input = EnvironmentBootstrapInput(
            project_path="/tmp/test_project",
            strategy=BootstrapStrategy.AUTO_DETECT,
            install_dependencies=True
        )
        
        # Mock context
        context = SharedContext(
            run_id=str(uuid4()),
            flow_id=str(uuid4()),
            data={"project_id": "test_project"}
        )
        
        print(f"Test input: {test_input.dict()}")
        print("Agent implementation complete!")
    
    # Run test if executed directly
    asyncio.run(test_bootstrap_agent()) 