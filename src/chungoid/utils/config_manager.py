"""Configuration Management System

This module provides comprehensive configuration management capabilities for the autonomous
agentic coding system, implementing hierarchical configuration sources with secure secret handling.

Key Features:
- Hierarchical configuration precedence (Env Vars → Project → Global → Defaults)
- Secure secret handling via environment variables with masking
- YAML-based configuration files with Pydantic validation
- Runtime configuration updates with caching
- Type-safe configuration models
- Global singleton pattern for system-wide access

Design Principles:
- Security first - secrets only in environment variables
- Type safety through Pydantic models
- Performance optimization with intelligent caching
- Clear precedence hierarchy for predictable behavior
- Runtime reconfiguration without restarts

Configuration Hierarchy (highest to lowest precedence):
1. Environment Variables (CHUNGOID_* prefix) - for secrets and overrides
2. Project Configuration (~/.chungoid/config.yaml) - project-specific settings
3. Global Configuration ({project}/.chungoid/config.yaml) - user-wide defaults
4. Hardcoded Defaults - fallback values

Architecture:
- ConfigurationManager: Main singleton service with caching
- Configuration Models: Type-safe Pydantic models for all settings
- EnvironmentMapper: Secure environment variable handling
- ConfigurationLoader: File loading and parsing with validation
"""

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
import yaml

from pydantic import BaseModel, Field, SecretStr, validator
from pydantic.config import ConfigDict

from .exceptions import ChungoidError

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Exceptions
# ============================================================================

class ConfigurationError(ChungoidError):
    """Base exception for configuration management operations."""
    pass

class ConfigurationValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass

class ConfigurationFileError(ConfigurationError):
    """Raised when configuration file operations fail."""
    pass

class SecretAccessError(ConfigurationError):
    """Raised when secret access fails or is unauthorized."""
    pass

# ============================================================================
# Configuration Models
# ============================================================================

class LLMConfiguration(BaseModel):
    """Configuration for LLM providers and settings."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Provider settings
    provider: str = Field(default="openai", description="LLM provider (openai, anthropic, etc.)")
    api_key: Optional[SecretStr] = Field(None, description="API key (from environment only)")
    api_base_url: Optional[str] = Field(None, description="Custom API base URL")
    
    # Model preferences
    default_model: str = Field(default="gpt-4o-mini-2024-07-18", description="Default model to use")
    fallback_model: str = Field(default="gpt-3.5-turbo", description="Fallback model if default fails")
    
    # Performance settings
    timeout: int = Field(default=60, ge=1, le=600, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.1, le=30.0, description="Delay between retries")
    rate_limit_rpm: int = Field(default=60, ge=1, description="Rate limit requests per minute")
    
    # Cost management
    max_tokens_per_request: int = Field(default=4000, ge=1, description="Maximum tokens per request")
    enable_cost_tracking: bool = Field(default=True, description="Track API usage costs")
    monthly_budget_limit: Optional[float] = Field(None, ge=0, description="Monthly budget limit in USD")

class ChromaDBConfiguration(BaseModel):
    """Configuration for ChromaDB connections and settings."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Connection settings
    host: str = Field(default="localhost", description="ChromaDB host")
    port: int = Field(default=8000, ge=1, le=65535, description="ChromaDB port")
    auth_token: Optional[SecretStr] = Field(None, description="Authentication token (from environment only)")
    use_ssl: bool = Field(default=False, description="Use SSL connection")
    
    # Database settings
    default_collection_prefix: str = Field(default="chungoid", description="Default collection name prefix")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    
    # Performance settings
    connection_timeout: int = Field(default=30, ge=1, le=300, description="Connection timeout in seconds")
    query_timeout: int = Field(default=60, ge=1, le=600, description="Query timeout in seconds")
    max_batch_size: int = Field(default=100, ge=1, le=1000, description="Maximum batch size for operations")
    
    # Data management
    auto_cleanup_enabled: bool = Field(default=True, description="Enable automatic cleanup of old data")
    cleanup_retention_days: int = Field(default=30, ge=1, description="Retention period for cleanup")

class ProjectConfiguration(BaseModel):
    """Configuration for project-specific settings."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Project identification
    name: Optional[str] = Field(None, description="Project name")
    project_type: Optional[str] = Field(None, description="Project type (python, javascript, etc.)")
    description: Optional[str] = Field(None, description="Project description")
    
    # Project behavior
    auto_detect_dependencies: bool = Field(default=True, description="Automatically detect dependencies")
    auto_detect_project_type: bool = Field(default=True, description="Automatically detect project type")
    enable_smart_caching: bool = Field(default=True, description="Enable intelligent caching")
    
    # File and directory settings
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["__pycache__", "node_modules", ".git", "*.pyc", "*.log"],
        description="File patterns to exclude from analysis"
    )
    include_hidden_files: bool = Field(default=False, description="Include hidden files in analysis")
    max_file_size_mb: int = Field(default=10, ge=1, le=100, description="Maximum file size to analyze (MB)")
    
    # Analysis preferences
    analysis_depth: int = Field(default=5, ge=1, le=20, description="Maximum directory depth for analysis")
    enable_content_analysis: bool = Field(default=True, description="Enable file content analysis")
    preferred_language_models: List[str] = Field(
        default_factory=list,
        description="Preferred language models for this project"
    )

class AgentConfiguration(BaseModel):
    """Configuration for agent behavior and settings."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Execution settings
    default_timeout: int = Field(default=300, ge=30, le=3600, description="Default agent timeout in seconds")
    max_concurrent_agents: int = Field(default=5, ge=1, le=20, description="Maximum concurrent agents")
    enable_parallel_execution: bool = Field(default=True, description="Enable parallel agent execution")
    
    # Retry and resilience
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts for failed agents")
    retry_exponential_backoff: bool = Field(default=True, description="Use exponential backoff for retries")
    base_retry_delay: float = Field(default=2.0, ge=0.1, le=60.0, description="Base retry delay in seconds")
    
    # Checkpoint and state management
    enable_automatic_checkpoints: bool = Field(default=True, description="Enable automatic checkpoint creation")
    checkpoint_frequency: int = Field(default=5, ge=1, le=50, description="Checkpoint frequency (every N stages)")
    enable_state_compression: bool = Field(default=True, description="Compress large execution states")
    
    # Monitoring and observability
    enable_performance_monitoring: bool = Field(default=True, description="Enable agent performance monitoring")
    log_agent_outputs: bool = Field(default=True, description="Log detailed agent outputs")
    enable_health_checks: bool = Field(default=True, description="Enable agent health monitoring")
    
    # Agent-specific overrides
    agent_timeouts: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-agent timeout overrides"
    )
    agent_retry_limits: Dict[str, int] = Field(
        default_factory=dict,
        description="Per-agent retry limit overrides"
    )

class LoggingConfiguration(BaseModel):
    """Configuration for logging and observability."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Basic logging settings
    level: str = Field(default="INFO", description="Default log level")
    enable_file_logging: bool = Field(default=True, description="Enable logging to files")
    log_directory: str = Field(default="logs", description="Log file directory")
    
    # Log formatting
    enable_structured_logging: bool = Field(default=True, description="Use structured JSON logging")
    include_timestamps: bool = Field(default=True, description="Include timestamps in logs")
    include_caller_info: bool = Field(default=False, description="Include caller information")
    
    # Log rotation and retention
    max_log_size_mb: int = Field(default=100, ge=1, le=1000, description="Maximum log file size (MB)")
    log_retention_days: int = Field(default=7, ge=1, le=365, description="Log retention period")
    compress_old_logs: bool = Field(default=True, description="Compress rotated log files")
    
    # Security and privacy
    mask_secrets: bool = Field(default=True, description="Mask secrets in log output")
    log_api_requests: bool = Field(default=False, description="Log API request details")
    enable_debug_logging: bool = Field(default=False, description="Enable debug-level logging")

class SystemConfiguration(BaseModel):
    """Master configuration containing all system settings."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )
    
    # Configuration metadata
    config_version: str = Field(default="1.0.0", description="Configuration schema version")
    last_updated: Optional[str] = Field(None, description="Last configuration update timestamp")
    
    # Sub-configurations
    llm: LLMConfiguration = Field(default_factory=LLMConfiguration)
    chromadb: ChromaDBConfiguration = Field(default_factory=ChromaDBConfiguration)
    project: ProjectConfiguration = Field(default_factory=ProjectConfiguration)
    agents: AgentConfiguration = Field(default_factory=AgentConfiguration)
    logging: LoggingConfiguration = Field(default_factory=LoggingConfiguration)
    
    # System-wide settings
    environment: str = Field(default="development", description="Runtime environment")
    enable_telemetry: bool = Field(default=False, description="Enable anonymous telemetry")
    data_directory: str = Field(default="~/.chungoid", description="Data directory path")

# ============================================================================
# Environment Variable Mapping
# ============================================================================

class EnvironmentVariableMapper:
    """Maps environment variables to configuration fields with secure handling."""
    
    # Environment variable mappings
    ENV_MAPPINGS = {
        # LLM Configuration
        "CHUNGOID_LLM_PROVIDER": "llm.provider",
        "CHUNGOID_LLM_API_KEY": "llm.api_key",
        "CHUNGOID_LLM_API_BASE_URL": "llm.api_base_url",
        "CHUNGOID_LLM_DEFAULT_MODEL": "llm.default_model",
        "CHUNGOID_LLM_TIMEOUT": "llm.timeout",
        "CHUNGOID_LLM_MAX_RETRIES": "llm.max_retries",
        
        # Anthropic specific
        "CHUNGOID_ANTHROPIC_API_KEY": "llm.api_key",
        "ANTHROPIC_API_KEY": "llm.api_key",  # Common Anthropic env var
        
        # OpenAI specific
        "CHUNGOID_OPENAI_API_KEY": "llm.api_key", 
        "OPENAI_API_KEY": "llm.api_key",  # Common OpenAI env var
        
        # ChromaDB Configuration
        "CHUNGOID_CHROMADB_HOST": "chromadb.host",
        "CHUNGOID_CHROMADB_PORT": "chromadb.port",
        "CHUNGOID_CHROMADB_AUTH_TOKEN": "chromadb.auth_token",
        "CHUNGOID_CHROMADB_USE_SSL": "chromadb.use_ssl",
        
        # Agent Configuration
        "CHUNGOID_AGENT_TIMEOUT": "agents.default_timeout",
        "CHUNGOID_AGENT_MAX_CONCURRENT": "agents.max_concurrent_agents",
        "CHUNGOID_AGENT_MAX_RETRIES": "agents.max_retries",
        
        # System Configuration
        "CHUNGOID_ENVIRONMENT": "environment",
        "CHUNGOID_DATA_DIRECTORY": "data_directory",
        "CHUNGOID_LOG_LEVEL": "logging.level",
        "CHUNGOID_ENABLE_DEBUG": "logging.enable_debug_logging",
    }
    
    # Secret fields that should be masked in logs
    SECRET_FIELDS = {
        "llm.api_key",
        "chromadb.auth_token",
        "CHUNGOID_LLM_API_KEY",
        "CHUNGOID_ANTHROPIC_API_KEY", 
        "CHUNGOID_OPENAI_API_KEY",
        "CHUNGOID_CHROMADB_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY"
    }
    
    @classmethod
    def load_from_environment(cls) -> Dict[str, Any]:
        """Load configuration values from environment variables."""
        env_config = {}
        
        for env_var, config_path in cls.ENV_MAPPINGS.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion
                converted_value = cls._convert_env_value(value, config_path)
                cls._set_nested_value(env_config, config_path, converted_value)
                
                # Log (with masking for secrets)
                if cls._is_secret_field(env_var) or cls._is_secret_field(config_path):
                    logger.debug(f"Loaded secret environment variable: {env_var} -> {config_path}")
                else:
                    logger.debug(f"Loaded environment variable: {env_var}={value} -> {config_path}")
        
        return env_config
    
    @classmethod
    def _convert_env_value(cls, value: str, config_path: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Boolean conversion
        if config_path.endswith(('.enable_', '.use_', '.auto_', '.compress_', '.mask_')):
            return value.lower() in ('true', '1', 'yes', 'on')
        
        # Integer conversion
        if config_path.endswith(('.timeout', '.port', '.retries', '.max_', '.days', '.size_mb')):
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Invalid integer value for {config_path}: {value}")
                return value
        
        # Float conversion  
        if config_path.endswith(('.delay', '.limit')):
            try:
                return float(value)
            except ValueError:
                logger.warning(f"Invalid float value for {config_path}: {value}")
                return value
        
        # Secret string conversion
        if cls._is_secret_field(config_path):
            from pydantic import SecretStr
            return SecretStr(value)
        
        return value
    
    @classmethod
    def _set_nested_value(cls, config_dict: Dict[str, Any], path: str, value: Any) -> None:
        """Set a value in a nested dictionary using dot notation."""
        keys = path.split('.')
        current = config_dict
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @classmethod
    def _is_secret_field(cls, field_path: str) -> bool:
        """Check if a field path represents a secret."""
        return field_path in cls.SECRET_FIELDS or 'api_key' in field_path.lower() or 'token' in field_path.lower()

# ============================================================================
# Configuration Loading and Management
# ============================================================================

class ConfigurationLoader:
    """Handles loading and parsing of configuration files."""
    
    @staticmethod
    def load_yaml_file(file_path: Path) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        try:
            if not file_path.exists():
                logger.debug(f"Configuration file not found: {file_path}")
                return {}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f) or {}
            
            logger.debug(f"Loaded configuration from: {file_path}")
            return content
            
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {file_path}: {e}")
            raise ConfigurationFileError(f"Invalid YAML syntax in {file_path}: {e}") from e
        except Exception as e:
            logger.error(f"Error loading configuration file {file_path}: {e}")
            raise ConfigurationFileError(f"Failed to load {file_path}: {e}") from e
    
    @staticmethod
    def save_yaml_file(file_path: Path, config_data: Dict[str, Any]) -> None:
        """Save configuration to a YAML file."""
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
            
            logger.info(f"Saved configuration to: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration file {file_path}: {e}")
            raise ConfigurationFileError(f"Failed to save {file_path}: {e}") from e
    
    @staticmethod
    def merge_configurations(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries with deep merging."""
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        merged = {}
        for config in configs:
            merged = deep_merge(merged, config)
        
        return merged

# ============================================================================
# Main Configuration Manager
# ============================================================================

class ConfigurationManager:
    """
    Singleton configuration manager providing hierarchical configuration with caching.
    
    Implements the following precedence (highest to lowest):
    1. Environment Variables (secrets and overrides)
    2. Project Configuration (.chungoid/config.yaml in project root)
    3. Global Configuration (~/.chungoid/config.yaml)
    4. Default Values (hardcoded in Pydantic models)
    """
    
    _instance: Optional['ConfigurationManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ConfigurationManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager (only once due to singleton)."""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._config: Optional[SystemConfiguration] = None
        self._config_cache_time: float = 0
        self._cache_ttl: float = 300  # 5 minutes cache TTL
        self._project_root: Optional[Path] = None
        
        logger.info("ConfigurationManager initialized")
    
    def get_config(self, force_reload: bool = False) -> SystemConfiguration:
        """
        Get the current system configuration with intelligent caching.
        
        Args:
            force_reload: Force reload from all sources, ignoring cache
            
        Returns:
            Complete system configuration
        """
        current_time = time.time()
        
        # Check if we need to reload
        if (force_reload or 
            self._config is None or 
            (current_time - self._config_cache_time) > self._cache_ttl):
            
            self._load_configuration()
            self._config_cache_time = current_time
        
        return self._config
    
    def reload_configuration(self) -> SystemConfiguration:
        """Force reload configuration from all sources."""
        return self.get_config(force_reload=True)
    
    def set_project_root(self, project_root: Path) -> None:
        """Set the project root path for project-specific configuration."""
        self._project_root = project_root
        logger.debug(f"Set project root to: {project_root}")
        # Force reload to pick up project-specific config
        self.reload_configuration()
    
    def get_secret(self, secret_path: str) -> Optional[str]:
        """
        Securely retrieve a secret value with automatic masking.
        
        Args:
            secret_path: Dot notation path to secret (e.g., 'llm.api_key')
            
        Returns:
            Secret value or None if not found
        """
        config = self.get_config()
        
        try:
            # Navigate to the secret using dot notation
            value = config
            for part in secret_path.split('.'):
                value = getattr(value, part)
            
            # Handle SecretStr objects
            if hasattr(value, 'get_secret_value'):
                return value.get_secret_value()
            
            return str(value) if value is not None else None
            
        except (AttributeError, KeyError):
            logger.warning(f"Secret not found: {secret_path}")
            return None
    
    def update_configuration(self, updates: Dict[str, Any], persist: bool = False) -> None:
        """
        Update configuration at runtime.
        
        Args:
            updates: Configuration updates in nested dictionary format
            persist: Whether to persist changes to project config file
        """
        try:
            # Apply updates to current config
            if self._config is None:
                self.get_config()  # Load initial config
            
            # Create updated config dict
            current_dict = self._config.dict()
            merged_dict = ConfigurationLoader.merge_configurations(current_dict, updates)
            
            # Validate new configuration
            new_config = SystemConfiguration(**merged_dict)
            
            # Update cached config
            self._config = new_config
            self._config_cache_time = time.time()
            
            logger.info("Configuration updated successfully")
            
            # Persist if requested
            if persist and self._project_root:
                self._save_project_config(updates)
                
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            raise ConfigurationError(f"Configuration update failed: {e}") from e
    
    def _load_configuration(self) -> None:
        """Load configuration from all sources with proper precedence."""
        try:
            logger.debug("Loading configuration from all sources")
            
            # 1. Start with defaults (implicit in Pydantic models)
            configs_to_merge = []
            
            # 2. Load global configuration
            global_config_path = Path.home() / '.chungoid' / 'config.yaml'
            global_config = ConfigurationLoader.load_yaml_file(global_config_path)
            if global_config:
                configs_to_merge.append(global_config)
                logger.debug("Loaded global configuration")
            
            # 3. Load project configuration
            if self._project_root:
                project_config_path = self._project_root / '.chungoid' / 'config.yaml'
                project_config = ConfigurationLoader.load_yaml_file(project_config_path)
                if project_config:
                    configs_to_merge.append(project_config)
                    logger.debug("Loaded project configuration")
            
            # 4. Load environment variables (highest precedence)
            env_config = EnvironmentVariableMapper.load_from_environment()
            if env_config:
                configs_to_merge.append(env_config)
                logger.debug("Loaded environment configuration")
            
            # Merge all configurations
            if configs_to_merge:
                merged_config = ConfigurationLoader.merge_configurations(*configs_to_merge)
            else:
                merged_config = {}
            
            # Create and validate final configuration
            self._config = SystemConfiguration(**merged_config)
            
            logger.info("Configuration loaded successfully from all sources")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fall back to defaults
            self._config = SystemConfiguration()
            logger.warning("Using default configuration due to load failure")
    
    def _save_project_config(self, config_updates: Dict[str, Any]) -> None:
        """Save configuration updates to project config file."""
        if not self._project_root:
            logger.warning("Cannot save project config - no project root set")
            return
        
        try:
            project_config_path = self._project_root / '.chungoid' / 'config.yaml'
            
            # Load existing project config
            existing_config = ConfigurationLoader.load_yaml_file(project_config_path)
            
            # Merge with updates
            updated_config = ConfigurationLoader.merge_configurations(existing_config, config_updates)
            
            # Save updated config
            ConfigurationLoader.save_yaml_file(project_config_path, updated_config)
            
            logger.info(f"Saved updated project configuration to: {project_config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save project configuration: {e}")
            raise ConfigurationFileError(f"Failed to save project config: {e}") from e

# ============================================================================
# Global Configuration Access
# ============================================================================

# Global singleton instance
_config_manager: Optional[ConfigurationManager] = None

def get_config() -> SystemConfiguration:
    """Get the global configuration instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager.get_config()

def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def reload_config() -> SystemConfiguration:
    """Force reload the global configuration."""
    return get_config_manager().reload_configuration()

def get_secret(secret_path: str) -> Optional[str]:
    """Securely get a secret value from configuration."""
    return get_config_manager().get_secret(secret_path)

# ============================================================================
# Utility Functions
# ============================================================================

def mask_secret(value: str) -> str:
    """Mask a secret value for safe logging."""
    if not value or len(value) < 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"

def validate_configuration(config_dict: Dict[str, Any]) -> SystemConfiguration:
    """Validate a configuration dictionary."""
    try:
        return SystemConfiguration(**config_dict)
    except Exception as e:
        raise ConfigurationValidationError(f"Configuration validation failed: {e}") from e

def create_default_config_file(file_path: Path, include_examples: bool = True) -> None:
    """Create a default configuration file with examples."""
    default_config = {
        'llm': {
            'provider': 'openai',
            'default_model': 'gpt-4o-mini-2024-07-18',
            'timeout': 60,
            'max_retries': 3
        },
        'chromadb': {
            'host': 'localhost',
            'port': 8000,
            'connection_timeout': 30
        },
        'agents': {
            'default_timeout': 300,
            'max_concurrent_agents': 5,
            'enable_automatic_checkpoints': True
        },
        'logging': {
            'level': 'INFO',
            'enable_file_logging': True,
            'mask_secrets': True
        }
    }
    
    if include_examples:
        default_config['_examples'] = {
            '_note': 'Remove this section in production',
            'environment_variables': [
                'CHUNGOID_LLM_API_KEY=your-api-key-here',
                'CHUNGOID_LLM_PROVIDER=openai',
                'CHUNGOID_CHROMADB_HOST=localhost'
            ],
            'secret_handling': 'Never put API keys in config files - use environment variables only'
        }
    
    ConfigurationLoader.save_yaml_file(file_path, default_config)
    logger.info(f"Created default configuration file: {file_path}")

# Example usage and testing support
if __name__ == "__main__":
    # Test the configuration system
    print("Configuration Management System - Test Mode")
    
    # Create configuration manager
    config_manager = ConfigurationManager()
    
    # Get configuration
    config = config_manager.get_config()
    print(f"Loaded configuration with LLM provider: {config.llm.provider}")
    print(f"Agent timeout: {config.agents.default_timeout}")
    print(f"ChromaDB host: {config.chromadb.host}")
    
    # Test secret handling
    api_key = config_manager.get_secret('llm.api_key')
    if api_key:
        print(f"API key loaded: {mask_secret(api_key)}")
    else:
        print("No API key configured")
    
    print("Configuration system ready!") 