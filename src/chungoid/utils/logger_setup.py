"""Configures logging for the application, including JSON formatting."""

import logging
import json
import os
from logging.handlers import RotatingFileHandler
from .config_manager import get_config, ConfigurationError
from pathlib import Path

# Environment Variable Defaults
# DEFAULT_LOG_LEVEL = "INFO"
# DEFAULT_LOG_FORMAT = "text" # Options: "text", "json"
# DEFAULT_LOG_FILE = "chungoid_mcp_server.log" # Default log file name

# Read configuration from environment variables
# LOG_LEVEL_ENV = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
# LOG_FORMAT_ENV = os.getenv("LOG_FORMAT", DEFAULT_LOG_FORMAT).lower()
# LOG_FILE_ENV = os.getenv("LOG_FILE", DEFAULT_LOG_FILE)


# Define a custom JSON Formatter
class JsonFormatter(logging.Formatter):
    """Formats log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats the log record into a JSON string."""
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),  # Use getMessage() for formatted message
            # Add other fields as needed, e.g., pathname, lineno
            "pathname": record.pathname,
            "lineno": record.lineno,
        }
        # Include exception info if available
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_entry["stack_info"] = self.formatStack(record.stack_info)

        # Handle potential non-serializable objects in args (basic handling)
        # More robust handling might involve custom serializers or filtering
        # For now, convert args to string representation if not directly serializable
        # This avoids crashing the logger if complex objects are logged
        if record.args:
            try:
                # Attempt direct serialization if possible (e.g., for simple types)
                json.dumps(record.args)
                log_entry["args"] = record.args
            except TypeError:
                # Fallback to string representation for complex types
                log_entry["args"] = tuple(repr(arg) for arg in record.args)

        return json.dumps(log_entry, ensure_ascii=False)


# Central logging setup function
def setup_logging(
    level: str | None = None, # Re-enable level, make it optional
    # log_file_path: str = "app.log",
    # max_bytes: int = 10 * 1024 * 1024, # Keep these if you want them configurable per call
    # backup_count: int = 5,             # Keep these if you want them configurable per call
    # use_json_formatter: bool = True,
):
    """Configures root logger based on settings from config.yaml or defaults.

    Reads logging configuration (format, file, max_bytes, backup_count)
    from the loaded configuration. The log level can be overridden by the 'level' param.

    Args:
        level: Optional log level string (e.g., "DEBUG", "INFO"). If None, uses config.
    #    max_bytes: Maximum size of the log file before rotation.
    #    backup_count: Number of backup log files to keep.
    """
    # Get configuration using the new ConfigurationManager
    try:
        system_config = get_config()
        log_config = system_config.logging
    except ConfigurationError as e:
        # Fall back to basic logging if config fails
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error(f"Failed to load configuration, using basic logging: {e}")
        return

    # Use provided level if given, otherwise use config
    effective_level = level if level is not None else log_config.level
    
    # Use enable_structured_logging to determine if JSON format should be used
    use_json_formatter = log_config.enable_structured_logging
    
    # Configure log file path - use log_directory and construct default filename
    if log_config.enable_file_logging:
        log_file_path_from_config = f"{log_config.log_directory}/chungoid.log"
    else:
        log_file_path_from_config = None
    
    # Convert max_log_size_mb to bytes for RotatingFileHandler
    max_bytes = log_config.max_log_size_mb * 1024 * 1024
    
    # Use log_retention_days to estimate backup count (roughly 1 backup per day)
    backup_count = min(log_config.log_retention_days, 30)  # Cap at 30 backups

    try:
        log_level_val = getattr(logging, effective_level.upper(), logging.INFO) # Use effective_level
    except AttributeError:
        print(f"Warning: Invalid log level '{effective_level}'. Defaulting to INFO.")
        log_level_val = logging.INFO

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level_val)  # Set the minimum level for the root logger

    # Clear existing handlers to avoid duplicates if called multiple times
    # Use list comprehension to avoid modifying list while iterating
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Choose formatter
    if use_json_formatter:
        formatter = JsonFormatter()
        stream_formatter = JsonFormatter()  # Use JSON for console too if JSON is enabled
    else:
        # Use a standard text formatter if JSON is disabled
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)
        stream_formatter = logging.Formatter(log_format)  # Match console format

    # Console Handler (always add)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_val)  # Console logs at the specified level
    console_handler.setFormatter(stream_formatter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler (optional)
    if log_file_path_from_config: # Check the original config value
        try:
            # NEW: Use current working directory as project root instead of calculating from __file__
            # This is more reliable when the package is installed in editable mode
            project_root = Path.cwd()
            
            # Look for indicators that we're in the right project directory
            # Check if we can find the chungoid-core directory or config.yaml
            if (project_root / "chungoid-core" / "config.yaml").exists():
                # We're in the main project directory
                pass
            elif (project_root / "config.yaml").exists():
                # We might be in chungoid-core subdirectory, go up one level
                project_root = project_root.parent
            elif "chungoid-core" in str(project_root):
                # We might be inside chungoid-core, navigate to the main project
                parts = project_root.parts
                try:
                    chungoid_index = parts.index("chungoid-core")
                    if chungoid_index > 0:
                        project_root = Path(*parts[:chungoid_index])
                except ValueError:
                    pass
            
            # If the configured path is not absolute, make it relative to project_root
            if not Path(log_file_path_from_config).is_absolute():
                resolved_log_file_path = project_root / log_file_path_from_config
            else:
                resolved_log_file_path = Path(log_file_path_from_config) # Use as is if absolute
            
            # Convert to string for os.path.dirname and RotatingFileHandler
            log_file_path_str = str(resolved_log_file_path)

            # Ensure log directory exists
            log_dir = os.path.dirname(log_file_path_str)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                # Use print for this initial setup phase as logger might not be fully ready
                print(f"Log directory created: {log_dir}")

            # Create file handler - No longer re-reading from config here
            file_handler = RotatingFileHandler(
                log_file_path_str, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)  # Use the chosen formatter
            file_handler.setLevel(logging.DEBUG) 
            root_logger.addHandler(file_handler)
            # Use a specific logger to announce file logging
            logging.getLogger(__name__).info(f"File logging configured to {log_file_path_str}")
        except Exception as e:
            # Use a specific logger for this error too
            logging.getLogger(__name__).error(
                f"Error setting up file logging: {e}. File logging disabled.", exc_info=True
            )
    else:
        logging.getLogger(__name__).info("File logging is disabled (no path provided in config).")

    logging.getLogger(__name__).info("Logging setup complete. Effective Level: %s", effective_level.upper())


# Example usage (typically called once at application startup)
if __name__ == "__main__":
    # Example: Configure logging - now reads from environment or defaults
    # Set environment variables here for testing if needed
    # os.environ['LOG_LEVEL'] = 'DEBUG' # This would now be CHUNGOID_LOGGING_LEVEL
    # os.environ['LOG_FORMAT'] = 'json' # This would now be CHUNGOID_LOGGING_FORMAT
    # os.environ['LOG_FILE'] = './test_app.log' # This would now be CHUNGOID_LOGGING_FILE

    # Get config for display purposes in example
    try:
        system_config = get_config()
        log_config = system_config.logging
        print("Configuring logger using settings from ConfigurationManager. Example values:")
        print(f"  Level: {log_config.level}, Structured: {log_config.enable_structured_logging}, Directory: {log_config.log_directory}")
    except ConfigurationError as e:
        print(f"Failed to load configuration: {e}")
        print("Using default logging setup")

    setup_logging()

    # Get loggers for different modules
    main_logger = logging.getLogger(__name__)  # Gets root logger if name is __main__
    module_logger = logging.getLogger("my_module")

    # Log messages
    main_logger.debug("This is a debug message from main.")  # Won't show if level is INFO
    main_logger.info("This is an info message from main.")
    module_logger.warning("This is a warning from my_module.")
    module_logger.error("This is an error from my_module.", extra={"custom_field": 123})

    # Example of logging an exception
    try:
        result = 1 / 0
    except ZeroDivisionError:
        main_logger.exception("Caught an expected exception.")

    print("Check console output and 'test_app.log' for log messages.")

    # Clean up test log file
    if os.path.exists("./test_app.log"):
        os.remove("./test_app.log")
