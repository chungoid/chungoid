"""Configures logging for the application, including JSON formatting."""

import logging
import json
import os
from logging.handlers import RotatingFileHandler
from .config_loader import get_config

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
    # Remove parameters that are now read from environment
    # level: str = "INFO",
    # log_file_path: str = "app.log",
    # max_bytes: int = 10 * 1024 * 1024, # Keep these if you want them configurable per call
    # backup_count: int = 5,             # Keep these if you want them configurable per call
    # use_json_formatter: bool = True,
):
    """Configures root logger based on settings from config.yaml or defaults.

    Reads logging configuration (level, format, file, max_bytes, backup_count)
    from the loaded configuration.

    # Args: # <<< Remove Args if parameters are removed, or update if kept >>>
    #    max_bytes: Maximum size of the log file before rotation.
    #    backup_count: Number of backup log files to keep.
    """
    # Get configuration using the loader
    config = get_config()
    log_config = config.get("logging", {})

    # Get settings from config, falling back to defaults if necessary (though loader handles defaults)
    level = log_config.get("level", "INFO")
    log_format_type = log_config.get("format", "text").lower()
    log_file_path = log_config.get("file", None)  # Default to None if not in config
    max_bytes = log_config.get("max_bytes", 10 * 1024 * 1024)
    backup_count = log_config.get("backup_count", 5)

    use_json_formatter = log_format_type == "json"

    # Use environment variables read at module level # <<< REMOVE THESE LINES >>>
    # level = LOG_LEVEL_ENV
    # use_json_formatter = LOG_FORMAT_ENV == "json"
    # log_file_path = LOG_FILE_ENV

    try:
        log_level = getattr(logging, level.upper(), logging.INFO)
    except AttributeError:
        print(f"Warning: Invalid log level '{level}'. Defaulting to INFO.")
        log_level = logging.INFO

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)  # Set the minimum level for the root logger

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
    console_handler.setLevel(log_level)  # Console logs at the specified level
    console_handler.setFormatter(stream_formatter)
    root_logger.addHandler(console_handler)

    # Rotating File Handler (optional)
    if log_file_path:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file_path)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                print(f"Created log directory: {log_dir}")

            # Create file handler
            log_file = config.get("logging", {}).get("file", "chungoid_mcp_server.log")
            max_bytes = config.get("logging", {}).get("max_bytes", 10 * 1024 * 1024)  # Default 10MB
            backup_count = config.get("logging", {}).get("backup_count", 5)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)  # Use the chosen formatter
            # <<< SET HANDLER LEVEL EXPLICITLY TO DEBUG >>>
            file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)
            logging.info(f"File logging configured to {log_file}")  # Use standard logging here
        except Exception as e:
            logging.error(
                f"Error setting up file logging: {e}. File logging disabled.", exc_info=True
            )
    else:
        logging.getLogger(__name__).info("File logging is disabled (no path provided).")

    logging.getLogger(__name__).info("Logging setup complete. Level: %s", level.upper())


# Example usage (typically called once at application startup)
if __name__ == "__main__":
    # Example: Configure logging - now reads from environment or defaults
    # Set environment variables here for testing if needed
    # os.environ['LOG_LEVEL'] = 'DEBUG' # This would now be CHUNGOID_LOGGING_LEVEL
    # os.environ['LOG_FORMAT'] = 'json' # This would now be CHUNGOID_LOGGING_FORMAT
    # os.environ['LOG_FILE'] = './test_app.log' # This would now be CHUNGOID_LOGGING_FILE

    # Get config for display purposes in example
    config = get_config().get("logging", {})
    print("Configuring logger using settings from config_loader. Example values:")
    print(
        f"  Level: {config.get('level', 'N/A')}, Format: {config.get('format', 'N/A')}, File: {config.get('file', 'N/A')}"
    )

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
