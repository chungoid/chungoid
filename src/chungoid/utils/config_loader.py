"""Loads and provides access to configuration settings from config.yaml."""

import yaml
import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# <<< Define Custom Exception >>>
class ConfigError(Exception):
    """Custom exception for critical configuration loading errors."""
    pass

# Define the default configuration structure
DEFAULT_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "text",
        "file": "chungoid_mcp_server.log",
        "max_bytes": 10 * 1024 * 1024,
        "backup_count": 5,
    },
    "chromadb": {
        # Preferred unified keys (new)
        "mode": "auto",             # auto | persistent | http
        "server_url": "",           # http://localhost:8000 (only for http mode)

        # Legacy/back-compat keys (still consumed if present)
        "client_type": "http",      # deprecated – will be mapped to mode if mode not set
        "host": "localhost",        # used to build server_url if not provided
        "port": 8000,
        "persist_path": "./chroma",
    },
    # Add other default sections if needed
    # "server": {
    #     "host": "127.0.0.1",
    #     "port": 8888
    # }
}

CONFIG_FILENAME = "config.yaml"
_config: Optional[Dict[str, Any]] = None


def _deep_merge(source: Dict, destination: Dict) -> Dict:
    """Recursively merges source dict into destination dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            # Get node or create one
            node = destination.setdefault(key, {})
            _deep_merge(value, node)
        else:
            destination[key] = value
    return destination


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Loads configuration from YAML file, merges with defaults, and allows environment overrides.

    Args:
        config_path: Optional path to the config file. Defaults to 'config.yaml' in the project root.

    Returns:
        The loaded and merged configuration dictionary.
    """
    global _config

    if _config is not None:
        logger.debug("Configuration already loaded. Returning cached version.")
        return _config

    # Start with deep copy of defaults
    # Use json dump/load for a simple deep copy
    import json

    loaded_config = json.loads(json.dumps(DEFAULT_CONFIG))

    # Determine config file path
    if config_path is None:
        # Assume project root is parent of 'utils' dir
        project_root = Path(__file__).parent.parent.resolve()
        config_file = project_root / CONFIG_FILENAME
    else:
        config_file = Path(config_path).resolve()

    logger.info(f"Attempting to load configuration from: {config_file}")

    # Load from YAML file if exists
    if config_file.is_file():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            if yaml_config and isinstance(yaml_config, dict):
                # Merge loaded config into defaults (loaded values take precedence)
                loaded_config = _deep_merge(yaml_config, loaded_config)
                logger.info(f"Successfully loaded and merged config from {config_file}")
            else:
                logger.warning(
                    f"Config file {config_file} is empty or not a dictionary. Using defaults."
                )
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_file}: {e}. Using defaults.")
        except IOError as e:
            logger.error(f"Error reading config file {config_file}: {e}. Using defaults.")
        except Exception as e:
            logger.exception(
                f"Unexpected error loading config file {config_file}: {e}. Using defaults."
            )
    else:
        logger.warning(f"Config file not found at {config_file}. Using default settings.")

    # Apply environment variable overrides (optional, but good practice)
    # Example: CHUNGOID_LOGGING_LEVEL overrides logging.level
    for section, settings in DEFAULT_CONFIG.items():
        for key in settings:
            env_var_name = f"CHUNGOID_{section.upper()}_{key.upper()}"
            env_value = os.getenv(env_var_name)
            if env_value is not None:
                logger.info(
                    f"Overriding config '{section}.{key}' with environment variable {env_var_name}."
                )
                # Attempt to convert type based on default type
                default_value = loaded_config[section][key]
                try:
                    if isinstance(default_value, bool):
                        loaded_config[section][key] = env_value.lower() in ("true", "1", "yes")
                    elif isinstance(default_value, int):
                        loaded_config[section][key] = int(env_value)
                    elif isinstance(default_value, float):
                        loaded_config[section][key] = float(env_value)
                    else:
                        loaded_config[section][key] = env_value  # Keep as string
                except ValueError:
                    logger.error(
                        f"Could not convert environment variable {env_var_name}='{env_value}' to type {type(default_value)}. Using string value."
                    )
                    loaded_config[section][key] = env_value

    # --- Normalise chromadb config (derive missing values, env overrides) ---
    chroma_cfg = loaded_config.setdefault("chromadb", {})

    # Current chroma_cfg["mode"] reflects (default -> file -> generic CHUNGOID_ envs)
    # Now apply specific overrides and derivations in order:

    # 1. Let CHROMA_MODE (short env var) override if set
    env_short_mode_val = os.getenv("CHROMA_MODE")
    if env_short_mode_val:
        chroma_cfg["mode"] = env_short_mode_val.lower()
        logger.info(f"ChromaDB mode updated by CHROMA_MODE to: '{chroma_cfg['mode']}'.")

    # 2. If mode is still "auto" (meaning no specific mode from file, generic CHUNGOID_ envs, or CHROMA_MODE)
    #    Derive from CHROMA_API_IMPL.
    #    Use DEFAULT_CONFIG["chromadb"]["mode"] as fallback if "mode" key is somehow missing from chroma_cfg.
    if chroma_cfg.get("mode", DEFAULT_CONFIG["chromadb"]["mode"]) == "auto":
        logger.info("ChromaDB mode is 'auto'; deriving from CHROMA_API_IMPL...")
        if os.getenv("CHROMA_API_IMPL", "").lower() == "http":
            chroma_cfg["mode"] = "http"
        else:
            chroma_cfg["mode"] = "persistent"
        logger.info(f"ChromaDB mode derived as '{chroma_cfg['mode']}'.")

    # 3. CHUNGOID_CHROMADB_MODE (long env var) has ultimate precedence for mode.
    #    This overrides any previous setting (default, file, generic, CHROMA_MODE, or auto-derivation).
    env_chungoid_mode_val = os.getenv("CHUNGOID_CHROMADB_MODE")
    if env_chungoid_mode_val:
        chroma_cfg["mode"] = env_chungoid_mode_val.lower()
        logger.info(f"ChromaDB mode ultimately set by CHUNGOID_CHROMADB_MODE to: '{chroma_cfg['mode']}'.")

    # Handle server_url: CHUNGOID_ version takes precedence, then CHROMA_ version.
    # This is processed after mode is finalized, as server_url might be needed for http mode.
    # Initial server_url in chroma_cfg is from (default -> file -> generic CHUNGOID_ envs)
    
    env_chungoid_server_url = os.getenv("CHUNGOID_CHROMADB_SERVER_URL")
    env_short_server_url = os.getenv("CHROMA_SERVER_URL")

    if env_chungoid_server_url:
        chroma_cfg["server_url"] = env_chungoid_server_url
        logger.info(f"ChromaDB server_url set by CHUNGOID_CHROMADB_SERVER_URL to: '{chroma_cfg['server_url']}'.")
    elif env_short_server_url: # Only if CHUNGOID_ version wasn't set
        chroma_cfg["server_url"] = env_short_server_url
        logger.info(f"ChromaDB server_url set by CHROMA_SERVER_URL to: '{chroma_cfg['server_url']}'.")
    
    # Fallback for server_url if still empty and mode is http (using legacy host/port if available)
    if chroma_cfg.get("mode") == "http" and not chroma_cfg.get("server_url"):
        # Use .get from chroma_cfg for host/port first (might have been set by generic envs), then DEFAULT_CONFIG
        host = chroma_cfg.get("host", DEFAULT_CONFIG["chromadb"]["host"])
        port = chroma_cfg.get("port", DEFAULT_CONFIG["chromadb"]["port"])
        chroma_cfg["server_url"] = f"http://{host}:{port}"
        logger.info(f"ChromaDB server_url constructed for http mode using host/port: '{chroma_cfg['server_url']}'.")


    _config = loaded_config
    logger.debug(f"Final configuration: {_config}")
    return _config


def get_config() -> Dict[str, Any]:
    """Returns the loaded configuration dictionary, loading it if necessary."""
    if _config is None:
        return load_config()
    return _config


# Example Usage (for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Simulate loading from project root
    config = load_config()
    print("\nLoaded Configuration:")
    print(yaml.dump(config, indent=2))

    # Example of accessing a setting
    log_level = config.get("logging", {}).get("level", "DEFAULT_LEVEL")
    print(f"\nLog Level from config: {log_level}")

    # Test environment override (uncomment to test)
    # os.environ['CHUNGOID_LOGGING_LEVEL'] = 'DEBUG'
    # _config = None # Force reload
    # config_reloaded = get_config()
    # print("\nReloaded Configuration (with env override):")
    # print(yaml.dump(config_reloaded, indent=2))
    # log_level_reloaded = config_reloaded.get("logging", {}).get("level", "DEFAULT_LEVEL")
    # print(f"\nLog Level from reloaded config: {log_level_reloaded}")
