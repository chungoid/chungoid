import os
from pathlib import Path

import pytest
import yaml

from chungoid.utils import config_loader


def test_load_default_config(tmp_path, monkeypatch):
    """When no config.yaml exists, defaults should be returned."""
    # Ensure loader looks at tmp dir
    monkeypatch.chdir(tmp_path)
    cfg = config_loader.load_config()
    assert cfg["logging"]["level"] == "INFO"
    assert cfg["chromadb"]["mode"] in {"persistent", "http"}


def test_load_custom_file(tmp_path):
    # write a custom yaml file
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("""\
logging:
  level: DEBUG
chromadb:
  mode: http
  server_url: http://example.com:9000
""")
    config_loader._config = None  # reset cache
    cfg = config_loader.load_config(str(cfg_path))
    assert cfg["logging"]["level"] == "DEBUG"
    assert cfg["chromadb"]["mode"] == "http"
    assert cfg["chromadb"]["server_url"] == "http://example.com:9000"


def test_env_override(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("logging:\n  level: WARNING\n")
    monkeypatch.setenv("CHUNGOID_LOGGING_LEVEL", "ERROR")
    config_loader._config = None
    cfg = config_loader.load_config(str(cfg_path))
    assert cfg["logging"]["level"] == "ERROR"

    # Clean for other tests
    monkeypatch.delenv("CHUNGOID_LOGGING_LEVEL") 