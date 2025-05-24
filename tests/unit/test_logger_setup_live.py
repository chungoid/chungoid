import logging
from importlib import reload
from pathlib import Path

import pytest

import chungoid.utils.logger_setup as logger_setup
from chungoid.utils import config_manager


def test_setup_logging_creates_rotating_handler(tmp_path, monkeypatch):
    # Prepare a custom config file with JSON format and DEBUG level
    cfg_yaml = (
        """\
logging:
  level: DEBUG
  format: text
  file: {log_file}
  max_bytes: 1024
  backup_count: 1
"""
    ).format(log_file=tmp_path / "app.log")

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_yaml)

    # Ensure loader picks our config
    config_manager._config = None  # reset cache
    cfg = config_manager.load_config(str(cfg_path))

    # Reload logger_setup to ensure it picks new config
    reload(logger_setup)

    # Call setup_logging
    logger_setup.setup_logging()

    root = logging.getLogger()
    # Expect at least one RotatingFileHandler
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers)
    # root level DEBUG per config
    assert root.level == logging.DEBUG

    # Emit a log and ensure file created
    logging.getLogger(__name__).info("hello world")
    assert (tmp_path / "app.log").exists() 