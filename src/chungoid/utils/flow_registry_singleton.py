from pathlib import Path
from .flow_registry import FlowRegistry
import os

_FLOW_REG_MODE = os.getenv("FLOW_REGISTRY_MODE", "persistent")
if _FLOW_REG_MODE not in {"persistent", "http", "memory"}:
    _FLOW_REG_MODE = "memory"
if "PYTEST_CURRENT_TEST" in os.environ or os.getenv("CHROMA_API_IMPL") == "http" or _FLOW_REG_MODE == "persistent" and os.getenv("CHROMA_SERVER_HOST"):
    _FLOW_REG_MODE = "memory"

_flow_registry = FlowRegistry(project_root=Path.cwd(), chroma_mode=_FLOW_REG_MODE) 