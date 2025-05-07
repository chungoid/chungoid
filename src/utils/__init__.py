"""Compatibility shim.

This temporary package maps legacy imports like
`import utils.prompt_manager` to the new module path
`chungoid.utils.prompt_manager` while we migrate the codebase.

Remove once all references are updated.  (Tracked in roadmap A3.)
"""
from importlib import import_module
import sys

_target_pkg = import_module("chungoid.utils")

# Re-export all public symbols at package level
globals().update(vars(_target_pkg))

# Ensure submodules are discoverable as utils.xyz
for _name, _mod in sys.modules.items():
    if _name.startswith("chungoid.utils"):
        shim_name = _name.replace("chungoid.utils", "utils", 1)
        sys.modules[shim_name] = _mod

# List of modules in the new location (chungoid.utils)
_actual_utils_modules = [
    "analysis_utils",
    "chroma_client_factory",
    "chroma_utils",
    "common_utils", # Assuming this exists or will exist
    "config_loader",
    "exceptions",
    "file_utils",   # Assuming this exists or will exist
    "logger_setup",
    "prompt_manager",
    "security",
    "state_manager",
    "template_helpers",
    "token_utils",  # Assuming this exists or will exist
    "tool_adapters",
]

# Iterate over a copy of the dictionary's items to avoid RuntimeError
for _module_name_key, _module_obj in list(sys.modules.items()):
    if _module_name_key.startswith("utils."):
        _submodule_name = _module_name_key.split('.', 1)[1]
        if _submodule_name in _actual_utils_modules:
            _target_module_full_name = f"chungoid.utils.{_submodule_name}"
            try:
                # Ensure the actual module is imported
                _imported_target_module = import_module(_target_module_full_name)
                # Make the old 'utils.submodule' reference point to the new 'chungoid.utils.submodule'
                sys.modules[_module_name_key] = _imported_target_module
            except ImportError:
                # If chungoid.utils.submodule can't be imported,
                # it's an issue, but we'll pass to maintain original shim's observed behavior.
                # A warning log here could be useful in the future.
                pass

# No del statement to avoid potential NameErrors during its execution 