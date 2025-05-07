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

del import_module, sys, _name, _mod, _target_pkg, shim_name 