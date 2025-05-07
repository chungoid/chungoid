from importlib import import_module

# Lazy import to avoid heavy deps on package import

def __getattr__(name):
    if name == "ChungoidEngine":
        engine_module = import_module("chungoid.engine")
        return engine_module.ChungoidEngine
    raise AttributeError(name)

__all__ = ["ChungoidEngine"]
