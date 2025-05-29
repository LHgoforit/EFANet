"""
src
===

Root Python package for the EFANet project.
Exposes a lightweight *model registry* so that experiment scripts can
instantiate networks with a single helper call, irrespective of where
the actual class is implemented.

Example
-------
>>> from src import build_model
>>> model = build_model("EFANet", channels=64, num_blocks=13)

Adding a new model
------------------
Decorate the class with `@register_model("MyNet")` inside its module::

    from src import register_model

    @register_model("MyNet")
    class MyNet(nn.Module):
        ...

The class becomes instantly discoverable through `build_model`.
"""

from __future__ import annotations
from typing import Callable, Dict, Type, Any

__all__ = [
    "register_model",
    "build_model",
    "__version__",
]

# ------------------------------------------------------------------
# Version helper (falls back to '0.0.0' if metadata is absent)
# ------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version

    __version__: str = _pkg_version("efanet")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# ------------------------------------------------------------------
# Model registry
# ------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str) -> Callable[[Type], Type]:
    """Class decorator used to register a model into the global registry."""

    def _wrapper(cls: Type) -> Type:
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise KeyError(f"Duplicate model registration for key '{name}'")
        _MODEL_REGISTRY[key] = cls
        return cls

    return _wrapper


def build_model(name: str, *args: Any, **kwargs: Any):
    """
    Instantiate a model by name.
    All positional/keyword arguments are forwarded to the model ctor.

    Parameters
    ----------
    name : str
        Human-readable key defined at registration time
        (case-insensitive).
    *args, **kwargs
        Constructor arguments.

    Returns
    -------
    nn.Module
        Instantiated PyTorch model.
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered.")
    return _MODEL_REGISTRY[key](*args, **kwargs)


# ------------------------------------------------------------------
# Register home-grown architectures
# ------------------------------------------------------------------
from src.efanet.efanet import EFANet  # noqa:E402  (import late to avoid cycles)

register_model("EFANet")(EFANet)
