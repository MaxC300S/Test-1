"""Compatibility utilities for smoothing over PyTorch platform quirks."""
from __future__ import annotations

import logging
import types
from typing import Any

import torch

LOGGER = logging.getLogger(__name__)


def ensure_torch_classes() -> None:
    """Ensure ``torch.classes`` exists even on builds missing the C++ bindings.

    On some Windows CPU-only distributions ``torch.classes`` raises a
    ``RuntimeError`` referencing ``__path__._path`` when libraries attempt to
    introspect the package.  Streamlit's module loader triggers that code path
    when it walks imports, which ultimately bubbles up as the error shown in the
    user's screenshot.  We install a lightweight stub that exposes the
    ``load_library`` API so that PyTorch callers continue to behave, while also
    surfacing a warning for observability.
    """

    try:
        _ = torch.classes  # type: ignore[attr-defined]
    except AttributeError:
        stub = types.SimpleNamespace()
        _install_stub(stub)
        torch.classes = stub  # type: ignore[assignment]
        LOGGER.warning("torch.classes attribute missing; installed compatibility stub")
    except RuntimeError as exc:
        message = str(exc)
        if "__path__._path" not in message and "torch::class" not in message:
            raise
        stub = types.SimpleNamespace()
        _install_stub(stub)
        torch.classes = stub  # type: ignore[assignment]
        LOGGER.warning("torch.classes unavailable (%s); installed compatibility stub", message)


def _install_stub(stub: Any) -> None:
    def _noop_load_library(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial stub
        LOGGER.debug("torch.classes.load_library called in stub mode with args=%s kwargs=%s", args, kwargs)

    stub.load_library = _noop_load_library
    stub.__doc__ = "Compatibility stub for torch.classes"


ensure_torch_classes()
