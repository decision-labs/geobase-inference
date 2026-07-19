"""Lightweight test doubles for optional native model runtimes."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

torch = ModuleType("torch")
torch.Tensor = np.ndarray
torch.float32 = np.float32
torch.from_numpy = np.asarray
torch.cuda = SimpleNamespace(
    is_available=lambda: False,
    empty_cache=lambda: None,
)
sys.modules["torch"] = torch
