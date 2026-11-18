"""CPU/GPU backend switch.

The same pipeline code runs on pandas/scipy/networkx (CPU) or on
RAPIDS cuDF/cupy/cuGraph (GPU). Pick with the MIXTAPE_BACKEND env var:
  auto (default) -> GPU if RAPIDS is importable, else CPU
  gpu            -> force GPU (error if RAPIDS missing)
  cpu            -> force CPU
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType


def gpu_available() -> bool:
    try:
        import cudf  # noqa: F401
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class Backend:
    name: str          # "cpu" | "gpu"
    df: ModuleType     # pandas | cudf
    xp: ModuleType     # numpy | cupy
    sparse: ModuleType  # scipy.sparse | cupyx.scipy.sparse

    @property
    def is_gpu(self) -> bool:
        return self.name == "gpu"

    def to_pandas(self, frame):
        return frame.to_pandas() if self.is_gpu else frame

    def to_numpy(self, arr):
        return arr.get() if self.is_gpu and hasattr(arr, "get") else arr

    def from_pandas(self, frame):
        return self.df.from_pandas(frame) if self.is_gpu else frame


@lru_cache(maxsize=None)
def get_backend(name: str | None = None) -> Backend:
    name = (name or os.environ.get("MIXTAPE_BACKEND", "auto")).lower()
    if name == "auto":
        name = "gpu" if gpu_available() else "cpu"
    if name == "gpu":
        if not gpu_available():
            raise RuntimeError(
                "MIXTAPE_BACKEND=gpu but RAPIDS (cudf/cupy) is not installed. "
                "Use Colab or WSL2 - see README."
            )
        import cudf
        import cupy
        import cupyx.scipy.sparse as cps
        return Backend("gpu", cudf, cupy, cps)
    if name == "cpu":
        import numpy
        import pandas
        import scipy.sparse as sps
        return Backend("cpu", pandas, numpy, sps)
    raise ValueError(f"unknown backend {name!r}")
