"""A minimal ``pyedflib.EdfReader``-compatible EDF reader built on ``edfio``.

``pyedflib`` is a C extension whose last PyPI release dates back to
2025-06 (no wheels for newer Python versions, source builds required),
while ``edfio`` is actively maintained and pure Python.  This adapter
exposes exactly the subset of the ``pyedflib.EdfReader`` API that this
package uses (in :class:`~torch_ecg.databases.base.NSRRDataBase` and
its subclasses, e.g. the SHHS reader), so that neither their code nor
the returned values change.
"""

from typing import List

import numpy as np
from edfio import read_edf

__all__ = [
    "_EdfReader",
]


class _EdfReader:
    """Read EDF/EDF+ files via ``edfio`` with a ``pyedflib.EdfReader``-
    compatible interface (the subset used by this package)."""

    def __init__(self, file_name: str) -> None:
        self._edf = read_edf(file_name)

    def getSignalLabels(self) -> List[str]:
        return [str(sig.label) for sig in self._edf.signals]

    def getSampleFrequency(self, chn: int) -> float:
        return float(self._edf.signals[chn].sampling_frequency)

    def getPhysicalDimension(self, chn: int) -> str:
        return str(self._edf.signals[chn].physical_dimension)

    def getTransducer(self, chn: int) -> str:
        return str(self._edf.signals[chn].transducer_type)

    def getPrefilter(self, chn: int) -> str:
        return str(self._edf.signals[chn].prefiltering)

    def readSignal(self, chn: int, digital: bool = False) -> np.ndarray:
        sig = self._edf.signals[chn]
        return sig.digital if digital else sig.data

    def _close(self) -> None:
        # edfio reads the whole file eagerly into memory,
        # so there is nothing to close
        self._edf = None  # type: ignore[assignment]
