"""Dataset adapters."""

from .arc_json import ARCDataset
from .augmented_arc import AugmentedARCDataset

__all__ = ["ARCDataset", "AugmentedARCDataset"]
