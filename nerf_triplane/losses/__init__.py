"""Training losses for the talking-head pipeline.

Each loss is opt-in via a `--lambda_*` CLI flag (default 0.0 = disabled),
so adding a file here never changes the default training behavior.
"""

from .temporal_loss import TemporalConsistencyLoss
from .dssim import DSSIMLoss
from .embedding_smoothness import EmbeddingSmoothnessLoss
from .geometry import EikonalDensityLoss
from .lip_sync import LipSyncLoss
from .disentangle import NegativeContrastDisentangleLoss

__all__ = [
    "TemporalConsistencyLoss",
    "DSSIMLoss",
    "EmbeddingSmoothnessLoss",
    "EikonalDensityLoss",
    "LipSyncLoss",
    "NegativeContrastDisentangleLoss",
]
