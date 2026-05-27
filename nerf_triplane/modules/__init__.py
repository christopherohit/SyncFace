"""Architectural modules for the full talking-head pipeline.

Each module here is a self-contained `nn.Module` that the network can
opt into via a CLI flag. Modules are deliberately decoupled from
`NeRFNetwork` so Stage A (identity-free pretraining) can re-use the
shared submodules across identities.
"""

from .motion_aligner import MotionAligner
from .deformation import DeformationMLP
from .mouth_branch import MouthBranch
from .head_stabilizer import HeadStabilizer, head_pose_smoothness_loss

__all__ = [
    "MotionAligner",
    "DeformationMLP",
    "MouthBranch",
    "HeadStabilizer",
    "head_pose_smoothness_loss",
]
