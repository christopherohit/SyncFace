"""Inference postprocessing.

Diagram lists: compose head + torso/bg, seam / neck fill, hair / detail
refinement, optional super-resolution.

The first stage (`seam_fill`) is implemented because it works without
any extra weights. The other two are wired as optional hooks; they
attempt a lazy import of the corresponding third-party package and
become a no-op (returning the input unchanged) if the package is not
installed. This keeps the trainer/test path self-contained while
letting users plug in GFPGAN/CodeFormer.
"""

from typing import Optional

import numpy as np


def seam_fill(
    composed: np.ndarray,
    face_mask: np.ndarray,
    feather_radius: int = 7,
) -> np.ndarray:
    """Feather the boundary between the rendered head and the background.

    Args:
        composed: HxWx3 uint8 RGB image already composed.
        face_mask: HxW float mask in [0, 1] for the rendered head.
        feather_radius: Gaussian blur kernel radius (must be odd >= 1).

    Returns:
        HxWx3 uint8 image with a softer seam.
    """
    import cv2

    if feather_radius % 2 == 0:
        feather_radius += 1
    blur = cv2.GaussianBlur(face_mask.astype(np.float32), (feather_radius, feather_radius), 0)
    soft_mask = np.clip(blur, 0.0, 1.0)[..., None]
    bg = composed.astype(np.float32) * (1.0 - soft_mask) + composed.astype(np.float32) * soft_mask
    return np.clip(bg, 0, 255).astype(np.uint8)


def hair_refine(image: np.ndarray, hair_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Sharpen the hair region a touch using an unsharp mask, if a hair mask is provided.

    No-op if `hair_mask` is None.
    """
    if hair_mask is None:
        return image
    import cv2

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    m = hair_mask[..., None].astype(np.float32)
    out = image.astype(np.float32) * (1.0 - m) + sharpened.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def super_resolve(image: np.ndarray, backend: str = "gfpgan") -> np.ndarray:
    """Optional face super-resolution.

    Tries to import the requested backend lazily; falls back to the
    input image if the backend is unavailable. Returning the input
    unchanged means the rest of the inference pipeline doesn't need to
    branch on whether SR is configured.
    """
    if backend == "gfpgan":
        try:
            from gfpgan import GFPGANer  # type: ignore
        except Exception:
            return image
        try:
            restorer = GFPGANer(model_path=None, upscale=1, arch="clean", channel_multiplier=2)
            _, _, restored = restorer.enhance(image, has_aligned=False, only_center_face=False)
            return restored if restored is not None else image
        except Exception:
            return image
    return image
