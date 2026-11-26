"""Synthetic horizontal motion blur shared by the blur-study training and evaluation code.

Severity is specified as a kernel width at a reference image width (31 px at 846 px, the
prepared Sintel width) and scaled to each source image's own width, so the blur covers the same
fraction of the field of view everywhere. It is always applied to the source-resolution image,
before any crop or resize.
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_BLUR_PX = 31
DEFAULT_REFERENCE_WIDTH = 846
BORDER = cv2.BORDER_REFLECT_101


def kernel_width(image_width: int, blur_px: float = DEFAULT_BLUR_PX,
                 reference_width: float = DEFAULT_REFERENCE_WIDTH) -> int:
    """Return the odd horizontal box-kernel width for an image of ``image_width`` pixels."""
    if image_width <= 0 or blur_px <= 0 or reference_width <= 0:
        raise ValueError("image_width, blur_px and reference_width must be positive")
    scaled = blur_px / reference_width * image_width
    # Nearest odd integer, matching the original eval/motion_blur.py rounding.
    return max(1, 2 * round((scaled - 1) / 2) + 1)


def apply_horizontal_blur(image: np.ndarray, width: int, strict: bool = False) -> np.ndarray:
    """Blur ``image`` (H x W or H x W x C) with a ``width`` x 1 box filter.

    With ``strict`` the call raises if the result equals the input (e.g. a constant image),
    which the evaluator uses to guarantee that every "blurred" view was actually changed.
    """
    if width < 1 or width % 2 == 0:
        raise ValueError(f"kernel width must be a positive odd integer, got {width}")
    blurred = cv2.blur(image, (width, 1), borderType=BORDER)
    if strict and np.array_equal(image, blurred):
        raise ValueError("Blur did not change the image")
    return blurred


def blur_source_image(image: np.ndarray, blur_px: float = DEFAULT_BLUR_PX,
                      reference_width: float = DEFAULT_REFERENCE_WIDTH,
                      strict: bool = False) -> tuple[np.ndarray, int]:
    """Blur a source-resolution image at the study severity; return (image, kernel width)."""
    width = kernel_width(image.shape[1], blur_px, reference_width)
    return apply_horizontal_blur(image, width, strict=strict), width
