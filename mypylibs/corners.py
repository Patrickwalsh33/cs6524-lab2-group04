from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from mypylibs.frames import PATTERN, Image

type Corners = NDArray[np.float32]

# Stop cornerSubPix after 30 iterations or when a corner moves less than 0.001 px.
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_corners(
    frames: list[Image],
    pattern: tuple[int, int] = PATTERN,
    save_dir: Path | None = None,
) -> tuple[list[Corners], tuple[int, int]]:

    image_points = []
    h, w = frames[0].shape[:2]

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Full (non fast-check) detection, adaptive threshold helps with uneven lighting.
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if not found:
            print(f"  frame {i}: board not found, skipped")
            continue

        # Refine each corner within an 11x11 window (winSize is the half-size).
        # Squares are roughly 20+ px wide at this resolution, so 5 px stays inside one square.
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)
        image_points.append(corners)

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            vis = cv2.drawChessboardCorners(frame.copy(), pattern, corners, found)
            cv2.imwrite(str(save_dir / f"corners_{i:02d}.png"), vis)

    print(f"  corners detected in {len(image_points)}/{len(frames)} frames ({pattern[0] * pattern[1]} corners each)")
    return image_points, (w, h)
