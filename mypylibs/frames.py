"""Step 1: extract sharp, diverse calibration frames from a checkerboard video."""

from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

type Image = NDArray[np.uint8]

# Inner corners (columns, rows). The printed board has 9x6 squares -> 8x5 inner corners.
PATTERN = (8, 5)


def read_video(path: Path) -> list[Image]:
    """Return all frames of a video as a list of BGR images."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def sharpness(gray: Image) -> float:
    """Variance of the Laplacian: higher means sharper (less motion blur)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_calibration_frames(
    video_path: Path,
    n_frames: int = 20,
    pattern: tuple[int, int] = PATTERN,
    save_dir: Path | None = None,
) -> list[Image]:
    """Pick `n_frames` sharp frames, spread across the video, where the board is visible.

    1. Score every frame by sharpness and keep only those where the board is found
       (fast check, the precise corner extraction happens in step 2).
    2. Split the video timeline into `n_frames` equal chunks and take the sharpest
       candidate from each chunk, so the selected views cover the different poses.
    """
    frames = read_video(video_path)

    candidates: list[tuple[int, float]] = []  # (frame_index, sharpness)
    for idx, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_FAST_CHECK)
        if found:
            candidates.append((idx, sharpness(gray)))

    if len(candidates) < n_frames:
        msg = f"{video_path}: board only detected in {len(candidates)} frames, need {n_frames}"
        raise RuntimeError(msg)

    # Take the sharpest frame in each time chunk (blur is judged relative to its
    # neighbours, so slow and fast parts of the video both contribute a view).
    edges = np.linspace(0, len(frames), n_frames + 1)
    selected = []
    for lo, hi in pairwise(edges):
        in_chunk = [c for c in candidates if lo <= c[0] < hi]
        if in_chunk:
            selected.append(max(in_chunk, key=lambda c: c[1])[0])

    # Some chunks may have had no detections, top up with the sharpest remaining frames.
    remaining = sorted((c for c in candidates if c[0] not in selected), key=lambda c: -c[1])
    selected.extend(idx for idx, _ in remaining[: n_frames - len(selected)])
    selected.sort()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for idx in selected:
            cv2.imwrite(str(save_dir / f"frame_{idx:04d}.png"), frames[idx])

    print(
        f"{video_path.name}: {len(frames)} frames, "
        f"board found in {len(candidates)}, selected {len(selected)}: {selected}"
    )
    return [frames[i] for i in selected]
