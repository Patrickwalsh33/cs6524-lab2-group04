"""Bonus: back-project one depth map into a coloured 3D point cloud (.ply) for CloudCompare."""

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from mypylibs.frames import Image

MIN_DEPTH_CM = 20
# A pixel whose depth differs from its 5x5 neighbourhood median by more than this fraction sits on a
# depth edge, where SGBM blends foreground and background into "flying pixels" between the two.
MAX_EDGE_JUMP = 0.05


def write_ply(path: Path, image: Image, depth: NDArray[np.float32], P: NDArray[np.float64], max_depth: float) -> int:
    """Write the reliable pixels of `depth` as a binary PLY with colours from `image`. Returns the point count."""
    f, cx, cy = P[0, 0], P[0, 2], P[1, 2]
    filled = np.nan_to_num(depth).astype(np.float32)
    median = cv2.medianBlur(filled, 5)
    keep = (filled > MIN_DEPTH_CM) & (filled < max_depth) & (np.abs(filled - median) < MAX_EDGE_JUMP * filled)

    v, u = np.nonzero(keep)
    z = filled[keep]
    # Pinhole back-projection. Y and Z are negated so the cloud is upright when opened in CloudCompare
    # (OpenCV's camera looks down +Z with +Y pointing down).
    xyz = np.column_stack(((u - cx) * z / f, -(v - cy) * z / f, -z)).astype(np.float32)
    rgb = image[keep][:, ::-1]  # BGR -> RGB

    vertices = np.empty(len(z), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    for i, axis in enumerate("xyz"):
        vertices[axis] = xyz[:, i]
    for i, channel in enumerate("rgb"):
        vertices[channel] = rgb[:, i]

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as fh:
        fh.write(header.encode())
        fh.write(vertices.tobytes())
    return len(vertices)
