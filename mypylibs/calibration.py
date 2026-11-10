"""Step 3: estimate the camera intrinsics K and distortion D from the checkerboard corners."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from mypylibs.corners import Corners
from mypylibs.frames import PATTERN

SQUARE_SIZE_CM = 2.5  # side of one printed square; sets the unit of every 3D quantity later on
TARGET_ERROR_PX = 0.5
MIN_FRAMES = 15


@dataclass
class Calibration:
    K: NDArray[np.float64]  # 3x3 intrinsic matrix
    D: NDArray[np.float64]  # distortion coefficients (k1, k2, p1, p2, k3)
    mean_error: float  # mean reprojection error over all corners, in pixels
    per_frame_error: list[float]
    frames_used: list[int]  # indices (into the step-2 corner list) kept for the final solve


def board_points(pattern: tuple[int, int] = PATTERN, square: float = SQUARE_SIZE_CM) -> NDArray[np.float32]:
    """3D corner positions on the board plane Z=0, in cm, in findChessboardCorners order.

    Row by row, x varying fastest: (0,0,0), (s,0,0), ..., ((cols-1)s, (rows-1)s, 0).
    """
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return pts


def _solve(
    obj_pts: list[NDArray[np.float32]], img_pts: list[Corners], image_size: tuple[int, int]
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[float], float]:
    """Run calibrateCamera and compute the per-frame and overall mean reprojection error."""
    _, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)

    per_frame, all_dists = [], []
    for obj, img, rvec, tvec in zip(obj_pts, img_pts, rvecs, tvecs, strict=True):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        dists = np.linalg.norm(projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        per_frame.append(float(dists.mean()))
        all_dists.append(dists)
    return K, D, per_frame, float(np.concatenate(all_dists).mean())


def calibrate(image_points: list[Corners], image_size: tuple[int, int]) -> Calibration:
    """Calibrate, then drop the worst frame and re-solve while above target (keeping >= MIN_FRAMES)."""
    obj = board_points()
    kept = list(range(len(image_points)))

    while True:
        K, D, per_frame, mean_err = _solve([obj] * len(kept), [image_points[i] for i in kept], image_size)
        print(f"  calibrated on {len(kept)} frames: mean reprojection error {mean_err:.3f} px")
        if mean_err <= TARGET_ERROR_PX or len(kept) <= MIN_FRAMES:
            break
        worst = int(np.argmax(per_frame))
        print(f"    dropping frame {kept[worst]} (error {per_frame[worst]:.3f} px)")
        kept.pop(worst)

    return Calibration(K, D.ravel(), mean_err, per_frame, kept)


def report(cal: Calibration) -> None:
    """Print the parameters the report asks for."""
    K, D = cal.K, cal.D
    print(f"  fx={K[0, 0]:.2f}  fy={K[1, 1]:.2f}  cx={K[0, 2]:.2f}  cy={K[1, 2]:.2f}  skew={K[0, 1]:.4f}")
    print(f"  k1={D[0]:.4f}  k2={D[1]:.4f}  k3={D[4]:.4f}  p1={D[2]:.5f}  p2={D[3]:.5f}")
    status = "OK" if cal.mean_error <= TARGET_ERROR_PX else "ABOVE TARGET"
    print(f"  mean reprojection error {cal.mean_error:.3f} px ({status}, target <= {TARGET_ERROR_PX})")


def save(cal: Calibration, out_dir: Path, cam_id: str) -> None:
    """Write K and D as human-readable CSV (numpy.savetxt), as required by the README."""
    np.savetxt(out_dir / f"{cam_id}_K.csv", cal.K, delimiter=",", fmt="%.6f")
    np.savetxt(out_dir / f"{cam_id}_D.csv", cal.D.reshape(1, -1), delimiter=",", fmt="%.8f")
