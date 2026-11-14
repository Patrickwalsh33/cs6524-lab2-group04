"""Steps 5 + 6: inter-frame pose (SIFT -> F -> E -> R, t), rectification, disparity and depth video."""

from pathlib import Path

import cv2
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mypylibs.calibration import Calibration
from mypylibs.frames import Image, read_video
from mypylibs.pointcloud import write_ply

type Matrix = NDArray[np.float64]

# Consecutive frames (30 fps) are only a few mm apart: the pose is unstable and disparities ~1 px.
# Pairing frame t with t + FRAME_GAP gives a usable baseline (tested 5/10/15: 10-15 most stable).
FRAME_GAP = 10
FPS = 30
# recoverPose only returns the *direction* of t (||t|| = 1): the true scale is unobservable from images
# alone. We assume a steady sideways hand speed to turn ||t|| into centimetres.
CAMERA_SPEED_CM_S = 15.0
BASELINE_CM = CAMERA_SPEED_CM_S * FRAME_GAP / FPS

MIN_MATCHES = 30
# Rectification + horizontal block matching assume a sideways move; reject pairs whose estimated
# motion is mostly forward/backward or that rotated a lot (usually a bad pose estimate).
MIN_SIDEWAYS = 0.9  # |t_x| of the unit translation
MAX_ROTATION_DEG = 5.0
NUM_DISPARITIES = 64  # must be a multiple of 16
BLOCK_SIZE = 5
DEPTH_MAX_CM = 500  # fixed colour scale so colours mean the same distance in every frame

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher()
sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=NUM_DISPARITIES,
    blockSize=BLOCK_SIZE,
    P1=8 * BLOCK_SIZE**2,
    P2=32 * BLOCK_SIZE**2,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
)


def relative_pose(gray1: Image, gray2: Image, K: Matrix) -> tuple[Matrix, Matrix] | None:
    """Step 5: SIFT matches -> essential E -> (R, unit t). None if the pose is unusable for stereo."""
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return None

    # Lowe's ratio test: keep a match only if it is clearly better than the second-best candidate.
    good = [m for m, n in matcher.knnMatch(des1, des2, k=2) if m.distance < 0.75 * n.distance]
    if len(good) < MIN_MATCHES:
        return None
    pts1 = np.float64([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float64([kp2[m.trainIdx].pt for m in good])

    # findEssentialMat (5-point, uses K) instead of findFundamentalMat + E = K^T F K: the 8-point F route
    # gave far more flipped/degenerate poses on our scenes (e.g. 180 deg rotations).
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

    rotation_deg = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    if abs(t[0, 0]) < MIN_SIDEWAYS or rotation_deg > MAX_ROTATION_DEG:
        return None
    return R, t


def depth_map(
    frame1: Image, frame2: Image, K: Matrix, R: Matrix, t: Matrix
) -> tuple[Image, NDArray[np.float32], Matrix]:
    """Step 6: rectify the pair, run SGBM, convert disparity to depth Z = f * B / d (cm).

    Returns the rectified left BGR image, its depth map (NaN where disparity is invalid) and the
    rectified 3x4 projection matrix (whose f, cx, cy back-project the depth into 3D).
    """
    h, w = frame1.shape[:2]
    no_dist = np.zeros(5)
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K, no_dist, K, no_dist, (w, h), R, t * BASELINE_CM)
    rect1 = cv2.remap(frame1, *cv2.initUndistortRectifyMap(K, no_dist, R1, P1, (w, h), cv2.CV_32FC1), cv2.INTER_LINEAR)
    rect2 = cv2.remap(frame2, *cv2.initUndistortRectifyMap(K, no_dist, R2, P2, (w, h), cv2.CV_32FC1), cv2.INTER_LINEAR)

    # P2[0, 3] = -f * B: negative when camera 2 is to the right. If the camera moved left, swap the
    # roles so the matcher always sees a (left, right) pair and disparities stay positive.
    left, right = (rect1, rect2) if P2[0, 3] < 0 else (rect2, rect1)
    left_gray, right_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in (left, right))
    disparity = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16  # SGBM returns fixed-point (x16)

    focal = P1[0, 0]
    depth = np.full_like(disparity, np.nan)
    valid = disparity > 0
    depth[valid] = focal * BASELINE_CM / disparity[valid]
    return left, depth, P1


def process_scene(video: Path, cal: Calibration, out_path: Path) -> None:
    """Write a side-by-side video (rectified frame | depth heatmap with a cm colour bar).

    Also writes a point cloud (.ply, same name) of the first usable pair from the middle of the video.
    """
    frames = read_video(video)
    # Frames are NOT undistorted: our D has huge k2/k3 (the board never reached the image corners, so
    # they are overfit) and undistorting made pose estimation clearly worse. Lens distortion at this
    # resolution is small, so we treat the camera as a pinhole with K only.
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    h, w = grays[0].shape

    fig, (ax_img, ax_depth) = plt.subplots(1, 2, figsize=(2 * w / 100 + 1.5, h / 100 + 0.6), dpi=100)
    img_artist = ax_img.imshow(np.zeros((h, w, 3), np.uint8))
    cmap = plt.get_cmap("turbo_r").with_extremes(bad="black")  # red = near, blue = far, black = no depth
    depth_artist = ax_depth.imshow(np.full((h, w), np.nan), cmap=cmap, vmin=0, vmax=DEPTH_MAX_CM)
    fig.colorbar(depth_artist, ax=ax_depth, label="Depth (cm)")
    ax_img.set_title("Rectified frame")
    ax_depth.set_title("Depth")
    for ax in (ax_img, ax_depth):
        ax.axis("off")
    fig.tight_layout()

    fig.canvas.draw()
    fw, fh = fig.canvas.get_width_height()
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (fw, fh))

    written = 0
    ply_path: Path | None = out_path.with_suffix(".ply")
    for i in range(len(grays) - FRAME_GAP):
        pose = relative_pose(grays[i], grays[i + FRAME_GAP], cal.K)
        if pose is None:
            continue
        rect, depth, P = depth_map(frames[i], frames[i + FRAME_GAP], cal.K, *pose)
        if ply_path is not None and i >= len(grays) // 2:
            n_points = write_ply(ply_path, rect, depth, P, DEPTH_MAX_CM)
            print(f"  {ply_path.name}: {n_points} points from frame {i}")
            ply_path = None
        img_artist.set_data(cv2.cvtColor(rect, cv2.COLOR_BGR2RGB))
        depth_artist.set_data(depth)
        fig.canvas.draw()
        writer.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR))
        written += 1

    writer.release()
    plt.close(fig)
    print(f"  {video.name} -> {out_path.name}: {written}/{len(grays) - FRAME_GAP} frame pairs")
