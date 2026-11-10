from pathlib import Path

from mypylibs.corners import detect_corners
from mypylibs.frames import extract_calibration_frames

ROOT_IN = Path("in")
ROOT_OUT = Path("out")


def process_camera(cam_id: str) -> None:
    calib_video = ROOT_IN / f"{cam_id}_calibration.mp4"

    # Step 1: extract calibration frames
    frames = extract_calibration_frames(calib_video, n_frames=20, save_dir=ROOT_OUT / "frames" / cam_id)

    # Step 2: detect + sub-pixel refine checkerboard corners (image points + size feed step 3)
    detect_corners(frames, save_dir=ROOT_OUT / "corners" / cam_id)


if __name__ == "__main__":
    ROOT_OUT.mkdir(exist_ok=True)
    for video in sorted(ROOT_IN.glob("*_calibration.mp4")):
        process_camera(video.name.split("_")[0])
