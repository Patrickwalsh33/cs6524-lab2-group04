import glob
import os

from mypylibs.frames import extract_calibration_frames

ROOT_IN = 'in'
ROOT_OUT = 'out'


def process_camera(cam_id: str):
    calib_video = os.path.join(ROOT_IN, f'{cam_id}_calibration.mp4')

    # Step 1: extract calibration frames
    frames = extract_calibration_frames(
        calib_video, n_frames=20,
        save_dir=os.path.join(ROOT_OUT, 'frames', cam_id))


if __name__ == '__main__':
    os.makedirs(ROOT_OUT, exist_ok=True)
    for video in sorted(glob.glob(os.path.join(ROOT_IN, '*_calibration.mp4'))):
        cam_id = os.path.basename(video).split('_')[0]
        process_camera(cam_id)
