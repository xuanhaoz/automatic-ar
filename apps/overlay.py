"""overlay – render marker outlines onto dataset frames.

Usage:
    python -m apps.overlay <path_to_data_folder> <solution_file_name> [-save-video]

Mirrors C++ overlay app.
"""

import argparse
import sys

import cv2
import numpy as np

from automatic_ar.aruco_serdes import read_detections_file
from automatic_ar.dataset import Dataset
from automatic_ar.multicam_mapper import MultiCamMapper


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Overlay solution marker outlines on dataset frames.'
    )
    parser.add_argument('folder',   help='Path to dataset folder')
    parser.add_argument('solution', help='Solution file name (relative to folder)')
    parser.add_argument('-save-video', dest='save_video', action='store_true',
                        help='Save overlay videos for each camera')
    args = parser.parse_args()

    folder_path       = args.folder
    solution_file     = args.solution
    solution_path     = folder_path + '/' + solution_file
    save_video        = args.save_video

    dataset   = Dataset(folder_path)
    num_cams  = dataset.get_num_cams()
    frame_nums = dataset.get_frame_nums()

    mcm = MultiCamMapper()
    if not mcm.read_solution_file(solution_path):
        print(f'Cannot read solution file: {solution_path}', file=sys.stderr)
        return 1

    detections_path = folder_path + '/aruco.detections'
    detections = read_detections_file(detections_path)

    image_sizes = mcm.get_image_sizes()
    writers = []
    if save_video:
        for i in range(num_cams):
            w, h = image_sizes[i] if i < len(image_sizes) else (640, 480)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out_path = f'{folder_path}/overlayed_cam_{i}.avi'
            vw = cv2.VideoWriter(out_path, fourcc, 24.0, (w, h))
            writers.append(vw)

    for frame_index, frame_num in enumerate(frame_nums):
        frames = dataset.get_frame(frame_num)

        # Markers from detections for this frame
        frame_det = detections[frame_index] if frame_index < len(detections) else []

        for cam in range(num_cams):
            img = frames[cam]
            if img is None:
                continue

            # Draw solution overlay
            img = mcm.overlay_markers(img, frame_index, cam)

            # Draw raw detections
            if cam < len(frame_det):
                for mid, corners in frame_det[cam]:
                    pts = corners.reshape(4, 2).astype(int)
                    for j in range(4):
                        cv2.line(img, tuple(pts[j]), tuple(pts[(j + 1) % 4]),
                                 (255, 0, 0), 1)

            cv2.imshow(str(cam), img)
            if writers and cam < len(writers):
                writers[cam].write(img)

        cv2.waitKey(1)

    cv2.destroyAllWindows()
    for vw in writers:
        vw.release()

    return 0


if __name__ == '__main__':
    sys.exit(main())
