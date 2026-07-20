"""find_solution – initialise and optimise a multi-camera solution.

Usage:
    python -m apps.find_solution <folder_path> <marker_size> [options]

Options:
    -subseqs              only use frames specified in subseqs.txt
    -exclude-cams N …     exclude camera indices from initialisation
    -with-huber           use Huber robust loss in optimisation
    -thresh T             IPPE ambiguity threshold (default: 2.0)
    -tracking-only        skip camera/marker pose optimisation

Mirrors C++ find_solution app.
"""

import argparse
import sys
import time

from automatic_ar.cam_config import CamConfig
from automatic_ar.initializer import Initializer
from automatic_ar.multicam_mapper import MultiCamMapper


def build_solution_name(args, excluded_cams, with_huber, use_subseqs,
                        tracking_only, threshold, set_threshold) -> str:
    name = ''
    if tracking_only:
        name += '_tracking_only'
    if use_subseqs:
        name += '_subseqs'
    if with_huber:
        name += '_with_huber'
    if excluded_cams:
        name += '_excluded_cams'
        for cid in sorted(excluded_cams):
            name += f'_{cid}'
    if set_threshold:
        name += f'_thresh_{threshold:.1f}'
    return name + '.solution'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Initialise and optimise a multi-camera AR solution.'
    )
    parser.add_argument('folder',      help='Path to dataset folder')
    parser.add_argument('marker_size', type=float,
                        help='Physical marker size in metres')
    parser.add_argument('-subseqs',    action='store_true',
                        help='Restrict to subsequences defined in subseqs.txt')
    parser.add_argument('-exclude-cams', dest='exclude_cams', nargs='*', type=int,
                        default=[], metavar='CAM')
    parser.add_argument('-with-huber', dest='with_huber', action='store_true')
    parser.add_argument('-thresh',     dest='thresh', type=float, default=None)
    parser.add_argument(
        '-tracking-only', dest='tracking_only', action='store_true')
    args = parser.parse_args()

    folder_path = args.folder
    marker_size = args.marker_size
    excluded_cams = set(args.exclude_cams or [])
    use_subseqs = args.subseqs
    with_huber = args.with_huber
    set_threshold = args.thresh is not None
    threshold = args.thresh if set_threshold else 2.0
    tracking_only = args.tracking_only

    solution_suffix = build_solution_name(
        args, excluded_cams, with_huber, use_subseqs,
        tracking_only, threshold, set_threshold
    )
    initial_path = folder_path + '/initial' + solution_suffix
    final_path = folder_path + '/final' + solution_suffix

    detections_path = folder_path + '/aruco.detections'
    cam_configs = CamConfig.read_cam_configs(folder_path)

    subseqs = None
    if use_subseqs:
        subseqs = MultiCamMapper.read_subseqs(folder_path + '/subseqs.txt')
        print(f'Using subsequences: {subseqs[0]} ... {subseqs[-1]}')

    print('Reading detections …')
    detections = Initializer.read_detections_file(detections_path, subseqs)

    print('Initialising poses …')
    t0 = time.perf_counter()
    initializer = Initializer(
        marker_size=marker_size,
        cam_configs=cam_configs,
        excluded_cams=excluded_cams,
        detections=detections,
        threshold=threshold,
    )

    mcm = MultiCamMapper.from_initializer(initializer)
    mcm.set_optmize_flag_cam_intrinsics(True)
    if with_huber:
        mcm.set_with_huber(True)
    if tracking_only:
        mcm.set_optmize_flag_cam_poses(False)
        mcm.set_optmize_flag_marker_poses(False)

    init_time = time.perf_counter() - t0
    print(f'Initialisation took {init_time:.2f}s')

    print('Writing initial solution …')
    mcm.write_solution_file(initial_path)
    mcm.write_text_solution_file(initial_path + '.yaml')

    print('Running optimisation …')
    t0 = time.perf_counter()
    mcm.solve()
    solve_time = time.perf_counter() - t0

    minutes = int(solve_time // 60)
    seconds = round(solve_time - minutes * 60)
    print(f'The algorithm took: {minutes} minutes {seconds} seconds')

    print('Writing final solution …')
    mcm.write_solution_file(final_path)
    mcm.write_text_solution_file(final_path + '.yaml')
    print('Done.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
