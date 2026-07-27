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
import numpy as np

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
    
    # If no calibration files found, use defaults and track them
    used_default_calibrations = False
    if not cam_configs:
        print('⚠ No camera calibration files (calib.xml) found.')
        print('  Creating default camera configurations with ~60° FOV.')
        print('  These will be optimized from marker observations.')
        
        # Count cameras by looking at directory structure
        import os
        cam_dirs = sorted([
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d)) and d.isdigit()
        ], key=int)
        num_cameras = len(cam_dirs)
        
        if num_cameras == 0:
            print('Error: No camera directories found (expected: 0/, 1/, etc.)')
            return 1
        
        cam_configs = CamConfig.create_default_configs(
            folder_path=folder_path,
            num_cameras=num_cameras,
            fov_degrees=60.0
        )
        used_default_calibrations = True
        print(f'  Created default configs for {num_cameras} camera(s)')
    
    # Store initial calibration values for later comparison
    initial_calibrations = {}
    for cam_id, cam_cfg in enumerate(cam_configs):
        initial_calibrations[cam_id] = {
            'K': cam_cfg.cam_mat.copy(),
            'D': cam_cfg.dist_coeffs.copy(),
        }

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
    
    # Save optimized camera calibrations back to calib.yaml files
    print('Saving optimized camera calibrations …')
    _save_optimized_calibrations(mcm, folder_path)
    
    # Compare calibrations if we used defaults
    if used_default_calibrations:
        print('\n' + '='*70)
        print('CAMERA CALIBRATION ESTIMATION vs. OPTIMISATION RESULTS')
        print('='*70)
        _report_calibration_changes(mcm, initial_calibrations, folder_path)
    
    print('Done.')
    return 0


def _save_optimized_calibrations(mcm, folder_path) -> None:
    """Save optimized camera calibrations back to calib.yaml files.
    
    Args:
        mcm: Optimized MultiCamMapper object
        folder_path: Path to dataset folder
    """
    import os
    from pathlib import Path
    import numpy as np
    
    final_mat_arrays = mcm.get_mat_arrays()
    final_K_dict = final_mat_arrays.get('cam_mats', {})
    final_D_dict = final_mat_arrays.get('dist_coeffs', {})
    
    if not final_K_dict:
        print('Warning: Could not retrieve optimized calibrations')
        return
    
    # Get the original cam_configs to get image sizes
    cam_configs = CamConfig.read_cam_configs(folder_path)
    
    # Save each camera's optimized calibration
    saved_count = 0
    for cam_id in sorted(final_K_dict.keys()):
        if cam_id < len(cam_configs):
            image_size = cam_configs[cam_id].image_size
        else:
            # Fallback: try to detect from images
            image_size = (1280, 720)
        
        final_K = final_K_dict[cam_id]
        final_D = final_D_dict.get(cam_id, np.zeros(5, dtype=np.float64))
        
        # Create CamConfig object with optimized values
        optimized_config = CamConfig(final_K, final_D, image_size)
        
        # Save to calib.yaml in camera folder
        cam_dir = Path(folder_path) / str(cam_id)
        cam_dir.mkdir(parents=True, exist_ok=True)
        calib_path = cam_dir / 'calib.yaml'
        
        if optimized_config.to_file(str(calib_path)):
            print(f'  ✓ Camera {cam_id}: saved to {calib_path}')
            saved_count += 1
        else:
            print(f'  ✗ Camera {cam_id}: failed to save')
    
    print(f'Saved optimized calibrations for {saved_count} camera(s)')


def _report_calibration_changes(mcm, initial_calibrations, folder_path):
    """Report differences between initial and final camera calibrations.
    
    Args:
        mcm: Optimized MultiCamMapper object
        initial_calibrations: Dict of initial K and D matrices
        folder_path: Path to dataset folder
    """
    import os
    from pathlib import Path
    
    final_mat_arrays = mcm.get_mat_arrays()
    final_K_dict = final_mat_arrays.get('cam_mats', {})
    final_D_dict = final_mat_arrays.get('dist_coeffs', {})
    
    if not initial_calibrations or not final_K_dict:
        print('Could not compare calibrations (missing data)')
        return
    
    # Create output file for calibration comparison
    comparison_path = os.path.join(folder_path, 'calibration_optimization_report.txt')
    
    with open(comparison_path, 'w') as f:
        f.write('CAMERA CALIBRATION OPTIMIZATION REPORT\n')
        f.write('=' * 80 + '\n\n')
        f.write('This report shows how initial estimated calibrations changed after\n')
        f.write('bundle adjustment optimization using marker observations.\n\n')
        
        # Process each camera
        for cam_id in sorted(initial_calibrations.keys()):
            if cam_id not in final_K_dict:
                continue
            
            init_K = initial_calibrations[cam_id]['K']
            init_D = initial_calibrations[cam_id]['D']
            final_K = final_K_dict[cam_id]
            final_D = final_D_dict[cam_id]
            
            fx_init, cx_init = init_K[0, 0], init_K[0, 2]
            fy_init, cy_init = init_K[1, 1], init_K[1, 2]
            
            fx_final, cx_final = final_K[0, 0], final_K[0, 2]
            fy_final, cy_final = final_K[1, 1], final_K[1, 2]
            
            f.write(f'\nCAMERA {cam_id}\n')
            f.write('-' * 80 + '\n')
            
            # Focal length
            fx_change = ((fx_final - fx_init) / fx_init * 100) if fx_init != 0 else 0
            fy_change = ((fy_final - fy_init) / fy_init * 100) if fy_init != 0 else 0
            
            f.write(f'Focal Length X (fx):\n')
            f.write(f'  Initial:  {fx_init:12.4f} pixels\n')
            f.write(f'  Final:    {fx_final:12.4f} pixels\n')
            f.write(f'  Change:   {fx_final - fx_init:+12.4f} pixels ({fx_change:+.2f}%)\n\n')
            
            f.write(f'Focal Length Y (fy):\n')
            f.write(f'  Initial:  {fy_init:12.4f} pixels\n')
            f.write(f'  Final:    {fy_final:12.4f} pixels\n')
            f.write(f'  Change:   {fy_final - fy_init:+12.4f} pixels ({fy_change:+.2f}%)\n\n')
            
            # Principal point
            cx_change = abs(cx_final - cx_init)
            cy_change = abs(cy_final - cy_init)
            
            f.write(f'Principal Point X (cx):\n')
            f.write(f'  Initial:  {cx_init:12.4f} pixels\n')
            f.write(f'  Final:    {cx_final:12.4f} pixels\n')
            f.write(f'  Change:   {cx_final - cx_init:+12.4f} pixels\n\n')
            
            f.write(f'Principal Point Y (cy):\n')
            f.write(f'  Initial:  {cy_init:12.4f} pixels\n')
            f.write(f'  Final:    {cy_final:12.4f} pixels\n')
            f.write(f'  Change:   {cy_final - cy_init:+12.4f} pixels\n\n')
            
            # Distortion coefficients
            f.write(f'Distortion Coefficients (k1, k2, p1, p2, k3):\n')
            f.write(f'  Initial:  {init_D[0]:+.6e} {init_D[1]:+.6e} {init_D[2]:+.6e} {init_D[3]:+.6e} {init_D[4]:+.6e}\n')
            f.write(f'  Final:    {final_D[0]:+.6e} {final_D[1]:+.6e} {final_D[2]:+.6e} {final_D[3]:+.6e} {final_D[4]:+.6e}\n')
            f.write(f'  Changes:  {final_D[0]-init_D[0]:+.6e} {final_D[1]-init_D[1]:+.6e} {final_D[2]-init_D[2]:+.6e} {final_D[3]-init_D[3]:+.6e} {final_D[4]-init_D[4]:+.6e}\n')
            
            # Summary
            avg_focal_change = abs(fx_change + fy_change) / 2
            f.write(f'\nSUMMARY:\n')
            f.write(f'  Average focal length change: {avg_focal_change:.2f}%\n')
            f.write(f'  Principal point shift: ({cx_change:.2f}, {cy_change:.2f}) pixels\n')
            f.write(f'  Distortion optimized: Yes\n')
        
        f.write('\n' + '=' * 80 + '\n')
        f.write('INTERPRETATION:\n')
        f.write('- Small changes (<5%) in focal length suggest initial estimate was good\n')
        f.write('- Large changes (>10%) suggest either poor initial estimate or\n')
        f.write('  significant camera differences from assumed 60° FOV\n')
        f.write('- Principal point shifts typically <50 pixels for well-centered cameras\n')
        f.write('- Non-zero distortion indicates actual lens distortion detected\n')
        f.write('=' * 80 + '\n')
    
    # Print summary to console
    print(f'\n✓ Calibration report saved to: {comparison_path}')
    
    # Also print brief summary to console
    for cam_id in sorted(initial_calibrations.keys()):
        if cam_id not in final_K_dict:
            continue
        
        init_K = initial_calibrations[cam_id]['K']
        final_K = final_K_dict[cam_id]
        
        fx_init, fx_final = init_K[0, 0], final_K[0, 0]
        fx_change = ((fx_final - fx_init) / fx_init * 100) if fx_init != 0 else 0
        
        print(f'  Camera {cam_id}: fx {fx_init:.1f}px → {fx_final:.1f}px ({fx_change:+.2f}%)')


if __name__ == '__main__':
    sys.exit(main())
