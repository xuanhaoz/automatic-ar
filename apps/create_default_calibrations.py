"""create_default_calibrations – Create default camera calibration files.

Usage:
    python -m apps.create_default_calibrations <dataset_folder> [--fov DEGREES]

Options:
    --fov DEGREES    Field of view in degrees (default: 60)

This script creates default camera calibration files (calib.xml) for datasets
that don't have them. The calibrations can then be optimized from marker
observations using find_solution with intrinsics optimization enabled.

Example:
    python -m apps.create_default_calibrations data/
    python -m apps.create_default_calibrations data/ --fov 45
"""

import argparse
import cv2
import os
import sys
from pathlib import Path

import numpy as np

from automatic_ar.cam_config import CamConfig


def create_calibrations_for_dataset(dataset_path: str, fov_degrees: float = 60.0) -> int:
    """Create default calib.xml files for all cameras in dataset.
    
    Args:
        dataset_path: Path to dataset folder
        fov_degrees: Field of view in degrees
    
    Returns:
        0 on success, 1 on failure
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.is_dir():
        print(f'Error: Dataset folder not found: {dataset_path}')
        return 1
    
    # Find all camera directories (numbered subdirectories)
    camera_dirs = sorted([
        d for d in dataset_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    ], key=lambda d: int(d.name))
    
    if not camera_dirs:
        print('Error: No camera directories found (expected: 0/, 1/, etc.)')
        return 1
    
    print(f'Found {len(camera_dirs)} camera directory(ies)')
    
    # Detect image size from first available image
    image_size = None
    for cam_dir in camera_dirs:
        for ext in ['*.jpg', '*.png', '*.JPG', '*.PNG']:
            images = list(cam_dir.glob(ext))
            if images:
                try:
                    img = cv2.imread(str(images[0]))
                    if img is not None:
                        h, w = img.shape[:2]
                        image_size = (w, h)
                        print(f'Detected image size: {image_size[0]}×{image_size[1]} from {images[0].name}')
                        break
                except Exception as e:
                    print(f'Warning: Could not read {images[0]}: {e}')
        
        if image_size:
            break
    
    if not image_size:
        print('Error: Could not detect image size from dataset')
        print('       Please ensure at least one image file exists (.jpg or .png)')
        return 1
    
    # Create calibration file for each camera
    created_count = 0
    for cam_idx, cam_dir in enumerate(camera_dirs):
        calib_path = cam_dir / 'calib.xml'
        
        if calib_path.exists():
            print(f'  Camera {cam_idx}: calib.xml already exists, skipping')
            continue
        
        try:
            # Create default config
            config = CamConfig.create_default_config(image_size, fov_degrees)
            
            # Write to file
            storage = cv2.FileStorage(str(calib_path), cv2.FileStorage_WRITE)
            storage.write('image_height', image_size[1])
            storage.write('image_width', image_size[0])
            storage.write('camera_matrix', config.cam_mat)
            storage.write('distortion_coefficients', config.dist_coeffs)
            storage.release()
            
            fx = config.cam_mat[0, 0]
            print(f'  Camera {cam_idx}: Created calib.xml with fx={fx:.1f}px (FOV={fov_degrees}°)')
            created_count += 1
            
        except Exception as e:
            print(f'  Camera {cam_idx}: Error creating calib.xml: {e}')
            return 1
    
    print(f'\n✓ Created {created_count} calibration file(s)')
    print(f'\nNext steps:')
    print(f'  1. Run find_solution to initialize and optimize:')
    print(f'     python -m apps.find_solution {dataset_path} <marker_size>')
    print(f'  2. Check calibration_optimization_report.txt for calibration changes')
    print(f'\nNote: The created calibrations are estimates with ~{fov_degrees}° FOV.')
    print(f'      The actual FOV will be refined during optimization.')
    
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Create default camera calibration files for a dataset.'
    )
    parser.add_argument('dataset', help='Path to dataset folder')
    parser.add_argument('--fov', dest='fov', type=float, default=60.0,
                       help='Field of view in degrees (default: 60)')
    args = parser.parse_args()
    
    return create_calibrations_for_dataset(args.dataset, args.fov)


if __name__ == '__main__':
    sys.exit(main())
