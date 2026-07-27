"""Plot marker distance history from `marker_distance_history.csv`.

Usage:
    python -m apps.plot_marker_distance_history <history_csv_or_folder> [options]

The input file is expected to contain:
    frame_num, reference_marker_id, marker_id, distance_m

If a folder is provided, the script looks for:
    <folder>/marker_distance_history.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _marker_color(marker_id: int) -> Tuple[int, int, int]:
    """Stable pseudo-random BGR color per marker."""
    return (
        (37 * marker_id + 80) % 256,
        (67 * marker_id + 120) % 256,
        (97 * marker_id + 160) % 256,
    )


def _resolve_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_dir():
        path = path / 'marker_distance_history.csv'
    return path


def _read_history(csv_path: Path) -> Tuple[Dict[int, List[Tuple[int, float]]], Optional[int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f'History file not found: {csv_path}')

    series: Dict[int, List[Tuple[int, float]]] = {}
    reference_marker_id = None

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        required = {'frame_num', 'reference_marker_id', 'marker_id', 'distance_m'}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f'CSV is missing required columns: {", ".join(sorted(missing))}'
            )

        for row in reader:
            frame_num = int(row['frame_num'])
            marker_id = int(row['marker_id'])
            ref_raw = row['reference_marker_id'].strip()
            if ref_raw:
                reference_marker_id = int(ref_raw)
            dist_raw = row['distance_m'].strip()
            if not dist_raw:
                continue
            distance = float(dist_raw)
            if np.isnan(distance):
                continue
            series.setdefault(marker_id, []).append((frame_num, distance))

    for marker_id in series:
        series[marker_id].sort(key=lambda p: p[0])

    return series, reference_marker_id


def _draw_plot(
    series: Dict[int, List[Tuple[int, float]]],
    reference_marker_id: Optional[int],
    width: int = 1400,
    height: int = 800,
) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    if not series:
        cv2.putText(img, 'No distance data found', (60, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img

    pad_l, pad_r, pad_t, pad_b = 90, 30, 60, 90
    x0, x1 = pad_l, width - pad_r
    y0, y1 = height - pad_b, pad_t

    all_frames = [fn for points in series.values() for fn, _ in points]
    all_dists = [d for points in series.values() for _, d in points]
    frame_min = min(all_frames)
    frame_max = max(all_frames)
    dist_min = 0.0
    dist_max = max(all_dists) if all_dists else 1.0
    if dist_max <= dist_min:
        dist_max = dist_min + 1.0

    cv2.rectangle(img, (x0, y1), (x1, y0), (220, 220, 220), 1)
    title = 'Marker distance history'
    if reference_marker_id is not None:
        title += f' (ref=m{reference_marker_id})'
    cv2.putText(img, title, (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    def to_xy(frame_num: int, dist: float) -> Tuple[int, int]:
        xf = (frame_num - frame_min) / max(1, frame_max - frame_min)
        yf = (dist - dist_min) / max(1e-12, dist_max - dist_min)
        x = x0 + int(xf * (x1 - x0))
        y = y0 - int(yf * (y0 - y1))
        return x, y

    # Axes labels
    cv2.putText(img, f'frame {frame_min}', (x0 - 10, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'frame {frame_max}', (x1 - 80, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, '0.0 m', (20, y0 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, f'{dist_max:.3f} m', (15, y1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, 'distance (m)', (15, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, 'frame number', (x0 + 10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Ticks
    for frac in np.linspace(0.0, 1.0, 5):
        y = y0 - int(frac * (y0 - y1))
        val = dist_min + frac * (dist_max - dist_min)
        cv2.line(img, (x0 - 4, y), (x0, y), (0, 0, 0), 1)
        cv2.putText(img, f'{val:.2f}', (20, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    for frac in np.linspace(0.0, 1.0, 5):
        x = x0 + int(frac * (x1 - x0))
        val = frame_min + frac * (frame_max - frame_min)
        cv2.line(img, (x, y0), (x, y0 + 4), (0, 0, 0), 1)
        cv2.putText(img, f'{int(round(val))}', (x - 10, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Plot each marker as a line strip.
    legend_items = []
    for marker_id, points in sorted(series.items()):
        color = _marker_color(marker_id)
        if len(points) >= 2:
            poly = np.array([to_xy(fn, dist) for fn, dist in points], dtype=np.int32)
            cv2.polylines(img, [poly], False, color, 2)
        for fn, dist in points:
            x, y = to_xy(fn, dist)
            cv2.circle(img, (x, y), 3, color, -1)
        legend_items.append((marker_id, color))

    # Legend
    legend_x = x1 - 220
    legend_y = 52
    cv2.rectangle(img, (legend_x - 10, legend_y - 20), (x1 - 10, legend_y + 24 * len(legend_items) + 10),
                  (245, 245, 245), -1)
    cv2.rectangle(img, (legend_x - 10, legend_y - 20), (x1 - 10, legend_y + 24 * len(legend_items) + 10),
                  (180, 180, 180), 1)
    cv2.putText(img, 'legend', (legend_x, legend_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for idx, (marker_id, color) in enumerate(legend_items):
        y = legend_y + 22 + idx * 22
        cv2.line(img, (legend_x, y), (legend_x + 18, y), color, 2)
        cv2.putText(img, f'm{marker_id}', (legend_x + 24, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Plot marker distance history vs frame number.'
    )
    parser.add_argument(
        'input',
        help='Path to marker_distance_history.csv or the dataset folder containing it',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output image path (default: <input>.png or <folder>/marker_distance_history.png)',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot in a window',
    )
    args = parser.parse_args()

    csv_path = _resolve_input_path(args.input)
    series, reference_marker_id = _read_history(csv_path)
    img = _draw_plot(series, reference_marker_id)

    output_path = Path(args.output) if args.output else csv_path.with_suffix('.png')
    cv2.imwrite(str(output_path), img)
    print(f'Wrote plot to: {output_path}')

    if args.show:
        cv2.imshow('Marker Distance History', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
