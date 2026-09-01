"""Convert multi-view ScanNet 3D boxes to sampled COCO 2D annotations.

Each selected camera view is emitted as one COCO image. Image files are never
copied or moved; ``file_name`` points to the source path (relative to
``--data-root`` by default).
"""

import argparse
import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


LOGGER = logging.getLogger('scannet_3d_to_coco')


def _load_infos(path: Path):
    with path.open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or 'data_list' not in payload:
        raise ValueError(f'Expected MMDet3D info dict at {path}')
    return payload.get('metainfo', {}), payload['data_list']


def _sample_indices(num_views: int, count: int, mode: str, rng):
    if num_views <= 0:
        raise ValueError('A scene must contain at least one image')
    if count <= 0:
        raise ValueError('--num-views must be positive')
    if mode == 'random':
        return rng.choice(num_views, count, replace=count > num_views)
    return np.rint(np.linspace(0, num_views - 1, count)).astype(np.int64)


def _image_size(path: Path):
    with Image.open(path) as image:
        width, height = image.size
    return width, height


def _project_boxes(info, view_indices, image_sizes):
    """Project 3D boxes directly in the original image pixel coordinates."""
    instances = info.get('instances', [])
    if not instances:
        return [([], 0) for _ in view_indices]

    boxes = np.asarray([item['bbox_3d'][:6] for item in instances],
                       dtype=np.float32)
    labels = np.asarray([item['bbox_label_3d'] for item in instances],
                        dtype=np.int64)
    centers = boxes[:, :3]
    dimensions = boxes[:, 3:6]

    axis_align = np.asarray(info.get('axis_align_matrix', np.eye(4)),
                            dtype=np.float32)
    extrinsics = []
    for view_index in view_indices:
        lidar2cam = np.asarray(info['lidar2cam'][int(view_index)],
                               dtype=np.float32)
        extrinsics.append(np.linalg.inv(axis_align @ lidar2cam))
    extrinsics = np.stack(extrinsics)

    intrinsic = np.asarray(info['cam2img'], dtype=np.float32)
    projected = []
    for view_id, (width, height) in enumerate(image_sizes):
        rotation = extrinsics[view_id, :3, :3]
        translation = extrinsics[view_id, :3, 3]
        camera_centers = centers @ rotation.T + translation
        depth = camera_centers[:, 2]
        center_pixels = camera_centers @ intrinsic[:3, :3].T
        center_pixels = center_pixels[:, :2] / np.maximum(depth[:, None], 1e-5)
        visible = depth > 1e-5
        visible &= np.isfinite(center_pixels).all(axis=1)
        visible &= (center_pixels[:, 0] >= 0) & (center_pixels[:, 0] < width)
        visible &= (center_pixels[:, 1] >= 0) & (center_pixels[:, 1] < height)

        corner_signs = np.array([
            (-1., -1., -1.), (-1., -1., 1.), (-1., 1., -1.),
            (-1., 1., 1.), (1., -1., -1.), (1., -1., 1.),
            (1., 1., -1.), (1., 1., 1.)
        ], dtype=np.float32)
        corners = centers[:, None] + dimensions[:, None] * corner_signs[None] * 0.5
        camera_corners = corners @ rotation.T + translation
        corner_depth = camera_corners[..., 2]
        corner_pixels = camera_corners @ intrinsic[:3, :3].T
        corner_pixels = corner_pixels[..., :2] / np.maximum(corner_depth[..., None], 1e-5)
        corner_valid = (corner_depth > 1e-5)
        corner_valid &= np.isfinite(corner_pixels).all(axis=-1)
        pixel_min = np.where(corner_valid[..., None], corner_pixels, np.inf).min(axis=1)
        pixel_max = np.where(corner_valid[..., None], corner_pixels, -np.inf).max(axis=1)
        visible &= corner_valid.any(axis=1)
        pixel_min[:, 0] = np.clip(pixel_min[:, 0], 0, width)
        pixel_min[:, 1] = np.clip(pixel_min[:, 1], 0, height)
        pixel_max[:, 0] = np.clip(pixel_max[:, 0], 0, width)
        pixel_max[:, 1] = np.clip(pixel_max[:, 1], 0, height)
        visible &= pixel_max[:, 0] > pixel_min[:, 0]
        visible &= pixel_max[:, 1] > pixel_min[:, 1]
        pixel = np.concatenate([pixel_min, pixel_max], axis=1)
        view_annotations = []
        for bbox, label, is_valid in zip(pixel, labels, visible):
            if not is_valid:
                continue
            x1, y1, x2, y2 = bbox.tolist()
            x1, y1 = max(0.0, min(x1, width)), max(0.0, min(y1, height))
            x2, y2 = max(0.0, min(x2, width)), max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue
            view_annotations.append((int(label), [x1, y1, x2 - x1, y2 - y1]))
        projected.append((view_annotations, len(instances)))
    return projected


def convert(args):
    info_path = Path(args.ann_file)
    if not info_path.is_absolute():
        info_path = Path(args.data_root) / info_path
    LOGGER.info('Loading annotations from %s', info_path)
    metainfo, data_list = _load_infos(info_path)
    LOGGER.info('Loaded %d scenes from %s', len(data_list), info_path)
    categories = metainfo.get('categories')
    if not categories:
        raise ValueError('The info file does not contain metainfo.categories')
    categories = sorted(categories.items(), key=lambda item: item[1])
    category_names = {int(label): name for name, label in categories}
    rng = np.random.default_rng(args.seed)

    coco = {
        'info': {'description': 'Sampled ScanNet 3D-to-2D projections'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {'id': label + 1, 'name': name, 'supercategory': 'object'}
            for name, label in categories
        ],
    }
    stats = Counter()
    image_id = annotation_id = 1
    data_root = Path(args.data_root).resolve()
    for scene_index, info in enumerate(data_list):
        image_paths = info.get('img_paths', [])
        indices = _sample_indices(len(image_paths), args.num_views,
                                  args.sampling, rng)
        selected_paths = [Path(image_paths[int(index)]) for index in indices]
        resolved_paths = [p if p.is_absolute() else data_root / p
                          for p in selected_paths]
        sizes = [_image_size(path) for path in resolved_paths]
        projected = _project_boxes(info, indices, sizes)
        scene_id = info.get('scene_id')
        if scene_id is None and selected_paths:
            scene_id = selected_paths[0].parent.name
        scene_id = str(scene_id or f'scene_{scene_index:05d}')
        for view_index, (source_path, size, annotations) in enumerate(
                zip(selected_paths, sizes, projected)):
            width, height = size
            file_name = str(source_path if args.path_mode == 'relative' else
                            (data_root / source_path).resolve())
            image_record = {
                'id': image_id,
                'file_name': file_name,
                'width': width,
                'height': height,
                'scene_id': scene_id,
                'view_index': int(indices[view_index]),
            }
            coco['images'].append(image_record)
            view_annotations, total_instances = annotations
            stats['source_instances'] += total_instances
            stats['visible_instances'] += len(view_annotations)
            stats['images'] += 1
            for label, bbox in view_annotations:
                area = float(bbox[2] * bbox[3])
                coco['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': label + 1,
                    'bbox': [float(value) for value in bbox],
                    'area': area,
                    'iscrowd': 0,
                    'segmentation': [],
                })
                annotation_id += 1
            image_id += 1

        if ((scene_index + 1) % args.log_interval == 0 or
                scene_index + 1 == len(data_list)):
            LOGGER.info(
                'Progress %d/%d scenes (%.1f%%): %d images, %d annotations',
                scene_index + 1, len(data_list),
                100.0 * (scene_index + 1) / max(len(data_list), 1),
                stats['images'], len(coco['annotations']))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        json.dump(coco, handle, indent=2)
    audit = output.with_suffix('.audit.json')
    audit.write_text(json.dumps({
        'source': str(info_path),
        'num_scenes': len(data_list),
        'num_views_per_scene': args.num_views,
        'sampling': args.sampling,
        'seed': args.seed,
        'path_mode': args.path_mode,
        'stats': dict(stats),
        'categories': category_names,
    }, indent=2))
    LOGGER.info('Wrote %d images and %d annotations to %s',
                len(coco['images']), len(coco['annotations']), output)
    LOGGER.info('Wrote audit summary to %s', audit)
    LOGGER.info('Projection statistics: %s', dict(stats))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--ann-file', required=True,
                        help='MMDet3D info pkl, absolute or relative to data-root')
    parser.add_argument('--output', required=True, help='Output COCO JSON path')
    parser.add_argument('--num-views', required=True, type=int)
    parser.add_argument('--sampling', choices=('random', 'uniform'),
                        default='uniform')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--path-mode', choices=('relative', 'absolute'),
                        default='relative',
                        help='How to encode file_name in COCO JSON')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Print progress every N scenes (default: 50)')
    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    arguments = parse_args()
    if arguments.log_interval <= 0:
        raise ValueError('--log-interval must be positive')
    LOGGER.info(
        'Starting conversion: num_views=%d, sampling=%s, seed=%d, path_mode=%s',
        arguments.num_views, arguments.sampling, arguments.seed,
        arguments.path_mode)
    convert(arguments)
