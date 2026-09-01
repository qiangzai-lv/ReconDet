"""Export sampled ScanNet views with reconstruction keypoint annotations."""

import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

LOGGER = logging.getLogger('scannet_3d_to_coco_keypoints')
FACE_NORMALS = np.array([
    (-1., 0., 0.), (1., 0., 0.), (0., -1., 0.),
    (0., 1., 0.), (0., 0., -1.), (0., 0., 1.)
], dtype=np.float32)


def _sample_indices(n, count, mode, rng):
    if n <= 0 or count <= 0:
        raise ValueError('scene and --num-views must be positive')
    if mode == 'random':
        return rng.choice(n, count, replace=count > n)
    return np.rint(np.linspace(0, n - 1, count)).astype(np.int64)


def _load_points(root, info, axis):
    rel = info.get('aligned_pts_path')
    if rel is None:
        point_info = info.get('lidar_points', {})
        point_name = point_info.get('lidar_path', info.get('pts_path'))
        if point_name is None:
            raise KeyError('Info entry has neither aligned_pts_path nor point path')
        rel = str(Path('points') / point_name) if not str(point_name).startswith('points') else str(point_name)
        raw = np.fromfile(root / rel, np.float32)
        dim = point_info.get('num_pts_feats', 6)
        raw = raw.reshape(-1, dim)
        raw[:, :3] = raw[:, :3] @ axis[:3, :3].T + axis[:3, 3]
        return raw[:, :3]
    dim = info.get('lidar_points', {}).get('num_pts_feats', 6)
    return np.fromfile(root / rel, np.float32).reshape(-1, dim)[:, :3]


def _surface_points(points, center, size):
    lower, upper = center - size / 2., center + size / 2.
    inside = np.all((points >= lower) & (points <= upper), axis=1)
    candidates = points[inside]
    if len(candidates) == 0:
        return center[None] + FACE_NORMALS * (size[None] / 2.)
    face_centers = center[None] + FACE_NORMALS * (size[None] / 2.)
    distances = np.abs(candidates[:, None, :] - face_centers[None, :, :]).sum(axis=2)
    return candidates[distances.argmin(axis=0)]


def _project(points, extrinsic, intrinsic, width, height):
    camera = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    depth = camera[:, 2]
    pixels = camera @ intrinsic[:3, :3].T
    pixels = pixels[:, :2] / np.maximum(depth[:, None], 1e-5)
    return pixels, depth


def convert(args):
    root = Path(args.data_root).resolve()
    ann_path = Path(args.ann_file)
    if not ann_path.is_absolute():
        ann_path = root / ann_path
    with ann_path.open('rb') as handle:
        payload = pickle.load(handle)
    data_list = payload['data_list']
    if args.max_scenes > 0:
        data_list = data_list[:args.max_scenes]
    categories = sorted(payload['metainfo']['categories'].items(), key=lambda x: x[1])
    rng = np.random.default_rng(args.seed)
    coco = {'info': {'description': 'ScanNet reconstruction keypoints'},
            'licenses': [], 'images': [], 'annotations': [],
            'categories': [{'id': i + 1, 'name': n, 'supercategory': 'object'}
                           for n, i in categories]}
    stats = Counter(); image_id = annotation_id = 1
    for scene_index, info in enumerate(data_list):
        paths = [Path(p) for p in info['img_paths']]
        indices = _sample_indices(len(paths), args.num_views, args.sampling, rng)
        axis = np.asarray(info['axis_align_matrix'], dtype=np.float32)
        points = _load_points(root, info, axis)
        face_points = []
        object_centers = []
        object_labels = []
        for obj in info.get('instances', []):
            box = np.asarray(obj['bbox_3d'][:6], dtype=np.float32)
            face_points.append(_surface_points(points, box[:3], box[3:6]))
            object_centers.append(box[:3])
            object_labels.append(int(obj['bbox_label_3d']))
        scene_id = str(info.get('scene_id') or paths[0].parent.name)
        for view_pos, index in enumerate(indices):
            source = paths[int(index)]
            source_abs = source if source.is_absolute() else root / source
            with Image.open(source_abs) as image:
                width, height = image.size
            pose = np.asarray(info['lidar2cam'][int(index)], dtype=np.float32)
            extrinsic = np.linalg.inv(axis @ pose)
            camera_center = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
            image_record = {'id': image_id, 'file_name': str(source),
                            'width': width, 'height': height,
                            'scene_id': scene_id, 'view_index': int(index)}
            coco['images'].append(image_record)
            for points6, object_center, label in zip(
                    face_points, object_centers, object_labels):
                face_scores = (FACE_NORMALS * (camera_center - object_center)).sum(axis=1)
                all_face_2d, face_depth = _project(
                    points6, extrinsic, np.asarray(info['cam2img']), width, height)
                face_visible = face_depth > 1e-5
                face_visible &= np.isfinite(all_face_2d).all(axis=1)
                face_visible &= (all_face_2d[:, 0] >= 0) & (all_face_2d[:, 0] < width)
                face_visible &= (all_face_2d[:, 1] >= 0) & (all_face_2d[:, 1] < height)
                selected = np.flatnonzero(face_visible)
                selected = selected[np.argsort(-face_scores[selected])]
                if len(selected) < 3:
                    # Keep the four-point contract for edge views. Remaining
                    # faces are the best camera-facing points, even if clipped.
                    fallback = np.argsort(-face_scores)
                    selected = np.concatenate([selected,
                                               fallback[~np.isin(fallback, selected)]])
                selected = selected[:3]
                key3d = np.concatenate([object_center[None], points6[selected]], axis=0)
                key2d, depth = _project(key3d, extrinsic, np.asarray(info['cam2img']), width, height)
                center_visible = (depth[0] > 1e-5 and
                                  np.isfinite(key2d[0]).all() and
                                  0 <= key2d[0, 0] < width and
                                  0 <= key2d[0, 1] < height)
                if not center_visible:
                    stats['center_invisible'] += 1
                    continue
                visibility = np.zeros(4, dtype=np.int64)
                visibility[0] = 1
                face_visible = (depth[1:] > 1e-5)
                face_visible &= np.isfinite(key2d[1:]).all(axis=1)
                face_visible &= (key2d[1:, 0] >= 0) & (key2d[1:, 0] < width)
                face_visible &= (key2d[1:, 1] >= 0) & (key2d[1:, 1] < height)
                visibility[1:] = face_visible.astype(np.int64)
                flat = []
                for (x, y), visible in zip(key2d, visibility):
                    flat.extend([float(x), float(y), int(visible)])
                coco['annotations'].append({
                    'id': annotation_id, 'image_id': image_id,
                    'category_id': label + 1, 'keypoints': flat,
                    'keypoints_2d': key2d.reshape(-1).astype(float).tolist(),
                    'keypoints_visibility': visibility.tolist(),
                    'keypoints_3d': key3d.reshape(-1).astype(float).tolist(),
                    'keypoint_faces': selected.astype(int).tolist(),
                    'num_keypoints': 4,
                })
                annotation_id += 1; stats['objects'] += 1
            image_id += 1; stats['images'] += 1
        if (scene_index + 1) % args.log_interval == 0 or scene_index + 1 == len(data_list):
            LOGGER.info('Progress %d/%d scenes: %d images, %d objects',
                        scene_index + 1, len(data_list), stats['images'], stats['objects'])
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2))
    LOGGER.info('Wrote %d images and %d keypoint annotations to %s',
                len(coco['images']), len(coco['annotations']), output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True); p.add_argument('--ann-file', required=True)
    p.add_argument('--output', required=True); p.add_argument('--num-views', required=True, type=int)
    p.add_argument('--sampling', choices=('random', 'uniform'), default='uniform')
    p.add_argument('--seed', type=int, default=0); p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--max-scenes', type=int, default=-1,
                   help='Limit scenes for a smoke test; <=0 means all')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    convert(args)


if __name__ == '__main__':
    main()
