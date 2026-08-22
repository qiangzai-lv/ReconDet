#!/usr/bin/env python3
"""Estimate and cache VGGT-to-ScanNet scale factors per scene.

The scale is the ratio of robust whole-scene point-cloud extents.  The GT
cloud is axis-aligned with ScanNet's matrix before its extent is measured;
VGGT points remain in VGGT coordinates because scale is rotation/translation
invariant.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import mmengine
import numpy as np
import torch
import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.visualize_vggt_pointcloud import (  # noqa: E402
    align_gt_points,
    estimate_scene_scale,
    load_gt_points,
    load_images,
    load_vggt,
    point_cloud_range,
    reconstruct_vggt,
    select_views,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Cache VGGT-to-ScanNet scene scale factors.')
    parser.add_argument(
        '--config', default='configs/viewbbox/vggt_scale.py',
        help='Python configuration file.')
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Recompute scenes already present in the output file.')
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Process at most this many selected scenes.')
    parser.add_argument(
        '--scene-id', default=None,
        help='Process only this scene id.')
    return parser.parse_args()


def load_config(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    spec = importlib.util.spec_from_file_location('vggt_scale_config', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load config: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_id(item):
    return Path(item['lidar_points']['lidar_path']).stem


def config_signature(config):
    return dict(
        data_root=str(config.data_root), checkpoint=str(config.checkpoint),
        num_views=int(config.num_views), image_width=int(config.image_width),
        image_height=int(config.image_height),
        point_stride=int(config.point_stride),
        max_points=int(config.max_points), max_depth=float(config.max_depth))


def main():
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    config = load_config(args.config)
    data_root = Path(config.data_root)
    ann_files = config.ann_files
    if isinstance(ann_files, str):
        ann_files = (ann_files,)
    selected = []
    seen = set()
    for ann_file in ann_files:
        annotation = mmengine.load(data_root / ann_file)
        for item in annotation['data_list']:
            current_id = scene_id(item)
            if current_id in seen:
                continue
            if args.scene_id is not None and current_id != args.scene_id:
                continue
            selected.append((current_id, item, ann_file))
            seen.add(current_id)
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError('No scenes selected')

    output_file = Path(config.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    signature = config_signature(config)
    if output_file.is_file():
        cache = mmengine.load(output_file)
    else:
        cache = dict(version=1, config=str(Path(args.config)), scenes={})
    if cache.get('config_values') != signature:
        cache = dict(version=1, config=str(Path(args.config)), scenes={})
    scenes_cache = cache.setdefault('scenes', {})
    pending = [entry for entry in selected
               if args.overwrite or entry[0] not in scenes_cache]
    skipped = len(selected) - len(pending)
    if not pending:
        print(f'Finished: processed=0, skipped={skipped}, failed=0')
        return

    device = torch.device(config.device)
    model = load_vggt(config.checkpoint, device)
    processed = failed = 0
    for index, (current_id, item, ann_file) in tqdm.tqdm(enumerate(pending, start=1)):
        print(f'[{index}/{len(pending)}] {current_id}')
        try:
            view_indices = select_views(item, data_root, config.num_views)
            images = load_images(
                item, data_root, view_indices,
                config.image_width, config.image_height)
            gt_points, _ = load_gt_points(item, data_root)
            gt_points = align_gt_points(gt_points, item['axis_align_matrix'])
            _, _, _, vggt_range = reconstruct_vggt(
                model, images, config.point_stride, config.max_depth,
                config.max_points, device)
            scale, gt_range, _ = estimate_scene_scale(gt_points, vggt_range)
            scenes_cache[current_id] = dict(
                scale=float(scale),
                gt_lower=gt_range[0].tolist(),
                gt_upper=gt_range[1].tolist(),
                gt_span=gt_range[2].tolist(),
                gt_diagonal=float(gt_range[3]),
                vggt_lower=vggt_range[0].tolist(),
                vggt_upper=vggt_range[1].tolist(),
                vggt_span=vggt_range[2].tolist(),
                vggt_diagonal=float(vggt_range[3]),
                num_views=len(view_indices),
                view_indices=[int(value) for value in view_indices],
                ann_file=ann_file,
            )
            print(
                f'[{current_id}] scale={scale:.6f}, '
                f'GT diagonal={gt_range[3]:.6f}, '
                f'VGGT diagonal={vggt_range[3]:.6f}')
            processed += 1
            mmengine.dump(cache, output_file)
        except Exception as error:
            failed += 1
            print(f'[{current_id}] failed: {type(error).__name__}: {error}')
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    cache['num_scenes'] = len(scenes_cache)
    cache['config_values'] = signature
    mmengine.dump(cache, output_file)
    print(
        f'Finished: processed={processed}, skipped={skipped}, failed={failed}, '
        f'cached={len(scenes_cache)}, output={output_file}')


if __name__ == '__main__':
    main()
