from pathlib import Path

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, Compose
from PIL import Image

from mmdet3d.registry import TRANSFORMS


def read_pose_matrix(file_path):
    """
    Reads a 4x4 pose matrix from a text file.

    Args:
        file_path (str): The path to the text file containing the pose matrix.

    Returns:
        np.ndarray: A 4x4 NumPy array representing the pose matrix.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        matrix = [list(map(float, line.strip().split())) for line in lines]

        pose_matrix = np.array(matrix)

        if pose_matrix.shape != (4, 4):
            raise ValueError("The input file does not contain a valid 4x4 pose matrix.")

        return pose_matrix

    except Exception as e:
        print(f"Error reading pose matrix: {e}")
        return None


@TRANSFORMS.register_module()
class LoadFirstFramePose(BaseTransform):
    def transform(self, results: dict) -> dict:
        first_img_path = results['img_path'][0]
        pose_matrix = read_pose_matrix(str(Path(first_img_path).with_suffix('.txt')))
        if pose_matrix is None:
            raise ValueError(f'Could not load first-frame pose for {first_img_path}')

        results['pose_matrix'] = pose_matrix.astype(np.float32)
        return results


@TRANSFORMS.register_module()
class MultiViewPipeline_Tgt(BaseTransform):

    def __init__(self,
                 transforms: dict,
                 n_images: int,
                 mean: tuple = [123.675, 116.28, 103.53],
                 std: tuple = [58.395, 57.12, 57.375],
                 margin: int = 10,
                 depth_range: tuple = [0.5, 5.5],
                 loading: str = 'random',
                 nerf_target_views: int = 0,
                 sample_freq: int = 3,
                 tgt_transforms=None):
        self.transforms = Compose(transforms)
        self.depth_transforms = Compose(transforms[1])
        self.n_images = n_images
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.margin = margin
        self.depth_range = depth_range
        self.loading = loading
        self.sample_freq = sample_freq
        self.nerf_target_views = nerf_target_views
        self.tgt_transforms = Compose(tgt_transforms)

    def transform(self, results: dict) -> dict:

        imgs = []
        depths = []
        extrinsics = []

        if self.loading == 'random':
            ids = np.arange(len(results['img_info']))
            replace = True if self.n_images > len(ids) else False
            ids = np.random.choice(ids, self.n_images, replace=replace)
            if self.nerf_target_views != 0:
                target_id = np.random.choice(
                    ids, self.nerf_target_views, replace=False)
                ids = np.setdiff1d(ids, target_id)
                ids = ids.tolist()

        elif self.loading == 'gap':
            ids = np.arange(len(results['img_info']))
            src_1 = np.random.randint(0, len(ids) // 2 - self.nerf_target_views // 2 - 1, (1,))[
                0]  # choose one from first half of images
            src_3 = np.random.randint(len(ids) // 2, len(ids) - self.nerf_target_views // 2 - 1, (1,))[0]
            src_used_id = [src_1, src_1 + self.nerf_target_views // 2 + 1, src_3,
                           src_3 + self.nerf_target_views // 2 + 1]
            target_id = []
            for k in range(self.nerf_target_views // 2):
                target_id = target_id + [src_1 + 1 + k, src_3 + 1 + k]
            used_id = src_used_id + target_id
            replace = True if self.n_images > len(ids) else False
            rest_src = np.random.choice(np.setdiff1d(ids, np.array(used_id)), self.n_images - len(used_id),
                                        replace=replace)
            ids = rest_src.tolist() + src_used_id
            assert max(ids) < len(results['img_info'])

        else:
            assert ""

        size = (240, 320)
        src_img_paths = []
        for i in ids:
            _results = dict()
            _results['img_path'] = results['img_info'][i]['filename']
            src_img_paths.append(results['img_info'][i]['filename'])
            _results = self.transforms(_results)  # load and resize.
            imgs.append(_results['img'])  # after resize, image is (239, 320, 3)
            # normalize
            for key in _results.get('img_fields', ['img']):
                _results[key] = mmcv.imnormalize(_results[key], self.mean,
                                                 self.std, True)  # to_rgb=True
            _results['img_norm_cfg'] = dict(
                mean=self.mean, std=self.std, to_rgb=True)
            # pad
            for key in _results.get('img_fields', ['img']):
                padded_img = mmcv.impad(_results[key], shape=size, pad_val=0)
                _results[key] = padded_img
            _results['pad_shape'] = padded_img.shape  # (240, 320, 3)
            _results['pad_fixed_size'] = size  # (240, 320)
            ori_shape = _results['ori_shape']  # (968, 1296)
            aft_shape = _results['img_shape']  # (239, 320)
            # prepare the depth information
            if 'depth_info' in results.keys():
                if '.npy' in results['depth_info'][i]['filename']:
                    _results['depth'] = np.load(
                        results['depth_info'][i]['filename'])
                else:
                    _results['depth'] = np.asarray((Image.open(
                        results['depth_info'][i]['filename']))) / 1000
                    _results['depth'] = mmcv.imresize(
                        _results['depth'], (aft_shape[1], aft_shape[0]))
                depths.append(_results['depth'])

            extrinsics.append(results['lidar2img']['extrinsic'][i])

        for key in _results.keys():
            if key not in ['img', 'img_info']:
                results[key] = _results[key]
        results['img'] = imgs  # bug here.. imgs
        results['img_path'] = src_img_paths  # manually add in img_path

        if len(depths) != 0:
            results['depth'] = depths
        results['lidar2img']['extrinsic'] = extrinsics  # w2c src view.
        return results
