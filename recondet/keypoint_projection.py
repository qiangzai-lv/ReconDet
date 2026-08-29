import numpy as np
import torch


class BBoxKeypointProjector:
    def __init__(self):
        self.face_normals = (
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        )

    def __call__(self, centers, dimensions, extrinsics, intrinsic,
                 scale_factor, image_shape, valid_image_shape):
        normals = centers.new_tensor(self.face_normals)
        face_centers = (
            centers[:, None] + dimensions[:, None] * normals[None] * 0.5)
        keypoints_3d = torch.cat([face_centers, centers[:, None]], dim=1)

        if not isinstance(extrinsics, torch.Tensor):
            extrinsics = np.asarray(extrinsics)
        extrinsics = torch.as_tensor(
            extrinsics, dtype=centers.dtype, device=centers.device)
        rotation = extrinsics[:, :3, :3]
        translation = extrinsics[:, :3, 3]
        camera_points = torch.einsum(
            'vij,nkj->nvki', rotation, keypoints_3d)
        camera_points = camera_points + translation[None, :, None]
        depth = camera_points[..., 2]

        intrinsic = torch.as_tensor(
            intrinsic, dtype=centers.dtype, device=centers.device)
        if intrinsic.dim() == 2:
            intrinsic = intrinsic[:3, :3].unsqueeze(0).expand(
                rotation.shape[0], -1, -1)
        else:
            intrinsic = intrinsic[:, :3, :3]
        intrinsic = intrinsic.clone()
        scale_factor = torch.as_tensor(
            scale_factor, dtype=centers.dtype, device=centers.device)
        if scale_factor.numel() == 2:
            scale_factor = scale_factor.reshape(1, 2).expand(
                rotation.shape[0], -1)
        elif scale_factor.dim() == 1:
            if scale_factor.numel() != rotation.shape[0]:
                raise ValueError(
                    'scale_factor must be [2], [num_views, 2], or '
                    '[num_views]; got shape '
                    f'{tuple(scale_factor.shape)} for '
                    f'{rotation.shape[0]} views.')
            scale_factor = scale_factor[:, None].expand(-1, 2)
        elif (scale_factor.dim() != 2 or
              scale_factor.shape != (rotation.shape[0], 2)):
            raise ValueError(
                'scale_factor must be [2], [num_views, 2], or [num_views]; '
                f'got shape {tuple(scale_factor.shape)} for '
                f'{rotation.shape[0]} views.')
        # ``intrinsic[:, 0]`` is [num_views, 3]. Expand the per-view image
        # scale as [num_views, 1] so broadcasting scales each matrix row.
        intrinsic[:, 0] *= scale_factor[:, 0, None]
        intrinsic[:, 1] *= scale_factor[:, 1, None]
        pixels = torch.einsum('vij,nvkj->nvki', intrinsic, camera_points)
        pixels = pixels[..., :2] / depth[..., None].clamp_min(1e-5)

        image_height, image_width = image_shape
        keypoints_2d = pixels / pixels.new_tensor(
            [image_width, image_height])
        valid_height, valid_width = valid_image_shape[:2]
        in_image = depth > 1e-5
        in_image &= torch.isfinite(pixels).all(dim=-1)
        in_image &= pixels[..., 0] >= 0
        in_image &= pixels[..., 0] < valid_width
        in_image &= pixels[..., 1] >= 0
        in_image &= pixels[..., 1] < valid_height

        camera_centers = -torch.einsum(
            'vji,vj->vi', rotation, translation)
        camera_directions = (
            camera_centers[None, :, None] - face_centers[:, None])
        face_facing = torch.einsum(
            'nvfi,fi->nvf', camera_directions, normals) > 0
        center_facing = torch.ones_like(face_facing[..., :1])
        visible = in_image & torch.cat([face_facing, center_facing], dim=-1)
        keypoints_2d = keypoints_2d.masked_fill(~visible[..., None], 0.)

        # Project all eight corners to build an axis-aligned image bbox.
        corner_signs = centers.new_tensor([
            (-1., -1., -1.), (-1., -1., 1.), (-1., 1., -1.),
            (-1., 1., 1.), (1., -1., -1.), (1., -1., 1.),
            (1., 1., -1.), (1., 1., 1.)
        ])
        corners_3d = centers[:, None] + dimensions[:, None] * corner_signs[None] * 0.5
        corner_camera = torch.einsum(
            'vij,nkj->nvki', rotation, corners_3d)
        corner_camera = corner_camera + translation[None, :, None]
        corner_depth = corner_camera[..., 2]
        corner_pixels = torch.einsum(
            'vij,nvkj->nvki', intrinsic, corner_camera)
        corner_pixels = corner_pixels[..., :2] / corner_depth[..., None].clamp_min(1e-5)
        corner_valid = corner_depth > 1e-5
        corner_valid &= torch.isfinite(corner_pixels).all(dim=-1)

        pixel_min = corner_pixels.masked_fill(~corner_valid[..., None], float('inf')).amin(dim=2)
        pixel_max = corner_pixels.masked_fill(~corner_valid[..., None], float('-inf')).amax(dim=2)
        bbox_valid = corner_valid.any(dim=2)
        image_height, image_width = image_shape
        pixel_min[..., 0].clamp_(min=0, max=image_width)
        pixel_min[..., 1].clamp_(min=0, max=image_height)
        pixel_max[..., 0].clamp_(min=0, max=image_width)
        pixel_max[..., 1].clamp_(min=0, max=image_height)
        bboxes_2d = torch.stack([
            pixel_min[..., 0], pixel_min[..., 1],
            pixel_max[..., 0], pixel_max[..., 1]
        ], dim=-1)
        bboxes_2d = bboxes_2d / bboxes_2d.new_tensor(
            [image_width, image_height, image_width, image_height])
        bboxes_2d = bboxes_2d.masked_fill(~bbox_valid[..., None], 0.)
        return keypoints_3d, keypoints_2d, visible, bboxes_2d, bbox_valid
