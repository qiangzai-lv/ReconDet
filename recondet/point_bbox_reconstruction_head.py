import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from scipy.optimize import linear_sum_assignment

from mmdet3d.registry import MODELS


def box_face_centers(boxes) -> torch.Tensor:
    centers = boxes.gravity_center
    half_sizes = boxes.tensor[:, 3:6] * 0.5
    offsets = centers.new_zeros((len(boxes), 6, 3))
    for axis in range(3):
        offsets[:, 2 * axis, axis] = -half_sizes[:, axis]
        offsets[:, 2 * axis + 1, axis] = half_sizes[:, axis]
    return centers[:, None] + offsets


@MODELS.register_module()
class PointBBoxReconstructionHead(BaseModule):
    """Initialize box queries by clustering reconstructed 3D points."""

    def __init__(self,
                 n_classes: int,
                 num_box_queries: int = 256,
                 input_dim: int = 2048,
                 hidden_dim: int = 512,
                 cls_cost_weight: float = 1.0,
                 uv_cost_weight: float = 5.0,
                 cls_loss_weight: float = 1.0,
                 uv_loss_weight: float = 5.0,
                 xyz_loss_weight: float = 5.0,
                 cluster_temperature: float = 1.0,
                 background_weight: float = 0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.n_classes = n_classes
        self.num_box_queries = num_box_queries
        self.cls_cost_weight = cls_cost_weight
        self.uv_cost_weight = uv_cost_weight
        self.cls_loss_weight = cls_loss_weight
        self.uv_loss_weight = uv_loss_weight
        self.xyz_loss_weight = xyz_loss_weight
        if cluster_temperature <= 0:
            raise ValueError('cluster_temperature must be positive')
        self.cluster_temperature = cluster_temperature

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim))
        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.cls_head = nn.Linear(hidden_dim, n_classes + 1)
        self.uv_delta_head = nn.Linear(hidden_dim, 2)

        self.box_queries = nn.Embedding(num_box_queries, hidden_dim)
        self.cluster_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.point_key_proj = nn.Linear(hidden_dim, hidden_dim)

        cls_weights = torch.ones(n_classes + 1)
        cls_weights[-1] = background_weight
        self.register_buffer('cls_weights', cls_weights)
        nn.init.constant_(self.uv_delta_head.weight, 0.0)
        nn.init.constant_(self.uv_delta_head.bias, 0.0)

    def predict_uv_delta(self, object_features: torch.Tensor) -> torch.Tensor:
        """Predict the inverse-sigmoid UV update for object-query decoding."""
        features = self.feature_proj(object_features.float())
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.uv_delta_head(features.float())

    def forward(self, object_features: torch.Tensor,
        reference_uv: torch.Tensor,
        reference_uv_layers=None) -> Dict[str, torch.Tensor]:
        features = self.feature_proj(object_features.float())
        with torch.autocast(device_type=features.device.type, enabled=False):
            xyz = self.xyz_head(features.float())
        predictions = dict(
            xyz=xyz,
            uv=reference_uv,
            cls_logits=self.cls_head(features),
            features=features)
        if reference_uv_layers:
            predictions['uv_layers'] = reference_uv_layers
        return predictions

    @staticmethod
    @torch.no_grad()
    def _foreground_weighted_fps(points: torch.Tensor,
                                 foreground: torch.Tensor,
                                 num_samples: int) -> torch.Tensor:
        batch_size, num_points, _ = points.shape
        if num_points == 0:
            raise ValueError('Cannot initialize queries from zero points')

        indices = torch.empty(
            batch_size, num_samples, device=points.device, dtype=torch.long)
        min_distances = torch.full(
            (batch_size, num_points), float('inf'), device=points.device)
        batch_ids = torch.arange(batch_size, device=points.device)
        current = foreground.argmax(dim=-1)
        sampling_weight = foreground.clamp_min(1e-6)

        for sample_id in range(num_samples):
            indices[:, sample_id] = current
            centers = points[batch_ids, current]
            distances = (points - centers[:, None]).square().sum(dim=-1)
            min_distances = torch.minimum(min_distances, distances)
            current = (min_distances * sampling_weight).argmax(dim=-1)
        return indices

    def reconstruct_boxes(self, predictions: Dict[str, torch.Tensor]):
        point_features = predictions['features'].flatten(1, 2)
        point_xyz_vggt = predictions['xyz'].flatten(1, 2)
        batch_size, num_points, hidden_dim = point_features.shape
        foreground = predictions['cls_logits'].softmax(dim=-1)[
            ..., :-1].amax(dim=-1).flatten(1)

        with torch.autocast(
                device_type=point_xyz_vggt.device.type, enabled=False):
            vggt_to_aligned = predictions['vggt_to_aligned'].float()
            point_homogeneous = torch.cat(
                (point_xyz_vggt.float(),
                 torch.ones_like(point_xyz_vggt[..., :1])), dim=-1)
            point_xyz = torch.einsum(
                'bij,bnj->bni', vggt_to_aligned,
                point_homogeneous)[..., :3]

            seed_indices = self._foreground_weighted_fps(
                point_xyz.detach(), foreground.detach(),
                self.num_box_queries)
            batch_ids = torch.arange(
                batch_size, device=point_features.device)[:, None]
            seed_xyz = point_xyz[batch_ids, seed_indices]
            seed_features = point_features[batch_ids, seed_indices]

            learned_queries = self.box_queries.weight[None].expand(
                batch_size, -1, -1)
            queries = self.cluster_query_proj(
                learned_queries.float() + seed_features.float())
            keys = self.point_key_proj(point_features.float())
            feature_logits = torch.einsum(
                'bqd,bnd->bqn', queries, keys) / math.sqrt(hidden_dim)
            spatial_distances = torch.cdist(
                seed_xyz.float(), point_xyz.float()).square()
            assignment_logits = (
                feature_logits -
                spatial_distances / self.cluster_temperature)
            assignments = assignment_logits.softmax(dim=1)
            assignments = assignments * foreground.float()[:, None]
            cluster_weights = assignments / assignments.sum(
                dim=-1, keepdim=True).clamp_min(1e-6)

            centers = torch.einsum(
                'bqn,bnd->bqd', cluster_weights, point_xyz.float())
            cluster_features = torch.einsum(
                'bqn,bnd->bqd', cluster_weights,
                point_features.float())
            query_features = cluster_features + learned_queries.float()
        return centers, query_features, cluster_weights

    @staticmethod
    def _camera_tensors(metadata: dict, reference: torch.Tensor,
                        num_views: int):
        projection = metadata['lidar2img']
        extrinsics = torch.as_tensor(
            np.asarray(projection['extrinsic']), device=reference.device,
            dtype=torch.float32)
        intrinsics = torch.as_tensor(
            np.asarray(projection['intrinsic']), device=reference.device,
            dtype=torch.float32)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0).expand(
                len(extrinsics), -1, -1)

        available_views = min(num_views, len(extrinsics), len(intrinsics))
        extrinsics = extrinsics[:available_views]
        intrinsics = intrinsics[:available_views]

        if extrinsics.shape[-2:] == (3, 4):
            homogeneous_extrinsics = extrinsics.new_zeros(
                available_views, 4, 4)
            homogeneous_extrinsics[:, :3] = extrinsics
            homogeneous_extrinsics[:, 3, 3] = 1.0
            extrinsics = homogeneous_extrinsics
        elif extrinsics.shape[-2:] != (4, 4):
            raise ValueError('Camera extrinsics must be 3x4 or 4x4')

        if intrinsics.shape[-2:] == (3, 3):
            homogeneous_intrinsics = intrinsics.new_zeros(
                available_views, 4, 4)
            homogeneous_intrinsics[:, :3, :3] = intrinsics
            homogeneous_intrinsics[:, 3, 3] = 1.0
            intrinsics = homogeneous_intrinsics
        elif intrinsics.shape[-2:] != (4, 4):
            raise ValueError('Camera intrinsics must be 3x3 or 4x4')

        scale_factors = metadata.get('view_scale_factors')
        if scale_factors is None:
            scale_factors = metadata.get('scale_factor')
        if scale_factors is None:
            raise KeyError('Projection requires per-view resize scale factors')
        scale_factors = torch.as_tensor(
            np.asarray(scale_factors), device=reference.device,
            dtype=torch.float32)
        if scale_factors.ndim == 1:
            scale_factors = scale_factors.unsqueeze(0).expand(
                available_views, -1)
        if len(scale_factors) < available_views or scale_factors.shape[-1] < 2:
            raise ValueError('Invalid per-view resize scale factors')
        scale_factors = scale_factors[:available_views, :2]

        resize_matrices = torch.eye(
            4, device=reference.device, dtype=torch.float32).unsqueeze(0).repeat(
                available_views, 1, 1)
        resize_matrices[:, 0, 0] = scale_factors[:, 0]
        resize_matrices[:, 1, 1] = scale_factors[:, 1]
        projection_matrices = resize_matrices @ intrinsics @ extrinsics

        image_shape = metadata.get(
            'batch_input_shape', metadata.get('pad_shape'))
        if image_shape is None or len(image_shape) < 2:
            raise KeyError('Projection requires the padded input image shape')
        image_height, image_width = int(image_shape[0]), int(image_shape[1])
        if image_height <= 0 or image_width <= 0:
            raise ValueError('Padded input image shape must be positive')
        return (extrinsics, projection_matrices,
                (image_height, image_width))

    @staticmethod
    def _empty_target(reference: torch.Tensor):
        return dict(
            uv=reference.new_empty((0, 2)),
            xyz=reference.new_empty((0, 3)),
            labels=torch.empty(0, device=reference.device, dtype=torch.long))

    def _build_view_targets(self, data_sample, num_views: int,
                            reference: torch.Tensor,
                            aligned_to_vggt: torch.Tensor) -> List[dict]:
        gt_instances = data_sample.gt_instances_3d
        boxes = gt_instances.bboxes_3d
        labels = gt_instances.labels_3d.to(reference.device)
        faces = box_face_centers(boxes).to(reference).float()
        extrinsics, projection_matrices, image_shape = self._camera_tensors(
            data_sample.metainfo, reference, num_views)
        available_views = len(extrinsics)
        image_height, image_width = image_shape
        targets = []

        for view_id in range(available_views):
            if len(boxes) == 0:
                targets.append(self._empty_target(reference))
                continue
            extrinsic = extrinsics[view_id]
            rotation = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]
            camera_center = -(rotation.transpose(0, 1) @ translation)

            box_centers = boxes.gravity_center.to(reference)
            face_ids = []
            for box_id in range(len(boxes)):
                for axis in range(3):
                    face_ids.append(
                        2 * axis + int(
                            camera_center[axis] >= box_centers[box_id, axis]))
            face_ids = torch.as_tensor(
                face_ids, device=reference.device, dtype=torch.long)
            box_ids = torch.arange(
                len(boxes), device=reference.device).repeat_interleave(3)
            xyz = faces[box_ids, face_ids]
            xyz_homogeneous = torch.cat(
                (xyz, torch.ones_like(xyz[:, :1])), dim=-1)
            xyz_vggt = (
                xyz_homogeneous @ aligned_to_vggt.transpose(0, 1))[..., :3]
            projected = xyz_homogeneous @ projection_matrices[
                view_id].transpose(0, 1)
            depth = projected[:, 2:3]
            pixels = projected[:, :2] / depth.clamp_min(1e-5)
            uv = pixels / pixels.new_tensor(
                (image_width, image_height))
            visible = depth[:, 0] > 1e-5
            visible &= torch.isfinite(uv).all(dim=-1)
            visible &= (uv > 0.0).all(dim=-1) & (uv < 1.0).all(dim=-1)
            targets.append(dict(
                uv=uv[visible],
                xyz=xyz_vggt[visible],
                labels=labels[box_ids[visible]]))

        while len(targets) < num_views:
            targets.append(self._empty_target(reference))
        return targets

    def _match(self, prediction: dict, target: dict):
        if target['labels'].numel() == 0:
            empty = torch.empty(
                0, device=prediction['uv'].device, dtype=torch.long)
            return empty, empty
        class_probability = prediction['cls_logits'].softmax(dim=-1)
        cost = (
            -self.cls_cost_weight * class_probability[:, target['labels']] +
            self.uv_cost_weight * torch.cdist(
                prediction['uv'], target['uv'], p=1))
        prediction_ids, target_ids = linear_sum_assignment(
            cost.detach().float().cpu().numpy())
        return (
            torch.as_tensor(prediction_ids, device=cost.device,
                            dtype=torch.long),
            torch.as_tensor(target_ids, device=cost.device,
                            dtype=torch.long))

    def loss(self, predictions: Dict[str, torch.Tensor],
             batch_data_samples) -> Dict[str, torch.Tensor]:
        aligned_to_vggt = predictions['aligned_to_vggt']
        uv_layers = predictions.get('uv_layers')
        if not uv_layers:
            uv_layers = [predictions['uv']]

        def loss_for_uv(uv_prediction):
            batch_size, num_views, num_queries = uv_prediction.shape[:3]
            zero = uv_prediction.sum() * 0.0
            cls_losses, uv_losses = [], []
            for batch_id in range(batch_size):
                targets = self._build_view_targets(
                    batch_data_samples[batch_id], num_views,
                    uv_prediction[batch_id], aligned_to_vggt[batch_id])
                for view_id, target in enumerate(targets):
                    prediction = {
                        'uv': uv_prediction[batch_id, view_id],
                        'cls_logits': predictions['cls_logits'][
                            batch_id, view_id],
                    }
                    prediction_ids, target_ids = self._match(
                        prediction, target)
                    class_target = torch.full(
                        (num_queries,), self.n_classes,
                        device=prediction_ids.device, dtype=torch.long)
                    if prediction_ids.numel() > 0:
                        class_target[prediction_ids] = target['labels'][
                            target_ids]
                        uv_losses.append(F.l1_loss(
                            prediction['uv'][prediction_ids],
                            target['uv'][target_ids]))
                    cls_losses.append(F.cross_entropy(
                        prediction['cls_logits'], class_target,
                        weight=self.cls_weights.to(
                            prediction['cls_logits'])))
            return (
                torch.stack(cls_losses).mean() * self.cls_loss_weight,
                (torch.stack(uv_losses).mean() if uv_losses else zero) *
                self.uv_loss_weight)

        def loss_for_xyz():
            xyz_prediction = predictions['xyz']
            uv_prediction = predictions['uv']
            batch_size, num_views = xyz_prediction.shape[:2]
            zero = xyz_prediction.sum() * 0.0
            xyz_losses = []
            for batch_id in range(batch_size):
                targets = self._build_view_targets(
                    batch_data_samples[batch_id], num_views,
                    uv_prediction[batch_id], aligned_to_vggt[batch_id])
                for view_id, target in enumerate(targets):
                    prediction = {
                        'uv': uv_prediction[batch_id, view_id],
                        'cls_logits': predictions['cls_logits'][
                            batch_id, view_id],
                    }
                    prediction_ids, target_ids = self._match(
                        prediction, target)
                    if prediction_ids.numel() > 0:
                        xyz_losses.append(F.smooth_l1_loss(
                            xyz_prediction[batch_id, view_id, prediction_ids],
                            target['xyz'][target_ids]))
            if not xyz_losses:
                return zero
            return torch.stack(xyz_losses).mean() * self.xyz_loss_weight

        layer_losses = [loss_for_uv(uv) for uv in uv_layers]
        losses = dict(
            cls_loss=torch.stack([loss[0] for loss in layer_losses]).mean(),
            uv_loss=torch.stack([loss[1] for loss in layer_losses]).sum(),
            xyz_loss=loss_for_xyz())
        for layer_id, (_, uv_loss) in enumerate(layer_losses):
            losses[f'uv_loss_layer_{layer_id}'] = uv_loss
        return losses
