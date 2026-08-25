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
    """Reconstruct six box faces from independently matched point queries."""

    def __init__(self,
                 n_classes: int,
                 num_box_queries: int = 256,
                 input_dim: int = 2048,
                 hidden_dim: int = 512,
                 cls_cost_weight: float = 1.0,
                 uv_cost_weight: float = 5.0,
                 cls_loss_weight: float = 1.0,
                 uv_loss_weight: float = 5.0,
                 background_weight: float = 0.1,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.n_classes = n_classes
        self.num_box_queries = num_box_queries
        self.cls_cost_weight = cls_cost_weight
        self.uv_cost_weight = uv_cost_weight
        self.cls_loss_weight = cls_loss_weight
        self.uv_loss_weight = uv_loss_weight

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim))
        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.cls_head = nn.Linear(hidden_dim, n_classes + 1)
        self.uv_delta_head = nn.Linear(hidden_dim, 2)

        self.box_queries = nn.Embedding(num_box_queries, hidden_dim)
        self.face_embeddings = nn.Embedding(6, hidden_dim)
        self.face_query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.point_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.face_offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3))

        cls_weights = torch.ones(n_classes + 1)
        cls_weights[-1] = background_weight
        self.register_buffer('cls_weights', cls_weights)
        nn.init.constant_(self.uv_delta_head.weight, 0.0)
        nn.init.constant_(self.uv_delta_head.bias, 0.0)
        nn.init.constant_(self.face_offset_head[-1].weight, 0.0)
        nn.init.constant_(self.face_offset_head[-1].bias, 0.0)

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

    def reconstruct_boxes(self, predictions: Dict[str, torch.Tensor]):
        point_features = predictions['features'].flatten(1, 2)
        point_xyz = predictions['xyz'].flatten(1, 2)
        batch_size, num_points, hidden_dim = point_features.shape

        face_queries = (
            self.box_queries.weight[:, None] +
            self.face_embeddings.weight[None]).reshape(-1, hidden_dim)
        queries = self.face_query_proj(face_queries)
        keys = self.point_key_proj(point_features)
        attention = torch.einsum(
            'fd,bnd->bfn', queries, keys) / math.sqrt(hidden_dim)
        foreground = predictions['cls_logits'].softmax(dim=-1)[
            ..., :-1].amax(dim=-1).flatten(1)
        attention = attention + foreground.clamp_min(1e-6).log()[:, None]
        attention = attention.softmax(dim=-1)

        with torch.autocast(device_type=point_xyz.device.type, enabled=False):
            attended_centers = torch.einsum(
                'bfn,bnd->bfd', attention.float(), point_xyz.float()).reshape(
                    batch_size, self.num_box_queries, 6, 3)
        face_features = torch.einsum(
            'bfn,bnd->bfd', attention, point_features).reshape(
                batch_size, self.num_box_queries, 6, hidden_dim)
        face_context = face_features + face_queries.reshape(
            1, self.num_box_queries, 6, hidden_dim)
        with torch.autocast(device_type=face_context.device.type, enabled=False):
            face_centers = attended_centers + self.face_offset_head(
                face_context.float())
        box_features = face_context.mean(dim=2)

        lower = torch.stack((face_centers[..., 0, 0],
                             face_centers[..., 2, 1],
                             face_centers[..., 4, 2]), dim=-1)
        upper = torch.stack((face_centers[..., 1, 0],
                             face_centers[..., 3, 1],
                             face_centers[..., 5, 2]), dim=-1)
        centers = 0.5 * (lower + upper)
        sizes = (upper - lower).abs().clamp_min(1e-3)
        return centers, sizes, box_features, face_centers

    @staticmethod
    def _camera_tensors(metadata: dict, reference: torch.Tensor):
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
        return extrinsics, intrinsics

    @staticmethod
    def _empty_target(reference: torch.Tensor):
        return dict(
            uv=reference.new_empty((0, 2)),
            labels=torch.empty(0, device=reference.device, dtype=torch.long))

    def _build_view_targets(self, data_sample, num_views: int,
                            reference: torch.Tensor) -> List[dict]:
        gt_instances = data_sample.gt_instances_3d
        boxes = gt_instances.bboxes_3d
        labels = gt_instances.labels_3d.to(reference.device)
        faces = box_face_centers(boxes).to(reference).float()
        extrinsics, intrinsics = self._camera_tensors(
            data_sample.metainfo, reference)
        available_views = min(num_views, len(extrinsics))
        targets = []

        ori_shape = data_sample.metainfo.get('ori_shape')
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
            camera_xyz = xyz @ rotation.transpose(0, 1) + translation
            pixels_h = camera_xyz @ intrinsics[
                view_id, :3, :3].transpose(0, 1)
            pixels = pixels_h[:, :2] / pixels_h[:, 2:].clamp_min(1e-5)

            if isinstance(ori_shape, (tuple, list)) and len(ori_shape) >= 2:
                height, width = float(ori_shape[0]), float(ori_shape[1])
            else:
                intrinsic = intrinsics[view_id]
                width = max(float(intrinsic[0, 2]) * 2.0, 1.0)
                height = max(float(intrinsic[1, 2]) * 2.0, 1.0)
            uv = pixels / pixels.new_tensor((width, height))
            visible = camera_xyz[:, 2] > 1e-4
            visible &= torch.isfinite(uv).all(dim=-1)
            visible &= (uv >= 0.0).all(dim=-1) & (uv <= 1.0).all(dim=-1)
            targets.append(dict(
                uv=uv[visible],
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
                    uv_prediction[batch_id])
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

        layer_losses = [loss_for_uv(uv) for uv in uv_layers]
        losses = dict(
            cls_loss=torch.stack([loss[0] for loss in layer_losses]).mean(),
            uv_loss=torch.stack([loss[1] for loss in layer_losses]).sum())
        for layer_id, (_, uv_loss) in enumerate(layer_losses):
            losses[f'uv_loss_layer_{layer_id}'] = uv_loss
        return losses
