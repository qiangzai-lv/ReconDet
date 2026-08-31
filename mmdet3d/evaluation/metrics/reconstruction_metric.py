from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmdet3d.registry import METRICS


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.prod(np.clip(boxes1[:, 2:] - boxes1[:, :2], 0.0, None), axis=1)
    area2 = np.prod(np.clip(boxes2[:, 2:] - boxes2[:, :2], 0.0, None), axis=1)
    return inter / np.maximum(area1[:, None] + area2[None, :] - inter, 1e-12)


@METRICS.register_module()
class ReconstructionMetric(BaseMetric):
    """Evaluate view-level reconstruction points assigned by 2D detections."""

    def __init__(self,
                 iou_thr: float = 0.5,
                 distance_thresholds: Sequence[float] = (0.10, 0.25, 0.50),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.iou_thr = float(iou_thr)
        self.distance_thresholds = tuple(float(x) for x in distance_thresholds)

    @staticmethod
    def _as_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _field(value, name):
        if isinstance(value, dict):
            return value[name]
        return getattr(value, name)

    @staticmethod
    def _to_pixel_boxes(boxes, image_shape, normalized=False):
        boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else np.array(boxes, copy=True)
        if (boxes.numel() if isinstance(boxes, torch.Tensor) else boxes.size) == 0:
            return boxes
        if normalized:
            height, width = image_shape[:2]
            scale = boxes.new_tensor([width, height, width, height]) if isinstance(boxes, torch.Tensor) else np.array([width, height, width, height])
            boxes = boxes * scale
        return boxes

    def _evaluate_view(self, result: dict) -> Dict[str, float]:
        pred_boxes = self._as_numpy(result['pred_bboxes_2d']).reshape(-1, 4)
        pred_labels = self._as_numpy(result['pred_labels_2d']).reshape(-1)
        pred_scores = self._as_numpy(result['pred_scores_2d']).reshape(-1)
        query_indices = self._as_numpy(result['pred_query_indices']).reshape(-1)
        pred_points = self._as_numpy(result['pred_points_3d']).reshape(-1, 3)
        gt_boxes = self._as_numpy(result['gt_bboxes_2d']).reshape(-1, 4)
        gt_labels = self._as_numpy(result['gt_labels_2d']).reshape(-1)
        gt_centers = self._as_numpy(result['gt_centers_3d']).reshape(-1, 3)
        gt_visible = self._as_numpy(result.get(
            'gt_visible', np.ones((len(gt_boxes),), dtype=bool))).reshape(-1)

        valid_pred = ((query_indices >= 0) &
                      (query_indices < len(pred_points)))
        valid_pred &= np.isfinite(pred_boxes).all(axis=1)
        valid_pred &= (pred_boxes[:, 2:] > pred_boxes[:, :2]).all(axis=1)
        if len(pred_points):
            safe_indices = query_indices.clip(min=0, max=len(pred_points) - 1)
            valid_pred &= np.isfinite(pred_points[safe_indices]).all(axis=1)
        else:
            valid_pred[:] = False
        valid_gt = np.isfinite(gt_boxes).all(axis=1)
        valid_gt &= np.isfinite(gt_centers).all(axis=1)
        valid_gt &= gt_visible.astype(bool)
        valid_gt &= (gt_boxes[:, 2:] > gt_boxes[:, :2]).all(axis=1)
        pred_ids = np.flatnonzero(valid_pred)
        gt_ids = np.flatnonzero(valid_gt)
        distances = []
        used_gt = set()
        if len(pred_ids) and len(gt_ids):
            ious = _box_iou(pred_boxes[pred_ids], gt_boxes[gt_ids])
            for local_pred in np.argsort(-pred_scores[pred_ids]):
                candidates = [
                    j for j, gt_id in enumerate(gt_ids)
                    if gt_id not in used_gt
                    and pred_labels[pred_ids[local_pred]] == gt_labels[gt_id]
                    and ious[local_pred, j] >= self.iou_thr
                ]
                if not candidates:
                    continue
                best = max(candidates, key=lambda j: ious[local_pred, j])
                used_gt.add(int(gt_ids[best]))
                point = pred_points[int(query_indices[pred_ids[local_pred]])]
                distances.append(float(np.linalg.norm(point - gt_centers[gt_ids[best]])))

        distances = np.asarray(distances, dtype=np.float64)
        metrics = {
            'center_mae': float(distances.mean()) if len(distances) else 0.0,
            'center_rmse': float(np.sqrt(np.mean(distances**2))) if len(distances) else 0.0,
            'center_median': float(np.median(distances)) if len(distances) else 0.0,
            'matched_pairs': float(len(distances)),
            'match_recall': float(len(distances) / len(gt_ids)) if len(gt_ids) else 0.0,
        }
        for threshold in self.distance_thresholds:
            metrics[f'within_{threshold:g}m'] = (
                float((distances <= threshold).mean()) if len(distances) else 0.0)
        return metrics

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_views = self._field(data_sample, 'pred_instances_2d_views')
            reconstruction_views = self._field(
                data_sample, 'pred_reconstruction')
            gt = self._field(data_sample, 'gt_instances_3d')
            if isinstance(data_sample, dict):
                metainfo = data_sample.get('metainfo', data_sample)
            else:
                metainfo = data_sample.metainfo
            image_shapes = metainfo.get('img_shape', metainfo.get('batch_input_shape'))
            if isinstance(image_shapes, np.ndarray):
                image_shapes = image_shapes.tolist()
            if image_shapes is None:
                raise KeyError('ReconstructionMetric requires img_shape metadata')
            if image_shapes and not isinstance(
                    image_shapes[0], (list, tuple, np.ndarray)):
                image_shapes = [image_shapes] * len(pred_views)
            if len(pred_views) != len(reconstruction_views):
                raise ValueError('2D and reconstruction view counts must match')
            for view_id, (pred, reconstruction) in enumerate(
                    zip(pred_views, reconstruction_views)):
                self.results.append({
                    'pred_bboxes_2d': self._field(pred, 'bboxes'),
                    'pred_labels_2d': self._field(pred, 'labels'),
                    'pred_scores_2d': self._field(pred, 'scores'),
                    'pred_query_indices': self._field(pred, 'query_indices'),
                    'pred_points_3d': self._field(reconstruction, 'points_3d'),
                    'gt_bboxes_2d': self._to_pixel_boxes(
                        self._field(gt, 'bboxes_2d')[:, view_id],
                        image_shapes[view_id], normalized=True),
                    'gt_labels_2d': self._field(gt, 'labels_3d'),
                    'gt_centers_3d': self._field(gt, 'centers_3d'),
                    'gt_visible': self._field(gt, 'bboxes_2d_visible')[:, view_id],
                })

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        if not results:
            return {}
        per_view = [self._evaluate_view(result) for result in results]
        matched = sum(item['matched_pairs'] for item in per_view)
        metrics = {}
        for key in per_view[0]:
            if key == 'matched_pairs':
                metrics[key] = matched
            elif key == 'match_recall':
                metrics[key] = float(np.mean([item[key] for item in per_view]))
            elif key.startswith('within_'):
                metrics[key] = float(np.mean([item[key] for item in per_view]))
            else:
                weighted = sum(item[key] * item['matched_pairs'] for item in per_view)
                metrics[key] = float(weighted / matched) if matched else 0.0
        return metrics
