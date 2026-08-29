from copy import deepcopy
from pathlib import Path
from typing import Sequence

import cv2
import torch
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.dist import is_main_process
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from mmengine.structures import InstanceData
from torch import nn


class GroundingDINOSemanticEncoder(nn.Module):
    """GroundingDINO with frozen encoders and a trainable detection decoder."""

    def __init__(self,
                 config: str,
                 checkpoint: str,
                 classes: Sequence[str],
                 print_score_thr: float = 0.3,
                 visualization_dir: str = None,
                 visualization_interval: int = 100) -> None:
        super().__init__()
        if not classes:
            raise ValueError('GroundingDINO classes must not be empty.')
        config_path = Path(config).expanduser()
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        cfg = Config.fromfile(str(config_path))
        model_cfg = deepcopy(cfg.model)
        model_cfg['_scope_'] = 'mmdet'
        data_preprocessor_cfg = model_cfg['data_preprocessor']
        image_mean = data_preprocessor_cfg['mean']
        image_std = data_preprocessor_cfg['std']
        if model_cfg.get('backbone', {}).get('init_cfg') is not None:
            model_cfg.backbone.init_cfg = None

        register_all_modules(init_default_scope=False)
        self.model = MMDET_MODELS.build(model_cfg)
        self.model.bbox_head.init_weights()
        checkpoint_data = CheckpointLoader.load_checkpoint(
            checkpoint, map_location='cpu')
        state_dict = checkpoint_data.get('state_dict', checkpoint_data)
        state_dict = {
            name.removeprefix('module.'): value
            for name, value in state_dict.items()
        }
        query_weight = state_dict.get('query_embedding.weight')
        if (query_weight is not None and
                query_weight.shape != self.model.query_embedding.weight.shape and
                query_weight.shape[1:] ==
                self.model.query_embedding.weight.shape[1:] and
                query_weight.shape[0] >= self.model.num_queries):
            state_dict['query_embedding.weight'] = query_weight[
                :self.model.num_queries].clone()
        load_state_dict(self.model, state_dict, strict=False)
        self.model._is_init = True

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for module in (self.model.query_embedding, self.model.decoder,
                       self.model.bbox_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.model.eval()

        self.classes = tuple(classes)
        self.print_score_thr = print_score_thr
        if visualization_interval < 1:
            raise ValueError('visualization_interval must be at least 1.')
        self.visualization_interval = visualization_interval
        self.training_iteration = 0
        self.visualization_dir = (
            Path(visualization_dir).expanduser()
            if visualization_dir is not None else None)
        self.visualization_iteration = 0
        self.register_buffer(
            'image_mean',
            torch.tensor(image_mean).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'image_std',
            torch.tensor(image_std).view(1, 3, 1, 1),
            persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        for module_name in ('backbone', 'neck', 'encoder', 'language_model'):
            module = getattr(self.model, module_name, None)
            if module is not None:
                module.eval()
        return self

    @staticmethod
    def _image_shape(data_sample, view_index: int, fallback):
        shape = data_sample.metainfo.get('img_shape', fallback)
        if (isinstance(shape, (list, tuple)) and shape and
                isinstance(shape[0], (list, tuple))):
            shape = shape[view_index]
        return int(shape[0]), int(shape[1])

    def _make_data_sample(self, source_sample, view_index: int,
                          padded_shape) -> DetDataSample:
        image_shape = self._image_shape(
            source_sample, view_index, padded_shape)
        data_sample = DetDataSample()
        data_sample.set_metainfo({
            'img_shape': image_shape,
            'ori_shape': image_shape,
            'batch_input_shape': tuple(padded_shape),
            'pad_shape': tuple(padded_shape),
            'scale_factor': (1.0, 1.0),
        })
        data_sample.text = self.classes
        data_sample.custom_entities = True
        return data_sample

    def _normalize_images(self, images, batch_data_samples):
        batch_size, num_views, channels, height, width = images.shape

        flattened = images.reshape(
            batch_size * num_views, channels, height, width).float()
        flattened = (flattened - self.image_mean) / self.image_std

        for batch_index, data_sample in enumerate(batch_data_samples):
            image_height, image_width = self._image_shape(
                data_sample, 0, (height, width))
            start = batch_index * num_views
            end = start + num_views
            if image_height < height:
                flattened[start:end, :, image_height:, :] = 0
            if image_width < width:
                flattened[start:end, :, :, image_width:] = 0
        return flattened

    def _save_visualization(self, image, prediction, source_sample,
                            batch_index: int, view_index: int) -> None:
        if self.visualization_dir is None or not is_main_process():
            return

        image_height, image_width = self._image_shape(
            source_sample, view_index, image.shape[-2:])
        rgb_image = image[:, :image_height, :image_width]
        rgb_image = rgb_image.permute(1, 2, 0).detach().cpu().numpy()
        bgr_image = rgb_image.clip(0, 255).astype('uint8')[..., ::-1].copy()

        instances = prediction.pred_instances
        keep = instances.scores >= self.print_score_thr
        instances = instances[keep].cpu()
        label_names = getattr(instances, 'label_names', [])
        colors = [
            (230, 159, 0), (86, 180, 233), (0, 158, 115),
            (240, 228, 66), (0, 114, 178), (213, 94, 0),
            (204, 121, 167), (128, 128, 128)
        ]
        for index, (bbox, score, label) in enumerate(zip(
                instances.bboxes, instances.scores, instances.labels)):
            x1, y1, x2, y2 = [int(round(value)) for value in bbox.tolist()]
            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))
            label_index = int(label)
            color = colors[label_index % len(colors)]
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), color, 2)

            label_name = (
                label_names[index] if index < len(label_names)
                else self.classes[label_index])
            text = f'{label_name} {float(score):.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(y1, text_height + baseline + 2)
            cv2.rectangle(
                bgr_image,
                (x1, text_y - text_height - baseline - 2),
                (min(x1 + text_width + 4, image_width - 1), text_y + 2),
                color, -1)
            cv2.putText(
                bgr_image, text, (x1 + 2, text_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                cv2.LINE_AA)

            if 'keypoints' in instances:
                points = instances.keypoints[index]
                point_colors = [(255, 255, 255), (0, 0, 255),
                                (0, 255, 0), (255, 0, 0)]
                for point_index, point in enumerate(points):
                    px = int(round(float(point[0]) * image_width))
                    py = int(round(float(point[1]) * image_height))
                    if 0 <= px < image_width and 0 <= py < image_height:
                        radius = 8 if point_index == 0 else 4
                        cv2.circle(
                            bgr_image, (px, py), radius,
                            point_colors[point_index % len(point_colors)], -1)
                        if point_index == 0:
                            cv2.circle(
                                bgr_image, (px, py), radius, (0, 0, 0), 2)

        iteration_dir = (
            self.visualization_dir /
            f'iter_{self.visualization_iteration:06d}')
        iteration_dir.mkdir(parents=True, exist_ok=True)
        output_path = iteration_dir / (
            f'batch_{batch_index:02d}_view_{view_index:03d}_pred.jpg')
        if not cv2.imwrite(str(output_path), bgr_image):
            raise IOError(f'Failed to save visualization to {output_path}.')

    def _save_gt_visualization(self, image, source_sample,
                               batch_index: int, view_index: int) -> None:
        if self.visualization_dir is None or not is_main_process():
            return
        image_height, image_width = self._image_shape(
            source_sample, view_index, image.shape[-2:])
        rgb_image = image[:, :image_height, :image_width]
        rgb_image = rgb_image.permute(1, 2, 0).detach().cpu().numpy()
        bgr_image = rgb_image.clip(0, 255).astype('uint8')[..., ::-1].copy()

        gt_instances = source_sample.gt_instances_3d
        gt_boxes = gt_instances.bboxes_2d
        gt_points = gt_instances.keypoints_2d
        gt_visible = getattr(gt_instances, 'keypoints_visible', None)
        gt_box_visible = getattr(gt_instances, 'bboxes_2d_visible', None)
        gt_labels = getattr(gt_instances, 'labels_3d', None)
        if gt_boxes.ndim == 3:
            gt_boxes = gt_boxes[:, view_index]
        if gt_points.ndim == 4:
            gt_points = gt_points[:, view_index]
        if gt_visible is not None and gt_visible.ndim == 3:
            gt_visible = gt_visible[:, view_index]
        if gt_box_visible is not None and gt_box_visible.ndim == 2:
            gt_box_visible = gt_box_visible[:, view_index]
        scene_id = source_sample.metainfo.get(
            'scene_id', source_sample.metainfo.get(
                'sample_idx', source_sample.metainfo.get('scan_id', 'unknown')))
        if isinstance(scene_id, (list, tuple)):
            scene_id = scene_id[view_index]
        for gt_index, bbox in enumerate(gt_boxes):
            bbox_size = bbox[2:] - bbox[:2]
            if (gt_box_visible is not None and
                    not bool(gt_box_visible[gt_index])):
                continue
            if not bool(torch.isfinite(bbox).all()) or not bool(
                    (bbox_size > 0).all()):
                continue
            coords = bbox * bbox.new_tensor(
                [image_width, image_height, image_width, image_height])
            x1, y1, x2, y2 = [int(round(float(value))) for value in coords]
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label_index = (int(gt_labels[gt_index])
                           if gt_labels is not None else -1)
            label_name = (self.classes[label_index]
                          if 0 <= label_index < len(self.classes)
                          else str(label_index))
            text = f'GT {label_name} scene={scene_id}'
            cv2.putText(bgr_image, text, (max(x1, 0), max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                        cv2.LINE_AA)
            points = gt_points[gt_index]
            point_mask = (gt_visible[gt_index].bool()
                          if gt_visible is not None else
                          torch.ones(len(points), dtype=torch.bool))
            for point_index, (point, visible) in enumerate(
                    zip(points, point_mask)):
                if not bool(visible):
                    continue
                px = int(round(float(point[0]) * image_width))
                py = int(round(float(point[1]) * image_height))
                if 0 <= px < image_width and 0 <= py < image_height:
                    is_center = point_index == len(points) - 1
                    radius = 8 if is_center else 4
                    color = (0, 0, 255) if is_center else (255, 0, 255)
                    cv2.circle(bgr_image, (px, py), radius, color, -1)
                    if is_center:
                        cv2.circle(
                            bgr_image, (px, py), radius, (0, 0, 0), 2)

        iteration_dir = self.visualization_dir / (
            f'iter_{self.visualization_iteration:06d}')
        iteration_dir.mkdir(parents=True, exist_ok=True)
        output_path = iteration_dir / (
            f'batch_{batch_index:02d}_view_{view_index:03d}_gt.jpg')
        if not cv2.imwrite(str(output_path), bgr_image):
            raise IOError(f'Failed to save visualization to {output_path}.')

    def _make_training_samples(self, batch_data_samples, num_views,
                               padded_shape):
        image_height, image_width = padded_shape
        samples = []
        for source_sample in batch_data_samples:
            source_instances = source_sample.gt_instances_3d
            # 3D ScanNet annotations use ``labels_3d``. GroundingDINO's
            # per-view samples require the generic 2D ``labels`` field.
            source_labels = getattr(source_instances, 'labels_3d', None)
            if source_labels is None:
                source_labels = getattr(source_instances, 'labels', None)
            if source_labels is None:
                boxes_3d = getattr(source_instances, 'bboxes_3d', None)
                num_boxes = len(boxes_3d) if boxes_3d is not None else 0
                label_device = getattr(boxes_3d, 'device', None)
                if label_device is None:
                    label_device = getattr(boxes_3d, 'tensor', None)
                    label_device = getattr(label_device, 'device', None)
                source_labels = torch.zeros(
                    (num_boxes,), dtype=torch.long,
                    device=label_device)
            for view_index in range(num_views):
                sample = self._make_data_sample(
                    source_sample, view_index, padded_shape)
                instances = InstanceData()
                bboxes = source_instances.bboxes_2d
                if bboxes.ndim == 3:
                    bboxes = bboxes[:, view_index]
                bbox_visible = source_instances.bboxes_2d_visible
                if bbox_visible.ndim == 2:
                    bbox_visible = bbox_visible[:, view_index]
                bbox_size = bboxes[:, 2:] - bboxes[:, :2]
                valid = bbox_visible.bool()
                valid &= torch.isfinite(bboxes).all(dim=-1)
                valid &= (bbox_size > 0).all(dim=-1)

                keypoints = source_instances.keypoints_2d
                visible = source_instances.keypoints_visible
                if keypoints.ndim == 4:
                    keypoints = keypoints[:, view_index]
                    visible = visible[:, view_index]

                instances.labels = source_labels[valid].clone()
                instances.bboxes_2d = bboxes[valid]
                instances.bboxes = bboxes[valid] * bboxes.new_tensor(
                    [image_width, image_height, image_width, image_height])
                instances.keypoints_2d = keypoints[valid]
                instances.keypoints_visible = visible[valid]
                sample.gt_instances = instances
                samples.append(sample)
        return samples

    def _normalize_view(self, image, source_sample, view_index):
        normalized = (image[None].float() - self.image_mean) / self.image_std
        image_height, image_width = self._image_shape(
            source_sample, view_index, image.shape[-2:])
        if image_height < image.shape[-2]:
            normalized[:, :, image_height:, :] = 0
        if image_width < image.shape[-1]:
            normalized[:, :, :, image_width:] = 0
        return normalized

    @torch.no_grad()
    def _visualize_training_batch(self, images, batch_data_samples):
        was_training = self.model.training
        self.model.eval()
        padded_shape = images.shape[-2:]
        for batch_index, source_sample in enumerate(batch_data_samples):
            for view_index in range(images.shape[1]):
                image = images[batch_index, view_index]
                data_sample = self._make_data_sample(
                    source_sample, view_index, padded_shape)
                prediction = self.model.predict(
                    self._normalize_view(image, source_sample, view_index),
                    [data_sample],
                    rescale=False)[0]
                self._save_gt_visualization(
                    image, source_sample, batch_index, view_index)
                self._save_visualization(
                    image, prediction, source_sample, batch_index, view_index)

        self.visualization_iteration += 1
        if was_training:
            self.model.train()
            for module_name in ('backbone', 'neck', 'encoder',
                                'language_model'):
                module = getattr(self.model, module_name, None)
                if module is not None:
                    module.eval()

    def loss(self, images, batch_data_samples):
        self.training_iteration += 1
        batch_size, num_views = images.shape[:2]
        padded_shape = images.shape[-2:]
        if (self.visualization_dir is not None and
                is_main_process() and
                self.training_iteration % self.visualization_interval == 0):
            self._visualize_training_batch(images, batch_data_samples)

        normalized = self._normalize_images(images, batch_data_samples)
        samples = self._make_training_samples(
            batch_data_samples, num_views, padded_shape)
        losses = self.model.loss(normalized, samples)
        return losses

    @torch.no_grad()
    def predict_and_print(self, images, batch_data_samples) -> None:
        was_training = self.model.training
        self.model.eval()
        batch_size, num_views = images.shape[:2]
        padded_shape = images.shape[-2:]
        flattened = self._normalize_images(images, batch_data_samples)

        flat_samples = []
        view_indices = []
        for batch_index, source_sample in enumerate(batch_data_samples):
            for view_index in range(num_views):
                flat_samples.append(self._make_data_sample(
                    source_sample, view_index, padded_shape))
                view_indices.append((batch_index, view_index))

        predictions = self.model.predict(
            flattened, flat_samples, rescale=False)
        for prediction, (batch_index, view_index) in zip(
                predictions, view_indices):
            instances = prediction.pred_instances
            keep = instances.scores >= self.print_score_thr
            kept_indices = keep.nonzero(as_tuple=False).flatten().tolist()
            label_names = getattr(instances, 'label_names', [])
            output = {
                'bboxes': instances.bboxes[keep].detach().cpu(),
                'scores': instances.scores[keep].detach().cpu(),
                'labels': instances.labels[keep].detach().cpu(),
                'label_names': [label_names[index]
                                for index in kept_indices],
            }
            print(
                f'[GroundingDINO] batch={batch_index} view={view_index} '
                f'score_thr={self.print_score_thr}: {output}',
                flush=True)
            self._save_visualization(
                images[batch_index, view_index], prediction,
                batch_data_samples[batch_index], batch_index, view_index)
        self.visualization_iteration += 1
        if was_training:
            self.model.train()
            for module_name in ('backbone', 'neck', 'encoder',
                                'language_model'):
                module = getattr(self.model, module_name, None)
                if module is not None:
                    module.eval()
