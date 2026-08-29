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
from mmengine.runner import load_checkpoint
from mmengine.structures import InstanceData
from torch import nn


class GroundingDINOSemanticEncoder(nn.Module):
    """Frozen GroundingDINO used for inspecting per-view 2D predictions."""

    def __init__(self,
                 config: str,
                 checkpoint: str,
                 classes: Sequence[str],
                 view_chunk_size: int = 1,
                 print_score_thr: float = 0.3,
                 visualization_dir: str = None) -> None:
        super().__init__()
        if not classes:
            raise ValueError('GroundingDINO classes must not be empty.')
        if view_chunk_size < 1:
            raise ValueError('view_chunk_size must be at least 1.')

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
        load_checkpoint(self.model, checkpoint, map_location='cpu')
        self.model._is_init = True

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for module in (self.model.decoder, self.model.bbox_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.model.eval()

        self.classes = tuple(classes)
        self.view_chunk_size = view_chunk_size
        self.print_score_thr = print_score_thr
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

        iteration_dir = (
            self.visualization_dir /
            f'iter_{self.visualization_iteration:06d}')
        iteration_dir.mkdir(parents=True, exist_ok=True)
        output_path = iteration_dir / (
            f'batch_{batch_index:02d}_view_{view_index:03d}.jpg')
        if not cv2.imwrite(str(output_path), bgr_image):
            raise IOError(f'Failed to save visualization to {output_path}.')

    def _make_training_samples(self, batch_data_samples, num_views,
                               padded_shape, view_start=0, view_end=None):
        image_height, image_width = padded_shape
        if view_end is None:
            view_end = num_views
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
            num_instances = len(source_labels)
            for view_index in range(view_start, view_end):
                sample = self._make_data_sample(
                    source_sample, view_index, padded_shape)
                instances = InstanceData()
                instances.labels = source_labels.clone()
                bboxes = source_instances.bboxes_2d
                if bboxes.ndim == 3:
                    bboxes = bboxes[:, view_index]
                instances.bboxes_2d = bboxes
                instances.bboxes = bboxes * bboxes.new_tensor(
                    [image_width, image_height, image_width, image_height])
                keypoints = source_instances.keypoints_2d
                visible = source_instances.keypoints_visible
                if keypoints.ndim == 4:
                    keypoints = keypoints[:, view_index]
                    visible = visible[:, view_index]
                instances.keypoints_2d = keypoints
                instances.keypoints_visible = visible
                if num_instances == 0:
                    instances.bboxes = instances.bboxes.reshape(0, 4)
                    instances.labels = instances.labels.reshape(0)
                sample.gt_instances = instances
                samples.append(sample)
        return samples

    def loss(self, images, batch_data_samples):
        batch_size, num_views = images.shape[:2]
        padded_shape = images.shape[-2:]
        total_views = num_views
        losses = {}
        for view_start in range(0, num_views, self.view_chunk_size):
            view_end = min(view_start + self.view_chunk_size, num_views)
            chunk_images = images[:, view_start:view_end]
            chunk_views = view_end - view_start
            normalized = self._normalize_images(
                chunk_images, batch_data_samples)
            samples = self._make_training_samples(
                batch_data_samples, num_views, padded_shape,
                view_start=view_start, view_end=view_end)
            chunk_losses = self.model.loss(
                normalized, samples, freeze_feature_extractor=True)
            weight = chunk_views / total_views
            for name, value in chunk_losses.items():
                weighted_value = value * weight
                if name in losses:
                    losses[name] = losses[name] + weighted_value
                else:
                    losses[name] = weighted_value
        return losses

    @torch.no_grad()
    def predict_and_print(self, images, batch_data_samples) -> None:
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

        for start in range(0, len(flat_samples), self.view_chunk_size):
            end = min(start + self.view_chunk_size, len(flat_samples))
            predictions = self.model.predict(
                flattened[start:end], flat_samples[start:end], rescale=False)
            for prediction, (batch_index, view_index) in zip(
                    predictions, view_indices[start:end]):
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
