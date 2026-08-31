# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmdet.evaluation import eval_map
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.evaluation import indoor_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import get_box_type


@METRICS.register_module()
class IndoorMetric(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(IndoorMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_3d = data_sample['pred_instances_3d']
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            self.results.append((eval_ann_info, cpu_pred_3d))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = []
        pred_results = []

        for eval_ann, sinlge_pred_results in results:
            ann_infos.append(eval_ann)
            pred_results.append(sinlge_pred_results)

        # ``indoor_eval`` cannot format AP tables when every prediction is
        # empty. This is expected when 3D detection is intentionally disabled
        # while evaluating the 2D branch.
        if pred_results and all(
                len(pred.get('scores_3d', ())) == 0
                for pred in pred_results):
            empty_metrics = {}
            for iou_thr in self.iou_thr:
                suffix = f'{iou_thr:.2f}'
                for class_name in self.dataset_meta['classes']:
                    empty_metrics[f'{class_name}_AP_{suffix}'] = 0.0
                    empty_metrics[f'{class_name}_rec_{suffix}'] = 0.0
                empty_metrics[f'mAP_{suffix}'] = 0.0
                empty_metrics[f'mAR_{suffix}'] = 0.0
            return empty_metrics

        # some checkpoints may not record the key "box_type_3d"
        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta.get('box_type_3d', 'depth'))

        ret_dict = indoor_eval(
            ann_infos,
            pred_results,
            self.iou_thr,
            self.dataset_meta['classes'],
            logger=logger,
            box_mode_3d=box_mode_3d)

        return ret_dict


@METRICS.register_module()
class Indoor2DMetric(BaseMetric):
    """indoor 2d predictions evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 iou_thr: Union[float, List[float]] = [0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None):
        super(Indoor2DMetric, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            gt_instances = data_sample['gt_instances_3d']
            if isinstance(gt_instances, dict):
                gt_bboxes = gt_instances['bboxes_2d']
                gt_labels = gt_instances['labels_3d']
                gt_visible = gt_instances['bboxes_2d_visible']
            else:
                gt_bboxes = gt_instances.bboxes_2d
                gt_labels = gt_instances.labels_3d
                gt_visible = gt_instances.bboxes_2d_visible
            if gt_bboxes.ndim == 2:
                gt_bboxes = gt_bboxes[:, None]
            if gt_visible.ndim == 1:
                gt_visible = gt_visible[:, None]
            num_views = gt_bboxes.shape[1]
            if isinstance(data_sample, dict):
                metainfo = data_sample.get('metainfo', data_sample)
            else:
                metainfo = data_sample.metainfo
            image_shape = metainfo.get(
                'img_shape', metainfo.get('batch_input_shape'))
            if isinstance(image_shape[0], (list, tuple)):
                image_shapes = image_shape
            else:
                image_shapes = [image_shape] * num_views

            valid_boxes, valid_labels = [], []
            for view_id in range(num_views):
                boxes = gt_bboxes[:, view_id]
                visible = gt_visible[:, view_id].bool()
                height, width = image_shapes[view_id][:2]
                boxes = boxes * boxes.new_tensor([width, height, width, height])
                valid = visible & torch.isfinite(boxes).all(dim=-1)
                valid &= (boxes[:, 2:] > boxes[:, :2]).all(dim=-1)
                valid &= gt_labels >= 0
                valid_boxes.append(boxes[valid])
                valid_labels.append(gt_labels[valid])
            ann = dict(
                labels=(torch.cat(valid_labels) if valid_labels else
                        gt_labels[:0]).cpu().numpy(),
                bboxes=(torch.cat(valid_boxes) if valid_boxes else
                        gt_bboxes.new_zeros((0, 4))).cpu().numpy())

            pred_bboxes = pred['bboxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            dets = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(pred_labels == label)[0]
                pred_bbox_scores = np.hstack(
                    [pred_bboxes[index], pred_scores[index].reshape((-1, 1))])
                dets.append(pred_bbox_scores)

            self.results.append((ann, dets))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        annotations, preds = zip(*results)
        eval_results = OrderedDict()
        for iou_thr_2d_single in self.iou_thr:
            mean_ap, _ = eval_map(
                preds,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr_2d_single,
                dataset=self.dataset_meta['classes'],
                logger=logger)
            eval_results['mAP_' + str(iou_thr_2d_single)] = mean_ap
        return eval_results
