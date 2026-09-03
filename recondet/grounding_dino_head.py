# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmengine.model import constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy
from mmdet.utils import InstanceList, reduce_mean
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.dense_heads.atss_vlfusion_head import (
    convert_grounding_to_cls_scores)
from mmdet.models.dense_heads.dino_head import DINOHead
from recondet.grounding_dino_3d_head import GroundingDINO3DHead


class ContrastiveEmbed(nn.Module):
    """text visual ContrastiveEmbed layer.

    Args:
        max_text_len (int, optional): Maximum length of text.
        log_scale (Optional[Union[str, float]]):  The initial value of a
          learnable parameter to multiply with the similarity
          matrix to normalize the output.  Defaults to 0.0.
          - If set to 'auto', the similarity matrix will be normalized by
            a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel number.
          - If set to 'none' or ``None``, there is no normalization applied.
          - If set to a float number, the similarity matrix will be multiplied
            by ``exp(log_scale)``, where ``log_scale`` is learnable.
        bias (bool, optional): Whether to add bias to the output.
          If set to ``True``, a learnable bias that is initialized as -4.6
          will be added to the output. Useful when training from scratch.
          Defaults to False.
    """

    def __init__(self,
                 max_text_len: int = 256,
                 log_scale: Optional[Union[str, float]] = None,
                 bias: bool = False):
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale
        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True)
        elif log_scale not in ['auto', 'none', None]:
            raise ValueError(f'log_scale should be one of '
                             f'"auto", "none", None, but got {log_scale}')

        self.bias = None
        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True)

    def forward(self, visual_feat: Tensor, text_feat: Tensor,
                text_token_mask: Tensor) -> Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_feat (Tensor): Text features.
            text_token_mask (Tensor): A mask used for text feats.

        Returns:
            Tensor: Classification score.
        """
        res = visual_feat @ text_feat.transpose(-1, -2)
        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == 'auto':
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])
        if self.bias is not None:
            res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float('-inf'))

        new_res = torch.full((*res.shape[:-1], self.max_text_len),
                             float('-inf'),
                             device=res.device)
        new_res[..., :res.shape[-1]] = res

        return new_res


@MODELS.register_module()
class ReconGroundingDINOHead(DINOHead):
    """Head of the Grounding DINO: Marrying DINO with Grounded Pre-Training for
    Open-Set Object Detection.

    Args:
        contrastive_cfg (dict, optional): Contrastive config that contains
          keys like ``max_text_len``. Defaults to dict(max_text_len=256).
    """

    def __init__(self, contrastive_cfg=dict(max_text_len=256), **kwargs):
        self.contrastive_cfg = contrastive_cfg
        self.max_text_len = contrastive_cfg.get('max_text_len', 256)
        self.reconstruction_dims = kwargs.pop('reconstruction_dims', 512)
        self.point_range = kwargs.pop(
            'point_range', (-6.5, -9.0, -1.0, 6.5, 9.0, 4.5))
        self.point_3d_loss_weight = kwargs.pop('point_3d_loss_weight', 1.0)
        self.class_3d_loss_weight = kwargs.pop('class_3d_loss_weight', 1.0)
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize classification branch and regression branch of head."""
        fc_cls = ContrastiveEmbed(**self.contrastive_cfg)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters,we do not need to copy it.
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)])
        else:
            self.cls_branches = nn.ModuleList(
                [copy.deepcopy(fc_cls) for _ in range(self.num_pred_layer)])
            self.reg_branches = nn.ModuleList([
                copy.deepcopy(reg_branch) for _ in range(self.num_pred_layer)
            ])
        self.reconstruction_head = GroundingDINO3DHead(
            query_dims=self.reconstruction_dims,
            semantic_dims=self.embed_dims,
            point_range=self.point_range)

    def init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
    ) -> Tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](hidden_state,
                                                        memory_text,
                                                        text_token_mask)
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)
        self._last_cls_scores = all_layers_outputs_classes.detach()
        self._last_bbox_preds = all_layers_outputs_coords.detach()
        return all_layers_outputs_classes, all_layers_outputs_coords

    def _get_3d_targets_single(self, cls_score, bbox_pred, gt_instances,
                               img_meta):
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
        pred_boxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        assign_result = self.assigner.assign(
            pred_instances=InstanceData(
                scores=cls_score, bboxes=pred_boxes),
            gt_instances=gt_instances,
            img_meta=img_meta)

        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).flatten()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).flatten()
        assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        labels = cls_score.new_zeros(
            (bbox_pred.shape[0], self.max_text_len))
        label_weights = cls_score.new_ones(bbox_pred.shape[0])
        point_targets = bbox_pred.new_zeros((bbox_pred.shape[0], 3))
        point_weights = bbox_pred.new_zeros(bbox_pred.shape[0])
        if pos_inds.numel() > 0:
            labels[pos_inds] = gt_instances.positive_maps[assigned_gt_inds]
            point_targets[pos_inds] = gt_instances.centers_3d[
                assigned_gt_inds]
            point_weights[pos_inds] = 1.0
        return (labels, label_weights, point_targets, point_weights,
                pos_inds.numel(), neg_inds.numel())

    def _loss_3d(self, reconstruction_hidden_states, cls_scores_2d,
                 bbox_preds_2d, memory_text, text_token_mask,
                 batch_gt_instances, batch_img_metas, class_branch):
        query = reconstruction_hidden_states[-1]
        cls_scores_3d, points_3d = self.reconstruction_head(
            query, memory_text, text_token_mask, class_branch)

        targets = [
            self._get_3d_targets_single(
                cls_score, bbox_pred, gt_instances, img_meta)
            for cls_score, bbox_pred, gt_instances, img_meta in zip(
                cls_scores_2d, bbox_preds_2d, batch_gt_instances,
                batch_img_metas)
        ]
        labels = torch.stack([target[0] for target in targets])
        label_weights = torch.stack([target[1] for target in targets])
        point_targets = torch.stack([target[2] for target in targets])
        point_weights = torch.stack([target[3] for target in targets])
        num_total_pos = sum(target[4] for target in targets)
        num_total_neg = sum(target[5] for target in targets)

        text_masks = text_token_mask.new_zeros(
            (text_token_mask.shape[0], self.max_text_len))
        text_masks[:, :text_token_mask.shape[1]] = text_token_mask
        text_mask = (text_masks > 0).unsqueeze(1).expand(
            -1, cls_scores_3d.shape[1], -1)
        masked_cls_scores = torch.masked_select(
            cls_scores_3d, text_mask).contiguous()
        masked_labels = torch.masked_select(labels, text_mask)
        expanded_label_weights = label_weights[..., None].expand_as(text_mask)
        masked_label_weights = torch.masked_select(
            expanded_label_weights, text_mask)

        cls_avg_factor = (
            num_total_pos + num_total_neg * self.bg_cls_weight)
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores_3d.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(float(cls_avg_factor), 1.0)
        loss_cls = self.loss_cls(
            masked_cls_scores,
            masked_labels,
            masked_label_weights,
            avg_factor=cls_avg_factor)

        point_min, point_max = self.reconstruction_head.point_range
        point_targets = (point_targets - point_min) / (
            point_max - point_min)
        points_3d = (points_3d - point_min) / (point_max - point_min)
        point_error = (points_3d - point_targets).abs().mean(dim=-1)
        point_avg_factor = torch.clamp(
            reduce_mean(points_3d.new_tensor([num_total_pos])),
            min=1.0).item()
        loss_point = (
            point_error * point_weights).sum() / point_avg_factor
        return (
            self.class_3d_loss_weight * loss_cls,
            self.point_3d_loss_weight * loss_point)

    def predict_reconstruction(self, reconstruction_hidden_states,
                               memory_text, text_token_mask):
        query = reconstruction_hidden_states[-1]
        return self.reconstruction_head(
            query, memory_text, text_token_mask,
            self.cls_branches[self.num_pred_layer - 1])


    def predict(self,
                hidden_states: Tensor,
                references: List[Tensor],
                memory_text: Tensor,
                text_token_mask: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, num_queries, bs, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).
            batch_data_samples (SampleList): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            InstanceList: Detection results of each image
                after the post process.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]

        outs = self(hidden_states, references, memory_text, text_token_mask)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        all_layers_cls_scores: Tensor,
                        all_layers_bbox_preds: Tensor,
                        batch_img_metas: List[Dict],
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        rescale: bool = False) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            all_layers_cls_scores (Tensor):  Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs, num_queries,
                cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and shape (num_decoder_layers, bs, num_queries,
                4) with the last dimension arranged as (cx, cy, w, h).
            batch_img_metas (List[Dict]): _description_
            batch_token_positive_maps (list[dict], Optional): Batch token
                positive map. Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cls_scores = all_layers_cls_scores[-1]
        bbox_preds = all_layers_bbox_preds[-1]
        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred,
                                                   token_positive_maps,
                                                   img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                bbox_pred: Tensor,
                                token_positive_maps: dict,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        img_shape = img_meta['img_shape']

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            bbox_pred = bbox_pred[bbox_index]
            query_indices = bbox_index
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(max_per_img)
            bbox_pred = bbox_pred[indexes]
            query_indices = indexes
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))
        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels
        results.query_indices = query_indices
        return results

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             memory_text: Tensor, text_token_mask: Tensor,
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int],
             reconstruction_hidden_states: Tensor = None) -> dict:
        """Compute only the reconstruction 3D supervision losses."""
        if reconstruction_hidden_states is None:
            return {}

        batch_gt_instances = [
            data_sample.gt_instances for data_sample in batch_data_samples]
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples]
        cls_scores_2d, bbox_preds_2d = self(
            hidden_states, references, memory_text, text_token_mask)
        cls_scores_2d = cls_scores_2d[-1]
        bbox_preds_2d = bbox_preds_2d[-1]
        if dn_meta is not None:
            num_dn = dn_meta['num_denoising_queries']
            cls_scores_2d = cls_scores_2d[:, num_dn:]
            bbox_preds_2d = bbox_preds_2d[:, num_dn:]
        loss_3d_cls, loss_3d_point = self._loss_3d(
            reconstruction_hidden_states,
            cls_scores_2d,
            bbox_preds_2d,
            memory_text,
            text_token_mask,
            batch_gt_instances,
            batch_img_metas,
            self.cls_branches[hidden_states.shape[0] - 1])
        losses = {}
        losses['loss_3d_cls'] = loss_3d_cls
        losses['loss_3d_point'] = loss_3d_point
        return losses


@MODELS.register_module()
class ReconGroundingDINOKeypointHead(ReconGroundingDINOHead):
    """Recondet head with GDINO fine-tuning bbox/keypoint branch layout."""

    def __init__(self, keypoint_center_loss_weight=1.0,
                 keypoint_face_loss_weight=1.0,
                 keypoint_inside_loss_weight=1.0, **kwargs):
        self.keypoint_center_loss_weight = keypoint_center_loss_weight
        self.keypoint_face_loss_weight = keypoint_face_loss_weight
        self.keypoint_inside_loss_weight = keypoint_inside_loss_weight
        super().__init__(**kwargs)

    def _init_layers(self):
        super()._init_layers()
        center_branch, face_branch = [], []
        for _ in range(self.num_reg_fcs):
            center_branch.extend([nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU()])
            face_branch.extend([nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU()])
        center_branch.append(nn.Linear(self.embed_dims, 2))
        face_branch.append(nn.Linear(self.embed_dims, 6))
        center_module, face_module = nn.Sequential(*center_branch), nn.Sequential(*face_branch)
        num_layers = self.num_pred_layer - int(self.as_two_stage)
        self.center_branches = nn.ModuleList([
            center_module if self.share_pred_layer else copy.deepcopy(center_module)
            for _ in range(num_layers)])
        self.face_offset_branches = nn.ModuleList([
            face_module if self.share_pred_layer else copy.deepcopy(face_module)
            for _ in range(num_layers)])

    def init_weights(self):
        super().init_weights()
        for branch in list(self.center_branches) + list(self.face_offset_branches):
            nn.init.constant_(branch[-1].weight, 0.)
            nn.init.constant_(branch[-1].bias, 0.)

    def forward(self, hidden_states, references, memory_text, text_token_mask):
        cls_scores, bbox_preds = super().forward(
            hidden_states, references, memory_text, text_token_mask)
        centers, offsets = [], []
        for layer_id in range(hidden_states.shape[0]):
            feat = hidden_states[layer_id]
            centers.append(self.center_branches[layer_id](feat).sigmoid())
            offsets.append(self.face_offset_branches[layer_id](feat))
        self._last_keypoint_centers = torch.stack(centers)
        self._last_keypoint_offsets = torch.stack(offsets)
        return cls_scores, bbox_preds
