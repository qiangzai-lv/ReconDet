"""Grounding-DINO head with center and face-offset regression."""

import copy
from itertools import permutations
from typing import List

import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import InstanceList
from .grounding_dino_head import GroundingDINOHead


@MODELS.register_module()
class GroundingDINOKeypointHead(GroundingDINOHead):
    """Grounding-DINO with a center branch and three face-offset branches.

    The inherited bbox/classification heads and matching remain unchanged.
    Keypoints are trained after the inherited matching has selected positive
    queries, which keeps pretrained Grounding-DINO behavior intact.
    """

    def __init__(self,
                 keypoint_center_loss_weight=1.0,
                 keypoint_face_loss_weight=1.0,
                 keypoint_inside_loss_weight=1.0,
                 **kwargs):
        self.keypoint_center_loss_weight = keypoint_center_loss_weight
        self.keypoint_face_loss_weight = keypoint_face_loss_weight
        self.keypoint_inside_loss_weight = keypoint_inside_loss_weight
        super().__init__(**kwargs)

    def _init_layers(self):
        super()._init_layers()
        center_branch = []
        face_branch = []
        for _ in range(self.num_reg_fcs):
            center_branch.extend([nn.Linear(self.embed_dims, self.embed_dims),
                                  nn.ReLU()])
            face_branch.extend([nn.Linear(self.embed_dims, self.embed_dims),
                                nn.ReLU()])
        center_branch.append(nn.Linear(self.embed_dims, 2))
        face_branch.append(nn.Linear(self.embed_dims, 6))
        center_module = nn.Sequential(*center_branch)
        face_module = nn.Sequential(*face_branch)
        # Grounding-DINO's extra prediction layer is reserved for encoder
        # proposals; keypoints are predicted only by decoder layers.
        num_decoder_layers = self.num_pred_layer - int(self.as_two_stage)
        self.center_branches = nn.ModuleList(
            [center_module if self.share_pred_layer else copy.deepcopy(center_module)
             for _ in range(num_decoder_layers)])
        self.face_offset_branches = nn.ModuleList(
            [face_module if self.share_pred_layer else copy.deepcopy(face_module)
             for _ in range(num_decoder_layers)])

    def init_weights(self):
        super().init_weights()
        for branch in list(self.center_branches) + list(self.face_offset_branches):
            nn.init.constant_(branch[-1].weight, 0.)
            nn.init.constant_(branch[-1].bias, 0.)

    def forward(self, hidden_states, references, memory_text, text_token_mask):
        cls_scores, bbox_preds = super().forward(
            hidden_states, references, memory_text, text_token_mask)
        centers = []
        offsets = []
        for layer_id in range(hidden_states.shape[0]):
            feat = hidden_states[layer_id]
            centers.append(self.center_branches[layer_id](feat).sigmoid())
            offsets.append(self.face_offset_branches[layer_id](feat))
        self._last_keypoint_centers = torch.stack(centers)
        self._last_keypoint_offsets = torch.stack(offsets)
        self._last_bbox_preds = bbox_preds
        self._last_cls_scores = cls_scores
        return cls_scores, bbox_preds

    @staticmethod
    def assign_face_points(pred: Tensor, target: Tensor,
                           visibility: Tensor):
        """Find the minimum-cost permutation for each batch item."""
        perms = pred.new_tensor(list(permutations(range(3))), dtype=torch.long)
        costs = []
        for perm in perms:
            costs.append((pred[:, perm] - target).abs().sum(-1) * visibility)
        costs = torch.stack(costs, dim=1).sum(-1)
        best_cost, best = costs.min(dim=1)
        return perms[best], best_cost

    @staticmethod
    def face_inside_penalty(points: Tensor, boxes: Tensor,
                            visibility: Tensor):
        """Hinge penalty for visible face points outside xyxy boxes."""
        x, y = points.unbind(-1)
        x1, y1, x2, y2 = boxes.unbind(-1)
        penalty = (torch.relu(x1[:, None] - x) + torch.relu(x - x2[:, None]) +
                   torch.relu(y1[:, None] - y) + torch.relu(y - y2[:, None]))
        return (penalty * visibility).sum() / visibility.sum().clamp_min(1.)

    def _keypoint_loss_layer(self, centers, offsets, bbox_preds, cls_scores,
                             batch_gt_instances, batch_img_metas, dn_meta):
        num_dn = int((dn_meta or {}).get('num_denoising_queries', 0))
        centers = centers[:, num_dn:]
        offsets = offsets[:, num_dn:].reshape(offsets.size(0), -1, 3, 2)
        bbox_preds = bbox_preds[:, num_dn:]
        cls_scores = cls_scores[:, num_dn:]
        center_losses, face_losses, inside_losses = [], [], []
        for img_id, (gt, meta) in enumerate(zip(batch_gt_instances,
                                                 batch_img_metas)):
            if not hasattr(gt, 'keypoints_2d') or len(gt.bboxes) == 0:
                continue
            h, w = meta['img_shape'][:2]
            factor = centers.new_tensor([w, h])
            gt_kp = gt.keypoints_2d.reshape(-1, 4, 2).to(centers)
            gt_vis = getattr(gt, 'keypoints_visibility',
                             centers.new_ones((len(gt_kp), 4))).to(centers)
            gt_center = gt_kp[:, 0] / factor
            gt_face = gt_kp[:, 1:] / factor
            gt_offsets = gt_face - gt_center[:, None]
            pred_box = bbox_preds[img_id]
            pred_box_xyxy = torch.stack((pred_box[:, 0] - pred_box[:, 2] / 2,
                                         pred_box[:, 1] - pred_box[:, 3] / 2,
                                         pred_box[:, 0] + pred_box[:, 2] / 2,
                                         pred_box[:, 1] + pred_box[:, 3] / 2), -1)
            pred_box_xyxy = pred_box_xyxy * centers.new_tensor([w, h, w, h])
            pred_inst = InstanceData(scores=cls_scores[img_id],
                                     bboxes=pred_box_xyxy)
            assign = self.assigner.assign(pred_inst, gt, meta)
            pos = torch.nonzero(assign.gt_inds > 0).squeeze(-1)
            if pos.numel() == 0:
                continue
            gt_ids = assign.gt_inds[pos].long() - 1
            pc = centers[img_id, pos]
            po = offsets[img_id, pos]
            tc = gt_center[gt_ids]
            to = gt_offsets[gt_ids]
            tv = gt_vis[gt_ids, 1:4]
            center_losses.append(torch.abs(pc - tc).mean())
            perm, _ = self.assign_face_points(po, to, tv)
            po_assigned = po[torch.arange(len(po), device=po.device)[:, None], perm]
            face_abs = torch.abs(po_assigned - to)
            face_losses.append((face_abs * tv[..., None]).sum() /
                               tv.sum().clamp_min(1.))
            pred_points = (pc[:, None] + po_assigned) * centers.new_tensor([w, h])
            boxes = gt.bboxes[gt_ids].to(pred_points)
            inside_losses.append(self.face_inside_penalty(pred_points, boxes, tv))
        zero = centers.sum() * 0. + offsets.sum() * 0.
        return (torch.stack(center_losses).mean() if center_losses else zero,
                torch.stack(face_losses).mean() if face_losses else zero,
                torch.stack(inside_losses).mean() if inside_losses else zero)

    def loss(self, hidden_states, references, memory_text, text_token_mask,
             enc_outputs_class, enc_outputs_coord, batch_data_samples, dn_meta):
        losses = super().loss(hidden_states, references, memory_text,
                              text_token_mask, enc_outputs_class,
                              enc_outputs_coord, batch_data_samples, dn_meta)
        batch_gt = [sample.gt_instances for sample in batch_data_samples]
        metas = [sample.metainfo for sample in batch_data_samples]
        centers = self._last_keypoint_centers
        offsets = self._last_keypoint_offsets
        for layer_id in range(centers.shape[0]):
            lc, lf, li = self._keypoint_loss_layer(
                centers[layer_id], offsets[layer_id],
                self._last_bbox_preds[layer_id],
                self._last_cls_scores[layer_id],
                batch_gt, metas, dn_meta)
            suffix = '' if layer_id == centers.shape[0] - 1 else f'.d{layer_id}'
            losses[f'loss_keypoint_center{suffix}'] = lc * self.keypoint_center_loss_weight
            losses[f'loss_keypoint_face{suffix}'] = lf * self.keypoint_face_loss_weight
            losses[f'loss_keypoint_inside{suffix}'] = li * self.keypoint_inside_loss_weight
        return losses

    def _predict_by_feat_single(self, cls_score, bbox_pred,
                                token_positive_maps, img_meta,
                                rescale=True):
        results = super()._predict_by_feat_single(
            cls_score, bbox_pred, token_positive_maps, img_meta, rescale)
        max_per_img = self.test_cfg.get('max_per_img', len(cls_score))
        if token_positive_maps is not None:
            from .grounding_dino_head import convert_grounding_to_cls_scores
            cls = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]
            _, indexes = cls.view(-1).topk(max_per_img)
            num_classes = cls.shape[-1]
            query_inds = indexes // num_classes
        else:
            scores, _ = cls_score.sigmoid().max(-1)
            _, query_inds = scores.topk(max_per_img)
        image_id = getattr(self, '_predict_image_id', 0)
        center = self._last_keypoint_centers[-1, image_id, query_inds]
        offsets = self._last_keypoint_offsets[-1, image_id, query_inds].reshape(-1, 3, 2)
        h, w = img_meta['img_shape'][:2]
        keypoints = torch.cat((center[:, None], center[:, None] + offsets), dim=1)
        keypoints = keypoints * keypoints.new_tensor([w, h])
        if rescale:
            keypoints /= keypoints.new_tensor(img_meta['scale_factor'])[:2]
        results.keypoints = keypoints
        return results

    def predict_by_feat(self, all_layers_cls_scores, all_layers_bbox_preds,
                        batch_img_metas, batch_token_positive_maps=None,
                        rescale=False):
        result_list = []
        for image_id, (meta, token_maps) in enumerate(
                zip(batch_img_metas, batch_token_positive_maps or [None] * len(batch_img_metas))):
            self._predict_image_id = image_id
            result_list.append(self._predict_by_feat_single(
                all_layers_cls_scores[-1, image_id],
                all_layers_bbox_preds[-1, image_id], token_maps, meta, rescale))
        return result_list
