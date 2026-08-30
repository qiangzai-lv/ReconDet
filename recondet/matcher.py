import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from mmdet3d.structures.ops.iou3d_calculator import axis_aligned_bbox_overlaps_3d


def _sanitize_prediction(value, min_value=None, max_value=None):
    value = torch.nan_to_num(
        value.float(), nan=0.0, posinf=1e6, neginf=-1e6)
    if min_value is not None or max_value is not None:
        value = value.clamp(min=min_value, max=max_value)
    return value


def _build_cost_matrix(all_centers, all_sizes, all_cls, all_objness,
                       gt_centers, gt_sizes, gt_labels, cost_weights):
    all_centers = _sanitize_prediction(all_centers, -100.0, 100.0)
    all_sizes = _sanitize_prediction(all_sizes, 1e-4, 100.0)
    all_cls = _sanitize_prediction(all_cls, -50.0, 50.0)
    all_objness = _sanitize_prediction(all_objness, -50.0, 50.0)
    gt_centers = gt_centers.float()
    gt_sizes = gt_sizes.float().clamp_min(1e-4)

    pred_boxes = UnifiedMatcher._center_size_pred_to_bbox(
        None, all_centers, all_sizes)
    gt_boxes = UnifiedMatcher._center_size_pred_to_bbox(
        None, gt_centers, gt_sizes)
    giou = axis_aligned_bbox_overlaps_3d(
        pred_boxes.unsqueeze(0), gt_boxes.unsqueeze(0), mode='giou').squeeze(0)
    giou = torch.nan_to_num(giou, nan=-1.0, posinf=1.0, neginf=-1.0)

    cost_class = -all_cls.sigmoid()[:, gt_labels]
    cost_center = torch.cdist(all_centers, gt_centers, p=1)
    cost_objness = -all_objness.sigmoid()
    total_cost = (
        cost_weights['cls'] * cost_class +
        cost_weights['center'] * cost_center +
        cost_weights['obj_ness'] * cost_objness -
        cost_weights['giou'] * giou)
    total_cost = torch.nan_to_num(
        total_cost, nan=1e6, posinf=1e6, neginf=1e6)
    return total_cost, giou


def _linear_sum_assignment(cost):
    cost_numpy = torch.nan_to_num(
        cost.detach().float(), nan=1e6, posinf=1e6,
        neginf=1e6).cpu().numpy()
    pred_indices, gt_indices = linear_sum_assignment(cost_numpy)
    return pred_indices, gt_indices


class UnifiedMatcher(nn.Module):
    def __init__(self, cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0}):
        super().__init__()
        self.cost_weights = cost_weights

    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, gt_centers, gt_sizes, gt_labels):
        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1)

        total_cost, _ = _build_cost_matrix(
            all_centers, all_sizes, all_cls, all_objness,
            gt_centers, gt_sizes, gt_labels, self.cost_weights)

        pred_indices, gt_indices = _linear_sum_assignment(total_cost)
        return torch.from_numpy(pred_indices).long().to(all_centers.device), torch.from_numpy(gt_indices).long().to(
            all_centers.device)

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0] / 2.0, centers[:, 1] - sizes[:, 1] / 2.0,
            centers[:, 2] - sizes[:, 2] / 2.0, centers[:, 0] + sizes[:, 0] / 2.0,
            centers[:, 1] + sizes[:, 1] / 2.0, centers[:, 2] + sizes[:, 2] / 2.0
        ], -1)


class UnifiedMatcherMoreThanOne(nn.Module):
    def __init__(self, cost_weights={'cls': 1.0, 'center': 0.0, 'obj_ness': 0.0, 'giou': 2.0}, matcher_iou_thres=0.25,
                 matcher_max_dynamic_samples=10):
        super().__init__()
        self.cost_weights = cost_weights
        self.iou_threshold = matcher_iou_thres,
        self.matcher_max_dynamic_samples = matcher_max_dynamic_samples

    @torch.no_grad()
    def _get_targets(self, all_centers, all_sizes, all_cls, all_objness, gt_centers, gt_sizes, gt_labels):
        if all_objness.dim() == 1:
            all_objness = all_objness.unsqueeze(-1)

        total_cost, giou = _build_cost_matrix(
            all_centers, all_sizes, all_cls, all_objness,
            gt_centers, gt_sizes, gt_labels, self.cost_weights)

        pred_indices, gt_indices = _linear_sum_assignment(total_cost)
        pred_indices = torch.from_numpy(pred_indices).long().to(all_centers.device)
        gt_indices = torch.from_numpy(gt_indices).long().to(all_centers.device)

        used_pred_mask = torch.zeros(giou.size(0), dtype=torch.bool, device=giou.device)
        used_pred_mask[pred_indices] = True

        iou_mask = giou > self.iou_threshold[0]

        dynamic_preds = []
        dynamic_gts = []

        max_iou_per_gt = giou.max(dim=0).values
        sorted_gt_indices = torch.argsort(max_iou_per_gt)

        for gt_idx in sorted_gt_indices:
            candidate_mask = iou_mask[:, gt_idx] & ~used_pred_mask
            candidate_preds = torch.nonzero(candidate_mask, as_tuple=True)[0]

            if candidate_preds.numel() == 0:
                continue

            giou_values = giou[candidate_preds, gt_idx]

            if self.matcher_max_dynamic_samples < len(giou_values):
                _, topk_indices = torch.topk(giou_values, k=self.matcher_max_dynamic_samples)
                selected_preds = candidate_preds[topk_indices]
            else:
                selected_preds = candidate_preds

            dynamic_preds.append(selected_preds)
            dynamic_gts.append(torch.full_like(selected_preds, gt_idx))

            used_pred_mask[selected_preds] = True

        if dynamic_preds:
            dynamic_preds = torch.cat(dynamic_preds)
            dynamic_gts = torch.cat(dynamic_gts)
        else:
            dynamic_preds = torch.empty(0, dtype=torch.long, device=giou.device)
            dynamic_gts = torch.empty(0, dtype=torch.long, device=giou.device)

        combined_preds = torch.cat([pred_indices, dynamic_preds])
        combined_gts = torch.cat([gt_indices, dynamic_gts])

        return combined_preds, combined_gts

    def _center_size_pred_to_bbox(self, centers, sizes):
        return torch.stack([
            centers[:, 0] - sizes[:, 0] / 2.0, centers[:, 1] - sizes[:, 1] / 2.0,
            centers[:, 2] - sizes[:, 2] / 2.0, centers[:, 0] + sizes[:, 0] / 2.0,
            centers[:, 1] + sizes[:, 1] / 2.0, centers[:, 2] + sizes[:, 2] / 2.0
        ], -1)
