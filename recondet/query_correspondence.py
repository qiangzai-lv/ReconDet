import torch


def assign_points_to_clusters(points, cluster_points):
    """Assign every candidate point to its nearest final cluster center."""
    return torch.cdist(points.float(), cluster_points.float()).argmin(dim=-1)


def select_candidate_indices(scores, threshold, minimum):
    """Select thresholded candidates, falling back to the top minimum."""
    indices = torch.nonzero(scores > threshold, as_tuple=False).flatten()
    if indices.numel() < minimum:
        indices = scores.topk(min(minimum, scores.numel())).indices
    return indices


def aggregate_cluster_view_references(points, bbox_centers, scores, view_ids,
                                      cluster_points, num_views,
                                      assignments=None):
    """Aggregate candidate bbox centers for each 3D cluster and view."""
    batch_size, num_clusters = cluster_points.shape[:2]
    if assignments is None:
        assignments = assign_points_to_clusters(points, cluster_points)

    group_ids = assignments * num_views + view_ids
    num_groups = num_clusters * num_views
    weights = scores.clamp_min(1e-6)
    weighted_centers = bbox_centers * weights[..., None]
    center_sums = bbox_centers.new_zeros((batch_size, num_groups, 2))
    center_sums.scatter_add_(
        1, group_ids[..., None].expand(-1, -1, 2), weighted_centers)
    weight_sums = scores.new_zeros((batch_size, num_groups))
    weight_sums.scatter_add_(1, group_ids, weights)

    references = center_sums / weight_sums.clamp_min(1e-6)[..., None]
    references = references.reshape(batch_size, num_clusters, num_views, 2)
    view_mask = weight_sums.reshape(
        batch_size, num_clusters, num_views).gt(0)
    return references, view_mask
