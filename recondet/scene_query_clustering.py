import torch


class WeightedFPSKMeans:

    def __init__(self, num_clusters=256, num_iterations=5):
        self.num_clusters = num_clusters
        self.num_iterations = num_iterations

    @staticmethod
    def _candidate_weight(class_scores):
        return class_scores.sigmoid().amax(dim=-1).clamp_min(1e-6)

    def _weighted_fps(self, points, weights):
        num_candidates = points.shape[0]
        num_centers = min(self.num_clusters, num_candidates)
        centers = points.new_empty(num_centers, dtype=torch.long)
        centers[0] = weights.argmax()
        min_distance = torch.full(
            (num_candidates,), float('inf'), device=points.device,
            dtype=points.dtype)
        for center_id in range(1, num_centers):
            distance = (points - points[centers[center_id - 1]]).square().sum(-1)
            min_distance = torch.minimum(min_distance, distance)
            priority = min_distance * weights
            priority[centers[:center_id]] = -torch.inf
            centers[center_id] = priority.argmax()
        return centers

    def __call__(self, points, queries, class_scores):
        batch_size, num_candidates, _ = points.shape
        weights = self._candidate_weight(class_scores)
        outputs_points = []
        outputs_queries = []
        outputs_scores = []
        for batch_id in range(batch_size):
            sample_points = points[batch_id]
            sample_queries = queries[batch_id]
            sample_scores = class_scores[batch_id]
            sample_weights = weights[batch_id]
            init_ids = self._weighted_fps(sample_points, sample_weights)
            centers = sample_points[init_ids].clone()
            center_queries = sample_queries[init_ids].clone()
            center_scores = sample_scores[init_ids].clone()

            for _ in range(self.num_iterations):
                distance = torch.cdist(sample_points, centers, p=2).square()
                assignment = distance.argmin(dim=1)
                new_centers = []
                new_queries = []
                new_scores = []
                for cluster_id in range(len(centers)):
                    members = assignment == cluster_id
                    if members.any():
                        member_weights = sample_weights[members]
                        normalized = member_weights / member_weights.sum().clamp_min(1e-6)
                        new_centers.append((normalized[:, None] * sample_points[members]).sum(0))
                        new_queries.append((normalized[:, None] * sample_queries[members]).sum(0))
                        new_scores.append((normalized[:, None] * sample_scores[members]).sum(0))
                    else:
                        new_centers.append(centers[cluster_id])
                        new_queries.append(center_queries[cluster_id])
                        new_scores.append(center_scores[cluster_id])
                centers = torch.stack(new_centers)
                center_queries = torch.stack(new_queries)
                center_scores = torch.stack(new_scores)

            if len(centers) < self.num_clusters:
                pad_count = self.num_clusters - len(centers)
                pad_ids = torch.arange(
                    pad_count, device=points.device) % num_candidates
                centers = torch.cat([centers, sample_points[pad_ids]], dim=0)
                center_queries = torch.cat(
                    [center_queries, sample_queries[pad_ids]], dim=0)
                center_scores = torch.cat(
                    [center_scores, sample_scores[pad_ids]], dim=0)
            outputs_points.append(centers[:self.num_clusters])
            outputs_queries.append(center_queries[:self.num_clusters])
            outputs_scores.append(center_scores[:self.num_clusters])

        return (torch.stack(outputs_points), torch.stack(outputs_queries),
                torch.stack(outputs_scores))
