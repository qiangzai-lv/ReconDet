import torch
import torch.nn as nn
from torch import no_grad


class GroundingDINO3DHead(nn.Module):

    def __init__(self, query_dims=512, semantic_dims=256,
                 point_range=(-6.5, -9.0, -1.0, 6.5, 9.0, 4.5)):
        super().__init__()
        self.class_query_projection = nn.Linear(query_dims, semantic_dims)
        self.point_head = nn.Linear(query_dims, 3)
        point_range = torch.as_tensor(point_range, dtype=torch.float32).reshape(2, 3)
        self.register_buffer('point_range', point_range, persistent=False)
        with no_grad():
            nn.init.zeros_(self.point_head.weight)
            nn.init.zeros_(self.point_head.bias)

    def forward(self, query, memory_text, text_token_mask, class_branch):
        class_query = self.class_query_projection(query)
        class_scores = class_branch(
            class_query, memory_text, text_token_mask)
        points_normalized = self.point_head(query).sigmoid()
        points_3d = self.point_range[0] + points_normalized * (
            self.point_range[1] - self.point_range[0])
        return class_scores, points_3d
