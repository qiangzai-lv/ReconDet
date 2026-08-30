import torch.nn as nn


class GroundingDINO3DHead(nn.Module):

    def __init__(self, query_dims=512, semantic_dims=256):
        super().__init__()
        self.class_query_projection = nn.Linear(query_dims, semantic_dims)
        self.point_head = nn.Linear(query_dims, 3)

    def forward(self, query, memory_text, text_token_mask, class_branch):
        class_query = self.class_query_projection(query)
        class_scores = class_branch(
            class_query, memory_text, text_token_mask)
        points_3d = self.point_head(query)
        return class_scores, points_3d
