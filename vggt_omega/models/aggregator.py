# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def inverse_sigmoid(value: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    value = value.clamp(eps, 1.0 - eps)
    return torch.log(value / (1.0 - value))


class FourierFeatures(nn.Module):
    """Fourier feature mapping for normalized UV coordinates."""

    def __init__(
        self,
        input_dim: int = 2,
        num_bands: int = 8,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_bands = num_bands
        self.include_input = include_input
        self.register_buffer(
            "frequencies",
            2.0 ** torch.arange(num_bands, dtype=torch.float32) * torch.pi,
            persistent=False,
        )

    @property
    def output_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + self.input_dim * self.num_bands * 2

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        uv = uv.float()
        encoded = uv.unsqueeze(-1) * self.frequencies.view(1, 1, 1, -1)
        features = [torch.sin(encoded), torch.cos(encoded)]
        if self.include_input:
            features.insert(0, uv.unsqueeze(-1))
        return torch.cat(features, dim=-1).flatten(start_dim=2)


class ObjectQueryBlock(nn.Module):
    """Read special tokens and sample patch tokens around an iterative UV."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float,
                 num_points: int) -> None:
        super().__init__()
        self.special_query_norm = nn.LayerNorm(embed_dim)
        self.special_memory_norm = nn.LayerNorm(embed_dim)
        self.special_cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True)
        self.deformable_query_norm = nn.LayerNorm(embed_dim)
        self.patch_memory_norm = nn.LayerNorm(embed_dim)
        self.deformable_attention = MultiScaleDeformableAttention(
            embed_dims=embed_dim,
            num_heads=num_heads,
            num_levels=1,
            num_points=num_points,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.deformable_attention.init_weights()

    def forward(
        self,
        queries: torch.Tensor,
        query_pos: torch.Tensor,
        reference_uv: torch.Tensor,
        special_memory: torch.Tensor,
        patch_memory: torch.Tensor,
        patch_grid_size: tuple[int, int],
        uv_delta_predictor: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_queries, embed_dim = queries.shape
        num_special_tokens = special_memory.shape[2]
        num_patch_tokens = patch_memory.shape[2]
        if num_patch_tokens != patch_grid_size[0] * patch_grid_size[1]:
            raise ValueError(
                "Patch memory does not match the provided spatial grid")

        per_view_queries = (queries + query_pos).reshape(
            batch_size * num_frames, num_queries, embed_dim)
        per_view_special_memory = special_memory.reshape(
            batch_size * num_frames, num_special_tokens, embed_dim)
        normalized_special_memory = self.special_memory_norm(
            per_view_special_memory)
        special_features, _ = self.special_cross_attention(
            self.special_query_norm(per_view_queries),
            normalized_special_memory,
            normalized_special_memory,
            need_weights=False,
        )
        queries = queries + special_features.view(
            batch_size, num_frames, num_queries, embed_dim)

        per_view_queries = queries.reshape(
            batch_size * num_frames, num_queries, embed_dim)
        per_view_query_pos = query_pos.reshape_as(per_view_queries)
        per_view_patch_memory = patch_memory.reshape(
            batch_size * num_frames, num_patch_tokens, embed_dim)
        per_view_reference_uv = reference_uv.reshape(
            batch_size * num_frames, num_queries, 1, 2)
        spatial_shapes = torch.as_tensor(
            [patch_grid_size], device=queries.device, dtype=torch.long)
        level_start_index = spatial_shapes.new_zeros(1)
        output_dtype = queries.dtype
        with torch.autocast(device_type=queries.device.type, enabled=False):
            deformable_features = self.deformable_attention(
                query=self.deformable_query_norm(per_view_queries).float(),
                value=self.patch_memory_norm(per_view_patch_memory).float(),
                identity=torch.zeros_like(per_view_queries, dtype=torch.float32),
                query_pos=per_view_query_pos.float(),
                reference_points=per_view_reference_uv.float(),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )
        queries = queries + deformable_features.to(output_dtype).view(
            batch_size, num_frames, num_queries, embed_dim)
        queries = queries + self.ffn(self.ffn_norm(queries))
        if uv_delta_predictor is not None:
            with torch.autocast(device_type=queries.device.type, enabled=False):
                uv_delta = uv_delta_predictor(queries.float())
                reference_uv = (
                    inverse_sigmoid(reference_uv.float()) + uv_delta).sigmoid()
        return queries, reference_uv


class ObjectQueryBranch(nn.Module):
    """Independent point-query stream attached to selected VGGT layers."""

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        layer_indices: tuple[int, ...],
        num_points: int,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.layer_indices = tuple(layer_indices)
        self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)
        self.reference_points = nn.Linear(embed_dim, 2)
        self.uv_delta_predictor = None
        self.uv_fourier = FourierFeatures(
            input_dim=2, num_bands=8, include_input=True)
        self.uv_position_proj = nn.Linear(
            self.uv_fourier.output_dim, embed_dim)
        self.blocks = nn.ModuleDict({
            str(layer_index): ObjectQueryBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_points=num_points,
            )
            for layer_index in self.layer_indices
        })
        self.output_norm = nn.LayerNorm(embed_dim)
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.0)

    def set_uv_delta_predictor(self, predictor: nn.Module) -> None:
        self.uv_delta_predictor = predictor

    def initialize(
        self, batch_size: int, num_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_pos, query = self.query_embedding.weight.chunk(2, dim=-1)
        query = query.view(1, 1, self.num_queries, -1).expand(
            batch_size, num_frames, -1, -1)
        query_pos = query_pos.view(1, 1, self.num_queries, -1).expand(
            batch_size, num_frames, -1, -1)
        reference_uv = self.reference_points(query_pos).sigmoid()
        return query, query_pos, reference_uv

    def update(
        self,
        layer_index: int,
        queries: torch.Tensor,
        query_pos: torch.Tensor,
        reference_uv: torch.Tensor,
        special_memory: torch.Tensor,
        patch_memory: torch.Tensor,
        patch_grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_frames, num_queries, _ = reference_uv.shape
        uv_position = self.uv_position_proj(
            self.uv_fourier(reference_uv.reshape(
                batch_size * num_frames, num_queries, 2)))
        uv_position = uv_position.view(
            batch_size, num_frames, num_queries, -1)
        decoder_query_pos = query_pos + uv_position
        return self.blocks[str(layer_index)](
            queries, decoder_query_pos, reference_uv, special_memory,
            patch_memory, patch_grid_size, self.uv_delta_predictor)


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.cached_layer_indices_ordered = tuple(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

        # Added after loading the released VGGT checkpoint by ReconDet. Keeping
        # it absent here preserves strict compatibility with that checkpoint.
        self.object_query_branch = None

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def initialize_object_query_branch(
        self,
        num_queries: int = 64,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_points: int = 4,
    ) -> None:
        if self.object_query_branch is not None:
            raise RuntimeError("Object query branch has already been initialized")
        self.object_query_branch = ObjectQueryBranch(
            num_queries=num_queries,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            layer_indices=self.cached_layer_indices_ordered,
            num_points=num_points,
        )

    def set_object_query_uv_delta_predictor(self, predictor: nn.Module) -> None:
        if self.object_query_branch is None:
            raise RuntimeError("Object query branch has not been initialized")
        self.object_query_branch.set_uv_delta_predictor(predictor)

    def forward(
        self,
        images: torch.Tensor,
        return_object_queries: bool = False,
        return_patch_attention: bool = False,
    ) -> tuple:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        patch_token_start = self.patch_token_start
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        reference_uv_layers = []
        if return_object_queries:
            if self.object_query_branch is None:
                raise RuntimeError("Object query branch has not been initialized")
            object_queries, object_query_pos, reference_uv = \
                self.object_query_branch.initialize(batch_size, num_frames)
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
                capture_attention=(
                    return_patch_attention and block_idx == 0),
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
                patch_token_start,
            )
            if (return_object_queries and
                    block_idx in self.object_query_branch.layer_indices):
                object_queries, reference_uv = \
                    self.object_query_branch.update(
                        block_idx,
                        object_queries,
                        object_query_pos,
                        reference_uv,
                        tokens[:, :, :patch_token_start],
                        tokens[:, :, patch_token_start:],
                        patch_grid_size,
                    )
                reference_uv_layers.append(reference_uv)
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        if return_patch_attention:
            attention_module = self.frame_blocks[0].attn
            if not hasattr(attention_module, "last_attention_mean"):
                raise RuntimeError("Frame attention weights were not captured")
            patch_attention = attention_module.last_attention_mean[:, patch_token_start:]
            del attention_module.last_attention_mean
            return outputs, patch_token_start, patch_attention

        if return_object_queries:
            object_queries = self.object_query_branch.output_norm(
                object_queries)
            return (outputs, patch_token_start, object_queries, reference_uv,
                    reference_uv_layers)
        return outputs, patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
        capture_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self.frame_blocks[block_idx](
            tokens,
            rope_sincos,
            capture_attention=capture_attention,
        )
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        patch_token_start: int,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self.inter_frame_blocks[block_idx](tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self.inter_frame_blocks[block_idx](camera_and_register_tokens, None)
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
