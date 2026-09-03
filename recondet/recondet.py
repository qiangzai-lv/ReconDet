from typing import List, Tuple, Union
import os

import torch
import numpy as np
from PIL import Image, ImageDraw
from mmengine.structures import InstanceData
from recondet.feature_projection import VGGTFeatureProjector

from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType
from recondet.detr3_models.helpers import GenericMLP
from recondet.detr3_models.position_embedding import PositionEmbeddingCoordsSine
from recondet.device import autocast, get_device
from recondet.geometry_attention import GeometryAwareDeformableDecoder
from recondet.grounding_dino_encoder import GroundingDINOSemanticEncoder
from recondet.scene_query_clustering import WeightedFPSKMeans
from recondet.query_correspondence import (
    aggregate_cluster_view_references, assign_points_to_clusters,
    select_candidate_indices)
from vggt_omega.models import VGGTOmega

device = get_device()


@MODELS.register_module()
class ReconDet(Base3DDetector):
    def __init__(
            self,
            bbox_head: ConfigType,
            train_cfg: OptConfigType = None,
            test_cfg: OptConfigType = None,
            data_preprocessor: OptConfigType = None,
            init_cfg: OptConfigType = None,
            decoder_cfg: OptConfigType = None,
            num_queries=128,
            token_dim=1024,
            test_only_last_layer=True,
            position_embedding="fourier",
            if_mix_precision=False,
            if_save_vggt_feature=False,
            enable_detection_loss=True,
            vggt_omega_checkpoint=None,
            grounding_dino_config=None,
            grounding_dino_checkpoint=None,
            semantic_classes=(),
            grounding_dino_print_score_thr=0.3,
            deformable_num_points=4,
            reconstruction_query_score_thr=0.1,
            debug_projection_vis=False,
            debug_projection_vis_interval=10,
            debug_projection_vis_dir=None
    ):

        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)

        self.semantic_encoder = GroundingDINOSemanticEncoder(
            config=grounding_dino_config,
            checkpoint=grounding_dino_checkpoint,
            classes=semantic_classes,
            print_score_thr=grounding_dino_print_score_thr)

        self.vggt_encoder = VGGTOmega()
        self.vggt_encoder.load_state_dict(
            torch.load(vggt_omega_checkpoint, map_location='cpu', weights_only=True)
        )
        self.vggt_encoder.camera_head = None
        dense_head = self.vggt_encoder.dense_head
        self.vggt_encoder.dense_head = None
        self.vggt_encoder.to(device)

        for param in self.vggt_encoder.parameters():
            param.requires_grad = False

        self.vggt_encoder.eval()

        self.decoder = GeometryAwareDeformableDecoder(
            embed_dims=token_dim,
            num_layers=decoder_cfg['dec_nlayers'],
            num_heads=decoder_cfg['dec_nhead'],
            feedforward_channels=decoder_cfg['dec_ffn_dim'],
            num_feature_levels=4,
            num_points=deformable_num_points,
            dropout=decoder_cfg['dec_dropout'])

        self.feature_projector = VGGTFeatureProjector(
            dense_head=dense_head,
            dim_in=dense_head.norm.normalized_shape[0],
            out_channels=[token_dim] * len(
                dense_head.intermediate_layer_idx))

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_queries = num_queries
        self.scene_query_clustering = WeightedFPSKMeans(
            num_clusters=num_queries, num_iterations=5)
        self.test_only_last_layer = test_only_last_layer

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=token_dim, pos_type=position_embedding, normalize=False
        )
        self.query_projection = GenericMLP(
            input_dim=token_dim,
            hidden_dims=[token_dim],
            output_dim=token_dim,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )
        self.if_mix_precision = if_mix_precision
        self.if_save_vggt_feature = if_save_vggt_feature
        self.enable_detection_loss = enable_detection_loss
        self.reconstruction_query_score_thr = reconstruction_query_score_thr
        # Temporary online-label debugging; remove after projection validation.
        self._projection_debug_iter = 0
        self._projection_debug_enabled = debug_projection_vis
        self._projection_debug_interval = debug_projection_vis_interval
        self._projection_debug_dir = debug_projection_vis_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'projection_debug'))

    @torch.no_grad()
    def _visualize_projection_labels(self, images, data_samples):
        if torch.distributed.is_available() and torch.distributed.is_initialized() \
                and torch.distributed.get_rank() != 0:
            return
        if not data_samples or not hasattr(data_samples[0], 'gt_instances_3d'):
            return
        instances = data_samples[0].gt_instances_3d
        if not all(hasattr(instances, key) for key in
                   ('bboxes_2d', 'bboxes_2d_visible', 'keypoints_2d',
                    'keypoints_visible')):
            return
        os.makedirs(self._projection_debug_dir, exist_ok=True)
        imgs = images[0].detach().float().cpu()
        # Visualization is called before extract_feat: inputs are commonly
        # uint8-like 0..255 tensors, but may already be normalized by the
        # data preprocessor. Restore display-space RGB values in both cases.
        if imgs.numel() and imgs.max() > 1.5:
            imgs = imgs / 255.0
        elif imgs.numel() and imgs.min() < 0:
            preprocessor = getattr(self, 'data_preprocessor', None)
            mean = getattr(preprocessor, 'mean', None)
            std = getattr(preprocessor, 'std', None)
            if mean is not None and std is not None:
                mean = torch.as_tensor(mean).cpu().view(1, -1, 1, 1)
                std = torch.as_tensor(std).cpu().view(1, -1, 1, 1)
                imgs = (imgs * std + mean) / 255.0
        imgs = imgs.clamp(0, 1)
        bboxes = instances.bboxes_2d.detach().cpu()
        bbox_valid = instances.bboxes_2d_visible.detach().cpu().bool()
        keypoints = instances.keypoints_2d.detach().cpu()
        keypoint_valid = instances.keypoints_visible.detach().cpu().bool()
        max_views = min(int(imgs.shape[0]), 10)
        colors = ('red', 'lime', 'cyan', 'yellow')
        for view in range(max_views):
            array = (imgs[view].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            image = Image.fromarray(array)
            draw = ImageDraw.Draw(image)
            height, width = array.shape[:2]
            for obj in range(len(bboxes)):
                if bbox_valid[obj, view]:
                    x1, y1, x2, y2 = (bboxes[obj, view] *
                                      torch.tensor([width, height, width, height])).tolist()
                    draw.rectangle((x1, y1, x2, y2), outline='red', width=2)
                points = keypoints[obj, view] * torch.tensor([width, height])
                for point, valid, color in zip(points, keypoint_valid[obj, view], colors):
                    if not valid:
                        continue
                    x, y = point.tolist(); radius = 3
                    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
            path = os.path.join(
                self._projection_debug_dir,
                f'iter_{self._projection_debug_iter:06d}_view_{view:02d}.jpg')
            image.save(path, quality=90)

    @torch.no_grad()
    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList, mode):

        if self.vggt_encoder.training:
            for param in self.vggt_encoder.parameters():
                param.requires_grad = False

            self.vggt_encoder.eval()

        with torch.no_grad():
            # The data preprocessor converts raw BGR uint8 images to RGB without
            # normalization. VGGT-Omega expects RGB values in [0, 1].
            img = batch_inputs_dict['imgs'].float().div(255.0)
            with autocast(img.device):
                aggregated_tokens_list, ps_idx = self.vggt_encoder.aggregator(
                    img)
                return aggregated_tokens_list, ps_idx, img

    def _cluster_reconstruction_queries(self, reconstruction_hidden,
                                        reconstruction_outputs, images):
        reconstruction_cls, reconstruction_points = reconstruction_outputs
        batch_size, num_views = images.shape[:2]
        query_count = reconstruction_points.shape[1]
        hidden = reconstruction_hidden[-1].reshape(
            batch_size, num_views, query_count, -1)
        points = reconstruction_points.reshape(
            batch_size, num_views, query_count, 3)
        classes = reconstruction_cls.reshape(
            batch_size, num_views, query_count, -1)
        head = self.semantic_encoder.model.bbox_head
        selection_scores = (
            head._last_cls_scores[-1].sigmoid().amax(-1).reshape(
                batch_size, num_views, query_count))
        reconstruction_scores = classes.sigmoid().amax(-1)
        bbox_centers = head._last_bbox_preds[-1, ..., :2].reshape(
            batch_size, num_views, query_count, 2)
        view_ids = torch.arange(num_views, device=points.device)[:, None]
        view_ids = view_ids.expand(num_views, query_count).reshape(-1)

        clustered = []
        for batch_id in range(batch_size):
            flat_selection_scores = selection_scores[batch_id].reshape(-1)
            ids = select_candidate_indices(
                flat_selection_scores, self.reconstruction_query_score_thr,
                self.num_queries)
            selected_points = points[batch_id].reshape(-1, 3)[ids]
            selected_hidden = hidden[batch_id].reshape(
                -1, hidden.shape[-1])[ids]
            selected_classes = classes[batch_id].reshape(
                -1, classes.shape[-1])[ids]
            cluster_xyz, cluster_query, cluster_scores = (
                self.scene_query_clustering(
                    selected_points[None], selected_hidden[None],
                    selected_classes[None]))
            selected_assignment = assign_points_to_clusters(
                selected_points[None], cluster_xyz)
            references, mask = aggregate_cluster_view_references(
                selected_points[None],
                bbox_centers[batch_id].reshape(-1, 2)[ids][None],
                reconstruction_scores[batch_id].reshape(-1)[ids][None],
                view_ids[ids][None], cluster_xyz,
                num_views, assignments=selected_assignment)
            candidate_clusters = torch.full(
                (num_views * query_count,), -1, dtype=torch.long,
                device=points.device)
            candidate_clusters[ids] = selected_assignment[0]
            candidate_clusters = candidate_clusters.reshape(
                num_views, query_count)
            clustered.append((cluster_xyz[0], cluster_query[0],
                              cluster_scores[0], references[0], mask[0],
                              candidate_clusters))
        outputs = [torch.stack(items) for items in zip(*clustered)]
        return tuple(output.detach() for output in outputs)

    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, batch_data_samples, query_xyz, query,
                         reference_points_2d, reference_view_mask):
        feature_maps = self.feature_projector(
            vggt_token_list,
            images,
            ps_idx)
        query_xyz = query_xyz.to(device=images.device, dtype=images.dtype)
        query = query.to(device=images.device, dtype=feature_maps[0].dtype)
        batch_inputs_dict['query_xyz'] = query_xyz
        return self.decoder(
            query,
            feature_maps,
            self.semantic_encoder.last_semantic_feature_maps,
            query_xyz,
            reference_points_2d,
            reference_view_mask,
            self.semantic_encoder.last_valid_ratios,
            self.pos_embedding,
            self.query_projection,
            self.bbox_head.center_heads)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        self._projection_debug_iter += 1
        if (self.training and self._projection_debug_enabled and
                self._projection_debug_interval > 0 and
                self._projection_debug_iter % self._projection_debug_interval == 0):
            self._visualize_projection_labels(batch_inputs_dict['imgs'], batch_data_samples)
        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'train')
        vggt_feature_maps = self.feature_projector(
            vggt_token_list, img, ps_idx)
        semantic_losses, reconstruction_hidden, reconstruction_outputs = (
            self.semantic_encoder.loss(
                batch_inputs_dict['imgs'],
                batch_data_samples,
                vggt_feature_maps=vggt_feature_maps,
                return_reconstruction=True))
        losses = {f'gdino_{name}': value
                  for name, value in semantic_losses.items()}
        if self.enable_detection_loss:
            query_xyz, query, _, references_2d, view_mask, _ = (
                self._cluster_reconstruction_queries(
                    reconstruction_hidden, reconstruction_outputs,
                    batch_inputs_dict['imgs']))
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                batch_data_samples, query_xyz, query, references_2d,
                view_mask)
            detection_losses = self.bbox_head.loss(
                box_features,
                batch_data_samples,
                batch_inputs_dict,
                refined_query_xyz=refined_query_xyz,
                **kwargs)
            losses.update({f'recondet_{name}': value
                           for name, value in detection_losses.items()})
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:

        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'test')
        reconstruction_outputs, view_predictions = (
            self.semantic_encoder.predict_reconstruction(
                batch_inputs_dict['imgs'], batch_data_samples,
                self.feature_projector(vggt_token_list, img, ps_idx)))
        reconstruction_hidden = (
            self.semantic_encoder.model._last_reconstruction_hidden_states)
        query_xyz, query, _, references_2d, view_mask, candidate_clusters = (
            self._cluster_reconstruction_queries(
                reconstruction_hidden, reconstruction_outputs,
                batch_inputs_dict['imgs']))
        box_features, refined_query_xyz = self.get_box_features(
            vggt_token_list, ps_idx, batch_inputs_dict, img,
            batch_data_samples, query_xyz, query, references_2d, view_mask)
        layer_ids = list(range(len(box_features)))
        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]]
            layer_ids = [layer_ids[-1]]
        results_list = self.bbox_head.predict(
            box_features,
            batch_data_samples,
            batch_inputs_dict,
            refined_query_xyz=refined_query_xyz,
            layer_ids=layer_ids,
            **kwargs)
        num_views = batch_inputs_dict['imgs'].shape[1]
        batch_size = len(batch_data_samples)
        # Detection queries are scene-level. Repeat their final 3D locations
        # for each view so query_indices and reconstruction points share the
        # same index space expected by the view-level metric.
        reconstruction_points = refined_query_xyz[-1].detach()
        reconstruction_points = reconstruction_points.repeat_interleave(
            num_views, dim=0)
        if len(view_predictions) != batch_size * num_views:
            raise ValueError(
                'Unexpected 2D prediction count: '
                f'{len(view_predictions)} for batch={batch_size}, '
                f'views={num_views}')

        scene_2d_predictions = []
        scene_view_predictions = []
        scene_reconstruction_predictions = []
        for batch_index, data_sample in enumerate(batch_data_samples):
            start = batch_index * num_views
            end = start + num_views
            view_predictions_for_scene = view_predictions[start:end]
            normalized_view_predictions = []
            for view_index, prediction in enumerate(view_predictions_for_scene):
                if hasattr(prediction, 'pred_instances'):
                    prediction = prediction.pred_instances
                query_indices = prediction.query_indices
                cluster_indices = torch.full_like(query_indices, -1)
                valid = ((query_indices >= 0) &
                         (query_indices < candidate_clusters.shape[-1]))
                cluster_indices[valid] = candidate_clusters[
                    batch_index, view_index, query_indices[valid]]
                normalized_view_predictions.append(
                    InstanceData(
                        bboxes=prediction.bboxes,
                        scores=prediction.scores,
                        labels=prediction.labels,
                        query_indices=cluster_indices))
            scene_view_predictions.append(normalized_view_predictions)
            scene_2d_predictions.append(
                InstanceData.cat(normalized_view_predictions))
            scene_reconstruction_predictions.append([
                InstanceData(points_3d=points)
                for points in reconstruction_points[start:end]
            ])
            data_sample.pred_instances_2d_views = normalized_view_predictions
            data_sample.pred_reconstruction = \
                scene_reconstruction_predictions[-1]
        predictions = self.add_pred_to_datasample(
            batch_data_samples, results_list, scene_2d_predictions)
        for data_sample, prediction, view_predictions_for_scene, \
                reconstruction_for_scene in zip(
                    predictions, scene_2d_predictions, scene_view_predictions,
                    scene_reconstruction_predictions):
            data_sample.pred_instances = prediction
            data_sample.pred_instances_2d_views = view_predictions_for_scene
            data_sample.pred_reconstruction = reconstruction_for_scene
        return predictions

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        vggt_token_list, ps_idx, img = self.extract_feat(
            batch_inputs_dict, batch_data_samples, 'train')

        _, reconstruction_hidden, reconstruction_outputs = (
            self.semantic_encoder.loss(
                batch_inputs_dict['imgs'], batch_data_samples,
                vggt_feature_maps=self.feature_projector(
                    vggt_token_list, img, ps_idx),
                return_reconstruction=True))
        query_xyz, query, _, references_2d, view_mask, _ = (
            self._cluster_reconstruction_queries(
                reconstruction_hidden, reconstruction_outputs,
                batch_inputs_dict['imgs']))
        box_features, refined_query_xyz = self.get_box_features(
            vggt_token_list, ps_idx, batch_inputs_dict, img,
            batch_data_samples, query_xyz, query, references_2d, view_mask)

        layer_ids = list(range(len(box_features)))
        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]]
            layer_ids = [layer_ids[-1]]

        results = self.bbox_head.forward(
            box_features, batch_inputs_dict, refined_query_xyz, layer_ids)
        return results
