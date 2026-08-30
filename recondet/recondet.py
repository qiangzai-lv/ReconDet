from typing import List, Tuple, Union

import torch
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
            deformable_num_points=4
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

    def _get_projection_cameras(self, batch_data_samples, images):
        extrinsics = []
        intrinsics = []
        image_height, image_width = images.shape[-2:]
        num_views = images.shape[1]
        for data_sample in batch_data_samples:
            metadata = data_sample.metainfo
            camera_info = metadata['lidar2img']
            sample_extrinsics = torch.as_tensor(
                camera_info['extrinsic'], device=images.device,
                dtype=torch.float32)
            sample_intrinsic = torch.as_tensor(
                camera_info['intrinsic'], device=images.device,
                dtype=torch.float32)[:3, :3].clone()
            ori_height, ori_width = metadata['ori_shape'][:2]
            image_scale = min(
                image_height / ori_height, image_width / ori_width)
            sample_intrinsic[:2] *= image_scale

            extrinsics.append(sample_extrinsics[:, :3])
            intrinsics.append(sample_intrinsic.expand(num_views, -1, -1))
        return torch.stack(extrinsics), torch.stack(intrinsics)

    @staticmethod
    def _format_gdino_losses(losses):
        aggregated = {}
        passthrough = {}
        for name, value in losses.items():
            if name.startswith('d') and '.' in name:
                layer_name, loss_name = name.split('.', 1)
                if layer_name[1:].isdigit():
                    aggregated.setdefault(loss_name, []).append(value)
                    continue
            passthrough[name] = value

        formatted = dict(passthrough)
        for name, values in aggregated.items():
            formatted[name] = torch.stack(values).sum()
        return formatted

    def _cluster_reconstruction_queries(self, reconstruction_hidden,
                                        reconstruction_outputs, images):
        reconstruction_cls, reconstruction_points = reconstruction_outputs
        batch_size, num_views = images.shape[:2]
        reconstruction_hidden = reconstruction_hidden[-1].reshape(
            batch_size, num_views * reconstruction_hidden.shape[2],
            reconstruction_hidden.shape[3])
        reconstruction_points = reconstruction_points.reshape(
            batch_size, num_views * reconstruction_points.shape[1], 3)
        reconstruction_cls = reconstruction_cls.reshape(
            batch_size, num_views * reconstruction_cls.shape[1],
            reconstruction_cls.shape[2])
        cluster_xyz, cluster_query, cluster_scores = (
            self.scene_query_clustering(
                reconstruction_points, reconstruction_hidden,
                reconstruction_cls))
        return cluster_xyz.detach(), cluster_query.detach(), cluster_scores.detach()

    def get_box_features(self, vggt_token_list, ps_idx, batch_inputs_dict,
                         images, batch_data_samples, query_xyz, query):
        feature_maps = self.feature_projector(
            vggt_token_list,
            images,
            ps_idx)
        camera_extrinsics, camera_intrinsics = self._get_projection_cameras(
            batch_data_samples, images)
        batch_inputs_dict['camera_extrinsics'] = camera_extrinsics
        batch_inputs_dict['camera_intrinsics'] = camera_intrinsics

        query_xyz = query_xyz.to(device=images.device, dtype=images.dtype)
        query = query.to(device=images.device, dtype=feature_maps[0].dtype)
        batch_inputs_dict['query_xyz'] = query_xyz
        return self.decoder(
            query,
            feature_maps,
            query_xyz,
            camera_extrinsics,
            camera_intrinsics,
            images.shape[-2:],
            self.pos_embedding,
            self.query_projection,
            self.bbox_head.center_heads)

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
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
        semantic_losses = self._format_gdino_losses(semantic_losses)
        losses = {f'gdino_{name}': value
                  for name, value in semantic_losses.items()}
        if self.enable_detection_loss:
            query_xyz, query = self._cluster_reconstruction_queries(
                reconstruction_hidden, reconstruction_outputs,
                batch_inputs_dict['imgs'])[:2]
            box_features, refined_query_xyz = self.get_box_features(
                vggt_token_list, ps_idx, batch_inputs_dict, img,
                batch_data_samples, query_xyz, query)
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
        reconstruction_outputs = self.semantic_encoder.predict_reconstruction(
            batch_inputs_dict['imgs'], batch_data_samples,
            self.feature_projector(vggt_token_list, img, ps_idx))
        reconstruction_hidden = (
            self.semantic_encoder.model._last_reconstruction_hidden_states)
        query_xyz, query = self._cluster_reconstruction_queries(
            reconstruction_hidden, reconstruction_outputs,
            batch_inputs_dict['imgs'])[:2]
        box_features, refined_query_xyz = self.get_box_features(
            vggt_token_list, ps_idx, batch_inputs_dict, img,
            batch_data_samples, query_xyz, query)
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
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
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
        query_xyz, query = self._cluster_reconstruction_queries(
            reconstruction_hidden, reconstruction_outputs,
            batch_inputs_dict['imgs'])[:2]
        box_features, refined_query_xyz = self.get_box_features(
            vggt_token_list, ps_idx, batch_inputs_dict, img,
            batch_data_samples, query_xyz, query)

        layer_ids = list(range(len(box_features)))
        if self.test_only_last_layer:
            box_features = [box_features[-1]]
            refined_query_xyz = [refined_query_xyz[-1]]
            layer_ids = [layer_ids[-1]]

        results = self.bbox_head.forward(
            box_features, batch_inputs_dict, refined_query_xyz, layer_ids)
        return results
