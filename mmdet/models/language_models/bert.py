# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

from mmdet.registry import MODELS


def generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Only token pairs in between two special tokens are attended to
    and thus the attention mask for these pairs is positive.

    Args:
        input_ids (torch.Tensor): input ids. Shape: [bs, num_token]
        special_tokens_mask (list): special tokens mask.

    Returns:
        Tuple(Tensor, Tensor):
        - attention_mask is the attention mask between each tokens.
          Only token pairs in between two special tokens are positive.
          Shape: [bs, num_token, num_token].
        - position_ids is the position id of tokens within each valid sentence.
          The id starts from 0 whenenver a special token is encountered.
          Shape: [bs, num_token]
    """
    input_ids = tokenized['input_ids']
    bs, num_token = input_ids.shape
    # special_tokens_mask:
    # bs, num_token. 1 for special tokens. 0 for normal tokens
    special_tokens_mask = torch.zeros((bs, num_token),
                                      device=input_ids.device).bool()

    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    # idxs: each row is a list of indices of special tokens
    idxs = torch.nonzero(special_tokens_mask)

    # generate attention mask and positional ids
    attention_mask = (
        torch.eye(num_token,
                  device=input_ids.device).bool().unsqueeze(0).repeat(
                      bs, 1, 1))
    position_ids = torch.zeros((bs, num_token), device=input_ids.device)
    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if (col == 0) or (col == num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1:col + 1,
                           previous_col + 1:col + 1] = True
            position_ids[row, previous_col + 1:col + 1] = torch.arange(
                0, col - previous_col, device=input_ids.device)
        previous_col = col

    return attention_mask, position_ids.to(torch.long)


@MODELS.register_module()
class BertModel(BaseModel):
    """BERT model for language embedding only encoder.

    Args:
        name (str, optional): name of the pretrained BERT model from
            HuggingFace. Defaults to bert-base-uncased.
        max_tokens (int, optional): maximum number of tokens to be
            used for BERT. Defaults to 256.
        pad_to_max (bool, optional): whether to pad the tokens to max_tokens.
             Defaults to True.
        use_sub_sentence_represent (bool, optional): whether to use sub
            sentence represent introduced in `Grounding DINO
            <https://arxiv.org/abs/2303.05499>`. Defaults to False.
        special_tokens_list (list, optional): special tokens used to split
            subsentence. It cannot be None when `use_sub_sentence_represent`
            is True. Defaults to None.
        add_pooling_layer (bool, optional): whether to adding pooling
            layer in bert encoder. Defaults to False.
        num_layers_of_embedded (int, optional): number of layers of
            the embedded model. Defaults to 1.
        use_checkpoint (bool, optional): whether to use gradient checkpointing.
             Defaults to False.
        enable_cache (bool, optional): whether to cache frozen text features
            during evaluation. Defaults to False.
        cache_size (int, optional): maximum number of unique caption batches
            retained in the cache. Defaults to 16.
    """

    def __init__(self,
                 name: str = 'bert-base-uncased',
                 max_tokens: int = 256,
                 pad_to_max: bool = True,
                 use_sub_sentence_represent: bool = False,
                 special_tokens_list: list = None,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False,
                 enable_cache: bool = False,
                 cache_size: int = 16,
                 **kwargs) -> None:

        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max
        if cache_size < 1:
            raise ValueError('cache_size must be at least 1.')
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self._text_feature_cache = OrderedDict()

        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          BertEncoder(
                              name,
                              add_pooling_layer=add_pooling_layer,
                              num_layers_of_embedded=num_layers_of_embedded,
                              use_checkpoint=use_checkpoint))]))

        self.use_sub_sentence_represent = use_sub_sentence_represent
        if self.use_sub_sentence_represent:
            assert special_tokens_list is not None, \
                'special_tokens should not be None \
                    if use_sub_sentence_represent is True'

            self.special_tokens = self.tokenizer.convert_tokens_to_ids(
                special_tokens_list)

    def clear_cache(self) -> None:
        self._text_feature_cache.clear()

    def train(self, mode: bool = True):
        if mode:
            self.clear_cache()
        return super().train(mode)

    def _apply(self, fn):
        self.clear_cache()
        return super()._apply(fn)

    def _forward_uncached(self, captions: Sequence[str]) -> dict:
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True).to(device)
        input_ids = tokenized.input_ids
        if self.use_sub_sentence_represent:
            attention_mask, position_ids = \
                generate_masks_with_special_tokens_and_transfer_map(
                    tokenized, self.special_tokens)
            token_type_ids = tokenized['token_type_ids']

        else:
            attention_mask = tokenized.attention_mask
            position_ids = None
            token_type_ids = None

        tokenizer_input = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids
        }
        language_dict_features = self.language_backbone(tokenizer_input)
        if self.use_sub_sentence_represent:
            language_dict_features['position_ids'] = position_ids
            language_dict_features[
                'text_token_mask'] = tokenized.attention_mask.bool()
        return language_dict_features

    @staticmethod
    def _deduplicate_captions(captions):
        unique_captions = []
        caption_indices = {}
        inverse_indices = []
        for caption in captions:
            if caption not in caption_indices:
                caption_indices[caption] = len(unique_captions)
                unique_captions.append(caption)
            inverse_indices.append(caption_indices[caption])
        return unique_captions, inverse_indices

    @staticmethod
    def _expand_cached_features(features, inverse_indices):
        reference = next(
            value for value in features.values()
            if isinstance(value, torch.Tensor))
        indices = torch.as_tensor(
            inverse_indices, device=reference.device, dtype=torch.long)
        unique_count = reference.shape[0]
        return {
            name: value.index_select(0, indices)
            if (isinstance(value, torch.Tensor) and value.ndim > 0 and
                value.shape[0] == unique_count) else value
            for name, value in features.items()
        }

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Encode captions, reusing frozen features for repeated text."""
        use_cache = (self.enable_cache and not self.training and
                     not torch.is_grad_enabled())
        if not use_cache:
            return self._forward_uncached(captions)

        unique_captions, inverse_indices = self._deduplicate_captions(captions)
        cache_key = tuple(unique_captions)
        language_dict_features = self._text_feature_cache.get(cache_key)
        if language_dict_features is None:
            language_dict_features = self._forward_uncached(unique_captions)
            language_dict_features = {
                name: value.detach() if isinstance(value, torch.Tensor)
                else value
                for name, value in language_dict_features.items()
            }
            self._text_feature_cache[cache_key] = language_dict_features
            while len(self._text_feature_cache) > self.cache_size:
                self._text_feature_cache.popitem(last=False)
        else:
            self._text_feature_cache.move_to_end(cache_key)

        return self._expand_cached_features(
            language_dict_features, inverse_indices)


class BertEncoder(nn.Module):
    """BERT encoder for language embedding.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
                Defaults to bert-base-uncased.
        add_pooling_layer (bool): whether to add a pooling layer.
        num_layers_of_embedded (int): number of layers of the embedded model.
                Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
    """

    def __init__(self,
                 name: str,
                 add_pooling_layer: bool = False,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        if BertConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # only encoder
        self.model = HFBertModel.from_pretrained(
            name, add_pooling_layer=add_pooling_layer, config=config)
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        outputs = self.model(
            input_ids=x['input_ids'],
            attention_mask=mask,
            position_ids=x['position_ids'],
            token_type_ids=x['token_type_ids'],
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:],
                               1).mean(1)
        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers_of_embedded
        if mask.dim() == 2:
            embedded = features * mask.unsqueeze(-1).float()
        else:
            embedded = features

        results = {
            'embedded': embedded,
            'masks': mask,
            'hidden': encoded_layers[-1]
        }
        return results
