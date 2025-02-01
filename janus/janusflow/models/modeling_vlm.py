# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from attrdict import AttrDict
from einops import rearrange
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from janus.janusflow.models.clip_encoder import CLIPVisionTower
from janus.janusflow.models.uvit import ShallowUViTEncoder, ShallowUViTDecoder
from janus.utils.cuda_memory_manager import monitor_memory
import torch.nn as nn


def model_name_to_cls(cls_name):

    if "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower
    elif "ShallowUViTEncoder" in cls_name:
        cls = ShallowUViTEncoder
    elif "ShallowUViTDecoder" in cls_name:
        cls = ShallowUViTDecoder
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionUnderstandEncoderConfig(PretrainedConfig):
    model_type = "vision_und_enc"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VisionGenerationEncoderConfig(PretrainedConfig):
    model_type = "vision_gen_enc"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class VisionGenerationDecoderConfig(PretrainedConfig):
    model_type = "vision_gen_dec"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_und_enc_config: VisionUnderstandEncoderConfig
    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_und_enc_config = kwargs.get("vision_und_enc_config", {})
        self.vision_und_enc_config = VisionUnderstandEncoderConfig(
            **vision_und_enc_config
        )

        vision_gen_enc_config = kwargs.get("vision_gen_enc_config", {})
        self.vision_gen_enc_config = VisionGenerationEncoderConfig(
            **vision_gen_enc_config
        )

        vision_gen_dec_config = kwargs.get("vision_gen_dec_config", {})
        self.vision_gen_dec_config = VisionGenerationDecoderConfig(
            **vision_gen_dec_config
        )

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):

    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        # vision understanding encoder
        vision_und_enc_config = config.vision_und_enc_config
        vision_und_enc_cls = model_name_to_cls(vision_und_enc_config.cls)
        self.vision_und_enc_model = vision_und_enc_cls(**vision_und_enc_config.params)

        # vision understanding aligner
        self.vision_und_enc_aligner = nn.Linear(1024, 2048, bias=True)

        # begin of understanding embedding
        self.beg_of_und_embed = nn.Parameter(torch.zeros(1, 2048))

        # vision generation encoder
        vision_gen_enc_config = config.vision_gen_enc_config
        vision_gen_enc_cls = model_name_to_cls(vision_gen_enc_config.cls)
        self.vision_gen_enc_model = vision_gen_enc_cls(**vision_gen_enc_config.params)

        # vision generation encoder aligner
        self.vision_gen_enc_aligner = nn.Linear(768, 2048, bias=True)

        # vision generation decoder
        vision_gen_dec_config = config.vision_gen_dec_config
        vision_gen_dec_cls = model_name_to_cls(vision_gen_dec_config.cls)
        self.vision_gen_dec_model = vision_gen_dec_cls(**vision_gen_dec_config.params)

        # language model
        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)

        # vision generation decoder aligner
        self.vision_gen_dec_aligner_norm = LlamaRMSNorm(
            2048, eps=language_config.rms_norm_eps
        )
        self.vision_gen_dec_aligner = nn.Linear(2048, 768, bias=True)

    @monitor_memory(warning_threshold_gb=1.5, track_stats=True)
    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.vision_und_enc_model(images)
        images_embeds = self.vision_und_enc_aligner(images_embeds)
        # print(images_embeds.shape, self.beg_of_und_embed.shape, images_seq_mask.shape, input_ids.shape)
        beg_of_und_embed = self.beg_of_und_embed[0].detach().clone()
        images_embeds = torch.cat(
            [
                beg_of_und_embed.view(1, 1, -1).repeat(images_embeds.shape[0], 1, 1),
                images_embeds,
            ],
            dim=1,
        )
        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds


AutoConfig.register("vision_und_enc", VisionUnderstandEncoderConfig)
AutoConfig.register("vision_gen_enc", VisionGenerationEncoderConfig)
AutoConfig.register("vision_gen_dec", VisionGenerationDecoderConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
