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

import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector


class VisionHead(torch.nn.Module):
    def __init__(self, params: AttrDict):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name: str) -> type:
    if "MlpProjector" in cls_name:
        cls = MlpProjector
    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower
    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        # Maintain backward compatibility with existing configs using "vision_head"
        cls = VisionHead
    else:
        raise ValueError(f"Invalid class name: {cls_name}")

    return cls


class BaseSubConfig(PretrainedConfig):
    model_type: str = "base"
    cls: str = ""
    params: AttrDict = AttrDict()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = AttrDict(kwargs.get("params", {}))


class VisionConfig(BaseSubConfig):
    model_type = "vision"


class AlignerConfig(BaseSubConfig):
    model_type = "aligner"


class GenVisionConfig(BaseSubConfig):
    model_type = "gen_vision"


class GenAlignerConfig(BaseSubConfig):
    model_type = "gen_aligner"


class GenHeadConfig(BaseSubConfig):
    model_type = "gen_head"


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig
    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision_config = VisionConfig(**kwargs.get("vision_config", {}))
        self.aligner_config = AlignerConfig(**kwargs.get("aligner_config", {}))
        self.gen_vision_config = GenVisionConfig(**kwargs.get("gen_vision_config", {}))
        self.gen_aligner_config = GenAlignerConfig(
            **kwargs.get("gen_aligner_config", {})
        )
        self.gen_head_config = GenHeadConfig(**kwargs.get("gen_head_config", {}))
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

        # Initialize vision components
        self.vision_model = model_name_to_cls(config.vision_config.cls)(
            **config.vision_config.params
        )
        self.aligner = model_name_to_cls(config.aligner_config.cls)(
            config.aligner_config.params
        )

        # Initialize generation components
        self.gen_vision_model = model_name_to_cls(config.gen_vision_config.cls)(
            **config.gen_vision_config.params
        )
        self.gen_aligner = model_name_to_cls(config.gen_aligner_config.cls)(
            config.gen_aligner_config.params
        )
        self.gen_head = model_name_to_cls(config.gen_head_config.cls)(
            config.gen_head_config.params
        )

        # Initialize embeddings and language model
        self.gen_embed = torch.nn.Embedding(
            config.gen_vision_config.params.image_token_size,
            config.gen_vision_config.params.n_embed,
        )
        self.language_model = LlamaForCausalLM(config.language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Prepares combined text and image embeddings for the language model.

        Args:
            input_ids: Token IDs for text inputs, shape [batch_size, seq_len]
            pixel_values: Image tensors, shape [batch_size, num_images, channels, height, width]
            images_seq_mask: Boolean mask indicating image positions in the text sequence,
                shape [batch_size, seq_len]
            images_emb_mask: Boolean mask for valid image tokens per image,
                shape [batch_size, num_images, tokens_per_image]

        Returns:
            Combined embeddings tensor of shape [batch_size, seq_len, embedding_dim]
        """
        batch_size, num_images = pixel_values.shape[:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        images_embeds = self.aligner(self.vision_model(images))  # [(b n), tokens, dim]

        # Reshape embeddings and masks
        images_embeds = rearrange(
            images_embeds, "(b n) t d -> b (n t) d", b=batch_size, n=num_images
        )
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # Validate mask compatibility
        assert torch.all(
            images_seq_mask.sum(dim=1) == images_emb_mask.sum(dim=1)
        ), "Masks must have matching number of image tokens"

        # Get text embeddings and replace image positions
        input_ids = input_ids.masked_fill(input_ids < 0, 0)  # Replace negatives
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor) -> torch.Tensor:
        """Generates image embeddings from token IDs for generation."""
        return self.gen_aligner(self.gen_embed(image_ids))


# Configuration registration
AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
