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
    PretrainedConfig,
)
from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector


class VisionHead(torch.nn.Module):
    """
    Vision head module for processing visual embeddings.
    """
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(params.n_embed, params.image_token_embed)
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(params.image_token_embed, params.image_token_size)

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x


def model_name_to_cls(cls_name):
    """
    Maps a class name to its corresponding class.
    """
    mapping = {
        "MlpProjector": MlpProjector,
        "CLIPVisionTower": CLIPVisionTower,
        "vision_head": VisionHead,
    }

    if "VQ" in cls_name:
        from janus.models.vq_model import VQ_models
        return VQ_models[cls_name]

    cls = mapping.get(cls_name)
    if cls is None:
        raise ValueError(f"Invalid class name: {cls_name}")
    return cls


class BaseConfig(PretrainedConfig):
    """
    Base configuration class for multi-modality components.
    """
    model_type = ""
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = AttrDict(kwargs.get("params", {}))


class VisionConfig(BaseConfig):
    model_type = "vision"


class AlignerConfig(BaseConfig):
    model_type = "aligner"


class GenVisionConfig(BaseConfig):
    model_type = "gen_vision"


class GenAlignerConfig(BaseConfig):
    model_type = "gen_aligner"


class GenHeadConfig(BaseConfig):
    model_type = "gen_head"


class MultiModalityConfig(PretrainedConfig):
    """
    Configuration for the multi-modality model.
    """
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
        self.gen_aligner_config = GenAlignerConfig(**kwargs.get("gen_aligner_config", {}))
        self.gen_head_config = GenHeadConfig(**kwargs.get("gen_head_config", {}))

        language_config = kwargs.get("language_config", {})
        self.language_config = (
            language_config if isinstance(language_config, LlamaConfig) else LlamaConfig(**language_config)
        )


class MultiModalityPreTrainedModel(PreTrainedModel):
    """
    Base class for multi-modality pre-trained models.
    """
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    """
    Multi-modality causal language model combining vision and language components.
    """
    def __init__(self, config: MultiModalityConfig):
        super().__init__(config)

        # Initialize vision model
        vision_cls = model_name_to_cls(config.vision_config.cls)
        self.vision_model = vision_cls(**config.vision_config.params)

        # Initialize aligner
        aligner_cls = model_name_to_cls(config.aligner_config.cls)
        self.aligner = aligner_cls(config.aligner_config.params)

        # Initialize generative vision model
        gen_vision_cls = model_name_to_cls(config.gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        # Initialize generative aligner
        gen_aligner_cls = model_name_to_cls(config.gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(config.gen_aligner_config.params)

        # Initialize generative head
        gen_head_cls = model_name_to_cls(config.gen_head_config.cls)
        self.gen_head = gen_head_cls(config.gen_head_config.params)

        # Generative embedding layer
        self.gen_embed = torch.nn.Embedding(
            config.gen_vision_config.params.image_token_size,
            config.gen_vision_config.params.n_embed,
        )

        # Language model
        self.language_model = LlamaForCausalLM(config.language_config)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """
        Prepares input embeddings by combining text and image embeddings.

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor): [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """
        bs, n = pixel_values.shape[:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")

        # Process images through vision model and aligner
        images_embeds = self.aligner(self.vision_model(images))  # [b x n, T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)  # [b, n x T2, D]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")  # [b, n x T2]

        # Prepare text embeddings
        input_ids[input_ids < 0] = 0  # Ignore negative IDs
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)  # [b, T, D]

        # Replace text embeddings with image embeddings where applicable
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]
        return inputs_embeds

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        """
        Prepares generative image embeddings.

        Args:
            image_ids (torch.LongTensor): Image token IDs.

        Returns:
            torch.Tensor: Generated image embeddings.
        """
        return self.gen_aligner(self.gen_embed(image_ids))


# Register configurations with Hugging Face's AutoConfig
AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)

# Register the multi-modality causal LM model
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
