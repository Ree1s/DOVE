# Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import (
    AttentionProcessor,
    CogVideoXAttnProcessor2_0,
    FusedCogVideoXAttnProcessor2_0,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero

from .token_merge import RestoreAdapter, Router, parse_token_merge_routes


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class TokenMergeCogVideoXTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        ofs_embed_dim (`int`, defaults to `512`):
            Output dimension of "ofs" embeddings used in CogVideoX-5b-I2B in version 1.5
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
        enable_token_merge (`bool`, defaults to `False`):
            Whether to enable token merge & routing when running the transformer.
        token_merge_routes (`str`, *optional*, defaults to `None`):
            Semi-colon separated specification of routing windows in the format ``start-end@ratio``.
        token_merge_default_ratio (`float`, defaults to `0.0`):
            Fallback ratio that applies to the full depth when no explicit routes are provided.
        token_merge_seed (`int`, defaults to `42`):
            Random seed forwarded to the router for stochastic merging.
        restore_adapter_expansion (`int`, defaults to `2`):
            Expansion factor for the post-merge `RestoreAdapter`.
    """

    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: Optional[int] = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: Optional[int] = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        enable_token_merge: bool = False,
        token_merge_routes: Optional[str] = None,
        token_merge_default_ratio: float = 0.0,
        token_merge_seed: int = 42,
        restore_adapter_expansion: int = 2,
        token_merge_window_size: int = 0,
        token_merge_window_stride: int = 1,
        token_merge_ratio_start: Optional[float] = None,
        token_merge_ratio_warmup_steps: int = 0,
        token_merge_ratio_schedule: str = "linear",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)

        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )  # same as time embeddings, for ofs

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.metric_heads = nn.ModuleList(
            [nn.Linear(inner_dim, 1) for _ in range(num_layers)]
        )
        for head in self.metric_heads:
            nn.init.zeros_(head.bias)
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            # For CogVideox 1.0
            output_dim = patch_size * patch_size * out_channels
        else:
            # For CogVideoX 1.5
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)

        self.gradient_checkpointing = False
        self._inner_dim = inner_dim
        self._warned_rotary_batch_mismatch = False

        self._configure_token_merge(
            enable_token_merge=enable_token_merge,
            routes_spec=token_merge_routes,
            default_ratio=token_merge_default_ratio,
            seed=token_merge_seed,
            restore_adapter_expansion=restore_adapter_expansion,
            window_size=token_merge_window_size,
            window_stride=token_merge_window_stride,
            ratio_start=token_merge_ratio_start,
            ratio_warmup_steps=token_merge_ratio_warmup_steps,
            ratio_schedule=token_merge_ratio_schedule,
        )

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _compute_frame_groups(self, num_frames: int) -> int:
        patch_size_t = getattr(self.config, "patch_size_t", None)
        if patch_size_t is None or patch_size_t <= 0:
            return max(1, num_frames)
        return max(1, num_frames // patch_size_t)

    def _compute_gram_matrix(self, features: torch.Tensor) -> torch.Tensor:
        """
        Computes the Gram matrix for a batch of features.
        Input shape: (B, N, C)
        Output shape: (B, C, C)
        """
        B, N, C = features.shape
        features_reshaped = features.permute(0, 2, 1)
        gram = torch.bmm(features_reshaped, features_reshaped.transpose(1, 2)) / (N * C)
        return gram

    def _configure_token_merge(
        self,
        enable_token_merge: bool,
        routes_spec: Optional[str],
        default_ratio: float,
        seed: int,
        restore_adapter_expansion: int,
        window_size: int,
        window_stride: int,
        ratio_start: Optional[float],
        ratio_warmup_steps: int,
        ratio_schedule: str,
    ) -> None:
        self.config.enable_token_merge = bool(enable_token_merge)
        self.config.token_merge_routes = routes_spec
        self.config.token_merge_default_ratio = float(default_ratio)
        self.config.token_merge_seed = int(seed)
        self.config.restore_adapter_expansion = int(restore_adapter_expansion)
        self.config.token_merge_window_size = int(window_size)
        self.config.token_merge_window_stride = int(window_stride)
        self.config.token_merge_ratio_start = None if ratio_start is None else float(ratio_start)
        self.config.token_merge_ratio_warmup_steps = int(max(0, ratio_warmup_steps))
        self.config.token_merge_ratio_schedule = ratio_schedule.lower()

        routes: List[Dict[str, Any]] = []
        num_layers = len(self.transformer_blocks)

        if enable_token_merge:
            parsed_routes = parse_token_merge_routes(routes_spec)
            if not parsed_routes and default_ratio > 0.0:
                parsed_routes = [
                    {
                        "start_layer": 0,
                        "end_layer": num_layers - 1,
                        "selection_ratio": float(default_ratio),
                        "ratio_end": float(default_ratio),
                    }
                ]

            for spec in parsed_routes:
                start_layer = int(max(0, min(spec["start_layer"], num_layers - 1)))
                end_layer = int(max(start_layer, min(spec["end_layer"], num_layers - 1)))
                ratio = float(spec["selection_ratio"])
                if ratio <= 0.0:
                    continue
                route_entry = {
                    "start_layer": start_layer,
                    "end_layer": end_layer,
                    "selection_ratio": ratio,
                }
                if "ratio_start" in spec:
                    route_entry["ratio_start"] = float(spec["ratio_start"])
                if "ratio_end" in spec:
                    route_entry["ratio_end"] = float(spec["ratio_end"])
                routes.append(route_entry)

        global_ratio_start = self.config.token_merge_ratio_start
        for route in routes:
            ratio_end = float(route.get("ratio_end", route["selection_ratio"]))
            ratio_start_value = route.get("ratio_start", None)
            if ratio_start_value is None:
                ratio_start_value = global_ratio_start if global_ratio_start is not None else ratio_end
            route["ratio_start"] = float(ratio_start_value)
            route["ratio_end"] = float(ratio_end)
            route["selection_ratio"] = float(ratio_end)

        self._routes = routes
        should_enable = enable_token_merge and len(routes) > 0
        self.enable_token_merge = should_enable
        self._token_merge_seed = seed
        self._warned_rotary_batch_mismatch = False
        self._token_merge_curriculum_progress = 1.0
        self._token_merge_curriculum_active = False

        if should_enable:
            self.router = Router(seed=seed, window_size=window_size, window_stride=window_stride)
            self.restore_adapter = RestoreAdapter(self._inner_dim, expansion=restore_adapter_expansion)
        else:
            self.router = None
            self.restore_adapter = None
        for head in self.metric_heads:
            head.requires_grad_(should_enable)

        if should_enable:
            needs_curriculum = any(
                abs(route["ratio_end"] - route["ratio_start"]) > 1e-6 for route in routes
            ) and self.config.token_merge_ratio_warmup_steps > 0
            self._token_merge_curriculum_active = needs_curriculum
            if self._token_merge_curriculum_active:
                self._apply_token_merge_curriculum_progress(0.0)

    def configure_token_merge(
        self,
        enable_token_merge: bool,
        routes_spec: Optional[str],
        default_ratio: float,
        seed: int,
        restore_adapter_expansion: int,
        window_size: int = 0,
        window_stride: int = 1,
        ratio_start: Optional[float] = None,
        ratio_warmup_steps: int = 0,
        ratio_schedule: str = "linear",
    ) -> None:
        self._configure_token_merge(
            enable_token_merge,
            routes_spec,
            default_ratio,
            seed,
            restore_adapter_expansion,
            window_size,
            window_stride,
            ratio_start,
            ratio_warmup_steps,
            ratio_schedule,
        )

    def freeze_parameters_to_routes(self) -> None:
        """Freeze all parameters except those belonging to routed transformer layers."""
        if not self._routes:
            logger.warning(
                "freeze_parameters_to_routes called but no token merge routes are configured. Skipping." 
            )
            return

        for param in self.parameters():
            param.requires_grad_(False)

        train_layers = set()
        for route in self._routes:
            train_layers.update(range(route["start_layer"], route["end_layer"] + 1))

        valid_layers = sorted(idx for idx in train_layers if 0 <= idx < len(self.transformer_blocks))
        if not valid_layers:
            logger.warning(
                "No valid transformer layers fall inside the configured routes; leaving all parameters frozen."
            )
            return

        for idx in valid_layers:
            for param in self.transformer_blocks[idx].parameters():
                param.requires_grad_(True)
            for param in self.metric_heads[idx].parameters():
                param.requires_grad_(True)

        if self.restore_adapter is not None:
            for param in self.restore_adapter.parameters():
                param.requires_grad_(True)

        logger.info("Token merge route training unfroze transformer layers %s", valid_layers)

    def _curriculum_interpolate(self, start: float, end: float, progress: float) -> float:
        schedule = getattr(self.config, "token_merge_ratio_schedule", "linear")
        progress = max(0.0, min(1.0, progress))
        if schedule == "cosine":
            weight = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:
            weight = progress
        return start + (end - start) * weight

    def _apply_token_merge_curriculum_progress(self, progress: float) -> None:
        if not self.enable_token_merge or not self._routes:
            return
        for route in self._routes:
            start = float(route.get("ratio_start", route["selection_ratio"]))
            end = float(route.get("ratio_end", route["selection_ratio"]))
            route["selection_ratio"] = float(self._curriculum_interpolate(start, end, progress))
        self._token_merge_curriculum_progress = max(0.0, min(1.0, progress))

    def update_token_merge_progress(self, global_step: int) -> None:
        if not self._token_merge_curriculum_active or not self.enable_token_merge:
            return
        total_steps = max(1, int(self.config.token_merge_ratio_warmup_steps))
        progress = float(global_step) / float(total_steps)
        prev_progress = getattr(self, "_token_merge_curriculum_progress", 0.0)
        progress = max(prev_progress, min(progress, 1.0))
        if abs(progress - prev_progress) < 1e-6 and progress >= 1.0:
            return
        self._apply_token_merge_curriculum_progress(progress)

    def _reduce_image_rotary_emb(
        self,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
        route_info,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if image_rotary_emb is None or route_info.pos_index is None:
            return image_rotary_emb

        pos_index = route_info.pos_index
        if pos_index.dim() != 2:
            return image_rotary_emb

        if not torch.all(pos_index == pos_index[0:1]):
            if not self._warned_rotary_batch_mismatch:
                logger.warning(
                    "Token merge generated different keep indices per sample while rotary embeddings are enabled; "
                    "disabling rotary embeddings for the routed layers."
                )
                self._warned_rotary_batch_mismatch = True
            return None

        gather_idx = pos_index[0].to(image_rotary_emb[0].device)
        cos, sin = image_rotary_emb
        cos_reduced = cos.index_select(0, gather_idx)
        sin_reduced = sin.index_select(0, gather_idx)
        return cos_reduced, sin_reduced

    def _build_restore_mask(
        self,
        route_info,
        text_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = route_info.orig_shape[0]
        image_token_count = route_info.orig_shape[1]
        mask_img = torch.zeros(batch_size, image_token_count, dtype=torch.bool, device=device)

        for batch_id, idx in enumerate(route_info.src_idx):
            if idx.numel() > 0:
                mask_img[batch_id, idx] = True

        if text_length > 0:
            mask_text = torch.zeros(batch_size, text_length, dtype=torch.bool, device=device)
            return torch.cat([mask_text, mask_img], dim=1)

        return mask_img

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        teacher_hidden_states: Optional[Dict[int, torch.Tensor]] = None,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape
        frame_groups = self._compute_frame_groups(num_frames)

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        routes = self._routes if self.enable_token_merge else []
        route_idx = 0
        active_route_info = None
        base_image_rotary_emb = image_rotary_emb
        rotary_emb_current = image_rotary_emb

        total_relational_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        distillation_loss_applied = False

        for i, block in enumerate(self.transformer_blocks):
            if (
                self.enable_token_merge
                and self.router is not None
                and active_route_info is None
                and route_idx < len(routes)
                and i == routes[route_idx]["start_layer"]
            ):
                combined_tokens = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                importance_map = None
                if self.metric_heads is not None:
                    metric_head = self.metric_heads[i]
                    image_tokens_only = combined_tokens[:, text_seq_length:]
                    if image_tokens_only.numel() > 0:
                        metric_input = image_tokens_only.to(metric_head.weight.dtype)
                        importance_map = metric_head(metric_input).squeeze(-1)
                reduced_tokens, active_route_info = self.router.tome_merge_and_route(
                    combined_tokens,
                    selection_ratio=routes[route_idx]["selection_ratio"],
                    text_length=text_seq_length,
                    num_frames=frame_groups,
                    importance_map=importance_map,
                )
                logger.info(
                    "TokenMerge start layer %s: combined_tokens %s -> reduced_tokens %s",
                    i,
                    list(combined_tokens.shape),
                    list(reduced_tokens.shape),
                )
                encoder_hidden_states = reduced_tokens[:, :text_seq_length]
                hidden_states = reduced_tokens[:, text_seq_length:]
                rotary_emb_current = self._reduce_image_rotary_emb(base_image_rotary_emb, active_route_info)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    rotary_emb_current,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=rotary_emb_current,
                    attention_kwargs=attention_kwargs,
                )

            if (
                teacher_hidden_states is not None
                and i in teacher_hidden_states
                and active_route_info is not None
                and getattr(active_route_info, "keep_idx", None) is not None
            ):
                h_S_pruned = hidden_states
                h_T_full = teacher_hidden_states[i].to(h_S_pruned.device, dtype=h_S_pruned.dtype)

                h_T_subset_list: List[torch.Tensor] = []
                B = h_T_full.shape[0]
                for b_idx in range(B):
                    keep_indices_b = active_route_info.keep_idx[b_idx].to(h_T_full.device)
                    h_T_subset_list.append(h_T_full[b_idx, keep_indices_b])

                if h_T_subset_list:
                    h_T_subset = torch.stack(h_T_subset_list, dim=0)
                    gram_S = self._compute_gram_matrix(h_S_pruned)
                    gram_T = self._compute_gram_matrix(h_T_subset.detach())
                    layer_loss = F.mse_loss(gram_S, gram_T)
                    total_relational_loss = total_relational_loss + layer_loss.to(total_relational_loss.dtype)
                    distillation_loss_applied = True

            if (
                self.enable_token_merge
                and self.router is not None
                and active_route_info is not None
                and route_idx < len(routes)
                and i == routes[route_idx]["end_layer"]
            ):
                combined_tokens = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                restored_tokens = self.router.end_route(combined_tokens, active_route_info)

                if self.restore_adapter is not None:
                    adapter = self.restore_adapter.to(device=restored_tokens.device, dtype=restored_tokens.dtype)
                    has_restored = any(idx.numel() > 0 for idx in active_route_info.src_idx)
                    if has_restored:
                        restore_mask = self._build_restore_mask(
                            active_route_info, text_seq_length, restored_tokens.device
                        )
                        restored_tokens = adapter(restored_tokens, restore_mask)
                logger.info(
                    "TokenMerge end layer %s: combined_tokens %s -> restored_tokens %s",
                    i,
                    list(combined_tokens.shape),
                    list(restored_tokens.shape),
                )

                encoder_hidden_states = restored_tokens[:, :text_seq_length]
                hidden_states = restored_tokens[:, text_seq_length:]
                active_route_info = None
                route_idx += 1
                rotary_emb_current = base_image_rotary_emb

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if distillation_loss_applied:
                return (output, total_relational_loss)
            return (output,)
        return Transformer2DModelOutput(
            sample=output,
            loss=total_relational_loss if distillation_loss_applied else None,
        )
