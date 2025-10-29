"""CogVideoX transformer subclass with optional token merge routing."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .token_merge import RestoreAdapter, Router

try:
    from diffusers.models.transformers.cogvideox_transformer3d import (
        CogVideoXTransformer3DModel as _BaseTransformer,
    )
    from diffusers.models.transformers.modeling_outputs import Transformer2DModelOutput
    from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
except ImportError as exc:  # pragma: no cover - handled dynamically
    _BaseTransformer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None
    logger = logging.get_logger(__name__)


if _BaseTransformer is None:  # pragma: no cover - executed when diffusers missing

    class CogVideoXTransformer3DTokenMerge:  # type: ignore[override]
        """Stub transformer that surfaces the missing dependency error."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "diffusers is required to use CogVideoXTransformer3DTokenMerge."
            ) from _IMPORT_ERROR

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> "CogVideoXTransformer3DTokenMerge":
            raise ImportError(
                "diffusers is required to use CogVideoXTransformer3DTokenMerge."
            ) from _IMPORT_ERROR

else:

    class CogVideoXTransformer3DTokenMerge(_BaseTransformer):
        """Drop-in CogVideoX transformer with optional token merge routing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.enable_token_merge: bool = False
            self.token_merge_routes: List[Dict[str, float]] = []
            self.router: Router = Router()
            self.restore_adapter: Optional[RestoreAdapter] = None
            self._token_merge_seed: int = 42

        # ------------------------------------------------------------------ #
        # Public configuration API
        # ------------------------------------------------------------------ #
        def configure_token_merge(
            self,
            *,
            enable: bool,
            routes: List[Dict[str, float]],
            seed: int = 42,
        ) -> None:
            """Enable/disable token merge and register route configuration."""

            self.enable_token_merge = bool(enable and routes)
            self.token_merge_routes = sorted(
                [
                    {
                        "start_layer": int(route["start_layer"]),
                        "end_layer": int(route["end_layer"]),
                        "selection_ratio": float(route["selection_ratio"]),
                    }
                    for route in routes
                    if route.get("selection_ratio", 0.0) > 0.0
                ],
                key=lambda r: (r["start_layer"], r["end_layer"]),
            )
            self._token_merge_seed = seed
            self.router = Router(seed=seed)
            self.restore_adapter = None

        # ------------------------------------------------------------------ #
        # Forward pass
        # ------------------------------------------------------------------ #
        def forward(  # type: ignore[override]
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            timestep: Union[int, float, torch.LongTensor],
            timestep_cond: Optional[torch.Tensor] = None,
            ofs: Optional[Union[int, float, torch.LongTensor]] = None,
            image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
        ):
            if not self.enable_token_merge or not self.token_merge_routes:
                return super().forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    timestep_cond=timestep_cond,
                    ofs=ofs,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=return_dict,
                )

            if attention_kwargs is not None:
                attention_kwargs = attention_kwargs.copy()
                lora_scale = attention_kwargs.pop("scale", 1.0)
            else:
                lora_scale = 1.0

            if USE_PEFT_BACKEND:
                scale_lora_layers(self, lora_scale)
            else:
                if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                    logger.warning(
                        "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                    )

            batch_size, num_frames, _, height, width = hidden_states.shape

            # 1. Time embedding
            timesteps = timestep
            t_emb = self.time_proj(timesteps)
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

            # 3. Transformer blocks with optional token merge
            routes_iter = iter(self.token_merge_routes)
            next_route = next(routes_iter, None)
            active_route: Optional[Dict[str, float]] = None
            route_info = None

            def maybe_start_route(layer_idx: int) -> None:
                nonlocal hidden_states, encoder_hidden_states, active_route, next_route, route_info
                if next_route is None or layer_idx != next_route["start_layer"]:
                    return

                tokens = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                merged_tokens, route_info_local = self.router.tome_merge_and_route(
                    tokens,
                    selection_ratio=next_route["selection_ratio"],
                    text_length=text_seq_length,
                    num_frames=num_frames,
                )

                encoder_hidden_states = merged_tokens[:, :text_seq_length]
                hidden_states = merged_tokens[:, text_seq_length:]
                active_route = next_route
                route_info = route_info_local
                next_route = next(routes_iter, None)

            def maybe_end_route(layer_idx: int) -> None:
                nonlocal hidden_states, encoder_hidden_states, active_route, route_info
                if active_route is None or layer_idx != active_route["end_layer"] or route_info is None:
                    return

                combined = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                restored = self.router.end_route(combined, route_info)
                encoder_hidden_states = restored[:, :text_seq_length]
                hidden_states = restored[:, text_seq_length:]

                has_restored_tokens = any(idx.numel() > 0 for idx in route_info.src_idx)
                if has_restored_tokens:
                    mask = torch.zeros(
                        restored.size(0),
                        restored.size(1),
                        dtype=torch.bool,
                        device=restored.device,
                    )
                    for b, idx in enumerate(route_info.src_idx):
                        if idx.numel() > 0:
                            mask[b, text_seq_length + idx] = True

                    if self.restore_adapter is None:
                        self.restore_adapter = RestoreAdapter(restored.size(-1))
                    self.restore_adapter = self.restore_adapter.to(
                        device=restored.device, dtype=restored.dtype
                    )
                    restored = self.restore_adapter(restored, mask)
                    encoder_hidden_states = restored[:, :text_seq_length]
                    hidden_states = restored[:, text_seq_length:]

                active_route = None
                route_info = None

            for i, block in enumerate(self.transformer_blocks):
                maybe_start_route(i)

                block_image_rotary_emb = None if active_route is not None else image_rotary_emb

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        block_image_rotary_emb,
                        attention_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=block_image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                    )

                maybe_end_route(i)

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
                unscale_lora_layers(self, lora_scale)

            if not return_dict:
                return (output,)
            return Transformer2DModelOutput(sample=output)

