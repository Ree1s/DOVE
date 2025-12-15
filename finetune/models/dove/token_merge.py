"""Token merge and routing utilities for DOVE.

This module extracts the ToMe-style token merge helpers from the reference
implementation shared by the user, removing dependencies on the SAT framework.
It provides pure PyTorch utilities that a customized CogVideoX transformer can
invoke during forward passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import nn


def _empty_index_list(batch_size: int, device: torch.device) -> List[torch.Tensor]:
    return [torch.empty(0, dtype=torch.long, device=device) for _ in range(batch_size)]


def bipartite_soft_matching_randframe(
    metric: torch.Tensor,
    num_frames: int,
    ratio: float,
    unm_pre: int,
    generator: torch.Generator,
    target_stride: int = 4,
    align_batch: bool = False,
    merge_mode: str = "replace",
    importance_map: Optional[torch.Tensor] = None,
    target_frame_override: Optional[int] = None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], Dict[str, object]]:
    """Random frame-based bipartite matching used by ToMe.

    The function partitions tokens into ``src`` (to be removed) and ``dst`` (to
    be kept) and returns **callable hooks** that can be applied to the token
    tensor. When ``ratio`` is zero the returned hooks act as no-ops.
    """
    batch_size, token_count, _ = metric.shape
    device = metric.device

    if num_frames <= 0 or token_count <= unm_pre:
        empty = _empty_index_list(batch_size, device)
        return (
            lambda x, mode=None: x,
            lambda x, mode=None: x,
            {"unm_num": token_count - unm_pre, "all_src_idx": empty, "all_dst_idx": empty},
        )

    tokens_per_frame = (token_count - unm_pre) // num_frames
    if tokens_per_frame <= 0:
        empty = _empty_index_list(batch_size, device)
        return (
            lambda x, mode=None: x,
            lambda x, mode=None: x,
            {"unm_num": token_count - unm_pre, "all_src_idx": empty, "all_dst_idx": empty},
        )

    dst_per_frame = int(tokens_per_frame * (1 - ratio))
    src_per_frame = tokens_per_frame - dst_per_frame
    if ratio == 0.0 or src_per_frame <= 0 or dst_per_frame <= 0:
        empty = _empty_index_list(batch_size, device)
        return (
            lambda x, mode=None: x,
            lambda x, mode=None: x,
            {"unm_num": token_count - unm_pre, "all_src_idx": empty, "all_dst_idx": empty},
        )

    def rand_indices(length: int, take: int, rng: torch.Generator) -> torch.Tensor:
        return torch.randperm(length, generator=rng, device=device)[:take]

    frame_offsets = unm_pre + torch.arange(num_frames, device=device, dtype=torch.long) * tokens_per_frame
    frame_tokens = frame_offsets[:, None] + torch.arange(tokens_per_frame, device=device, dtype=torch.long)[None, :]
    frame_start_list = [int(val) for val in frame_offsets.cpu().tolist()]

    all_dst_idx: List[torch.Tensor] = []
    all_src_idx: List[torch.Tensor] = []
    dst_tokens_batch: List[torch.Tensor] = []
    src_tokens_batch: List[torch.Tensor] = []

    if align_batch:
        if target_frame_override is not None:
            target_frame_common = target_frame_override % num_frames
        else:
            target_frame_common = torch.randint(0, num_frames, (1,), generator=generator, device=device).item()

        if importance_map is not None:
            start_idx_common = frame_start_list[target_frame_common]
            end_idx_common = start_idx_common + tokens_per_frame
            if importance_map.shape[1] == token_count:
                frame_importance = importance_map[:, start_idx_common:end_idx_common]
            else:
                rel_start = start_idx_common - unm_pre
                frame_importance = importance_map[:, rel_start : rel_start + tokens_per_frame]
            frame_scores = frame_importance.mean(dim=0)
            dst_indices_common = torch.topk(frame_scores, k=dst_per_frame, largest=True).indices
        else:
            dst_indices_common = rand_indices(tokens_per_frame, dst_per_frame, generator)

        keep_mask_common = torch.ones(tokens_per_frame, dtype=torch.bool, device=device)
        keep_mask_common[dst_indices_common] = False
        target_keep_idx = frame_tokens[target_frame_common][keep_mask_common]

        shared_src_indices: Dict[int, torch.Tensor] = {}
        for frame_id in range(num_frames):
            if frame_id == target_frame_common:
                continue
            if importance_map is not None:
                start_idx = frame_start_list[frame_id]
                end_idx = start_idx + tokens_per_frame
                if importance_map.shape[1] == token_count:
                    frame_importance = importance_map[:, start_idx:end_idx]
                else:
                    rel_start = start_idx - unm_pre
                    frame_importance = importance_map[:, rel_start : rel_start + tokens_per_frame]
                frame_scores = frame_importance.mean(dim=0)
                selected = torch.topk(frame_scores, k=src_per_frame, largest=False).indices
            else:
                selected = rand_indices(tokens_per_frame, src_per_frame, generator)
            shared_src_indices[frame_id] = frame_tokens[frame_id][selected]

    for b in range(batch_size):
        if align_batch:
            target_frame = target_frame_common
            dst_indices_in_frame = dst_indices_common
        else:
            if target_frame_override is not None:
                target_frame = target_frame_override % num_frames
            else:
                target_frame = torch.randint(0, num_frames, (1,), generator=generator, device=device).item()

            if importance_map is not None:
                start_idx = frame_start_list[target_frame]
                end_idx = start_idx + tokens_per_frame
                if importance_map.shape[1] == token_count:
                    frame_importance = importance_map[b, start_idx:end_idx]
                else:
                    rel_start = start_idx - unm_pre
                    frame_importance = importance_map[b, rel_start : rel_start + tokens_per_frame]
                dst_indices_in_frame = torch.topk(frame_importance, k=dst_per_frame, largest=True).indices
            else:
                dst_indices_in_frame = rand_indices(tokens_per_frame, dst_per_frame, generator)

        dst_idx = frame_tokens[target_frame][dst_indices_in_frame]
        src_idx_list: List[torch.Tensor] = []

        for frame_id in range(num_frames):
            if frame_id == target_frame:
                if align_batch:
                    src_idx_list.append(target_keep_idx)
                else:
                    keep_mask = torch.ones(tokens_per_frame, dtype=torch.bool, device=device)
                    keep_mask[dst_indices_in_frame] = False
                    src_idx_list.append(frame_tokens[frame_id][keep_mask])
            else:
                if align_batch:
                    src_idx_list.append(shared_src_indices[frame_id])
                else:
                    if importance_map is not None:
                        start_idx = frame_start_list[frame_id]
                        end_idx = start_idx + tokens_per_frame
                        if importance_map.shape[1] == token_count:
                            frame_importance = importance_map[b, start_idx:end_idx]
                        else:
                            rel_start = start_idx - unm_pre
                            frame_importance = importance_map[b, rel_start : rel_start + tokens_per_frame]
                        selected = torch.topk(frame_importance, k=src_per_frame, largest=False).indices
                    else:
                        selected = rand_indices(tokens_per_frame, src_per_frame, generator)
                    src_idx_list.append(frame_tokens[frame_id][selected])

        src_idx = torch.cat(src_idx_list)
        if src_idx.numel() > num_frames * src_per_frame:
            src_idx = src_idx[: num_frames * src_per_frame]

        all_dst_idx.append(dst_idx)
        all_src_idx.append(src_idx)
        dst_tokens_batch.append(metric[b, dst_idx])
        src_tokens_batch.append(metric[b, src_idx])

    dst_tokens = torch.stack(dst_tokens_batch) if dst_tokens_batch else metric.new_empty((batch_size, 0, metric.shape[-1]))
    src_tokens = torch.stack(src_tokens_batch) if src_tokens_batch else metric.new_empty((batch_size, 0, metric.shape[-1]))

    if dst_tokens.numel() > 0 and src_tokens.numel() > 0:
        similarity = torch.einsum("bsc,bdc->bsd", src_tokens, dst_tokens)
        if align_batch:
            similarity = similarity.mean(dim=0, keepdim=True).expand(batch_size, -1, -1)
        _, best_dst_indices = similarity.max(dim=2)
    else:
        best_dst_indices = torch.empty((batch_size, 0), dtype=torch.long, device=device)

    def merge_tokens(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        if x.shape[1] != token_count:
            return x
        keep_tensors: List[torch.Tensor] = []
        for b in range(batch_size):
            item_keep = torch.ones(token_count, dtype=torch.bool, device=x.device)
            if all_src_idx[b].numel() > 0:
                item_keep[all_src_idx[b]] = False
            keep_tensors.append(x[b, item_keep])
        return torch.stack(keep_tensors, dim=0)

    def unmerge_tokens(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        return x

    total_src_tokens = sum(idx.numel() for idx in all_src_idx)
    unm_num = token_count - total_src_tokens // max(batch_size, 1)

    return merge_tokens, unmerge_tokens, {
        "unm_num": unm_num,
        "all_src_idx": all_src_idx,
        "all_dst_idx": all_dst_idx,
        "best_dst_indices": best_dst_indices,
    }


def bipartite_soft_matching_rand2d(
    metric: torch.Tensor,
    width: int,
    height: int,
    stride_x: int,
    stride_y: int,
    remove_tokens: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    text_length: int = 0,
    importance_map: Optional[torch.Tensor] = None,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], Dict[str, object]]:
    """2D bipartite matching variant that works on spatial patches."""
    batch_size, total_tokens, _ = metric.shape
    device = metric.device
    image_tokens = total_tokens - text_length
    if image_tokens <= 0 or remove_tokens <= 0:
        empty = _empty_index_list(batch_size, device)
        return (
            lambda x, mode=None: x,
            lambda x, mode=None: x,
            {"unm_num": total_tokens, "all_src_idx": empty, "all_dst_idx": empty},
        )

    remove_tokens = min(remove_tokens, image_tokens)
    generator = generator or torch.Generator(device=device)

    if importance_map is not None:
        if importance_map.shape[1] == total_tokens:
            importance_img = importance_map[:, text_length:]
        else:
            importance_img = importance_map
        selected_list = [
            torch.argsort(importance_img[b], dim=-1)[:remove_tokens].to(device) for b in range(batch_size)
        ]
        all_src_idx = [sel + text_length for sel in selected_list]
    else:
        if no_rand:
            step = max(1, image_tokens // remove_tokens)
            selected = torch.arange(0, image_tokens, step, device=device)[:remove_tokens]
        else:
            selected = torch.randperm(image_tokens, generator=generator, device=device)[:remove_tokens]
        all_src_idx = [selected + text_length for _ in range(batch_size)]

    def merge_tokens(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        if x.shape[1] != total_tokens:
            return x
        kept = []
        for b in range(batch_size):
            keep_mask = torch.ones(total_tokens, dtype=torch.bool, device=x.device)
            if all_src_idx[b].numel() > 0:
                keep_mask[all_src_idx[b]] = False
            kept.append(x[b, keep_mask])
        return torch.stack(kept, dim=0)

    def unmerge_tokens(x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        return x

    empty_dst = _empty_index_list(batch_size, device)
    return merge_tokens, unmerge_tokens, {
        "unm_num": total_tokens - remove_tokens,
        "all_src_idx": all_src_idx,
        "all_dst_idx": empty_dst,
    }


@dataclass
class RouteInfo:
    src_tokens: torch.Tensor
    src_idx: List[torch.Tensor]
    keep_idx: List[torch.Tensor]
    best_dst_indices: Optional[torch.Tensor]
    pos_index: Optional[torch.Tensor]
    orig_shape: Tuple[int, int, int]
    text_length: int


class Router:
    """Router that orchestrates token masking, ToMe merging and restoration."""

    def __init__(self, seed: int = 42, window_size: int = 0, window_stride: int = 1) -> None:
        self.seed = seed
        self.window_size = max(0, int(window_size))
        self.window_stride = max(1, int(window_stride)) if self.window_size > 0 else 1
        self.next_window_start = 0

    def get_mask(
        self,
        tokens: torch.Tensor,
        mask_ratio: float = 0.0,
        l1_reg: float = 0.0,
        inverse: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = tokens.shape
        device = tokens.device
        keep = int(seq_len * (1 - mask_ratio))

        mags = tokens.abs().sum(dim=-1)
        min_val = mags.min(dim=1, keepdim=True).values
        max_val = mags.max(dim=1, keepdim=True).values
        norm = (mags - min_val) / (max_val - min_val + 1e-8)
        adjusted = 1.0 - norm if inverse else norm

        rand = torch.rand(batch_size, seq_len, device=device, dtype=tokens.dtype)
        scores = (1 - l1_reg) * rand + l1_reg * adjusted

        ids_shuffle = scores.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]

        mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return {"mask": mask, "ids_keep": ids_keep, "ids_shuffle": ids_shuffle}

    def set_window_params(self, window_size: int, window_stride: int) -> None:
        self.window_size = max(0, int(window_size))
        self.window_stride = max(1, int(window_stride)) if self.window_size > 0 else 1
        self.next_window_start = 0

    def tome_merge_and_route(
        self,
        tokens: torch.Tensor,
        selection_ratio: float,
        text_length: int = 0,
        num_frames: int = 0,
        importance_map: Optional[torch.Tensor] = None,
        use_psg_importance: bool = False,
    ) -> Tuple[torch.Tensor, RouteInfo]:
        if text_length > 0:
            text_tokens, image_tokens = tokens[:, :text_length], tokens[:, text_length:]
        else:
            text_tokens, image_tokens = None, tokens

        if use_psg_importance and importance_map is None and num_frames > 1:
            if image_tokens.shape[1] % max(num_frames, 1) == 0:
                importance_map = compute_psg_temporal_importance(image_tokens, num_frames)

        generator = torch.Generator(device=tokens.device)
        generator.manual_seed(self.seed)
        target_frame_override: Optional[int] = None
        window_size = min(self.window_size, num_frames) if num_frames > 0 else 0
        if window_size > 0 and num_frames > 0 and image_tokens.shape[1] % num_frames == 0:
            window_start = self.next_window_start % num_frames
            window_frames = [(window_start + i) % num_frames for i in range(window_size)]
            tokens_per_frame = image_tokens.shape[1] // num_frames
            if importance_map is not None and tokens_per_frame > 0:
                frame_scores = importance_map.reshape(tokens.shape[0], num_frames, tokens_per_frame).mean(dim=(0, 2))
                window_scores = frame_scores[window_frames]
                target_frame_override = window_frames[torch.argmin(window_scores).item()]
            else:
                target_frame_override = window_frames[0]
            self.next_window_start = (self.next_window_start + self.window_stride) % max(num_frames, 1)

        merge_fn, _, info = bipartite_soft_matching_randframe(
            image_tokens,
            num_frames,
            selection_ratio,
            0,
            generator,
            align_batch=True,
            importance_map=importance_map,
            target_frame_override=target_frame_override,
        )
        # else:
        #     remove = int(image_tokens.shape[1] * selection_ratio)
        #     merge_fn, _, info = bipartite_soft_matching_rand2d(
        #         image_tokens,
        #         width=0,
        #         height=0,
        #         stride_x=1,
        #         stride_y=1,
        #         remove_tokens=remove,
        #         generator=generator,
        #         text_length=0,
        #         importance_map=importance_map,
        #     )

        merged_image_tokens = merge_fn(image_tokens)
        merged_tokens = (
            torch.cat([text_tokens, merged_image_tokens], dim=1) if text_tokens is not None else merged_image_tokens
        )

        keep_idx = []
        for b in range(tokens.shape[0]):
            mask = torch.ones(image_tokens.shape[1], dtype=torch.bool, device=tokens.device)
            if info["all_src_idx"][b].numel() > 0:
                mask[info["all_src_idx"][b]] = False
            keep_idx.append(mask.nonzero(as_tuple=True)[0])

        src_copy = torch.stack(
            [
                image_tokens[b, idx] if idx.numel() > 0 else image_tokens.new_empty((0, image_tokens.shape[-1]))
                for b, idx in enumerate(info["all_src_idx"])
            ]
        )

        pos_index = torch.stack(keep_idx, dim=0) if keep_idx else None

        route_info = RouteInfo(
            src_tokens=src_copy,
            src_idx=info["all_src_idx"],
            keep_idx=keep_idx,
            best_dst_indices=info.get("best_dst_indices", None),
            pos_index=pos_index,
            orig_shape=image_tokens.shape,
            text_length=text_length,
        )
        return merged_tokens, route_info

    def end_route(self, tokens: torch.Tensor, route_info: RouteInfo) -> torch.Tensor:
        text_length = route_info.text_length
        src_copy = route_info.src_tokens
        src_idx = route_info.src_idx
        batch_size, image_token_count, channels = route_info.orig_shape

        if text_length > 0:
            text_tokens = tokens[:, :text_length]
            processed_tokens = tokens[:, text_length:]
        else:
            text_tokens = None
            processed_tokens = tokens

        restored = tokens.new_zeros((batch_size, image_token_count, channels))
        for b in range(batch_size):
            keep_indices = route_info.keep_idx[b]
            if keep_indices.numel() > 0:
                restored[b, keep_indices] = processed_tokens[b, : keep_indices.numel()]
            if src_idx[b].numel() > 0:
                restored[b, src_idx[b]] = src_copy[b, : src_idx[b].numel()]

        return torch.cat([text_tokens, restored], dim=1) if text_tokens is not None else restored

    @torch.no_grad()
    def end_route_inference(self, tokens: torch.Tensor, route_info: RouteInfo) -> torch.Tensor:
        return self.end_route(tokens, route_info)


class RestoreAdapter(nn.Module):
    """A lightweight adapter that only updates tokens reinserted after merging."""

    def __init__(self, hidden_size: int, expansion: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + 1, expansion * hidden_size),
            nn.GELU(),
            nn.Linear(expansion * hidden_size, hidden_size),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, hidden_states: torch.Tensor, restored_mask: torch.Tensor) -> torch.Tensor:
        gate = restored_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        normed = self.norm(hidden_states)
        mlp_inp = torch.cat([normed, gate], dim=-1)
        delta = self.mlp(mlp_inp)
        return hidden_states + delta * gate


__all__ = [
    "Router",
    "RouteInfo",
    "RestoreAdapter",
    "bipartite_soft_matching_randframe",
    "bipartite_soft_matching_rand2d",
    "compute_psg_temporal_importance",
]


def compute_psg_temporal_importance(
    image_tokens: torch.Tensor,
    num_frames: int,
    eps: float = 1e-8,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute a PSG-style temporal curvature importance map for image tokens.

    Args:
        image_tokens: [B, N_img, C], where N_img = num_frames * tokens_per_frame.
        num_frames:   Number of frames (F).
        eps:          Small constant for numerical stability.
        normalize:    If True, normalize importance to [0, 1] per batch.

    Returns:
        importance_map: [B, N_img]
    """
    B, N_img, C = image_tokens.shape
    device = image_tokens.device

    if num_frames <= 0 or N_img == 0:
        return image_tokens.new_ones(B, N_img, device=device)

    if N_img % num_frames != 0:
        return image_tokens.new_ones(B, N_img, device=device)

    tokens_per_frame = N_img // num_frames
    x = image_tokens.reshape(B, num_frames, tokens_per_frame, C)

    if num_frames == 1:
        return image_tokens.new_ones(B, N_img, device=device)

    v = x[:, 1:, :, :] - x[:, :-1, :, :]  # [B, F-1, P, C]

    if num_frames == 2:
        v_mag = v.norm(dim=-1)  # [B, 1, P]
        importance = v_mag.expand(B, num_frames, tokens_per_frame)
        if normalize:
            flat = importance.reshape(B, -1)
            min_vals = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
            max_vals = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
            importance = (importance - min_vals) / (max_vals - min_vals + eps)
        return importance.reshape(B, N_img)

    v1 = v[:, :-1, :, :]
    v2 = v[:, 1:, :, :]

    dot = (v1 * v2).sum(dim=-1)
    n1 = v1.norm(dim=-1)
    n2 = v2.norm(dim=-1)

    cos = dot / (n1 * n2 + eps)
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    curvature = torch.arccos(cos)  # [B, F-2, P]

    importance = image_tokens.new_zeros((B, num_frames, tokens_per_frame), device=device)
    importance[:, 1:-1, :] = curvature
    importance[:, 0, :] = curvature[:, 0, :]
    importance[:, -1, :] = curvature[:, -1, :]

    if normalize:
        flat = importance.reshape(B, -1)
        min_vals = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        max_vals = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
        importance = (importance - min_vals) / (max_vals - min_vals + eps)

    return importance.reshape(B, N_img)


def parse_token_merge_routes(spec: Optional[str]) -> List[Dict[str, float]]:
    """Parse a simple route specification string into structured configs.

    Expected syntax: ``"start-end@ratio;..."``. Whitespace is ignored.
    """
    routes: List[Dict[str, float]] = []
    if not spec:
        return routes

    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            layer_span, ratio_str = chunk.split("@", maxsplit=1)
            start_str, end_str = layer_span.split("-", maxsplit=1)
            ratio_part = ratio_str.strip()
            ratio_start = None
            ratio_end = None
            if "->" in ratio_part:
                ratio_start_str, ratio_end_str = ratio_part.split("->", maxsplit=1)
                ratio_start = float(ratio_start_str.strip())
                ratio_end = float(ratio_end_str.strip())
            else:
                ratio_end = float(ratio_part)
            route = {
                "start_layer": int(start_str.strip()),
                "end_layer": int(end_str.strip()),
                "selection_ratio": float(ratio_end),
            }
            if ratio_start is not None:
                route["ratio_start"] = ratio_start
            if ratio_end is not None:
                route["ratio_end"] = ratio_end
        except ValueError as exc:
            raise ValueError(
                f"Invalid token-merge route spec '{chunk}'. Expected 'start-end@ratio' or 'start-end@start->end'."
            ) from exc
        if route["start_layer"] > route["end_layer"]:
            raise ValueError(f"Route start {route['start_layer']} cannot be greater than end {route['end_layer']}.")
        ratio_vals = [route["selection_ratio"]]
        if "ratio_start" in route:
            ratio_vals.append(route["ratio_start"])
        if "ratio_end" in route:
            ratio_vals.append(route["ratio_end"])
        if not all(0.0 <= val <= 1.0 for val in ratio_vals):
            raise ValueError(f"Route ratio values must be in [0,1], got {ratio_vals}.")
        routes.append(route)

    return routes


__all__.append("parse_token_merge_routes")
