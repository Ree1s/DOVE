from functools import partial
from einops import rearrange, repeat
import numpy as np

import torch
from torch import nn
import torch.nn.functional as tnf

from sat.model.base_model import BaseModel, non_conflict
from sat.model.mixins import BaseMixin
from sat.transformer_defaults import HOOKS_DEFAULT, attention_fn_default
from sat.mpu.layers import ColumnParallelLinear
from sgm.util import instantiate_from_config

from sgm.modules.diffusionmodules.openaimodel import Timestep
from sgm.modules.diffusionmodules.util import (
    linear,
    timestep_embedding,
)
from sat.ops.layernorm import LayerNorm, RMSNorm

from typing import Tuple, Callable, Union, List, Optional


def do_nothing(x: torch.Tensor, mode:str=None):
    return x


def mps_gather_workaround(input, dim, index):
    """MPS设备上的gather操作替代方案"""
    if input.shape[-1] == 1:
        return torch.gather(
            input.unsqueeze(-1),
            dim - 1 if dim < 0 else dim,
            index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        return torch.gather(input, dim, index)

def bipartite_soft_matching_randframe(metric: torch.Tensor,
                                      F: int, ratio: float, unm_pre: int, generator: torch.Generator,
                                      target_stride: int = 4, align_batch: bool = False,
                                      merge_mode: str = "replace",
                                      importance_map: torch.Tensor | None = None) -> Tuple[Callable, Callable, dict]:
    """
    Partitions the multi-frame tokens into src and dst and merges ratio of src tokens from src to dst.
    Dst tokens are partitioned by choosing one random frame.

    Args:
        - metric [B, N, C]: metric to use for similarity.
        - F: frame number.
        - ratio: ratio of src tokens to be removed (by merging).
        - unm_pre: number of src tokens not merged at previous ToMe. Pre-sequence: [unm_pre|F_0|F_1|...]
        - generator: random number generator
        - target_stride: stride of target frame.
        - align_batch: whether to align similarity matching maps of samples in the batch. True when using PnP.
        - merge_mode: how to merge tokens. "mean": tokens -> Mean(src_token, dst_token); "replace": tokens -> dst_token.
        - importance_map: optional [B, N] or [B, N - unm_pre] importance to guide dst/src (higher keeps as dst).

    Returns:
        Merge and unmerge operation according to the matching result. Return a dict including other values.
    """
    
    # 📊 Original random frame selection method with optional importance
    B, N, _ = metric.shape
    
    # Helper function to create empty index lists
    def create_empty_indices():
        return [torch.empty(0, dtype=torch.long, device=metric.device) for _ in range(B)]
    
    # Compute pre-frame token number. N = unm_pre + tnum * F.
    if F <= 0:
        print(f"Warning: Invalid frame number F={F} in bipartite_soft_matching_randframe. Skipping.")
        empty = create_empty_indices()
        return do_nothing, do_nothing, {
            "unm_num": N - unm_pre,
            "all_src_idx": empty,
            "all_dst_idx": empty
        }

    # Ensure N > unm_pre if F > 0
    if N <= unm_pre:
         print(f"Warning: N ({N}) <= unm_pre ({unm_pre}). Cannot compute tokens per frame. Skipping merge.")
         empty = create_empty_indices()
         return do_nothing, do_nothing, {
             "unm_num": 0,
             "all_src_idx": empty,
             "all_dst_idx": empty
         }

    tnum = (N - unm_pre) // F # Calculate tokens per frame

    if tnum <= 0:
        print(f"Warning: tnum ({tnum}) <= 0. Cannot perform merge. Skipping.")
        empty = create_empty_indices()
        return do_nothing, do_nothing, {
            "unm_num": N - unm_pre,
            "all_src_idx": empty,
            "all_dst_idx": empty
        }

    # Compute the number of tokens to be merged.
    num_dst = int(tnum * (1 - ratio))
    num_src = tnum - num_dst

    # Special case: when ratio=0, no merging should occur
    if ratio == 0.0 or num_src <= 0:
        # Return identity operations with empty source indices
        empty = create_empty_indices()
        return do_nothing, do_nothing, {
            "unm_num": N - unm_pre,
            "all_src_idx": empty,
            "all_dst_idx": empty
        }

    if num_dst <= 0:
        print(f"Warning: num_dst ({num_dst}) <= 0. Skipping merge.")
        empty = create_empty_indices()
        return do_nothing, do_nothing, {
            "unm_num": N - unm_pre,
            "all_src_idx": empty,
            "all_dst_idx": empty
        }

    def rand_indices(n, nk, generator):
        return torch.randperm(n, device=metric.device)[:nk]

    # For each sample in the batch
    all_dst_idx = []
    all_src_idx = []
    all_dst_tokens = []
    all_src_tokens = []

    for b in range(B):
        # Pick target frame
        target_frame = torch.randint(0, F, (1,), device=metric.device).item()
        
        # Get indices for target frame tokens
        target_start = unm_pre + target_frame * tnum
        target_end = target_start + tnum
        target_indices = torch.arange(target_start, target_end, device=metric.device)
        
        # Select dst tokens from target frame
        if importance_map is not None and tnum > 0:
            # importance_map can be image-only [B, N - unm_pre] or full [B, N]
            if importance_map.shape[1] == N:
                frame_imp = importance_map[b, target_start:target_end]
            else:
                imp_start = target_start - unm_pre
                imp_end = imp_start + tnum
                frame_imp = importance_map[b, imp_start:imp_end]
            # Keep the most important as destinations
            dst_indices_in_frame = torch.topk(frame_imp, k=num_dst, largest=True).indices
        else:
            # Random selection fallback
            dst_indices_in_frame = rand_indices(tnum, num_dst, generator)
        dst_idx = target_indices[dst_indices_in_frame]
        
        # Collect src tokens from all other frames
        src_idx_list = []
        for f in range(F):
            if f == target_frame:
                # From target frame, select remaining tokens as src
                remaining_indices = torch.arange(tnum, device=metric.device)
                mask = torch.ones(tnum, dtype=torch.bool, device=metric.device)
                mask[dst_indices_in_frame] = False
                src_indices_in_frame = remaining_indices[mask]
                frame_start = unm_pre + f * tnum
                src_idx_list.append(frame_start + src_indices_in_frame)
            else:
                # From other frames, select tokens
                frame_start = unm_pre + f * tnum
                frame_end = frame_start + tnum
                frame_indices = torch.arange(frame_start, frame_end, device=metric.device)
                if importance_map is not None and tnum > 0:
                    if importance_map.shape[1] == N:
                        frame_imp = importance_map[b, frame_start:frame_end]
                    else:
                        imp_start = frame_start - unm_pre
                        imp_end = imp_start + tnum
                        frame_imp = importance_map[b, imp_start:imp_end]
                    # Pick least important as sources
                    selected_indices_in_frame = torch.topk(frame_imp, k=num_src, largest=False).indices
                else:
                    selected_indices_in_frame = rand_indices(tnum, num_src, generator)
                src_idx_list.append(frame_indices[selected_indices_in_frame])
        
        src_idx = torch.cat(src_idx_list)
        
        # Ensure we have the right number of src tokens
        if len(src_idx) > F * num_src:
            src_idx = src_idx[:F * num_src]
        
        all_dst_idx.append(dst_idx)
        all_src_idx.append(src_idx)
        
        # Get actual tokens
        dst_tokens = metric[b, dst_idx]
        src_tokens = metric[b, src_idx]
        
        all_dst_tokens.append(dst_tokens)
        all_src_tokens.append(src_tokens)

    # Stack tokens for batch processing
    dst_tokens = torch.stack(all_dst_tokens) if all_dst_tokens else metric.new_empty((B, 0, metric.shape[-1]))
    src_tokens = torch.stack(all_src_tokens) if all_src_tokens else metric.new_empty((B, 0, metric.shape[-1]))

    # Compute matching: importance-aware or cosine similarity
    if dst_tokens.numel() > 0 and src_tokens.numel() > 0:
        if importance_map is not None:
            # Deterministic mapping; not used in removal-only merge
            best_dst_indices = torch.zeros((B, src_tokens.shape[1]), dtype=torch.long, device=metric.device)
        else:
            similarity = torch.einsum('bsc,bdc->bsd', src_tokens, dst_tokens)  # [B, F*num_src, num_dst]
            if align_batch:
                similarity = similarity.mean(dim=0, keepdim=True).expand(B, -1, -1)
            _, best_dst_indices = similarity.max(dim=2)  # [B, F*num_src]
    else:
        best_dst_indices = torch.empty((B, 0), dtype=torch.long, device=metric.device)

    def merge_tokens(x: torch.Tensor, mode: str = None) -> torch.Tensor:
        if mode is None:
            mode = merge_mode
            
        B_x, N_x, C_x = x.shape
        if N_x != N:
            print(f"Warning: Token count mismatch in merge. Expected {N}, got {N_x}. Skipping merge.")
            return x

        # For "replace" mode: only remove src tokens, keep dst tokens unchanged
        output_tokens_list = []
        for b in range(B_x):
            current_item_src_idx = all_src_idx[b]
            item_keep_mask = torch.ones(N_x, dtype=torch.bool, device=x.device)
            if current_item_src_idx.numel() > 0:
                item_keep_mask[current_item_src_idx] = False
            kept_tokens_for_item_b = x[b, item_keep_mask]
            output_tokens_list.append(kept_tokens_for_item_b)
        return torch.stack(output_tokens_list, dim=0)

    def unmerge_tokens(x: torch.Tensor, mode: str = None) -> torch.Tensor:
        # Router handles restoration
        return x

    # Calculate final token count after merge
    total_src_tokens = sum(len(src_idx) for src_idx in all_src_idx)
    unm_num = N - total_src_tokens // B  # Average reduction per sample

    return merge_tokens, unmerge_tokens, {
        "unm_num": unm_num, 
        "all_src_idx": all_src_idx,
        "all_dst_idx": all_dst_idx
    }


def bipartite_soft_matching_rand2d(metric: torch.Tensor,
                                   w: int, h: int, sx: int, sy: int, r: int,
                                   no_rand: bool = False,
                                   generator: torch.Generator = None,
                                   trace_source: bool = False,
                                   source: torch.Tensor = None,
                                   text_length: int = 0,
                                   importance_map: torch.Tensor | None = None) -> Tuple[Callable, Callable, dict]:
    """
    Bipartite soft matching rand2d (token-removal style like randframe, no zero-masking).

    Args:
        metric: [B, N_total, C] tokens. If text_length > 0, they are at the front and will be preserved.
        w, h, sx, sy: kept for API compatibility (unused here).
        r: number of image tokens to remove (as src).
        no_rand: if True, use uniform selection; else random selection.
        generator: random generator.
        trace_source, source: kept for API compatibility (unused here).
        text_length: number of text tokens at the front (preserved).
        importance_map: optional [B, N_total] LR-importance map to guide removal (lower -> more removable).

    Returns:
        merge_fn, unmerge_fn, info where info contains 'all_src_idx' like randframe.
    """
    B, N_total, C = metric.shape

    # Operate on image tokens only; indices are relative to image segment
    N_img = N_total - text_length
    if N_img <= 0 or r <= 0:
        return do_nothing, do_nothing, {
            "unm_num": N_total,
            "all_src_idx": [torch.empty(0, dtype=torch.long, device=metric.device) for _ in range(B)],
            "all_dst_idx": [torch.empty(0, dtype=torch.long, device=metric.device) for _ in range(B)],
        }

    r = min(r, N_img)

    if generator is None:
        generator = torch.Generator(device=metric.device)
        generator.manual_seed(42)

    # Select indices within image tokens [0, N_img)
    if importance_map is not None:
        # Per-batch selection of lowest-importance image tokens
        if importance_map.shape[1] == N_total:
            importance_img = importance_map[:, text_length:]  # [B, N_img]
        else:
            # Assume importance provided for image-only tokens already
            importance_img = importance_map
        selected_list = []
        for b in range(B):
            imp_b = importance_img[b]
            sel_b = torch.argsort(imp_b, dim=-1)[:r]  # low importance first → sources
            selected_list.append(sel_b.to(metric.device))
        # Convert to global indices [0, N_total)
        all_src_idx = [sel + text_length for sel in selected_list]
    else:
        if no_rand:
            step = max(1, N_img // r)
            selected_img_idx = torch.arange(0, N_img, step, device=metric.device)[:r]
        else:
            selected_img_idx = torch.randperm(N_img, device=metric.device, generator=generator)[:r]
        # For token-removal merge, we need global indices into [0, N_total)
        selected_global_idx = selected_img_idx + text_length
        # Build per-batch src index lists
        all_src_idx = [selected_global_idx for _ in range(B)]

    # We do not use dst indices in removal-only mode
    all_dst_idx = [torch.empty(0, dtype=torch.long, device=metric.device) for _ in range(B)]

    def merge_tokens(x: torch.Tensor, mode: str = None) -> torch.Tensor:
        B_x, N_x, C_x = x.shape
        if N_x != N_total:
            return x
        output_tokens_list = []
        for b in range(B_x):
            current_item_src_idx = all_src_idx[b]
            keep_mask = torch.ones(N_x, dtype=torch.bool, device=x.device)
            if current_item_src_idx.numel() > 0:
                keep_mask[current_item_src_idx] = False
            kept = x[b, keep_mask]
            output_tokens_list.append(kept)
        return torch.stack(output_tokens_list, dim=0)

    def unmerge_tokens(x: torch.Tensor, mode: str = None) -> torch.Tensor:
        # For router-based approach, traditional unmerge is handled externally
        return x

    unm_num = N_total - (r // 1)  # Reduced by r tokens globally (since indices are global)

    return merge_tokens, unmerge_tokens, {
        "unm_num": unm_num,
        "all_src_idx": all_src_idx,
        "all_dst_idx": all_dst_idx,
    }


class ImagePatchEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        in_channels,
        hidden_size,
        patch_size,
        bias=True,
        text_hidden_size=None,
    ):
        super().__init__()
        # print(in_channels)
        # self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.proj_sr = nn.Conv2d(in_channels * 2, hidden_size, kernel_size=patch_size, stride=patch_size, bias=bias)

        # 复制原始层前16个通道的权重
        # self.proj_sr.weight.data[:, :in_channels, :, :] = self.proj.weight.data.clone()
        # # 将后16个通道的权重初始化为零
        # torch.nn.init.constant_(self.proj_sr.weight.data[:, in_channels:, :, :], 0)

        # # 如果使用了 bias，直接复制原有的 bias 值
        # if bias:
        #     self.proj_sr.bias.data = self.proj.bias.data.clone()

        if text_hidden_size is not None:
            self.text_proj = nn.Linear(text_hidden_size, hidden_size)
        else:
            self.text_proj = None

    def word_embedding_forward(self, input_ids, **kwargs):
        # now is 3d patch
        images = kwargs["images"]  # (b,t,c,h,w)
        B, T = images.shape[:2]
        emb = images.view(-1, *images.shape[2:])
        
        #--------
        # Debug
        #--------
        # emb_ori = emb
        # x_ori, _ = emb.chunk(2, dim=1)
        # emb = self.proj(x_ori)
        # emb_debug = self.proj_sr(emb_ori)  # ((b t),d,h/2,w/2)  [2 * 8, 16, 60, 90]
        # print(torch.sqrt((emb - emb_debug)**2).mean())
        
        emb = self.proj_sr(emb)  # ((b t),d,h/2,w/2)  [2 * 8, 32, 60, 90]
        emb = emb.view(B, T, *emb.shape[1:])
        emb = emb.flatten(3).transpose(2, 3)  # (b,t,n,d)
        emb = rearrange(emb, "b t n d -> b (t n) d")

        if self.text_proj is not None:
            text_emb = self.text_proj(kwargs["encoder_outputs"])
            emb = torch.cat((text_emb, emb), dim=1)  # (b,n_t+t*n_i,d)

        emb = emb.contiguous()
        return emb  # (b,n_t+t*n_i,d)

    def reinit(self, parent_model=None):
        w = self.proj_sr.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.proj_sr.bias, 0)
        del self.transformer.word_embeddings


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_height,
    grid_width,
    t_size,
    cls_token=False,
    height_interpolation=1.0,
    width_interpolation=1.0,
    time_interpolation=1.0,
):
    """
    grid_size: int of the grid height and width
    t_size: int of the temporal size
    return:
    pos_embed: [t_size*grid_size*grid_size, embed_dim] or [1+t_size*grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 4 * 3
    embed_dim_temporal = embed_dim // 4

    # spatial
    grid_h = np.arange(grid_height, dtype=np.float32) / height_interpolation
    grid_w = np.arange(grid_width, dtype=np.float32) / width_interpolation
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # temporal
    grid_t = np.arange(t_size, dtype=np.float32) / time_interpolation
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # concate: [T, H, W] order
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_height * grid_width, axis=1)  # [T, H*W, D // 4]
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, t_size, axis=0)  # [T, H*W, D // 4 * 3]

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    # pos_embed = pos_embed.reshape([-1, embed_dim])  # [T*H*W, D]

    return pos_embed  # [T, H*W, D]


def get_2d_sincos_pos_embed(embed_dim, grid_height, grid_width, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_height, dtype=np.float32)
    grid_w = np.arange(grid_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_height, grid_width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Basic3DPositionEmbeddingMixin(BaseMixin):
    """
    Routing-friendly absolute 3D sin-cos positional embedding with optional index-aware gather.
    """

    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        text_length=0,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.text_length = text_length
        self.compressed_num_frames = compressed_num_frames
        self.spatial_length = height * width
        self.num_patches = height * width * compressed_num_frames
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, int(text_length + self.num_patches), int(hidden_size)), requires_grad=False
        )
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation

    @torch.no_grad()
    def _gather_image_pos_by_index(self, pos_index_image: torch.Tensor) -> torch.Tensor:
        if pos_index_image.dim() != 2:
            raise ValueError(f"pos_index_image must be [B, L], got {tuple(pos_index_image.shape)}")
        img_pos_all = self.pos_embedding[0, self.text_length : self.text_length + self.num_patches]
        if pos_index_image.numel() > 0:
            max_idx = int(pos_index_image.max().item())
            if max_idx >= img_pos_all.size(0):
                raise IndexError(
                    f"pos_index_image index {max_idx} out of range for image positions {img_pos_all.size(0)}"
                )
        return img_pos_all[pos_index_image]

    def position_embedding_forward(self, position_ids, **kwargs):
        images = kwargs.get("images", None)
        pos_index_image = kwargs.get("pos_index_image", None)

        if pos_index_image is not None:
            batch = pos_index_image.size(0)
            if self.text_length > 0:
                pos_text = self.pos_embedding[:, : self.text_length].expand(batch, -1, -1).contiguous()
            else:
                pos_text = None
            pos_img = self._gather_image_pos_by_index(pos_index_image)
            return torch.cat([pos_text, pos_img], dim=1) if pos_text is not None else pos_img

        if images is not None and images.shape[1] == 1:
            return self.pos_embedding[:, : self.text_length + self.spatial_length]

        return self.pos_embedding[:, : self.text_length + kwargs["seq_length"]]

    def reinit(self, parent_model=None):
        del self.transformer.position_embeddings
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embedding.shape[-1],
            self.height,
            self.width,
            self.compressed_num_frames,
            height_interpolation=self.height_interpolation,
            width_interpolation=self.width_interpolation,
            time_interpolation=self.time_interpolation,
        )
        pos_embed = torch.from_numpy(pos_embed).float()
        pos_embed = rearrange(pos_embed, "t n d -> (t n) d")
        self.pos_embedding.data[:, -self.num_patches :].copy_(pos_embed)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class Rotary3DPositionEmbeddingMixin(BaseMixin):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        text_length,
        theta=10000,
        rot_v=False,
        learnable_pos_embed=False,
    ):
        super().__init__()
        self.rot_v = rot_v

        dim_t = hidden_size_head // 4
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

        grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        freqs = rearrange(freqs, "t h w d -> (t h w) d")

        freqs = freqs.contiguous()
        freqs_sin = freqs.sin()
        freqs_cos = freqs.cos()
        self.register_buffer("freqs_sin", freqs_sin)
        self.register_buffer("freqs_cos", freqs_cos)

        self.text_length = text_length
        if learnable_pos_embed:
            num_patches = height * width * compressed_num_frames + text_length
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, int(hidden_size)), requires_grad=True)
        else:
            self.pos_embedding = None

    def rotary(self, t: torch.Tensor, pos_index: Optional[torch.Tensor] = None):
        B, Hh, L, _ = t.shape
        if pos_index is None:
            freqs_cos = self.freqs_cos[:L].unsqueeze(0).unsqueeze(0)
            freqs_sin = self.freqs_sin[:L].unsqueeze(0).unsqueeze(0)
        else:
            flat_idx = pos_index.reshape(-1)
            gather_cos = self.freqs_cos.index_select(0, flat_idx).view(B, L, -1).unsqueeze(1)
            gather_sin = self.freqs_sin.index_select(0, flat_idx).view(B, L, -1).unsqueeze(1)
            freqs_cos, freqs_sin = gather_cos, gather_sin
        freqs_cos = freqs_cos.to(dtype=t.dtype)
        freqs_sin = freqs_sin.to(dtype=t.dtype)
        return t * freqs_cos + rotate_half(t) * freqs_sin

    def position_embedding_forward(self, position_ids, **kwargs):
        if self.pos_embedding is not None:
            return self.pos_embedding[:, :self.text_length + kwargs["seq_length"]]
        else:
            return None

    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        **kwargs,
    ):
        attention_fn_fallback = HOOKS_DEFAULT["attention_fn"]
        pos_index_image = kwargs.get("pos_index_image", None)
        query_layer[:, :, self.text_length :] = self.rotary(
            query_layer[:, :, self.text_length :], pos_index_image
        )
        key_layer[:, :, self.text_length :] = self.rotary(
            key_layer[:, :, self.text_length :], pos_index_image
        )
        if self.rot_v:
            value_layer[:, :, self.text_length :] = self.rotary(
                value_layer[:, :, self.text_length :], pos_index_image
            )

        return attention_fn_fallback(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def unpatchify(x, c, p, w, h, rope_position_ids=None, **kwargs):
    """
    x: (N, T/2 * S, patch_size**3 * C)
    imgs: (N, T, H, W, C)
    """
    if rope_position_ids is not None:
        assert NotImplementedError
        # do pix2struct unpatchify
        L = x.shape[1]
        x = x.reshape(shape=(x.shape[0], L, p, p, c))
        x = torch.einsum("nlpqc->ncplq", x)
        imgs = x.reshape(shape=(x.shape[0], c, p, L * p))
    else:
        b = x.shape[0]
        imgs = rearrange(x, "b (t h w) (c p q) -> b t c (h p) (w q)", b=b, h=h, w=w, c=c, p=p, q=p)

    return imgs


class FinalLayerMixin(BaseMixin):
    def __init__(
        self,
        hidden_size,
        time_embed_dim,
        patch_size,
        out_channels,
        latent_width,
        latent_height,
        elementwise_affine,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=elementwise_affine, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 2 * hidden_size, bias=True))

        self.spatial_length = latent_width * latent_height // patch_size**2
        self.latent_width = latent_width
        self.latent_height = latent_height

    def final_forward(self, logits, **kwargs):
        x, emb = logits[:, kwargs["text_length"] :, :], kwargs["emb"]  # x:(b,(t n),d)

        shift, scale = self.adaLN_modulation(emb).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return unpatchify(
            x,
            c=self.out_channels,
            p=self.patch_size,
            w=self.latent_width // self.patch_size,
            h=self.latent_height // self.patch_size,
            rope_position_ids=kwargs.get("rope_position_ids", None),
            **kwargs,
        )

    def reinit(self, parent_model=None):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)


class SwiGLUMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features, bias=False):
        super().__init__()
        self.w2 = nn.ModuleList(
            [
                ColumnParallelLinear(
                    in_features,
                    hidden_features,
                    gather_output=False,
                    bias=bias,
                    module=self,
                    name="dense_h_to_4h_gate",
                )
                for i in range(num_layers)
            ]
        )

    def mlp_forward(self, hidden_states, **kw_args):
        x = hidden_states
        origin = self.transformer.layers[kw_args["layer_id"]].mlp
        x1 = origin.dense_h_to_4h(x)
        x2 = self.w2[kw_args["layer_id"]](x)
        hidden = origin.activation_func(x2) * x1
        x = origin.dense_4h_to_h(hidden)
        return x


class AdaLNMixin(BaseMixin):
    def __init__(
        self,
        width,
        height,
        hidden_size,
        num_layers,
        time_embed_dim,
        compressed_num_frames,
        qk_ln=True,
        hidden_size_head=None,
        elementwise_affine=True,
        enable_routing: bool = False,
        routes: list | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.width = width
        self.height = height
        self.compressed_num_frames = compressed_num_frames
        self.enable_routing = enable_routing
        self.routes = routes if routes is not None else []

        self.adaLN_modulations = nn.ModuleList(
            [nn.Sequential(nn.SiLU(), nn.Linear(time_embed_dim, 12 * hidden_size)) for _ in range(num_layers)]
        )

        self.qk_ln = qk_ln
        if qk_ln:
            self.query_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )
            self.key_layernorm_list = nn.ModuleList(
                [
                    LayerNorm(hidden_size_head, eps=1e-6, elementwise_affine=elementwise_affine)
                    for _ in range(num_layers)
                ]
            )

    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        **kwargs,
    ):
        text_length = kwargs["text_length"]
        feature_tap = kwargs.get("feature_tap", None)
        layer_id = kwargs.get("layer_id", 0)
        if feature_tap is not None:
            feature_tap.capture_layer_input(layer_id, hidden_states, text_length)
        # hidden_states (b,(n_t+t*n_i),d)
        text_hidden_states = hidden_states[:, :text_length]  # (b,n,d)
        img_hidden_states = hidden_states[:, text_length:]  # (b,(t n),d)
        layer = self.transformer.layers[kwargs["layer_id"]]
        adaLN_modulation = self.adaLN_modulations[kwargs["layer_id"]]

        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            text_shift_msa,
            text_scale_msa,
            text_gate_msa,
            text_shift_mlp,
            text_scale_mlp,
            text_gate_mlp,
        ) = adaLN_modulation(kwargs["emb"]).chunk(12, dim=1)
        gate_msa, gate_mlp, text_gate_msa, text_gate_mlp = (
            gate_msa.unsqueeze(1),
            gate_mlp.unsqueeze(1),
            text_gate_msa.unsqueeze(1),
            text_gate_mlp.unsqueeze(1),
        )

        # self full attention (b,(t n),d)   b: batchsize; (t n): temp & spa; d: hidden_size
        img_attention_input = layer.input_layernorm(img_hidden_states)
        text_attention_input = layer.input_layernorm(text_hidden_states)
        img_attention_input = modulate(img_attention_input, shift_msa, scale_msa)
        text_attention_input = modulate(text_attention_input, text_shift_msa, text_scale_msa)

        # Determine effective height and width based on routing state
        effective_height = self.height
        effective_width = self.width
        
        # Safely access transformer attributes
        transformer = self.transformer
        route_info = getattr(transformer, 'current_route_info', None)
        enable_routing = getattr(transformer, 'enable_routing', False)
        routes = getattr(transformer, 'routes', [])
        
        # Check if routing is active
        if route_info is not None and enable_routing and routes:
            # Get current route index
            active_route_idx = getattr(transformer, 'current_route_idx', 0)
            
            if active_route_idx < len(routes):
                current_route = routes[active_route_idx]
                selection_ratio = current_route['selection_ratio']
                scaling_factor = np.sqrt(1 - selection_ratio)

                effective_height = int(self.height * scaling_factor)
                effective_width = int(self.width * scaling_factor)
                
                # Ensure dimensions are at least 1
                effective_height = max(1, effective_height)
                effective_width = max(1, effective_width)

        # Spatial LIEM
        _, thw, _ = img_attention_input.shape
        t = thw // (effective_height * effective_width)
        spa_fea = rearrange(img_attention_input, 'b (t h w) c -> (b t) c h w', h=effective_height, w=effective_width)
        spa_fea = layer.spa_local(spa_fea)

        # Temporal LIEM
        temp_fea = rearrange(spa_fea, '(b t) c h w -> (b h w) t c', h=effective_height, w=effective_width, t=t)
        temp_fea = layer.temp_local(temp_fea)

        img_attention_input = rearrange(temp_fea, '(b h w) t c -> b (t h w) c', h=effective_height, w=effective_width)

        attention_input = torch.cat((text_attention_input, img_attention_input), dim=1)  # (b,n_t+t*n_i,d)
        attention_output = layer.attention(attention_input, mask, **kwargs)
        text_attention_output = attention_output[:, :text_length]  # (b,n,d)
        img_attention_output = attention_output[:, text_length:]  # (b,(t n),d)

        if self.transformer.layernorm_order == "sandwich":
            text_attention_output = layer.third_layernorm(text_attention_output)
            img_attention_output = layer.third_layernorm(img_attention_output)
        img_hidden_states = img_hidden_states + gate_msa * img_attention_output  # (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_msa * text_attention_output  # (b,n,d)

        # mlp (b,(t n),d)
        img_mlp_input = layer.post_attention_layernorm(img_hidden_states)  # vision (b,(t n),d)
        text_mlp_input = layer.post_attention_layernorm(text_hidden_states)  # language (b,n,d)
        img_mlp_input = modulate(img_mlp_input, shift_mlp, scale_mlp)
        text_mlp_input = modulate(text_mlp_input, text_shift_mlp, text_scale_mlp)
        mlp_input = torch.cat((text_mlp_input, img_mlp_input), dim=1)  # (b,(n_t+t*n_i),d
        mlp_output = layer.mlp(mlp_input, **kwargs)
        img_mlp_output = mlp_output[:, text_length:]  # vision (b,(t n),d)
        text_mlp_output = mlp_output[:, :text_length]  # language (b,n,d)
        if self.transformer.layernorm_order == "sandwich":
            text_mlp_output = layer.fourth_layernorm(text_mlp_output)
            img_mlp_output = layer.fourth_layernorm(img_mlp_output)

        img_hidden_states = img_hidden_states + gate_mlp * img_mlp_output  # vision (b,(t n),d)
        text_hidden_states = text_hidden_states + text_gate_mlp * text_mlp_output  # language (b,n,d)

        hidden_states = torch.cat((text_hidden_states, img_hidden_states), dim=1)  # (b,(n_t+t*n_i),d)
        if feature_tap is not None:
            feature_tap.capture_layer_output(layer_id, hidden_states)

        return hidden_states

    def reinit(self, parent_model=None):
        for layer in self.adaLN_modulations:
            nn.init.constant_(layer[-1].weight, 0)
            nn.init.constant_(layer[-1].bias, 0)

    @non_conflict
    def attention_fn(
        self,
        query_layer,
        key_layer,
        value_layer,
        attention_mask,
        attention_dropout=None,
        log_attention_weights=None,
        scaling_attention_score=True,
        old_impl=attention_fn_default,
        **kwargs,
    ):
        if self.qk_ln:
            query_layernorm = self.query_layernorm_list[kwargs["layer_id"]]
            key_layernorm = self.key_layernorm_list[kwargs["layer_id"]]
            query_layer = query_layernorm(query_layer)
            key_layer = key_layernorm(key_layer)

        return old_impl(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attention_dropout=attention_dropout,
            log_attention_weights=log_attention_weights,
            scaling_attention_score=scaling_attention_score,
            **kwargs,
        )


str_to_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


import torch
from typing import List, Tuple

class Router:
    """
    Router for token masking, ToMe merging and restoration.

    Implements:
      - get_mask: stochastic token masking based on magnitude.
      - tome_merge_and_route: ToMe merge and routing information.
      - start_route: apply shuffle and mask.
      - end_route: restore tokens vectorized.

    Reference:
      - ToMe: Token Merging for Efficient Vision Transformers, ICLR 2023 citeref1
    """

    def __init__(self, seed: int = 42):
        """
        Initialize router with a fixed random seed.

        Args:
            seed (int): Seed for reproducible operations.

        初始化路由器并设置随机种子。
        Args:
            seed (int): 用于可重复操作的随机种子。
        """
        self.seed = seed

    def get_mask(self,
                 x: torch.Tensor,
                 mask_ratio: float = 0.0,
                 l1_reg: float = 0.0,
                 inverse: bool = False) -> dict:
        """
        English:
        Compute a boolean mask for stochastic token dropping based on normalized magnitudes
        and random noise.

        中文：
        基于归一化后令牌幅值和随机噪声计算用于随机丢弃令牌的布尔遮罩。

        Args:
            x (Tensor[B,N,C]): Input tokens.
            mask_ratio (float): Fraction of tokens to drop.
            l1_reg (float): Weight for magnitude-based regularization.
            inverse (bool): If True, low-magnitude tokens are preferred.

        Returns:
            dict: {{
                'mask': Tensor[B,N] (bool),
                'ids_keep': Tensor[B,K] (long),
                'ids_shuffle': Tensor[B,N] (long)
            }}
        """
        B, N, C = x.shape
        device, dtype = x.device, x.dtype
        k = int(N * (1 - mask_ratio))

        # Compute normalized magnitudes
        mags = x.abs().sum(dim=-1)
        minm = mags.min(dim=1, keepdim=True)[0]
        maxm = mags.max(dim=1, keepdim=True)[0]
        norm = (mags - minm) / (maxm - minm + 1e-8)
        adjusted = 1.0 - norm if inverse else norm

        # Combine random noise and magnitude score
        rand = torch.rand(B, N, device=device, dtype=dtype)
        scores = (1 - l1_reg) * rand + l1_reg * adjusted

        # Shuffle and mask
        ids_shuffle = scores.argsort(dim=1)
        ids_keep = ids_shuffle[:, :k]
        mask = torch.ones(B, N, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        return {'mask': mask, 'ids_keep': ids_keep, 'ids_shuffle': ids_shuffle}

    def tome_merge_and_route(self,
                             x: torch.Tensor,
                             selection_ratio: float,
                             text_length: int = 0,
                             F: int = 13) -> Tuple[torch.Tensor, dict]:
        """
        Merge tokens using bipartite_soft_matching (80% randframe, 20% rand2d) and
        record source indices for later restoration.
        """
        # Split text and image tokens
        if text_length > 0:
            text, img = x[:, :text_length], x[:, text_length:]
        else:
            text, img = None, x

        # Prepare RNG
        gen = torch.Generator(device=img.device)
        gen.manual_seed(self.seed)

        # Decide method: 80% randframe, 20% rand2d
        prob = torch.rand((), generator=gen, device=img.device).item()
        use_randframe = prob < 0.8

        if use_randframe:
            merge_fn, unmerge_fn, info = bipartite_soft_matching_randframe(
                img, F, selection_ratio, 0, gen, merge_mode="replace"
            )
            all_src_idx = info['all_src_idx']
        else:
            # Compute r from selection_ratio over image tokens
            B_img, N_img, C_img = img.shape
            r = int(N_img * selection_ratio)
            merge_fn, unmerge_fn, info = bipartite_soft_matching_rand2d(
                img, w=30, h=45, sx=3, sy=2, r=r, generator=gen, text_length=text_length
            )
            all_src_idx = info['all_src_idx']

        B, N_img, C = img.shape
        all_keep_idx = []
        for b in range(B):
            keep_mask = torch.ones(N_img, dtype=torch.bool, device=img.device)
            idx_b = all_src_idx[b]
            if idx_b.numel() > 0:
                keep_mask[idx_b] = False
            keep_idx = keep_mask.nonzero(as_tuple=True)[0]
            all_keep_idx.append(keep_idx)
        keep_idx_tensor = torch.stack(all_keep_idx, dim=0) if all_keep_idx else torch.empty(0, device=img.device, dtype=torch.long)

        # Extract source tokens to restore later
        src_copy = torch.stack([
            img[b, idx] if idx.numel() > 0 else img.new_empty((0, C))
            for b, idx in enumerate(all_src_idx)
        ])

        # Merge image tokens
        merged_img = merge_fn(img)
        merged_x = torch.cat([text, merged_img], dim=1) if text is not None else merged_img

        route_info = {
            'src_tokens': src_copy,
            'src_idx': all_src_idx,
            'keep_idx': all_keep_idx,
            'pos_index': keep_idx_tensor,
            'orig_shape': img.shape,
            'text_length': text_length
        }
        return merged_x, route_info

    def tome_merge_and_route_liem(self,
                                  x: torch.Tensor,
                                  selection_ratio: float,
                                  text_length: int = 0,
                                  F: int | None = None,
                                  height: int | None = None,
                                  width: int | None = None,
                                  lambda_add: float = 0.5,
                                  spa_fea: torch.Tensor | None = None,
                                  temp_fea: torch.Tensor | None = None) -> Tuple[torch.Tensor, dict]:
        """
        LIEM-based token removal routing.
        - Computes spatial importance and temporal confidence from spa_fea/temp_fea.
        - Builds an importance_map (higher=keep as dst, lower=remove as src).
        - Calls randframe/rand2d with importance_map to select src/dst and remove tokens.
        """
        # Split text and image tokens
        if text_length > 0:
            text, img = x[:, :text_length], x[:, text_length:]
        else:
            text, img = None, x

        B, N_img, C = img.shape
        if height is None or width is None:
            raise ValueError("height and width must be provided for LIEM routing")
        N_per_frame = height * width
        if N_per_frame <= 0:
            # Nothing to do
            return x, {
                'src_tokens': img.new_empty((B, 0, C)),
                'src_idx': [torch.empty(0, dtype=torch.long, device=img.device) for _ in range(B)],
                'orig_shape': img.shape,
                'text_length': text_length
            }
        if F is None:
            if N_img % N_per_frame != 0:
                raise ValueError(f"Image token count {N_img} is not divisible by HxW={N_per_frame}")
            F = N_img // N_per_frame
        if selection_ratio <= 0.0:
            # No reduction
            return x, {
                'src_tokens': img.new_empty((B, 0, C)),
                'src_idx': [torch.empty(0, dtype=torch.long, device=img.device) for _ in range(B)],
                'orig_shape': img.shape,
                'text_length': text_length
            }

        device, dtype = img.device, img.dtype
        eps = 1e-8

        if spa_fea is None or temp_fea is None:
            raise ValueError("spa_fea and temp_fea must be provided for LIEM routing")

        # spa_fea: [(B*T), C, H, W] after layer.spa_local
        # Spatial importance: use channel-mean response (already LIEM-weighted), optional 7x7 avg smoothing
        spa_mean = spa_fea.mean(dim=1, keepdim=False)  # [(B*T), H, W]
        spa_mean_2d = spa_mean.reshape(B * F, 1, height, width)
        A_spatial = tnf.avg_pool2d(spa_mean_2d, kernel_size=7, stride=1, padding=3)
        A_spatial = A_spatial.reshape(B, F, height, width)
        S_spatial = A_spatial.reshape(B, F, N_per_frame)
        # Normalize per frame
        s_min = S_spatial.amin(dim=2, keepdim=True)
        s_max = S_spatial.amax(dim=2, keepdim=True)
        S_spatial = (S_spatial - s_min) / (s_max - s_min + eps)

        # temp_fea: [(B*H*W), T, C] after layer.temp_local
        # Temporal confidence: channel-mean response per time
        temp_mean = temp_fea.mean(dim=-1)  # [(B*H*W), T]
        tmp = temp_mean.reshape(B, height * width, F)  # [B, N, F]
        # Smooth over time using avg_pool1d
        t_smooth = tnf.avg_pool1d(tmp, kernel_size=3, stride=1, padding=1)  # [B, N, F]
        W_temp = t_smooth.permute(0, 2, 1)  # [B, F, N]
        # Normalize per frame
        w_min = W_temp.amin(dim=2, keepdim=True)
        w_max = W_temp.amax(dim=2, keepdim=True)
        W_temp = (W_temp - w_min) / (w_max - w_min + eps)

        # Effective importance (additive coupling)
        S_eff = S_spatial + lambda_add * (1.0 - W_temp)  # [B, F, N]
        # Normalize again per frame to [0,1]
        e_min = S_eff.amin(dim=2, keepdim=True)
        e_max = S_eff.amax(dim=2, keepdim=True)
        S_eff = (S_eff - e_min) / (e_max - e_min + eps)

        # Convert to keep-importance: higher means keep
        S_keep = 1.0 - S_eff  # [B, F, N]
        importance_map_img = S_keep.reshape(B, F * N_per_frame)

        # Prepare RNG
        gen = torch.Generator(device=img.device)
        gen.manual_seed(self.seed)
        prob = torch.rand((), generator=gen, device=img.device).item()
        use_randframe = prob < 0.8

        if use_randframe:
            merge_fn, unmerge_fn, info = bipartite_soft_matching_randframe(
                img, F, selection_ratio, 0, gen, merge_mode="replace", importance_map=importance_map_img
            )
        else:
            r = int(N_img * selection_ratio)
            merge_fn, unmerge_fn, info = bipartite_soft_matching_rand2d(
                img, w=width, h=height, sx=1, sy=1, r=r, generator=gen, text_length=0, importance_map=importance_map_img
            )

        all_src_idx = info['all_src_idx']
        all_keep_idx = []
        for b in range(B):
            keep_mask = torch.ones(N_img, dtype=torch.bool, device=img.device)
            idx_b = all_src_idx[b]
            if idx_b.numel() > 0:
                keep_mask[idx_b] = False
            keep_idx = keep_mask.nonzero(as_tuple=True)[0]
            all_keep_idx.append(keep_idx)
        keep_idx_tensor = torch.stack(all_keep_idx, dim=0) if all_keep_idx else torch.empty(0, device=img.device, dtype=torch.long)
        # Copy sources for restoration
        src_copy = torch.stack([
            img[b, idx] if idx.numel() > 0 else img.new_empty((0, C))
            for b, idx in enumerate(all_src_idx)
        ])
        merged_img = merge_fn(img)
        merged_x = torch.cat([text, merged_img], dim=1) if text is not None else merged_img
        route_info = {
            'src_tokens': src_copy,
            'src_idx': all_src_idx,
            'keep_idx': all_keep_idx,
            'pos_index': keep_idx_tensor,
            'orig_shape': img.shape,
            'text_length': text_length
        }
        return merged_x, route_info

    def start_route(self,
                    x: torch.Tensor,
                    mask_info: dict) -> torch.Tensor:
        """
        Shuffle and mask tokens before routing (backward-compatible).
        """
        ids = mask_info['ids_shuffle']
        k = mask_info['ids_keep'].size(1)
        return x.gather(
            1,
            ids.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )[:, :k]

    def end_route(self,
                  masked_x: torch.Tensor,
                  route_info: dict) -> torch.Tensor:
        """
        English:
        🚀 Vectorized restore original token order using recorded source indices.

        中文：
        🚀 向量化恢复原始令牌序列使用记录的源索引。

        Args:
            masked_x (Tensor[B,N_reduced,C]): Tokens after processing.
            route_info (dict): Routing info from tome_merge_and_route.

        Returns:
            restored_x (Tensor[B,N_original,C]): Restored tokens.
        """
        text_length = route_info['text_length']
        src_copy = route_info['src_tokens']            # [B, N_src, C]
        all_src_idx = route_info['src_idx']            # list of length B
        B, N_img, C = route_info['orig_shape']

        # Split text and processed tokens
        if text_length > 0:
            text_tokens = masked_x[:, :text_length]
            proc_tokens = masked_x[:, text_length:]
        else:
            text_tokens = None
            proc_tokens = masked_x

        # 🚀 OPTIMIZATION 1: Pre-allocate with proper device and dtype
        device, dtype = proc_tokens.device, proc_tokens.dtype
        restored = torch.zeros((B, N_img, C), device=device, dtype=dtype)

        # 🚀 OPTIMIZATION 2: Vectorized batch processing
        # Create batch masks for all source positions at once
        batch_src_masks = torch.zeros((B, N_img), dtype=torch.bool, device=device)
        
        # Set source positions to True for all batches
        for b in range(B):
            idx_b = all_src_idx[b]
            if idx_b.numel() > 0:
                batch_src_masks[b, idx_b] = True

        # 🚀 OPTIMIZATION 3: Vectorized keep indices computation
        batch_keep_masks = ~batch_src_masks  # [B, N_img]
        
        # Process each batch item with vectorized operations
        for b in range(B):
            keep_indices = batch_keep_masks[b].nonzero(as_tuple=True)[0]
            
            # Fill non-source slots with processed tokens
            if keep_indices.numel() > 0:
                num_keep = keep_indices.numel()
                restored[b, keep_indices] = proc_tokens[b, :num_keep]
            
            # 🚀 OPTIMIZATION 4: Vectorized source token restoration
            idx_b = all_src_idx[b]
            if idx_b.numel() > 0:
                # Determine how many source tokens we can restore
                max_src_tokens = min(idx_b.numel(), src_copy.shape[1])
                
                if max_src_tokens > 0:
                    # Vectorized scatter operation instead of Python loop
                    valid_src_indices = idx_b[:max_src_tokens]
                    restored[b, valid_src_indices] = src_copy[b, :max_src_tokens]

        # Reattach text tokens if needed
        if text_tokens is not None:
            return torch.cat([text_tokens, restored], dim=1)
        else:
            return restored

    @torch.no_grad()
    def end_route_inference(
        self,
        masked_x: torch.Tensor,
        route_info: dict,
        original_x=None,
        mask_token: float = 0.0
    ) -> torch.Tensor:
        """
        Vectorized inference-only restoration (no gradients) of merged tokens.
        """
        text_length = route_info['text_length']
        src_copy = route_info['src_tokens']
        all_src_idx = route_info['src_idx']
        B, N_img, C = route_info['orig_shape']

        if text_length > 0:
            text_tokens = masked_x[:, :text_length]
            proc_tokens = masked_x[:, text_length:]
        else:
            text_tokens = None
            proc_tokens = masked_x

        device, dtype = proc_tokens.device, proc_tokens.dtype
        restored = torch.zeros((B, N_img, C), device=device, dtype=dtype)

        batch_src_masks = torch.zeros((B, N_img), dtype=torch.bool, device=device)
        for b in range(B):
            idx_b = all_src_idx[b]
            if idx_b.numel() > 0:
                batch_src_masks[b, idx_b] = True

        batch_keep_masks = ~batch_src_masks
        for b in range(B):
            keep_indices = batch_keep_masks[b].nonzero(as_tuple=True)[0]
            if keep_indices.numel() > 0:
                num_keep = keep_indices.numel()
                restored[b, keep_indices] = proc_tokens[b, :num_keep]

            idx_b = all_src_idx[b]
            if idx_b.numel() > 0:
                max_src_tokens = min(idx_b.numel(), src_copy.shape[1])
                if max_src_tokens > 0:
                    valid_src_indices = idx_b[:max_src_tokens]
                    restored[b, valid_src_indices] = src_copy[b, :max_src_tokens]

        if text_tokens is not None:
            return torch.cat([text_tokens, restored], dim=1)
        else:
            return restored


class RestoreAdapter(nn.Module):
    """
    A tiny masked adapter that only updates tokens restored at end_route.
    """

    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 1, expansion * d_model),
            nn.GELU(),
            nn.Linear(expansion * d_model, d_model),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, restored_mask: torch.Tensor) -> torch.Tensor:
        gate = restored_mask.to(dtype=x.dtype).unsqueeze(-1)
        normed = self.norm(x)
        inp = torch.cat([normed, gate], dim=-1)
        delta = self.mlp(inp)
        return x + delta * gate

class RouterMixin(BaseMixin):
    def __init__(self, enable_routing=False, routes=None, height=None, width=None):
        super().__init__()
        self.enable_routing = enable_routing
        self.routes = routes if routes is not None else []
        self.height = height
        self.width = width
        self.restore_adapter: Optional[RestoreAdapter] = None
        if enable_routing:
            self.router = Router()

    @non_conflict
    def word_embedding_forward(self, input_ids, old_impl=None, **kwargs):
        if old_impl is not None:
            return old_impl(input_ids, **kwargs)
        return None

    @non_conflict
    def layer_forward(
        self,
        hidden_states,
        mask,
        *args,
        old_impl=None,
        **kwargs,
    ):
        transformer = self.transformer
        enable_routing = getattr(transformer, "enable_routing", False)
        routes = getattr(transformer, "routes", [])
        if not enable_routing or not routes:
            return old_impl(hidden_states, mask, *args, **kwargs)

        layer_id = kwargs.get("layer_id", 0)
        text_length = kwargs.get("text_length", 0)
        feature_tap = kwargs.get("feature_tap", None)

        if not hasattr(transformer, "current_route_idx"):
            transformer.current_route_idx = 0
        if not hasattr(transformer, "current_route_info"):
            transformer.current_route_info = None

        current_route_idx = getattr(transformer, "current_route_idx", 0)
        if current_route_idx >= len(routes):
            return old_impl(hidden_states, mask, *args, **kwargs)

        route = routes[current_route_idx]

        if layer_id == route["start_layer_idx"]:
            img_token_count = hidden_states.shape[1] - text_length
            n_per_frame = (self.height or 1) * (self.width or 1)
            F = int(img_token_count // n_per_frame) if n_per_frame > 0 else 0

            merged_states, route_info = self.router.tome_merge_and_route(
                hidden_states,
                selection_ratio=route["selection_ratio"],
                text_length=text_length,
                F=F,
            )
            transformer.current_route_info = route_info

            if feature_tap is not None:
                feature_tap.capture_layer_input(
                    layer_id,
                    hidden_states,
                    text_length,
                    keep_index=route_info.get("pos_index"),
                    total_img_tokens=route_info["orig_shape"][1],
                )

            kwargs_merged = dict(kwargs)
            kwargs_merged["pos_index_image"] = route_info.get("pos_index", None)
            hidden_states = old_impl(merged_states, mask, *args, **kwargs_merged)

        elif layer_id == route["end_layer_idx"]:
            route_info = getattr(transformer, "current_route_info", None)
            kwargs_reduced = dict(kwargs)
            if route_info is not None:
                kwargs_reduced["pos_index_image"] = route_info.get("pos_index", None)
            hidden_states = old_impl(hidden_states, mask, *args, **kwargs_reduced)

            if route_info is not None:
                if self.training:
                    hidden_states = self.router.end_route(hidden_states, route_info)
                else:
                    hidden_states = self.router.end_route_inference(hidden_states, route_info)

                B = hidden_states.size(0)
                N_img = route_info["orig_shape"][1]
                restored_mask_img = torch.zeros(B, N_img, dtype=torch.bool, device=hidden_states.device)
                for b, idx in enumerate(route_info["src_idx"]):
                    if idx.numel() > 0:
                        restored_mask_img[b, idx] = True

                if text_length > 0:
                    restored_mask = torch.cat(
                        [
                            torch.zeros(B, text_length, dtype=torch.bool, device=hidden_states.device),
                            restored_mask_img,
                        ],
                        dim=1,
                    )
                else:
                    restored_mask = restored_mask_img

                if self.restore_adapter is None:
                    d_model = hidden_states.size(-1)
                    self.restore_adapter = RestoreAdapter(d_model, expansion=2).to(hidden_states.device)
                    self.restore_adapter = self.restore_adapter.to(hidden_states.dtype)

                hidden_states = self.restore_adapter(hidden_states, restored_mask)

                restore_idx_tensor = None
                src_idx_list = route_info["src_idx"]
                if src_idx_list:
                    lengths = [idx.numel() for idx in src_idx_list]
                    if lengths and all(l == lengths[0] for l in lengths):
                        count = lengths[0]
                        if count >= 0:
                            restore_idx_tensor = torch.stack(
                                [idx.to(hidden_states.device) for idx in src_idx_list], dim=0
                            ) if count > 0 else torch.zeros(
                                len(src_idx_list), 0, device=hidden_states.device, dtype=torch.long
                            )

                if feature_tap is not None:
                    feature_tap.capture_layer_output(layer_id, hidden_states, restore_index=restore_idx_tensor)

                transformer.current_route_idx += 1
                transformer.current_route_info = None

        else:
            route_info = getattr(transformer, "current_route_info", None)
            kwargs_mid = dict(kwargs)
            if route_info is not None:
                kwargs_mid["pos_index_image"] = route_info.get("pos_index", None)
            hidden_states = old_impl(hidden_states, mask, *args, **kwargs_mid)

        return hidden_states

    def forward_with_routing(self, hidden_states, **kwargs):
        # Keep this method for backward compatibility, but it's not used anymore
        return None, None, None

    def reinit(self, parent_model=None):
        pass


class DiffusionTransformer(BaseModel):
    def __init__(
        self,
        transformer_args,
        num_frames,
        time_compressed_rate,
        latent_width,
        latent_height,
        patch_size,
        in_channels,
        out_channels,
        hidden_size,
        num_layers,
        num_attention_heads,
        elementwise_affine,
        time_embed_dim=None,
        num_classes=None,
        modules={},
        input_time="adaln",
        adm_in_channels=None,
        parallel_output=True,
        height_interpolation=1.0,
        width_interpolation=1.0,
        time_interpolation=1.0,
        use_SwiGLU=False,
        use_RMSNorm=False,
        zero_init_y_embed=False,
        enable_routing=False,
        routes=None,
        **kwargs,
    ):
        self.latent_width = latent_width
        self.latent_height = latent_height
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.time_compressed_rate = time_compressed_rate
        self.spatial_length = latent_width * latent_height // patch_size**2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.model_channels = hidden_size
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else hidden_size
        self.num_classes = num_classes
        self.adm_in_channels = adm_in_channels
        self.input_time = input_time
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.is_decoder = transformer_args.is_decoder
        self.elementwise_affine = elementwise_affine
        self.height_interpolation = height_interpolation
        self.width_interpolation = width_interpolation
        self.time_interpolation = time_interpolation
        self.inner_hidden_size = hidden_size * 4
        self.zero_init_y_embed = zero_init_y_embed
        self.enable_routing = enable_routing
        self.routes = routes if routes is not None else []
        
        try:
            self.dtype = str_to_dtype[kwargs.pop("dtype")]
        except:
            self.dtype = torch.float32

        if use_SwiGLU:
            kwargs["activation_func"] = F.silu
        elif "activation_func" not in kwargs:
            approx_gelu = nn.GELU(approximate="tanh")
            kwargs["activation_func"] = approx_gelu

        if use_RMSNorm:
            kwargs["layernorm"] = RMSNorm
        else:
            kwargs["layernorm"] = partial(LayerNorm, elementwise_affine=elementwise_affine, eps=1e-6)

        transformer_args.num_layers = num_layers
        transformer_args.hidden_size = hidden_size
        transformer_args.num_attention_heads = num_attention_heads
        transformer_args.parallel_output = parallel_output
        super().__init__(args=transformer_args, transformer=None, **kwargs)

        # Forward routing-related attributes to the transformer instance
        self.transformer.enable_routing = self.enable_routing
        self.transformer.routes = self.routes
        self.transformer.current_route_info = None
        self.transformer.current_route_idx = 0

        module_configs = modules
        self._build_modules(module_configs)

        if use_SwiGLU:
            self.add_mixin(
                "swiglu", SwiGLUMixin(num_layers, hidden_size, self.inner_hidden_size, bias=False), reinit=True
            )

    def _build_modules(self, module_configs):
        model_channels = self.hidden_size
        # time_embed_dim = model_channels * 4
        time_embed_dim = self.time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.Sequential(
                    Timestep(model_channels),
                    nn.Sequential(
                        linear(model_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    ),
                )
            elif self.num_classes == "sequential":
                assert self.adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(self.adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
                if self.zero_init_y_embed:
                    nn.init.constant_(self.label_emb[0][2].weight, 0)
                    nn.init.constant_(self.label_emb[0][2].bias, 0)
            else:
                raise ValueError()

        pos_embed_config = module_configs["pos_embed_config"]
        self.add_mixin(
            "pos_embed",
            instantiate_from_config(
                pos_embed_config,
                height=self.latent_height // self.patch_size,
                width=self.latent_width // self.patch_size,
                compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate + 1,
                hidden_size=self.hidden_size,
            ),
            reinit=True,
        )

        patch_embed_config = module_configs["patch_embed_config"]
        self.add_mixin(
            "patch_embed",
            instantiate_from_config(
                patch_embed_config,
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                in_channels=self.in_channels,
            ),
            reinit=True,
        )
        if self.input_time == "adaln":
            adaln_layer_config = module_configs["adaln_layer_config"]
            self.add_mixin(
                "adaln_layer",
                instantiate_from_config(
                    adaln_layer_config,
                    height=self.latent_height // self.patch_size,
                    width=self.latent_width // self.patch_size,
                    hidden_size=self.hidden_size,
                    num_layers=self.num_layers,
                    compressed_num_frames=(self.num_frames - 1) // self.time_compressed_rate + 1,
                    hidden_size_head=self.hidden_size // self.num_attention_heads,
                    time_embed_dim=self.time_embed_dim,
                    elementwise_affine=self.elementwise_affine,
                    enable_routing=self.enable_routing,
                    routes=self.routes,
                ),
            )
        else:
            raise NotImplementedError

        final_layer_config = module_configs["final_layer_config"]
        self.add_mixin(
            "final_layer",
            instantiate_from_config(
                final_layer_config,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
                out_channels=self.out_channels,
                time_embed_dim=self.time_embed_dim,
                latent_width=self.latent_width,
                latent_height=self.latent_height,
                elementwise_affine=self.elementwise_affine,
            ),
            reinit=True,
        )

        if "lora_config" in module_configs:
            lora_config = module_configs["lora_config"]
            self.add_mixin("lora", instantiate_from_config(lora_config, layer_num=self.num_layers), reinit=True)

        # Add router mixin if routing is enabled
        if self.enable_routing:
            self.add_mixin(
                "router",
                RouterMixin(
                    enable_routing=self.enable_routing,
                    routes=self.routes,
                    height=self.latent_height // self.patch_size,
                    width=self.latent_width // self.patch_size,
                ),
                reinit=True,
            )

        # 注入 Router-LoRA（构造阶段，可选）
        if "router_lora_config" in module_configs and "router_lora" not in self.mixins:
            router_cfg = module_configs["router_lora_config"]
            self.add_mixin(
                "router_lora",
                instantiate_from_config(router_cfg, layer_num=self.num_layers),
                reinit=True,
            )

        self._module_cfgs = module_configs   # 记一份，后面用
        return
    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        # Reset routing state for each forward pass
        if self.enable_routing:
            self.transformer.current_route_idx = 0
            self.transformer.current_route_info = None
            
        # print('x shape:', x.shape)  # train phase: torch.Size([2, 8, 32, 60, 90]) 
        b, t, d, h, w = x.shape
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False, dtype=self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            # assert y.shape[0] == x.shape[0]
            assert x.shape[0] % y.shape[0] == 0
            y = y.repeat_interleave(x.shape[0] // y.shape[0], dim=0)
            emb = emb + self.label_emb(y)

        kwargs["seq_length"] = t * h * w // (self.patch_size**2)
        kwargs["images"] = x
        kwargs["emb"] = emb
        kwargs["encoder_outputs"] = context
        kwargs["text_length"] = context.shape[1]

        kwargs["input_ids"] = kwargs["position_ids"] = kwargs["attention_mask"] = torch.ones((1, 1)).to(x.dtype)
        output = super().forward(**kwargs)[0]

        return output
