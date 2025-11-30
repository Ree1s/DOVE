from pathlib import Path
import argparse
import logging
import time
import math

import torch
from torchvision import transforms
from torchvision.io import write_video
from tqdm import tqdm

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    AutoencoderKLCogVideoX,
)

from transformers import AutoTokenizer, T5EncoderModel

from finetune.models.dove.cogvideox_transformer3d_router import (
    TokenMergeCogVideoXTransformer3DModel,
)

from transformers import set_seed
from typing import Dict, Tuple
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from safetensors.torch import load_file

import json
import os
import cv2
from PIL import Image

from pathlib import Path
import pyiqa
import imageio.v3 as iio
import glob
import numpy as np

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logging.basicConfig(level=logging.INFO)

# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
fr_metrics = ['psnr', 'ssim', 'lpips', 'dists']


def no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper


def _load_transformer_state_dict(weights_root: Path) -> Dict[str, torch.Tensor]:
    transformer_dir = Path(weights_root)
    if (transformer_dir / "transformer").is_dir():
        transformer_dir = transformer_dir / "transformer"

    index_file = transformer_dir / "diffusion_pytorch_model.safetensors.index.json"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        shard_cache: Dict[Path, Dict[str, torch.Tensor]] = {}
        state_dict: Dict[str, torch.Tensor] = {}
        for weight_name, shard_name in index_data["weight_map"].items():
            shard_path = transformer_dir / shard_name
            if shard_path not in shard_cache:
                shard_cache[shard_path] = load_file(shard_path)
            state_dict[weight_name] = shard_cache[shard_path][weight_name]
        return state_dict

    safetensor_file = transformer_dir / "diffusion_pytorch_model.safetensors"
    if safetensor_file.exists():
        return load_file(safetensor_file)

    bin_file = transformer_dir / "pytorch_model.bin"
    if bin_file.exists():
        return torch.load(bin_file, map_location="cpu")

    raise FileNotFoundError(f"No transformer weights found under {transformer_dir}")


def is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in video_exts)


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(to_tensor(Image.fromarray(rgb)))
    cap.release()
    return torch.stack(frames)


def read_image_folder(folder_path):
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    frames = [to_tensor(Image.open(p).convert("RGB")) for p in image_files]
    return torch.stack(frames)


def load_sequence(path):
    # return a tensor of shape [F, C, H, W] // 0, 1
    if os.path.isdir(path):
        return read_image_folder(path)
    elif os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Treat image as a single-frame video
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)  # [1, C, H, W]
    raise ValueError(f"Unsupported input: {path}")

@no_grad
def compute_metrics(pred_frames, gt_frames, metrics_model, metric_accumulator, file_name):

    print(f"\n\n[{file_name}] Metrics:", end=" ")
    for name, model in metrics_model.items():
        scores = []
        for i in range(pred_frames.shape[0]):
            pred = pred_frames[i].unsqueeze(0)
            if gt_frames != None:
                gt = gt_frames[i].unsqueeze(0)
            if name in fr_metrics:
                score = model(pred, gt).item()
            else:
                score = model(pred).item()
            scores.append(score)
        val = sum(scores) / len(scores)
        metric_accumulator[name].append(val)
        print(f"{name.upper()}={val:.4f}", end="  ")
    print()


def save_frames_as_png(video, output_dir, fps=8):
    """
    Save video frames as PNG sequence.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_dir (str): directory to save PNG files
        fps (int): kept for API compatibility
    """
    video = video[0]  # Remove batch dimension
    video = video.permute(1, 2, 3, 0)  # [F, H, W, C]

    os.makedirs(output_dir, exist_ok=True)


def _parse_metric_layer_indices(spec: str | None):
    if not spec:
        return None
    indices = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.add(int(part))
        except ValueError:
            logging.warning("Skipping invalid metric head layer index: %s", part)
    return indices or None


def _parse_metric_layout(spec: str | None):
    if not spec:
        return None
    parts = [p.strip() for p in spec.replace("x", ",").split(",")]
    parts = [p for p in parts if p]
    if len(parts) != 3:
        logging.warning("Invalid metric_head_layout '%s', expected format F,H,W", spec)
        return None
    try:
        frames, height, width = [int(p) for p in parts]
        if frames <= 0 or height <= 0 or width <= 0:
            raise ValueError
    except ValueError:
        logging.warning("Invalid metric_head_layout '%s', values must be positive integers", spec)
        return None
    return (frames, height, width)


def _render_metric_heatmap(values: np.ndarray, output_resolution: int = 512) -> Image.Image:
    if values.ndim == 1:
        normalized = values - values.min()
        max_val = normalized.max()
        if max_val > 0:
            normalized = normalized / max_val
        width = int(math.ceil(math.sqrt(normalized.size)))
        height = int(math.ceil(normalized.size / width))
        grid = np.zeros((height * width,), dtype=np.float32)
        grid[: normalized.size] = normalized
        grid = grid.reshape(height, width)
    else:
        grid = values.astype(np.float32)
        grid = grid - grid.min()
        max_val = grid.max()
        if max_val > 0:
            grid = grid / max_val
    grid = (grid * 255.0).clip(0, 255).astype(np.uint8)
    heatmap = Image.fromarray(grid, mode="L")
    if output_resolution and (heatmap.size[0] != output_resolution or heatmap.size[1] != output_resolution):
        heatmap = heatmap.resize((output_resolution, output_resolution), Image.BILINEAR)
    return heatmap


def _tensor_to_image(t: torch.Tensor) -> Image.Image:
    """Convert [C,H,W] or [H,W] tensor in [0,1] to PIL Image."""
    if t.dim() == 3:
        t = t.clamp(0, 1)
        arr = (t * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr)
    t = t.clamp(0, 1)
    arr = (t * 255.0).byte().cpu().numpy()
    return Image.fromarray(arr)


def save_metric_head_records(
    records,
    sample_name: str,
    output_dir: Path,
    layout: Tuple[int, int, int] | None = None,
    lq_frame: torch.Tensor | None = None,
):
    if not records:
        logging.warning("No metric head activations captured for %s", sample_name)
        return
    sample_dir = Path(output_dir) / Path(sample_name).stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for layer_idx, tensors in records.items():
        metadata[str(layer_idx)] = []
        for call_idx, tensor in enumerate(tensors):
            tensor = tensor.float()
            # keep only first batch to avoid large dumps during tiled processing
            if tensor.dim() == 3:
                tensor = tensor[0]  # [tokens, 1] or [tokens]
            values = tensor.reshape(-1).cpu().numpy()
            base_name = sample_dir / f"layer{layer_idx:02d}_call{call_idx:02d}"
            np.save(f"{base_name}.npy", values)
            entry = {
                "call_index": call_idx,
                "num_tokens": int(values.size),
                "min": float(values.min()) if values.size else 0.0,
                "max": float(values.max()) if values.size else 0.0,
                "file_prefix": base_name.name,
            }
            if layout and values.size == layout[0] * layout[1] * layout[2]:
                frames, height, width = layout
                volume = values.reshape(frames, height, width)
                entry["layout"] = {"frames": frames, "height": height, "width": width}
                for frame_idx in range(frames):
                    frame_vals = volume[frame_idx]
                    heatmap = _render_metric_heatmap(frame_vals)
                    if lq_frame is not None:
                        # assume lq_frame shape [C,H,W] or [F,C,H,W]; pick matching frame if available
                        if lq_frame.dim() == 4 and frame_idx < lq_frame.shape[0]:
                            lq_img = _tensor_to_image(lq_frame[frame_idx])
                        elif lq_frame.dim() == 3:
                            lq_img = _tensor_to_image(lq_frame)
                        else:
                            lq_img = None
                        if lq_img is not None:
                            # resize lq to heatmap size, then concat horizontally
                            lq_img = lq_img.resize(heatmap.size, Image.BILINEAR)
                            combined = Image.new("RGB", (heatmap.width * 2, heatmap.height))
                            combined.paste(lq_img.convert("RGB"), (0, 0))
                            combined.paste(heatmap.convert("RGB"), (heatmap.width, 0))
                            combined.save(f"{base_name}_frame{frame_idx:02d}_heatmap.png")
                        else:
                            heatmap.save(f"{base_name}_frame{frame_idx:02d}_heatmap.png")
                    else:
                        heatmap.save(f"{base_name}_frame{frame_idx:02d}_heatmap.png")
            else:
                heatmap = _render_metric_heatmap(values)
                heatmap.save(f"{base_name}_heatmap.png")
            metadata[str(layer_idx)].append(entry)
    meta_path = sample_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_video_with_imageio_lossless(video, output_path, fps=8):
    """
    Save a video tensor to .mkv using imageio.v3.imwrite with ffmpeg backend.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_path (str): where to save the .mkv file
        fps (int): frames per second
    """
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='rgb24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '0'],
    )


def save_video_with_imageio(video, output_path, fps=8, format='yuv444p'):
    """
    Save a video tensor to .mp4 using imageio.v3.imwrite with ffmpeg backend.

    Args:
        video (torch.Tensor): shape [B, C, F, H, W], float in [0, 1]
        output_path (str): where to save the .mp4 file
        fps (int): frames per second
    """
    video = video[0]
    video = video.permute(1, 2, 3, 0)

    frames = (video * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    if format == 'yuv444p':
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv444p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '0'],
        )
    else:
        iio.imwrite(
            output_path,
            frames,
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            macro_block_size=None,
            ffmpeg_params=['-crf', '10'],
        )


def preprocess_video_match(
    video_path: Path | str,
    is_match: bool = False,
) -> torch.Tensor:
    """
    Loads a single video.

    Args:
        video_path: Path to the video file.
    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    video_reader = decord.VideoReader(uri=video_path.as_posix())
    video_num_frames = len(video_reader)
    frames = video_reader.get_batch(list(range(video_num_frames)))
    F, H, W, C = frames.shape
    original_shape = (F, H, W, C)
    
    pad_f = 0
    pad_h = 0
    pad_w = 0

    if is_match:
        remainder = (F - 1) % 8
        if remainder != 0:
            last_frame = frames[-1:]
            pad_f = 8 - remainder
            repeated_frames = last_frame.repeat(pad_f, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)

        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            # pad = (w_left, w_right, h_top, h_bottom)
            frames = torch.nn.functional.pad(frames, pad=(0, 0, 0, pad_w, 0, pad_h))  # pad right and bottom

    # to F, C, H, W
    return frames.float().permute(0, 3, 1, 2).contiguous(), pad_f, pad_h, pad_w, original_shape


def remove_padding_and_extra_frames(video, pad_F, pad_H, pad_W):
    if pad_F > 0:
        video = video[:, :, :-pad_F, :, :]
    if pad_H > 0:
        video = video[:, :, :, :-pad_H, :]
    if pad_W > 0:
        video = video[:, :, :, :, :-pad_W]
    
    return video


def make_temporal_chunks(F, chunk_len, overlap_t=8):
    """
    Args:
        F: total number of frames
        chunk_len: int, chunk length in time (excluding overlap)
        overlap: int, number of overlapping frames between chunks
    Returns:
        time_chunks: List of (start_t, end_t) tuples
    """
    if chunk_len == 0:
        return [(0, F)]

    effective_stride = chunk_len - overlap_t
    if effective_stride <= 0:
        raise ValueError("chunk_len must be greater than overlap")

    chunk_starts = list(range(0, F - overlap_t, effective_stride))
    if chunk_starts[-1] + chunk_len < F:
        chunk_starts.append(F - chunk_len)

    time_chunks = []
    for i, t_start in enumerate(chunk_starts):
        t_end = min(t_start + chunk_len, F)
        time_chunks.append((t_start, t_end))

    if len(time_chunks) >= 2 and time_chunks[-1][1] - time_chunks[-1][0] < chunk_len:
        last = time_chunks.pop()
        prev_start, _ = time_chunks[-1]
        time_chunks[-1] = (prev_start, last[1])

    return time_chunks


def make_spatial_tiles(H, W, tile_size_hw, overlap_hw=(32, 32)):
    """
    Args:
        H, W: height and width of the frame
        tile_size_hw: Tuple (tile_height, tile_width)
        overlap_hw: Tuple (overlap_height, overlap_width)
    Returns:
        spatial_tiles: List of (start_h, end_h, start_w, end_w) tuples
    """
    tile_height, tile_width = tile_size_hw
    overlap_h, overlap_w = overlap_hw

    if tile_height == 0 or tile_width == 0:
        return [(0, H, 0, W)]

    tile_stride_h = tile_height - overlap_h
    tile_stride_w = tile_width - overlap_w

    if tile_stride_h <= 0 or tile_stride_w <= 0:
        raise ValueError("Tile size must be greater than overlap")

    h_tiles = list(range(0, H - overlap_h, tile_stride_h))
    if not h_tiles or h_tiles[-1] + tile_height < H:
        h_tiles.append(H - tile_height)
    
     # Merge last row if needed
    if len(h_tiles) >= 2 and h_tiles[-1] + tile_height > H:
        h_tiles.pop()

    w_tiles = list(range(0, W - overlap_w, tile_stride_w))
    if not w_tiles or w_tiles[-1] + tile_width < W:
        w_tiles.append(W - tile_width)
    
    # Merge last column if needed
    if len(w_tiles) >= 2 and w_tiles[-1] + tile_width > W:
        w_tiles.pop()

    spatial_tiles = []
    for h_start in h_tiles:
        h_end = min(h_start + tile_height, H)
        if h_end + tile_stride_h > H:
            h_end = H
        for w_start in w_tiles:
            w_end = min(w_start + tile_width, W)
            if w_end + tile_stride_w > W:
                w_end = W
            spatial_tiles.append((h_start, h_end, w_start, w_end))
    return spatial_tiles


def get_valid_tile_region(t_start, t_end, h_start, h_end, w_start, w_end,
                          video_shape, overlap_t, overlap_h, overlap_w):
    _, _, F, H, W = video_shape

    t_len = t_end - t_start
    h_len = h_end - h_start
    w_len = w_end - w_start

    valid_t_start = 0 if t_start == 0 else overlap_t // 2
    valid_t_end = t_len if t_end == F else t_len - overlap_t // 2
    valid_h_start = 0 if h_start == 0 else overlap_h // 2
    valid_h_end = h_len if h_end == H else h_len - overlap_h // 2
    valid_w_start = 0 if w_start == 0 else overlap_w // 2
    valid_w_end = w_len if w_end == W else w_len - overlap_w // 2

    out_t_start = t_start + valid_t_start
    out_t_end = t_start + valid_t_end
    out_h_start = h_start + valid_h_start
    out_h_end = h_start + valid_h_end
    out_w_start = w_start + valid_w_start
    out_w_end = w_start + valid_w_end

    return {
        "valid_t_start": valid_t_start, "valid_t_end": valid_t_end,
        "valid_h_start": valid_h_start, "valid_h_end": valid_h_end,
        "valid_w_start": valid_w_start, "valid_w_end": valid_w_end,
        "out_t_start": out_t_start, "out_t_end": out_t_end,
        "out_h_start": out_h_start, "out_h_end": out_h_end,
        "out_w_start": out_w_start, "out_w_end": out_w_end,
    }


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    transformer_config: Dict,
    vae_scale_factor_spatial: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:

    grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

    if transformer_config.patch_size_t is None:
        base_num_frames = num_frames
    else:
        base_num_frames = (
            num_frames + transformer_config.patch_size_t - 1
        ) // transformer_config.patch_size_t
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=None,
        grid_size=(grid_height, grid_width),
        temporal_size=base_num_frames,
        grid_type="slice",
        max_size=(grid_height, grid_width),
        device=device,
    )

    return freqs_cos, freqs_sin
    
@no_grad
def process_video(
    pipe: CogVideoXPipeline,
    video: torch.Tensor,
    prompt: str = '',
    noise_step: int = 0,
    sr_noise_step: int = 399,
    empty_prompt_embedding: torch.Tensor = None,
):
    # SR the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.

    # Decode video
    video = video.to(pipe.vae.device, dtype=pipe.vae.dtype)
    latent_dist = pipe.vae.encode(video).latent_dist
    latent = latent_dist.sample() * pipe.vae.config.scaling_factor

    patch_size_t = pipe.transformer.config.patch_size_t
    if patch_size_t is not None:
        ncopy = latent.shape[2] % patch_size_t
        # Copy the first frame ncopy times to match patch_size_t
        first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
        latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)

        assert latent.shape[2] % patch_size_t == 0

    batch_size, num_channels, num_frames, height, width = latent.shape

    # Get prompt embeddings
    if prompt == "" and empty_prompt_embedding is not None:
        # Use pre-loaded empty prompt embedding
        prompt_embedding = empty_prompt_embedding.to(latent.device, dtype=latent.dtype)
        # Expand to match batch size if needed
        if prompt_embedding.shape[0] != batch_size:
            prompt_embedding = prompt_embedding.repeat(batch_size, 1, 1)
    else:
        # Encode the prompt
        prompt_token_ids = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = pipe.text_encoder(
            prompt_token_ids.to(latent.device)
        )[0]
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

    latent = latent.permute(0, 2, 1, 3, 4)

    # Add noise to latent (Select)
    if noise_step != 0:
        noise = torch.randn_like(latent)
        add_timesteps = torch.full(
            (batch_size,),
            fill_value=noise_step,
            dtype=torch.long,
            device=latent.device,
        )
        latent = pipe.scheduler.add_noise(latent, noise, add_timesteps)
    
    timesteps = torch.full(
        (batch_size,),
        fill_value=sr_noise_step,
        dtype=torch.long,
        device=latent.device,
    )

    # Prepare rotary embeds
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    transformer_config = pipe.transformer.config
    rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=transformer_config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=latent.device,
        )
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # Predict noise
    predicted_noise = forward_transformer_with_profile(
        latent,
        prompt_embedding,
        timesteps,
        rotary_emb,
    )
    
    latent_generate = pipe.scheduler.get_velocity(
        predicted_noise, latent, timesteps
    )

    # generate video
    if patch_size_t is not None and ncopy > 0:
        latent_generate = latent_generate[:, ncopy:, :, :, :]

    # [B, C, F, H, W]
    video_generate = pipe.decode_latents(latent_generate)
    video_generate = (video_generate * 0.5 + 0.5).clamp(0.0, 1.0)
    
    return video_generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSR using DOVE")

    parser.add_argument("--input_dir", type=str)

    parser.add_argument("--input_json", type=str, default=None)

    parser.add_argument("--gt_dir", type=str, default=None)

    parser.add_argument("--eval_metrics", type=str, default='') # 'psnr,ssim,lpips,dists,clipiqa,musiq,maniqa,niqe'

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")

    parser.add_argument("--output_path", type=str, default="./results", help="The path save generated video")

    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")

    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")

    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")

    parser.add_argument("--upscale_mode", type=str, default="bilinear")

    parser.add_argument("--upscale", type=int, default=4)

    parser.add_argument("--noise_step", type=int, default=0)

    parser.add_argument("--sr_noise_step", type=int, default=399)

    parser.add_argument("--token_merge_routes", type=str, default=None,
                    help="Token merge routes, e.g. '10-17@0.36;28-35@0.36'.")
    parser.add_argument("--token_merge_default_ratio", type=float, default=0.0)
    parser.add_argument("--token_merge_seed", type=int, default=42)
    parser.add_argument("--token_merge_restore_adapter_expansion", type=int, default=2)
    parser.add_argument("--token_merge_window_size", type=int, default=0)
    parser.add_argument("--token_merge_window_stride", type=int, default=1)
    parser.add_argument("--token_merge_ratio_start", type=float, default=None)
    parser.add_argument("--token_merge_ratio_warmup_steps", type=int, default=0)
    parser.add_argument("--token_merge_ratio_schedule", type=str, default="linear")
    parser.add_argument("--token_merge_use_psg_importance", action="store_true")
    parser.add_argument("--token_merge_layer_gate_group_size", type=int, default=0)
    parser.add_argument("--token_merge_layer_gate_keep_per_group", type=int, default=0)
    parser.add_argument("--token_merge_layer_gate_tau", type=float, default=1.0)
    parser.add_argument("--token_merge_layer_gate_logit_scale", type=float, default=1.0)
    parser.add_argument("--visualize_metric_heads", action="store_true", help="Save importance maps from token-merge metric heads.")
    parser.add_argument("--metric_head_layers", type=str, default=None, help="Comma-separated list of transformer layer indices to visualize.")
    parser.add_argument("--metric_head_output_dir", type=str, default=None, help="Directory to store metric-head visualizations (defaults to output_path/metric_heads).")
    parser.add_argument("--metric_head_layout", type=str, default=None, help="Optional frames,height,width layout to reshape metric vectors (e.g., '2,80,120').")

    parser.add_argument("--profile_transformer", action="store_true",
                    help="Record latency (and FLOPs for the first call if torch.profiler is available) of transformer forward pass")

    parser.add_argument("--is_cpu_offload", action="store_true", help="Enable CPU offload for the model")

    parser.add_argument("--is_vae_st", action="store_true", help="Enable VAE slicing and tiling")

    parser.add_argument("--png_save", action="store_true", help="Save output as PNG sequence")

    parser.add_argument("--save_format", type=str, default="yuv444p", help="Save output as PNG sequence")

    # Crop and Tiling Parameters
    parser.add_argument("--tile_size_hw", type=int, nargs=2, default=(0, 0), help="Tile size for spatial tiling (height, width)")

    parser.add_argument("--overlap_hw", type=int, nargs=2, default=(32, 32))

    parser.add_argument("--chunk_len", type=int, default=0, help="Chunk length for temporal chunking")

    parser.add_argument("--overlap_t", type=int, default=8)

    args = parser.parse_args()

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError("Invalid dtype. Choose from 'float16', 'bfloat16', or 'float32'.")
    
    if args.chunk_len > 0:
        print(f"Chunking video into {args.chunk_len} frames with {args.overlap_t} overlap")
        overlap_t = args.overlap_t
    else:
        overlap_t = 0
    if args.tile_size_hw != (0, 0):
        print(f"Tiling video into {args.tile_size_hw} frames with {args.overlap_hw} overlap")
        overlap_hw = args.overlap_hw
    else:
        overlap_hw = (0, 0)

    # Set seed
    set_seed(args.seed)

    try:
        import torch.profiler as torch_profiler
    except (ImportError, ModuleNotFoundError):
        torch_profiler = None

    profile_state = {
        "times_ms": [],
        "flops": None,
        "profiled": False,
    }

    def forward_transformer_with_profile(hidden_states, encoder_hidden_states, timestep, image_rotary_emb):
        if hidden_states.device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        result = None
        if args.profile_transformer and not profile_state["profiled"] and torch_profiler is not None:
            try:
                with torch_profiler.profile(
                    activities=[
                        torch_profiler.ProfilerActivity.CPU,
                        torch_profiler.ProfilerActivity.CUDA
                        if hidden_states.device.type == "cuda"
                        else torch_profiler.ProfilerActivity.CPU,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_flops=True,
                    with_modules=False,
                ) as prof:
                    result = pipe.transformer(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        return_dict=False,
                    )[0]
                profile_state["flops"] = prof.key_averages().total_average().flops
                profile_state["profiled"] = True
            except Exception as profiling_error:
                logging.warning("Transformer profiling failed: %s", profiling_error)
        if result is None:
            result = pipe.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

        if hidden_states.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1e3
        if args.profile_transformer:
            profile_state["times_ms"].append(elapsed_ms)
        return result

    # Load empty prompt embedding if exists
    empty_prompt_embedding = None
    empty_prompt_path = Path("pretrained_models/prompt_embeddings/e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.safetensors")
    if empty_prompt_path.exists():
        try:
            empty_prompt_embedding = load_file(str(empty_prompt_path))["prompt_embedding"]
            print(f"Loaded empty prompt embedding from {empty_prompt_path}")
        except Exception as e:
            print(f"Warning: Failed to load empty prompt embedding: {e}")
            empty_prompt_embedding = None
    else:
        print(f"Empty prompt embedding not found at {empty_prompt_path}")

    if args.input_json is not None:
        with open(args.input_json, 'r') as f:
            video_prompt_dict = json.load(f)
    else:
        video_prompt_dict = {}
    
    # Get all video files from input directory
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(args.input_dir, f'*{ext}')))
    video_files = sorted(video_files)  # Sort files for consistent ordering

    if not video_files:
        raise ValueError(f"No video files found in {args.input_dir}")
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.model_path, subfolder="text_encoder")
    vae = AutoencoderKLCogVideoX.from_pretrained(args.model_path, subfolder="vae", torch_dtype=dtype)

    transformer_config = TokenMergeCogVideoXTransformer3DModel.load_config(
        args.model_path, subfolder="transformer"
    )
    transformer = TokenMergeCogVideoXTransformer3DModel.from_config(transformer_config)
    state_dict = _load_transformer_state_dict(Path(args.model_path))
    cfg = transformer.config
    routes_spec = args.token_merge_routes or getattr(cfg, "token_merge_routes", None)
    if routes_spec is None:
        cfg_path = Path(args.model_path) / "token_merge_config.json"
        if cfg_path.exists():
            try:
                cfg_data = json.loads(cfg_path.read_text())
                routes_spec = cfg_data.get("token_merge_routes", None)
                args.token_merge_default_ratio = cfg_data.get("token_merge_default_ratio", args.token_merge_default_ratio)
                args.token_merge_seed = cfg_data.get("token_merge_seed", args.token_merge_seed)
                args.token_merge_restore_adapter_expansion = cfg_data.get("token_merge_restore_adapter_expansion", args.token_merge_restore_adapter_expansion)
                args.token_merge_window_size = cfg_data.get("token_merge_window_size", args.token_merge_window_size)
                args.token_merge_window_stride = cfg_data.get("token_merge_window_stride", args.token_merge_window_stride)
                args.token_merge_ratio_start = cfg_data.get("token_merge_ratio_start", args.token_merge_ratio_start)
                args.token_merge_ratio_warmup_steps = cfg_data.get("token_merge_ratio_warmup_steps", args.token_merge_ratio_warmup_steps)
                args.token_merge_ratio_schedule = cfg_data.get("token_merge_ratio_schedule", args.token_merge_ratio_schedule)
                args.token_merge_use_psg_importance = cfg_data.get("token_merge_use_psg_importance", args.token_merge_use_psg_importance)
                args.token_merge_layer_gate_group_size = cfg_data.get(
                    "token_merge_layer_gate_group_size", args.token_merge_layer_gate_group_size
                )
                args.token_merge_layer_gate_keep_per_group = cfg_data.get(
                    "token_merge_layer_gate_keep_per_group", args.token_merge_layer_gate_keep_per_group
                )
                args.token_merge_layer_gate_tau = cfg_data.get(
                    "token_merge_layer_gate_tau", args.token_merge_layer_gate_tau
                )
                args.token_merge_layer_gate_logit_scale = cfg_data.get(
                    "token_merge_layer_gate_logit_scale", args.token_merge_layer_gate_logit_scale
                )
            except Exception as err:
                logging.warning("Failed to parse token_merge_config.json: %s", err)

    transformer.configure_token_merge(
        enable_token_merge=True if routes_spec else False,
        routes_spec=routes_spec,
        default_ratio=args.token_merge_default_ratio,
        seed=args.token_merge_seed,
        restore_adapter_expansion=args.token_merge_restore_adapter_expansion,
        window_size=args.token_merge_window_size,
        window_stride=args.token_merge_window_stride,
        ratio_start=args.token_merge_ratio_start,
        ratio_warmup_steps=args.token_merge_ratio_warmup_steps,
        ratio_schedule=args.token_merge_ratio_schedule,
        use_psg_importance=args.token_merge_use_psg_importance,
        layer_gate_group_size=args.token_merge_layer_gate_group_size if hasattr(args, "token_merge_layer_gate_group_size") else 0,
        layer_gate_keep_per_group=args.token_merge_layer_gate_keep_per_group if hasattr(args, "token_merge_layer_gate_keep_per_group") else 0,
        layer_gate_tau=args.token_merge_layer_gate_tau if hasattr(args, "token_merge_layer_gate_tau") else 1.0,
        layer_gate_logit_scale=args.token_merge_layer_gate_logit_scale if hasattr(args, "token_merge_layer_gate_logit_scale") else 1.0,
    )
    load_info = transformer.load_state_dict(state_dict, strict=False)
    if load_info.missing_keys:
        logging.warning("Missing token-merge keys when reloading transformer: %s", load_info.missing_keys)
    if load_info.unexpected_keys:
        logging.warning("Unexpected token-merge keys when reloading transformer: %s", load_info.unexpected_keys)
    transformer.to(dtype=dtype)
    # logging.info(
    #     "restore_adapter.mlp[0].weight sample (first 5 elems): %s",
    #     transformer.restore_adapter.mlp[0].weight.flatten()[:5],
    # )
    # logging.info(
    #     "restore_adapter.mlp[2].weight sample (first 5 elems): %s",
    #     transformer.restore_adapter.mlp[2].weight.flatten()[:5],
    # )
    # logging.info(
    #     "restore_adapter.norm.weight sample (first 5 elems): %s",
    #     transformer.restore_adapter.norm.weight.flatten()[:5],
    # )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")

    pipe = CogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    try:
        config = CogVideoXPipeline.load_config(args.model_path)
        pipe.register_to_config(**config)
    except Exception:
        pass

    metric_viz = None
    if args.visualize_metric_heads and getattr(transformer, "metric_heads", None) is not None:
        metric_layers = _parse_metric_layer_indices(args.metric_head_layers)
        metric_output_dir = Path(
            args.metric_head_output_dir or (Path(args.output_path) / "metric_heads")
        )
        metric_output_dir.mkdir(parents=True, exist_ok=True)
        metric_layout = _parse_metric_layout(args.metric_head_layout)
        metric_viz = {
            "layers": metric_layers,
            "output_dir": metric_output_dir,
            "buffer": {},
            "enabled": False,
            "hooks": [],
            "layout": metric_layout,
            "current_lq": None,
        }

        def make_metric_hook(layer_idx):
            def _hook(module, inputs, output):
                if not metric_viz["enabled"]:
                    return
                if metric_viz["layers"] is not None and layer_idx not in metric_viz["layers"]:
                    return
                metric_viz["buffer"].setdefault(layer_idx, []).append(output.detach().cpu())
            return _hook

        for idx, head in enumerate(transformer.metric_heads):
            if metric_layers is not None and idx not in metric_layers:
                continue
            metric_viz["hooks"].append(head.register_forward_hook(make_metric_hook(idx)))
    elif args.visualize_metric_heads:
        logging.warning("Metric head visualization enabled, but transformer has no metric heads.")

    # If you're using with lora, add this code
    if args.lora_path:
        print(f"Loading LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(
            args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1.0) # lora_scale = lora_alpha / rank

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    if args.is_cpu_offload:
        # pipe.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    
    if args.is_vae_st:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    
    # pipe.transformer.eval()
    # torch.set_grad_enabled(False)

    # 4. Set the metircs
    if args.eval_metrics != '':
        metrics_list = [m.strip().lower() for m in args.eval_metrics.split(',')]
        metrics_models = {}
        for name in metrics_list:
            try:
                metrics_models[name] = pyiqa.create_metric(name).to(pipe.device).eval()
            except Exception as e:
                print(f"Failed to initialize metric '{name}': {e}")
        metric_accumulator = {name: [] for name in metrics_list}
    else:
        metrics_models = None
        metric_accumulator = None
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.basename(video_path)
        prompt = video_prompt_dict.get(video_name, "")
        if os.path.exists(video_path):
            # Read video
            # [F, C, H, W]
            video, pad_f, pad_h, pad_w, original_shape = preprocess_video_match(video_path, is_match=True)
            H_, W_ = video.shape[2], video.shape[3]
            video = torch.nn.functional.interpolate(video, size=(H_*args.upscale, W_*args.upscale), mode=args.upscale_mode, align_corners=False)
            __frame_transform = transforms.Compose(
                [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)] # -1, 1
            )
            video = torch.stack([__frame_transform(f) for f in video], dim=0)
            video = video.unsqueeze(0)
            # [B, C, F, H, W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            if metric_viz:
                # store the LQ chunk in [F, C, H, W] scaled to [0,1] for visualization
                lq_chunk = video.clone()
                lq_chunk = lq_chunk[0]  # [F, C, H, W]
                lq_chunk = (lq_chunk * 0.5 + 0.5).clamp(0, 1)
                metric_viz["current_lq"] = lq_chunk

            _B, _C, _F, _H, _W = video.shape
            time_chunks = make_temporal_chunks(_F, args.chunk_len, overlap_t)
            spatial_tiles = make_spatial_tiles(_H, _W, args.tile_size_hw, overlap_hw)

            output_video = torch.zeros_like(video)
            write_count = torch.zeros_like(video, dtype=torch.int)

            if metric_viz:
                metric_viz["buffer"] = {}
                metric_viz["enabled"] = True

            print(f"Process video: {video_name} | Prompt: {prompt} | Frame: {_F} (ori: {original_shape[0]}; pad: {pad_f}) | Target Resolution: {_H}, {_W} (ori: {original_shape[1]*args.upscale}, {original_shape[2]*args.upscale}; pad: {pad_h}, {pad_w}) | Chunk Num: {len(time_chunks)*len(spatial_tiles)}")

            for t_start, t_end in time_chunks:
                for h_start, h_end, w_start, w_end in spatial_tiles:
                    video_chunk = video[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                    # print(f"video_chunk: {video_chunk.shape} | t: {t_start}:{t_end} | h: {h_start}:{h_end} | w: {w_start}:{w_end}")

                    # [B, C, F, H, W]
                    _video_generate = process_video(
                        pipe=pipe,
                        video=video_chunk,
                        prompt=prompt,
                        noise_step=args.noise_step,
                        sr_noise_step=args.sr_noise_step,
                        empty_prompt_embedding=empty_prompt_embedding,
                    )

                    region = get_valid_tile_region(
                        t_start, t_end, h_start, h_end, w_start, w_end,
                        video_shape=video.shape,
                        overlap_t=overlap_t,
                        overlap_h=overlap_hw[0],
                        overlap_w=overlap_hw[1],
                    )
                    output_video[:, :, region["out_t_start"]:region["out_t_end"],
                                    region["out_h_start"]:region["out_h_end"],
                                    region["out_w_start"]:region["out_w_end"]] = \
                    _video_generate[:, :, region["valid_t_start"]:region["valid_t_end"],
                                    region["valid_h_start"]:region["valid_h_end"],
                                    region["valid_w_start"]:region["valid_w_end"]]
                    write_count[:, :, region["out_t_start"]:region["out_t_end"],
                                    region["out_h_start"]:region["out_h_end"],
                                    region["out_w_start"]:region["out_w_end"]] += 1
            
            video_generate = output_video

            if (write_count == 0).any():
                print("Error: Lack of write in region !!!")
                exit()
            if (write_count > 1).any():
                print("Error: Write count > 1 in region !!!")
                exit()

            video_generate = remove_padding_and_extra_frames(video_generate, pad_f, pad_h*4, pad_w*4)
            file_name = os.path.basename(video_path)
            output_path = os.path.join(args.output_path, file_name)

            if metric_viz:
                metric_viz["enabled"] = False
                save_metric_head_records(
                    metric_viz["buffer"],
                    video_name,
                    metric_viz["output_dir"],
                    metric_viz["layout"],
                    lq_frame=metric_viz.get("current_lq", None),
                )

            if metrics_models is not None:
                #  [1, C, F, H, W] -> [F, C, H, W]
                pred_frames = video_generate[0]
                pred_frames = pred_frames.permute(1, 0, 2, 3).contiguous()
                if args.gt_dir is not None:
                    gt_frames = load_sequence(os.path.join(args.gt_dir, file_name))
                else:
                    gt_frames = None
                compute_metrics(pred_frames, gt_frames, metrics_models, metric_accumulator, file_name)

            if args.png_save:
                # Save as PNG sequence
                output_dir = output_path.rsplit('.', 1)[0]  # Remove extension
                save_frames_as_png(video_generate, output_dir, fps=args.fps)
            else:
                output_path = output_path.replace('.mkv', '.mp4')
                save_video_with_imageio(video_generate, output_path, fps=args.fps, format=args.save_format)
        else:
            print(f"Warning: {video_name} not found in {args.input_dir}")

    if metric_viz:
        for hook in metric_viz["hooks"]:
            hook.remove()

    if args.profile_transformer and profile_state["times_ms"]:
        avg_time = sum(profile_state["times_ms"]) / len(profile_state["times_ms"])
        logging.info(
            "Transformer forward latency: avg %.2f ms over %d calls (min %.2f ms, max %.2f ms)",
            avg_time,
            len(profile_state["times_ms"]),
            min(profile_state["times_ms"]),
            max(profile_state["times_ms"]),
        )
        if profile_state["flops"] is not None:
            logging.info("Transformer forward FLOPs (first profiled call): %.3e", profile_state["flops"])

    if metrics_models is not None:
        print("\n=== Overall Average Metrics ===")
        count = len(next(iter(metric_accumulator.values())))
        overall_avg = {metric: 0 for metric in metrics_list}
        out_name = 'metrics_'
        for metric in metrics_list:
            out_name += f"{metric}_"
            scores = metric_accumulator[metric]
            if scores:
                avg = sum(scores) / len(scores)
                overall_avg[metric] = avg
                print(f"{metric.upper()}: {avg:.4f}")

        out_name = out_name.rstrip('_') + '.json'
        out_path = os.path.join(args.output_path, out_name)
        output = {
            "per_sample": metric_accumulator,
            "average": overall_avg,
            "count": count
        }
        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)

    print("All videos processed.")
