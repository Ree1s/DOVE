import json
import os
import re
import shutil
import sys
from typing import Dict, Optional

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

script_path = os.path.abspath(sys.argv[0])
script_directory = os.path.dirname(script_path)
os.chdir(script_directory)

REPO_ROOT = os.path.abspath(os.path.join(script_directory, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from VideoScore2.eval.eval_methods.utils_dover.datasets.dover_datasets import (  # noqa: E402
    UnifiedFrameSampler,
    spatial_temporal_view_decomposition,
)
from VideoScore2.eval.eval_methods.utils_dover.models.evaluator import (  # noqa: E402
    DOVER as DoverModel,
)

video_exts = [".mp4", ".avi", ".mov", ".mkv"]

DOVER_SAMPLE_TYPES: Dict[str, Dict[str, int]] = {
    "technical": {
        "fragments_h": 7,
        "fragments_w": 7,
        "fsize_h": 32,
        "fsize_w": 32,
        "aligned": 32,
        "clip_len": 32,
        "frame_interval": 2,
        "num_clips": 3,
    },
    "aesthetic": {
        "size_h": 224,
        "size_w": 224,
        "clip_len": 32,
        "frame_interval": 2,
        "t_frag": 32,
        "num_clips": 1,
    },
}

DOVER_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
DOVER_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)


def is_video_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in video_exts)


def read_video_frames(video_path: str) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        frames.append(tensor)
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return torch.stack(frames)


def read_image_folder(folder_path: str) -> torch.Tensor:
    image_files = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not image_files:
        raise ValueError(f"No images found inside {folder_path}")
    frames = []
    for path in image_files:
        img = np.array(Image.open(path).convert("RGB"))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        frames.append(tensor)
    return torch.stack(frames)


def load_sequence(path: str) -> torch.Tensor:
    if os.path.isdir(path):
        return read_image_folder(path)
    if os.path.isfile(path):
        if is_video_file(path):
            return read_video_frames(path)
        if path.lower().endswith((".png", ".jpg", ".jpeg")):
            img = to_tensor(Image.open(path).convert("RGB"))
            return img.unsqueeze(0)
    raise ValueError(f"Unsupported input: {path}")


def natural_sort_key(value: str):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", value)]


def img2video(subfolder_path: str, output_path: str, fps: int = 8) -> None:
    img_tensor = read_image_folder(subfolder_path)
    img_tensor = img_tensor.permute(0, 2, 3, 1)
    frames = img_tensor.clamp(0, 255).to(torch.uint8).cpu().numpy()
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264rgb",
        pixelformat="rgb24",
        macro_block_size=None,
        ffmpeg_params=["-crf", "0"],
    )
    print(f"Video saved to {output_path}")


def build_dover_samplers(sample_types: Dict[str, Dict[str, int]]) -> Dict[str, UnifiedFrameSampler]:
    samplers: Dict[str, UnifiedFrameSampler] = {}
    for stype, cfg in sample_types.items():
        if "t_frag" in cfg:
            samplers[stype] = UnifiedFrameSampler(
                cfg["clip_len"] // cfg["t_frag"],
                cfg["t_frag"],
                cfg["frame_interval"],
                cfg["num_clips"],
            )
        else:
            samplers[stype] = UnifiedFrameSampler(
                cfg["clip_len"],
                cfg["num_clips"],
                cfg["frame_interval"],
            )
    return samplers


def load_dover_model(weights_path: str, device: torch.device) -> DoverModel:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"DOVER weights not found at {weights_path}")
    print(f"Loading DOVER weights from {weights_path}")
    model = DoverModel(
        backbone=dict(
            technical={"type": "swin_tiny_grpb"},
            aesthetic={"type": "conv_tiny"},
        ),
        backbone_preserve_keys="technical,aesthetic",
        divide_head=True,
        vqa_head=dict(in_channels=768, hidden_channels=64),
    )
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    cleaned_state = {}
    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned_state[new_key] = value
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading DOVER weights: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading DOVER weights: {unexpected}")
    model.to(device)
    model.eval()
    return model


def fuse_scores(tqe: float, aqe: float) -> float:
    x = (tqe - 0.1107) / 0.07355 * 0.6104 + (aqe + 0.08285) / 0.03774 * 0.3896
    return float(1.0 / (1.0 + np.exp(-x)))


def evaluate_video_with_dover(
    model: DoverModel,
    video_path: str,
    sample_types: Dict[str, Dict[str, int]],
    samplers: Dict[str, UnifiedFrameSampler],
    device: torch.device,
) -> Optional[float]:
    try:
        views, _ = spatial_temporal_view_decomposition(
            video_path, sample_types, samplers, is_train=False
        )
    except Exception as exc:
        print(f"Skipping {video_path}: {exc}")
        return None

    inputs = {}
    for branch, clip in views.items():
        sample_cfg = sample_types.get(branch, {})
        num_clips = sample_cfg.get("num_clips", 1)
        clip = clip.float()
        normalized = ((clip.permute(1, 2, 3, 0) - DOVER_MEAN) / DOVER_STD).permute(
            3, 0, 1, 2
        )
        normalized = normalized.reshape(
            clip.shape[0], num_clips, -1, *clip.shape[2:]
        ).transpose(0, 1)
        inputs[branch] = normalized.to(device)

    with torch.no_grad():
        raw_scores = model(inputs)

    if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) < 2:
        return None

    tqe_score = raw_scores[0].mean().item()
    aqe_score = raw_scores[1].mean().item()
    fused = fuse_scores(tqe_score, aqe_score)
    return fused


def evaluate_with_dover(
    video_map: Dict[str, str], weights_path: str, device: torch.device
) -> Dict[str, float]:
    sample_types = {key: dict(value) for key, value in DOVER_SAMPLE_TYPES.items()}
    samplers = build_dover_samplers(sample_types)
    model = load_dover_model(weights_path, device)
    results: Dict[str, float] = {}
    for name, video_path in video_map.items():
        score = evaluate_video_with_dover(model, video_path, sample_types, samplers, device)
        if score is not None:
            results[name] = score
    return results


def process(pred_root: str, out_path: str, weights_path: str) -> None:
    pred_root = os.path.abspath(pred_root)
    out_path = os.path.abspath(out_path)

    if not os.path.isdir(pred_root):
        raise FileNotFoundError(f"Prediction directory not found: {pred_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_items = os.listdir(pred_root)
    if not all_items:
        print("Prediction directory is empty.")
        return

    pred_files: Dict[str, str] = {}
    for item in all_items:
        item_path = os.path.join(pred_root, item)
        if os.path.isdir(item_path) or is_video_file(item_path):
            name, _ = os.path.splitext(item)
            pred_files[name] = item_path

    if not pred_files:
        print("No valid folders or video files found in the specified directory.")
        return

    pred_names = sorted(pred_files.keys(), key=natural_sort_key)

    temp_dir = os.path.join(out_path, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    prepared_files: Dict[str, str] = {}

    for name in pred_names:
        src_path = pred_files[name]
        dst_path = os.path.join(temp_dir, f"{name}.mp4")
        try:
            if os.path.isdir(src_path):
                img2video(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
        except Exception as exc:
            print(f"Failed to prepare {name}: {exc}")
            continue

        if os.path.isfile(dst_path):
            prepared_files[name] = os.path.abspath(dst_path)

    if not prepared_files:
        print("No videos could be prepared for DOVER evaluation.")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return

    print(f"Evaluating {len(prepared_files)} videos with DOVER...")
    results = evaluate_with_dover(prepared_files, weights_path, device)
    count = len(results)

    if count > 0:
        overall_avg = float(np.mean(list(results.values())))
        print(results)
    else:
        overall_avg = None
        print("No valid samples were processed.")

    print(f"\nProcessed {count} samples.")
    print(f"Average score: {overall_avg}")
    output = {"per_sample": results, "average": overall_avg, "count": count}

    os.makedirs(out_path, exist_ok=True)
    file_out_path = os.path.join(out_path, "metrics_dover.json")

    with open(file_out_path, "w") as f:
        json.dump(output, f, indent=2)

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"Results saved to: {out_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to predicted results folder")
    parser.add_argument("--out", type=str, default="", help="Path to save JSON output (as directory)")
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to DOVER pretrained weights (.pth)",
    )
    args = parser.parse_args()

    out_dir = args.out if args.out else args.pred

    default_weights = os.path.join(
        REPO_ROOT,
        "VideoScore2",
        "eval",
        "eval_methods",
        "utils_dover",
        "pretrained_weights",
        "DOVER.pth",
    )
    weights_path = os.path.abspath(args.weights or default_weights)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"DOVER checkpoint not found at {weights_path}. Provide it via --weights."
        )

    process(args.pred, out_dir, weights_path)


if __name__ == "__main__":
    main()
