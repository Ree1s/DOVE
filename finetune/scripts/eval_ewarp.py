# 测量一下UDM10
import os
import numpy as np
import sys
script_path = os.path.abspath(sys.argv[0])
script_directory = os.path.dirname(script_path)
repo_root = os.path.abspath(os.path.join(script_directory, "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
os.chdir(script_directory)
import cv2
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from functools import partial
import shutil
import subprocess
import re
import imageio.v3 as iio
from argparse import Namespace

raft_parent = os.path.abspath(os.path.join(script_directory, "..", "utils"))
if raft_parent not in sys.path:
    sys.path.append(raft_parent)

from RAFT.raft import RAFT  # noqa: E402
from RAFT.utils.utils import InputPadder  # noqa: E402
from finetune.utils.optical_flow_utils import flow_warp, fbConsistencyCheck  # noqa: E402


# 0 ~ 1
to_tensor = transforms.ToTensor()
video_exts = ['.mp4', '.avi', '.mov', '.mkv']
video_metrics  = ['dover']


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

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def img2video(subfolder_path, output_path, fps=8):
    # 2025.4.19
    img_tensor = read_image_folder(subfolder_path)
    if img_tensor is None:
        print(f"Failed to read images from {subfolder_path}")
        return
    img_tensor = img_tensor.permute(0, 2, 3, 1)  # [F, H, W, C]
    frames = (img_tensor * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()  # [F, H, W, C]
    iio.imwrite(
        output_path,
        frames,
        fps=fps,
        codec='libx264rgb',
        pixelformat='rgb24',
        macro_block_size=None,
        ffmpeg_params=['-crf', '0'],
    )
    print(f"Video saved to {output_path}")

class EwarpCalculator:
    def __init__(self, model_path, device, small=False, mixed_precision=False, alternate_corr=False, iters=20):
        self.device = device
        self.iters = iters
        raft_args = Namespace(
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
            dropout=0.0,
        )
        self.model = RAFT(raft_args)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"RAFT checkpoint not found at {model_path}")
        state = torch.load(model_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        load_res = self.model.load_state_dict(cleaned, strict=False)
        missing, unexpected = load_res.missing_keys, load_res.unexpected_keys
        if missing:
            print(f"Warning: missing RAFT keys: {missing}")
        if unexpected:
            print(f"Warning: unexpected RAFT keys: {unexpected}")
        self.model.to(device)
        self.model.eval()

    def warp_once(self, src, tgt, flow_src_tgt, flow_tgt_src):
        mask = fbConsistencyCheck(flow_src_tgt, flow_tgt_src)
        warped = flow_warp(src, flow_src_tgt.permute(0, 2, 3, 1))
        diff = ((warped - tgt) / 255.0) ** 2
        diff_mean = diff.mean(dim=1, keepdim=True)
        denom = mask.sum()
        if denom <= 0:
            return None
        error = (diff_mean * mask).sum() / denom
        return float(error.item())

    def warp_error(self, frame_a, frame_b):
        if frame_a.size(-1) < 2 or frame_a.size(-2) < 2:
            return None
        image1 = frame_a.unsqueeze(0)
        image2 = frame_b.unsqueeze(0)
        padder = InputPadder(image1.shape)
        image1_pad, image2_pad = padder.pad(image1, image2)
        with torch.no_grad():
            _, flow12 = self.model(image1_pad, image2_pad, iters=self.iters, test_mode=True)
            _, flow21 = self.model(image2_pad, image1_pad, iters=self.iters, test_mode=True)
        flow12 = padder.unpad(flow12)
        flow21 = padder.unpad(flow21)
        err_fwd = self.warp_once(image1, image2, flow12, flow21)
        err_bwd = self.warp_once(image2, image1, flow21, flow12)
        valid_errors = [e for e in (err_fwd, err_bwd) if e is not None]
        if not valid_errors:
            return None
        return float(sum(valid_errors) / len(valid_errors))

    def __call__(self, video_path):
        frames = read_video_frames(video_path).to(self.device) * 255.0
        if frames.size(0) < 2:
            return None
        errors = []
        for idx in range(frames.size(0) - 1):
            err = self.warp_error(frames[idx], frames[idx + 1])
            if err is not None:
                errors.append(err)
        if not errors:
            return None
        return float(np.mean(errors))


def process(pred_root, out_path, args):

    pred_root = os.path.abspath(pred_root)
    out_path = os.path.abspath(out_path)
    all_items = os.listdir(pred_root)
    folders_count = 0
    videos_count = 0
    
    for item in all_items:
        item_path = os.path.join(pred_root, item)
        if os.path.isdir(item_path):
            folders_count += 1
        elif is_video_file(item_path):
            videos_count += 1
    
    is_folder_dominant = folders_count >= videos_count
    print(f"Found {folders_count} folders and {videos_count} videos. Folder dominant: {is_folder_dominant}")
    
    pred_files = {}
    if is_folder_dominant:
        for item in all_items:
            item_path = os.path.join(pred_root, item)
            if os.path.isdir(item_path):
                name = os.path.splitext(item)[0]
                pred_files[name] = item_path
    else:
        for item in all_items:
            item_path = os.path.join(pred_root, item)
            if is_video_file(item_path):
                name = os.path.splitext(item)[0]
                pred_files[name] = item_path
    
    if not pred_files:
        print("No valid folders or video files found in the specified directory.")
        return
    
    pred_names = sorted(pred_files.keys())
    
    input_path = pred_root  
    if is_folder_dominant:
        input_path = os.path.abspath(os.path.join(out_path, "temp"))
        os.makedirs(input_path, exist_ok=True)
        
        for name in pred_names:
            subfolder_path = pred_files[name]
            if os.path.isdir(subfolder_path):
                video_path = os.path.join(input_path, f"{name}.mp4")
                img2video(subfolder_path, video_path)
                pred_files[name] = video_path
    else:
        input_path = os.path.abspath(os.path.join(out_path, "temp"))
        os.makedirs(input_path, exist_ok=True)
        for name in pred_names:
            video_path = pred_files[name]
            if is_video_file(video_path):
                new_video_path = os.path.join(input_path, f"{name}.mp4")
                shutil.copy(video_path, new_video_path)
                pred_files[name] = new_video_path

    args.pred = input_path
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    calculator = EwarpCalculator(
        args.model,
        device,
        small=args.small,
        mixed_precision=args.mixed_precision,
        alternate_corr=args.alternate_corr,
        iters=args.iters,
    )
    results = {}
    for name, video_path in pred_files.items():
        score = calculator(video_path)
        if score is not None:
            results[name] = score
    if results:
        avg_score = float(np.mean(list(results.values())))
    else:
        avg_score = None

    count = len(results)
    if count > 0:
        print(results)
    else:
        print("No valid samples were processed.")
    overall_avg = avg_score

    print(f"\nProcessed {count} samples.")
    print(f"Average score: {overall_avg}")
    output = {
        "per_sample": results,
        "average": overall_avg,
        "count": count
    }

    os.makedirs(out_path, exist_ok=True)
    out_name = 'metrics_ewarp.json'
    file_out_path = os.path.join(out_path, out_name)

    with open(file_out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    output_folder = os.path.join(out_path,"temp")

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default='', help="Specify the path of generated videos")
    parser.add_argument("--metric", type=str, default='warping_error', help="Specify the metric to be used")
    parser.add_argument('--model', type=str, default='finetune/scripts/models/raft-things.pth',help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--iters', type=int, default=20, help='Number of RAFT update iterations')
    parser.add_argument('--out', type=str, default='', help='Path to save JSON output (as directory)')
    args = parser.parse_args()

    if args.out == '':
        out = args.pred
    else:
        out = args.out
    process(args.pred, out, args)
