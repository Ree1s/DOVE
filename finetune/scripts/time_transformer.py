#!/usr/bin/env python
import argparse
import time
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusers import CogVideoXPipeline

from finetune.models.dove.cogvideox_transformer3d_router import (
    TokenMergeCogVideoXTransformer3DModel,
)


def build_prompts(tokenizer, text_encoder, batch_size, device, dtype):
    prompts = ["A test prompt."] * batch_size
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        text_emb = text_encoder(tokens.input_ids)[0]
    return text_emb.to(dtype)


def build_latents(config, batch_size, device, dtype):
    num_frames = getattr(config, "sample_frames", 1)
    channels = getattr(config, "in_channels", 16)
    height = getattr(config, "sample_height", 60)
    width = getattr(config, "sample_width", 90)
    latents = torch.randn(
        batch_size,
        num_frames,
        channels,
        height,
        width,
        device=device,
        dtype=dtype,
    )
    return latents


def run_transformer(transformer, latents, text_emb, device, dtype):
    timesteps = torch.zeros(latents.size(0), dtype=torch.long, device=device)

    def _forward():
        with torch.no_grad():
            transformer(
                hidden_states=latents,
                encoder_hidden_states=text_emb,
                timestep=timesteps,
                return_dict=False,
            )

    return _forward


def benchmark(fn, warmup=5, iters=20):
    # Warm-up
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    return elapsed * 1e3  # milliseconds


def load_token_merge(base_transformer, args):
    tm_transformer = TokenMergeCogVideoXTransformer3DModel.from_config(
        base_transformer.config
    )
    state = base_transformer.state_dict()
    missing, unexpected = tm_transformer.load_state_dict(state, strict=False)
    if missing:
        print(f"[time_transformer] Missing keys (expected for metric heads): {len(missing)}")
    if unexpected:
        print(f"[time_transformer] Unexpected keys: {unexpected}")
    tm_transformer.to(base_transformer.device)
    tm_transformer.to(next(base_transformer.parameters()).dtype)
    tm_transformer.configure_token_merge(
        enable_token_merge=True,
        routes_spec=args.token_merge_routes,
        default_ratio=args.token_merge_default_ratio,
        seed=args.token_merge_seed,
        restore_adapter_expansion=args.token_merge_restore_adapter_expansion,
        window_size=args.token_merge_window_size,
        window_stride=args.token_merge_window_stride,
        ratio_start=args.token_merge_ratio_start,
        ratio_warmup_steps=args.token_merge_ratio_warmup_steps,
        ratio_schedule=args.token_merge_ratio_schedule,
        use_psg_importance=args.token_merge_use_psg_importance,
        layer_gate_group_size=args.token_merge_layer_gate_group_size,
        layer_gate_keep_per_group=args.token_merge_layer_gate_keep_per_group,
        layer_gate_tau=args.token_merge_layer_gate_tau,
        layer_gate_logit_scale=args.token_merge_layer_gate_logit_scale,
    )
    if args.token_merge_freeze_routes_only:
        tm_transformer.freeze_parameters_to_routes()
    return tm_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark DOVE transformer with/without token merge.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained CogVideoX model or DOVE checkpoint.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)

    # Token merge parameters
    parser.add_argument("--token_merge_routes", type=str, default="10-17@0.36;28-35@0.36")
    parser.add_argument("--token_merge_default_ratio", type=float, default=0.0)
    parser.add_argument("--token_merge_seed", type=int, default=42)
    parser.add_argument("--token_merge_restore_adapter_expansion", type=int, default=2)
    parser.add_argument("--token_merge_window_size", type=int, default=0)
    parser.add_argument("--token_merge_window_stride", type=int, default=1)
    parser.add_argument("--token_merge_ratio_start", type=float, default=None)
    parser.add_argument("--token_merge_ratio_warmup_steps", type=int, default=0)
    parser.add_argument("--token_merge_ratio_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--token_merge_freeze_routes_only", action="store_true")
    parser.add_argument("--token_merge_use_psg_importance", action="store_true")
    parser.add_argument("--token_merge_layer_gate_group_size", type=int, default=0)
    parser.add_argument("--token_merge_layer_gate_keep_per_group", type=int, default=0)
    parser.add_argument("--token_merge_layer_gate_tau", type=float, default=1.0)
    parser.add_argument("--token_merge_layer_gate_logit_scale", type=float, default=1.0)

    return parser.parse_args()


def get_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()

    device = torch.device(args.device)
    dtype = get_dtype(args.dtype)

    pipe = CogVideoXPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)

    base_transformer = pipe.transformer

    token_merge_transformer = load_token_merge(base_transformer, args)

    text_emb = build_prompts(pipe.tokenizer, pipe.text_encoder, args.batch_size, device, dtype)
    latents = build_latents(base_transformer.config, args.batch_size, device, dtype)

    base_forward = run_transformer(base_transformer, latents, text_emb, device, dtype)
    tm_forward = run_transformer(token_merge_transformer, latents, text_emb, device, dtype)

    base_ms = benchmark(base_forward, warmup=args.warmup, iters=args.iters)
    tm_ms = benchmark(tm_forward, warmup=args.warmup, iters=args.iters)

    print("=== Transformer Forward Benchmark ===")
    print(f"Model path: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"DType: {dtype}")
    print(f"Base transformer: {base_ms:.2f} ms")
    print(f"Token-merge transformer: {tm_ms:.2f} ms")
    print(f"Speed-up: {base_ms / tm_ms if tm_ms > 0 else float('inf'):.2f}x")


if __name__ == "__main__":
    main()
