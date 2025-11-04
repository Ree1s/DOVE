import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator


class Args(BaseModel):
    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["real-sr", "real-sr-image-video"]
    training_type: Literal["lora", "sft"] = "lora"

    ########## Output ##########
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all"] | None = None
    tracker_name: str = "VSR"

    ########## Data ###########
    data_root: Path
    image_data_root: Path | None = None
    caption_column: Path | None = None
    image_column: Path | None = None
    video_column: Path

    ########## Training #########
    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)
    crop_mode: str = "random_crop" # for sr

    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    lr_warmup_type: str = "linear"

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800
    stastic_frequency: int = 100
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_update_after_step: int = 0
    ema_update_every: int = 1

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None
    validation_dir: Path | None  # if set do_validation, should not be None
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None  # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    validation_ref_videos: str | None
    gen_fps: int = 15
    raw_test: bool = False # Whether to use raw image for validation
    num_inference_steps: int = 50
    eval_metric_list: str = '' # ["psnr", "ssim", "lpips", "dists", "clipiqa", "musiq", "maniqa", 'niqe']

    ########## SR ##########
    is_latent: bool = False
    is_prompt_latent: bool = False
    is_cache: bool = True
    prompt_cache: str = "prompt_embeddings"
    empty_prompt: bool = True
    empty_ratio: float = 0.0 # The ratio of empty prompt in the training set
    sr_noise_step: int = 399
    degradation_config: str = "configs/degradation.yaml"  # if set model_type == real-sr, should not be None
    is_image_video_mix: bool = False # Whether to use video and image mix for training
    image_step: int = 0
    video_step: int = 0
    use_optical_flow: bool = False # Whether to use optical flow
    is_learnable_fuse: bool = False # Whether to use learnable optical flow
    image_ratio: float = 0.0 # The ratio of image and video in the training set

    ########## Flow Match ##########
    noise_step: int = 700
    shift_t: float = 1.0

    ########## Token Merge ##########
    enable_token_merge: bool = False
    token_merge_routes: str | None = None
    token_merge_seed: int = 42
    token_merge_default_ratio: float = 0.0
    token_merge_restore_adapter_expansion: int = 2
    token_merge_window_size: int = 0
    token_merge_window_stride: int = 1
    token_merge_freeze_routes_only: bool = False

    ########## GAN ##########
    diffusion_gan_max_timestep: int = 1000
    gen_cls_loss_weight: float = 5e-3

    ########## Perceptual Loss ##########
    use_perceptual_loss: bool = False
    ea_dists_weight: float = 0.0
    dists_weight: float = 0.0
    ea_lpips_weight: float = 0.0
    lpips_weight: float = 0.0
    frame_diff_weight: float = 0.0


    @field_validator("image_column")
    def validate_image_column(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("model_type") == "i2v" and not v:
            logging.warning(
                "No `image_column` specified for i2v model. Will automatically extract first frames from videos as conditioning images."
            )
        return v

    @field_validator("validation_dir", "validation_videos")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        values = info.data
        if values.get("do_validation") and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "i2v" and not v:
            raise ValueError(
                "validation_images must be specified when do_validation is True and model_type is i2v"
            )
        return v

    @field_validator("validation_videos")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "v2v" and not v:
            raise ValueError(
                "validation_videos must be specified when do_validation is True and model_type is v2v"
            )
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            # if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
            #     raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("token_merge_routes")
    def validate_token_merge_routes(cls, v: str | None, info: ValidationInfo) -> str | None:
        """Validate token merge configuration and provide helpful warnings."""
        values = info.data

        if not values.get("enable_token_merge"):
            return v

        from finetune.models.dove.token_merge import parse_token_merge_routes

        routes_spec = v
        if routes_spec is not None and routes_spec.strip() == "":
            routes_spec = None

        parsed_routes = parse_token_merge_routes(routes_spec) if routes_spec else []
        default_ratio = values.get("token_merge_default_ratio", 0.0)

        if not parsed_routes and default_ratio <= 0.0:
            logging.warning(
                "Token merge enabled but no routes specified and default_ratio=0.0. "
                "Routing will be a no-op. Specify --token_merge_routes or --token_merge_default_ratio > 0."
            )

        for route in parsed_routes:
            ratio = route.get("selection_ratio", 0.0)
            if ratio > 0.7:
                logging.warning(
                    f"Token merge ratio {ratio:.1%} is very aggressive (>70%). "
                    f"This may significantly impact quality. Consider starting with ratios <0.5."
                )

        if default_ratio > 0.7:
            logging.warning(
                f"Default token merge ratio {default_ratio:.1%} is very aggressive (>70%). "
                f"This may significantly impact quality. Consider starting with ratios <0.5."
            )

        return v

    @field_validator("token_merge_window_stride")
    def validate_token_merge_window(cls, v: int, info: ValidationInfo) -> int:
        values = info.data
        window_size = values.get("token_merge_window_size", 0)
        if window_size > 0 and v <= 0:
            raise ValueError("token_merge_window_stride must be > 0 when windowing is enabled")
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(cls, v: Tuple[int, int, int], info: ValidationInfo) -> str:
        try:
            frames, height, width = v

            # # Check if (frames - 1) is multiple of 8
            # if (frames - 1) % 8 != 0:
            #     raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError(
                        "For cogvideox-5b models, height must be 480 and width must be 720"
                    )

            return v

        except ValueError as e:
            if (
                str(e) == "not enough values to unpack (expected 3, got 0)"
                or str(e) == "invalid literal for int() with base 10"
            ):
                raise ValueError("train_resolution must be in format 'frames x height x width'")
            raise e

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()
        # Required arguments
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--data_root", type=str, required=True)
        parser.add_argument("--image_data_root", type=str, default=None)
        parser.add_argument("--caption_column", type=str, default=None)
        parser.add_argument("--video_column", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--report_to", type=str, default=None)
        parser.add_argument("--crop_mode", type=str, default="random_crop") # for sr

        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--train_steps", type=int, default=None)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--beta3", type=float, default=0.98)
        parser.add_argument("--epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--use_ema", action="store_true", help="Enable Exponential Moving Average tracking of transformer parameters")
        parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay factor (closer to 1 means slower updates)")
        parser.add_argument("--ema_update_after_step", type=int, default=0, help="Number of steps before EMA starts updating")
        parser.add_argument("--ema_update_every", type=int, default=1, help="Update EMA weights every N optimizer steps")

        # Learning rate scheduler
        parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=100)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)
        parser.add_argument("--lr_warmup_type", type=str, default="linear")

        # Data loading
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=lambda x: x.lower() == 'true', default=True) # 固定数据到CUDA
        parser.add_argument("--image_column", type=str, default=None)

        # Model configuration
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--enable_slicing", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--enable_tiling", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--nccl_timeout", type=int, default=1800)
        parser.add_argument("--stastic_frequency", type=int, default=100)

        # LoRA parameters
        parser.add_argument("--rank", type=int, default=128)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument(
            "--target_modules", type=str, nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"]
        )

        # Checkpointing
        parser.add_argument("--checkpointing_steps", type=int, default=200)
        parser.add_argument("--checkpointing_limit", type=int, default=10)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)

        # Validation
        parser.add_argument("--do_validation", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--validation_steps", type=int, default=None)
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--validation_prompts", type=str, default=None)
        parser.add_argument("--validation_images", type=str, default=None)
        parser.add_argument("--validation_videos", type=str, default=None)
        parser.add_argument("--validation_ref_videos", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)
        parser.add_argument("--raw_test", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--num_inference_steps", type=int, default=50)
        parser.add_argument("--eval_metric_list", type=str, default='') # ["psnr", "ssim", "lpips", "dists", "clipiqa", "musiq", "maniqa", 'niqe']

        # SR parameters
        parser.add_argument("--is_latent", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--is_prompt_latent", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--is_cache", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--empty_prompt", type=lambda x: x.lower() == 'true', default=True)
        parser.add_argument("--empty_ratio", type=float, default=0.0) # The ratio of empty prompt in the training set
        parser.add_argument("--prompt_cache", type=str, default="prompt_embeddings")
        parser.add_argument("--sr_noise_step", type=int, default=399)
        parser.add_argument("--degradation_config", type=str, default="configs/degradation.yaml")
        parser.add_argument("--is_image_video_mix", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--image_step", type=int, default=0)
        parser.add_argument("--video_step", type=int, default=0)
        parser.add_argument("--use_optical_flow", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--is_learnable_fuse", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--image_ratio", type=float, default=0.0) # The ratio of image and video in the training set

        # Flow Match parameters
        parser.add_argument("--noise_step", type=int, default=700)
        parser.add_argument("--shift_t", type=float, default=1.0)

        # Token merge parameters
        parser.add_argument("--enable_token_merge", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument(
            "--token_merge_routes",
            type=str,
            default=None,
            help="Semicolon-separated routing specs: 'start-end@ratio;...' e.g. '10-15@0.3;20-25@0.5'",
        )
        parser.add_argument(
            "--token_merge_default_ratio",
            type=float,
            default=0.0,
            help="Default merge ratio applied to all layers when no explicit routes specified",
        )
        parser.add_argument(
            "--token_merge_seed",
            type=int,
            default=42,
            help="Random seed for stochastic token merging",
        )
        parser.add_argument(
            "--token_merge_restore_adapter_expansion",
            type=int,
            default=2,
            help="Expansion factor for RestoreAdapter MLP",
        )
        parser.add_argument(
            "--token_merge_window_size",
            type=int,
            default=0,
            help="Temporal sliding window size for routing (0 disables windowing)",
        )
        parser.add_argument(
            "--token_merge_window_stride",
            type=int,
            default=1,
            help="Stride in frames for sliding routing window progression",
        )
        parser.add_argument(
            "--token_merge_freeze_routes_only",
            type=lambda x: x.lower() == 'true',
            default=False,
            help="Freeze all transformer parameters outside configured token merge routes",
        )

        # GAN parameters
        parser.add_argument("--diffusion_gan_max_timestep", type=int, default=1000)
        parser.add_argument("--gen_cls_loss_weight", type=float, default=5e-3)

        # Perceptual Loss parameters
        parser.add_argument("--use_perceptual_loss", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--ea_dists_weight", type=float, default=0.0)
        parser.add_argument("--dists_weight", type=float, default=0.0)
        parser.add_argument("--ea_lpips_weight", type=float, default=0.0)
        parser.add_argument("--lpips_weight", type=float, default=0.0)
        parser.add_argument("--frame_diff_weight", type=float, default=0.0)

        args = parser.parse_args()

        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        if args.token_merge_routes is not None:
            args.token_merge_routes = args.token_merge_routes.strip() or None

        report_to = args.report_to
        if isinstance(report_to, str):
            report_to = report_to.lower()
            if report_to in ("none", "null", ""):
                args.report_to = None
            elif report_to not in {"tensorboard", "wandb", "all"}:
                raise ValueError(
                    f"Invalid value for --report_to: {args.report_to}. Expected one of ['tensorboard', 'wandb', 'all'] or 'none'."
                )

        return cls(**vars(args))
