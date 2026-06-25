import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.multiprocessing as mp
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from datasets import load_dataset, load_from_disk


if is_wandb_available():
    import wandb

check_min_version("0.38.0.dev0")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Manojb/stable-diffusion-2-1-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default="./model_out/controlnet",
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="condition",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument('--dataset_name', type=str, default='Luka-Wang/realsinglehumanpose', help='Dataset name')
    parser.add_argument('--dataset_split', type=str, default="test")
    parser.add_argument('--num_inference_steps', type=int, default=20, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory for generated images')

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

@torch.no_grad()
def worker(rank, num_gpus, indices, args):
    """
    Each worker runs on GPU `rank`, processing dataset samples at `indices`.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"[GPU {rank}] Loading models... ({len(indices)} samples to process)")

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    # Load models onto this GPU
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    ).to(device, dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    ).to(device, dtype=torch.float16)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    ).to(device, dtype=torch.float16)
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path, torch_dtype=torch.float16
    ).to(device)
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale

    # Precompute unconditional embeddings
    uncond_input = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device), return_dict=False)[0]

    print(f"[GPU {rank}] Models loaded. Starting inference...")

    for global_idx in tqdm(indices, desc=f"GPU {rank}", position=rank):
        condition_image = (
            dataset[global_idx][args.conditioning_image_column]
            .convert('RGB')
            .resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
        )
        prompt = dataset[global_idx][args.caption_column]

        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = text_encoder(text_input.input_ids.to(device), return_dict=False)[0]
        encoder_hidden_states = torch.cat([uncond_embeddings, text_embeddings], dim=0).to(dtype=torch.float16)

        controlnet_cond = transforms.ToTensor()(condition_image).unsqueeze(0).to(device, dtype=torch.float16)
        controlnet_cond_input = torch.cat([controlnet_cond, controlnet_cond], dim=0)

        latent_shape = (1, unet.config.in_channels, args.resolution // 8, args.resolution // 8)
        latents = torch.randn(latent_shape, device=device, dtype=torch.float16)

        scheduler.set_timesteps(num_inference_steps, device=device)
        latents = latents * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents, latents], dim=0)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            down_block_res_samples, mid_block_res_sample = controlnet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond_input,
                return_dict=False,
            )

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / vae.config.scaling_factor
        with torch.no_grad():
            image = vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image[0] * 255).round().astype(np.uint8)
        image = Image.fromarray(image)

        save_path = os.path.join(args.output_dir, f"{global_idx}.png")
        image.save(save_path)

    print(f"[GPU {rank}] Done. Processed {len(indices)} images.")


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine number of GPUs
    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA GPUs available."
    print(f"Using {num_gpus} GPUs for inference.")

    # Load dataset just to get its length
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    total_samples = len(dataset)
    del dataset  # free memory before spawning
    print(f"Total samples: {total_samples}")

    # Split indices across GPUs (no samples dropped)
    all_indices = list(range(total_samples))
    chunk_size = math.ceil(total_samples / num_gpus)
    indices_per_gpu = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, total_samples)
        indices_per_gpu.append(all_indices[start:end])

    for i, idx_list in enumerate(indices_per_gpu):
        print(f"  GPU {i}: {len(idx_list)} samples (indices {idx_list[0]}..{idx_list[-1]})")

    # Spawn one process per GPU
    mp.spawn(
        worker,
        args=(num_gpus, indices_per_gpu, args),  # note: mp.spawn prepends rank automatically
        nprocs=num_gpus,
        join=True,
    )

    print(f"All done. Images saved to {args.output_dir}/")


# mp.spawn passes `rank` as the first arg, but `indices_per_gpu` is the full list.
# We need a thin wrapper so each worker picks its own slice.
def _worker_wrapper(rank, num_gpus, indices_per_gpu, args):
    worker(rank, num_gpus, indices_per_gpu[rank], args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    assert num_gpus > 0, "No CUDA GPUs available."
    print(f"Using {num_gpus} GPUs for inference.")

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    total_samples = len(dataset)
    del dataset
    print(f"Total samples: {total_samples}")

    all_indices = list(range(total_samples))
    chunk_size = math.ceil(total_samples / num_gpus)
    indices_per_gpu = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = min(start + chunk_size, total_samples)
        if start < end:
            indices_per_gpu.append(all_indices[start:end])

    actual_gpus = len(indices_per_gpu)
    for i, idx_list in enumerate(indices_per_gpu):
        print(f"  GPU {i}: {len(idx_list)} samples (indices {idx_list[0]}..{idx_list[-1]})")

    mp.spawn(
        _worker_wrapper,
        args=(actual_gpus, indices_per_gpu, args),
        nprocs=actual_gpus,
        join=True,
    )

    print(f"All done. {total_samples} images saved to {args.output_dir}/")