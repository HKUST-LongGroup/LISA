from cleanfid import fid
import torch
import os
import shutil

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet inference script.")
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default="./model_out/controlnet",
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
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
    parser.add_argument('--dataset_name', type=str, default='Luka-Wang/sr_control', help='Dataset name')
    parser.add_argument('--dataset_split', type=str, default="val")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    input_path = args.controlnet_model_name_or_path + '/outputs/' + args.dataset_split

    # --------------- FID ---------------
    print("Computing FID...")
    real_images_dir = "./real/sr/"

    if not os.path.exists(real_images_dir):
        os.makedirs(real_images_dir, exist_ok=True)
        for idx in tqdm(range(len(dataset)), desc="Saving real images"):
            real_image = dataset[idx]["image"].convert("RGB")
            real_image.save(os.path.join(real_images_dir, f"{idx}.png"))
    else:
        print(f"Directory {real_images_dir} already exists. Skipping image creation.")

    fid_score = fid.compute_fid(input_path, real_images_dir)
    print(f"FID: {fid_score:.4f}")

    print("Done.")


    # --------------- PSNR & LPIPS ---------------
    print("Computing PSNR and LPIPS...")
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)

    psnr_values = []
    lpips_values = []

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Computing PSNR/LPIPS"):
            real_image = dataset[idx]["image"].convert("RGB")
            gen_path = os.path.join(input_path, f"{idx}.png")
            if not os.path.exists(gen_path):
                continue
            gen_image = Image.open(gen_path).convert("RGB")

            if gen_image.size != real_image.size:
                gen_image = gen_image.resize(real_image.size, Image.BICUBIC)

            real_t = pil_to_tensor(real_image).float().unsqueeze(0).to(device) / 255.0
            gen_t = pil_to_tensor(gen_image).float().unsqueeze(0).to(device) / 255.0

            psnr_values.append(psnr_metric(gen_t, real_t).item())

            lpips_values.append(lpips_metric(gen_t, real_t).item())

    psnr_score = float(np.mean(psnr_values)) if psnr_values else float("nan")
    lpips_score = float(np.mean(lpips_values)) if lpips_values else float("nan")
    print(f"PSNR: {psnr_score:.4f}")
    print(f"LPIPS: {lpips_score:.4f}")
    
    
    # --------------- Save Results ---------------
    print("Saving results to ./results.txt")
    with open("./results.txt", "a") as f:
        f.write(f"Dataset Name: {args.dataset_name}\n")
        f.write(f"Dataset Split: {args.dataset_split}\n")
        f.write(f"ControlNet Model Path: {args.controlnet_model_name_or_path}\n")
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"PSNR: {psnr_score:.4f}\n")
        f.write(f"LPIPS: {lpips_score:.4f}\n")
        f.write("-" * 50 + "\n")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)