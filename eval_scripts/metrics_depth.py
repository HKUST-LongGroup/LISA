from cleanfid import fid
import torch
import os
import shutil

from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.multimodal.clip_score import CLIPScore
import numpy as np
import argparse
from controlnet_aux import MidasDetector

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
    parser.add_argument('--dataset_name', type=str, default='Luka-Wang/depth_control', help='Dataset name')
    parser.add_argument('--dataset_split', type=str, default="val")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    input_path = args.controlnet_model_name_or_path + '/outputs/' + args.dataset_split
    # --------------- RMSE ---------------
    print("Computing RMSE (depth)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
    midas.to(device)

    def _depth_to_norm_array(pil_img, size):
        """Convert a depth PIL image to a [0,1]-normalized float32 numpy array of shape (H, W)."""
        d = pil_img.convert("L").resize(size, Image.BILINEAR)
        arr = np.asarray(d, dtype=np.float32)
        d_min, d_max = arr.min(), arr.max()
        if d_max - d_min < 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return (arr - d_min) / (d_max - d_min)

    rmse_scores = []
    eval_size = (512, 512)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Computing RMSE"):
            gen_image_path = os.path.join(input_path, f"{idx}.png")
            if not os.path.exists(gen_image_path):
                continue

            # Ground-truth depth map from the dataset's conditioning column
            gt_depth_pil = dataset[idx][args.conditioning_image_column]
            gt_depth = _depth_to_norm_array(gt_depth_pil, eval_size)

            # Estimated depth from the generated image
            gen_image = Image.open(gen_image_path).convert("RGB").resize(eval_size, Image.BILINEAR)
            pred_depth_pil = midas(gen_image)
            if isinstance(pred_depth_pil, tuple):
                pred_depth_pil = pred_depth_pil[0]
            pred_depth = _depth_to_norm_array(pred_depth_pil, eval_size)

            rmse_scores.append(float(np.sqrt(np.mean((gt_depth - pred_depth) ** 2))))

    rmse = float(np.mean(rmse_scores)) if rmse_scores else 0.0
    print(f"RMSE: {rmse:.4f}")

    # Free Midas memory before next stages
    del midas
    torch.cuda.empty_cache()


    # --------------- FID ---------------
    print("Computing FID...")
    real_images_dir = "./real/depth/"

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



    # --------------- CLIP Score---------------
    print("Computing CLIP Score...")

    safe_model_path = "./clip-vit-base-patch16-safetensors"

    def extract_tensor(out):
        if isinstance(out, torch.Tensor):
            return out
        for attr in ['image_embeds', 'text_embeds', 'pooler_output']:
            if hasattr(out, attr) and getattr(out, attr) is not None:
                return getattr(out, attr)
        return out[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric = CLIPScore(model_name_or_path=safe_model_path).to(device)

    # Monkey-patch to ensure plain Tensor output
    original_get_image_features = metric.model.get_image_features
    original_get_text_features = metric.model.get_text_features
    metric.model.get_image_features = lambda *args, **kwargs: extract_tensor(original_get_image_features(*args, **kwargs))
    metric.model.get_text_features = lambda *args, **kwargs: extract_tensor(original_get_text_features(*args, **kwargs))

    for idx in tqdm(range(len(dataset)), desc="Computing CLIP Score"):
        text = dataset[idx][args.caption_column]
        gen_image_path = os.path.join(input_path, f"{idx}.png")

        if not os.path.exists(gen_image_path):
            continue

        gen_image = Image.open(gen_image_path).convert("RGB")
        gen_image_tensor = pil_to_tensor(gen_image).to(device)

        metric.update(gen_image_tensor.unsqueeze(0), [text])

    clip_score = (metric.score / metric.n_samples).item()
    print(f"CLIP Score: {clip_score:.4f}")
    print("Done.")

    # --------------- Save Results ---------------
    print("Saving results to ./results.txt")
    with open("./results.txt", "a") as f:
        f.write(f"Dataset Name: {args.dataset_name}\n")
        f.write(f"Dataset Split: {args.dataset_split}\n")
        f.write(f"ControlNet Model Path: {args.controlnet_model_name_or_path}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"CLIP Score: {clip_score:.4f}\n")
        f.write("-" * 50 + "\n")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)