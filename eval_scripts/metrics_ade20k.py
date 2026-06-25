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
    parser.add_argument('--dataset_name', type=str, default='Luka-Wang/seg_control', help='Dataset name')
    parser.add_argument('--dataset_split', type=str, default="val")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    input_path = args.controlnet_model_name_or_path + '/outputs/' + args.dataset_split
    # --------------- mIoU ---------------
    print("Computing mIoU with Mask2Former (ADE20K-Semantic)...")
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model_name = "facebook/mask2former-swin-large-ade-semantic"
    seg_processor = AutoImageProcessor.from_pretrained(seg_model_name)
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(seg_model_name).to(device)
    seg_model.eval()

    NUM_CLASSES = 150  # ADE20K semantic classes
    IGNORE_INDEX = 255

    # Confusion matrix: rows = GT class, cols = Pred class
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    @torch.no_grad()
    def predict_semantic_map(pil_image):
        inputs = seg_processor(images=pil_image, return_tensors="pt").to(device)
        outputs = seg_model(**inputs)
        seg_map = seg_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[pil_image.size[::-1]]
        )[0]
        return seg_map.cpu().numpy().astype(np.int64)

    for idx in tqdm(range(len(dataset)), desc="Computing mIoU"):
        gen_image_path = os.path.join(input_path, f"{idx}.png")
        if not os.path.exists(gen_image_path):
            continue

        real_image = dataset[idx]["image"].convert("RGB")
        gen_image = Image.open(gen_image_path).convert("RGB")

        # Align sizes (predict generated at the same size as GT image)
        if gen_image.size != real_image.size:
            gen_image = gen_image.resize(real_image.size, Image.BILINEAR)

        gt_seg = predict_semantic_map(real_image)
        pred_seg = predict_semantic_map(gen_image)

        valid = (gt_seg >= 0) & (gt_seg < NUM_CLASSES) & \
                (pred_seg >= 0) & (pred_seg < NUM_CLASSES)
        gt_flat = gt_seg[valid]
        pred_flat = pred_seg[valid]

        # Vectorized confusion matrix update
        idx_flat = gt_flat * NUM_CLASSES + pred_flat
        bincount = np.bincount(idx_flat, minlength=NUM_CLASSES * NUM_CLASSES)
        confusion += bincount.reshape(NUM_CLASSES, NUM_CLASSES)

    tp = np.diag(confusion).astype(np.float64)
    gt_sum = confusion.sum(axis=1).astype(np.float64)
    pred_sum = confusion.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        iou_per_class = np.where(union > 0, tp / union, np.nan)
    miou = np.nanmean(iou_per_class)
    print(f"mIoU: {miou:.4f}")

    # Free segmentation model memory
    del seg_model, seg_processor
    torch.cuda.empty_cache()


    # --------------- FID ---------------
    print("Computing FID...")
    real_images_dir = "./real/ade20k/"

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
        f.write(f"mIoU: {miou:.4f}\n")
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"CLIP Score: {clip_score:.4f}\n")
        f.write("-" * 50 + "\n")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)