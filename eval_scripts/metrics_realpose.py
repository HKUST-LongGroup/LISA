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
from controlnet_aux import OpenposeDetector
import argparse
from scipy.optimize import linear_sum_assignment

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
    parser.add_argument('--dataset_name', type=str, default='Luka-Wang/realsinglehumanpose', help='Dataset name')
    parser.add_argument('--dataset_split', type=str, default="val")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args):
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    input_path = args.controlnet_model_name_or_path + '/outputs/' + args.dataset_split
    # --------------- PCK@0.2 ---------------
    PCK_THRESHOLD = 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    try:
        open_pose.body_estimation.model.to(device)
    except Exception:
        pass

    NUM_KPS = 18  # OpenPose body key points

    def infer_pose(image):
        if isinstance(image, Image.Image):
            img_rgb = np.array(image.convert("RGB"))
        else:
            img_rgb = image
        img_bgr = img_rgb[:, :, ::-1].copy()  # RGB -> BGR

        candidate, subset = open_pose.body_estimation(img_bgr)
        persons = []
        if subset is None or len(subset) == 0 or candidate is None or len(candidate) == 0:
            return persons

        candidate = np.asarray(candidate, dtype=np.float32)  # (N,4): x, y, score, id
        subset = np.asarray(subset, dtype=np.float32)        # (M, 20)

        for person in subset:
            kps = np.zeros((NUM_KPS, 2), dtype=np.float32)
            scores = np.zeros((NUM_KPS,), dtype=np.float32)
            for k in range(NUM_KPS):
                cidx = int(person[k])
                if cidx < 0 or cidx >= len(candidate):
                    scores[k] = 0.0
                    continue
                kps[k, 0] = candidate[cidx, 0]
                kps[k, 1] = candidate[cidx, 1]
                scores[k] = candidate[cidx, 2]

            vis_mask = scores > 0
            if vis_mask.sum() == 0:
                continue
            xs = kps[vis_mask, 0]
            ys = kps[vis_mask, 1]
            bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            persons.append({'keypoints': kps, 'scores': scores, 'bbox': bbox})
        return persons

    def bbox_iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, (b1[2]-b1[0])) * max(0.0, (b1[3]-b1[1]))
        a2 = max(0.0, (b2[2]-b2[0])) * max(0.0, (b2[3]-b2[1]))
        union = a1 + a2 - inter + 1e-9
        return inter / union

    def compute_pck_for_pair(gt_persons, pred_persons, threshold=0.2, score_thr=0.3):
        total_kps = 0
        correct_kps = 0
        if len(gt_persons) == 0:
            return None 

        n_gt, n_pr = len(gt_persons), len(pred_persons)
        if n_pr == 0:
            for g in gt_persons:
                vis = g['scores'] > score_thr
                total_kps += int(vis.sum())
            return (0, total_kps)

        cost = np.ones((n_gt, n_pr), dtype=np.float32)
        for i, g in enumerate(gt_persons):
            for j, p in enumerate(pred_persons):
                cost[i, j] = 1.0 - bbox_iou(g['bbox'], p['bbox'])
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_pred = set()

        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= 1.0: 
                continue
            g = gt_persons[i]; p = pred_persons[j]
            matched_pred.add(j)
            bw = max(1e-6, g['bbox'][2] - g['bbox'][0])
            bh = max(1e-6, g['bbox'][3] - g['bbox'][1])
            ref = np.sqrt(bw * bw + bh * bh) 
            vis = g['scores'] > score_thr
            if vis.sum() == 0:
                continue
            dist = np.linalg.norm(g['keypoints'] - p['keypoints'], axis=1)
            correct = (dist <= threshold * ref) & vis
            correct_kps += int(correct.sum())
            total_kps += int(vis.sum())

        matched_gt = {i for i, j in zip(row_ind, col_ind) if cost[i, j] < 1.0}
        for i, g in enumerate(gt_persons):
            if i in matched_gt:
                continue
            vis = g['scores'] > score_thr
            total_kps += int(vis.sum())
        return (correct_kps, total_kps)

    pck_values = []
    total_correct = 0
    total_count = 0
    for idx in tqdm(range(len(dataset)), desc="Computing PCK"):
        gen_image_path = os.path.join(input_path, f"{idx}.png")
        if not os.path.exists(gen_image_path):
            continue
        real_image = dataset[idx]["image"].convert("RGB")
        gen_image = Image.open(gen_image_path).convert("RGB")

        try:
            gt_persons = infer_pose(real_image)
            pred_persons = infer_pose(gen_image)
        except Exception as e:
            print(f"[WARN] pose inference failed at idx={idx}: {e}")
            continue

        res = compute_pck_for_pair(gt_persons, pred_persons, threshold=PCK_THRESHOLD, score_thr=0.1)
        if res is None:
            continue
        c, t = res
        if t > 0:
            pck_values.append(c / t)
            total_correct += c
            total_count += t

    mean_pck = float(np.mean(pck_values)) if pck_values else 0.0
    micro_pck = (total_correct / total_count) if total_count > 0 else 0.0
    print(f"PCK@{PCK_THRESHOLD} (per-image mean): {mean_pck:.4f}")
    print(f"PCK@{PCK_THRESHOLD} (micro): {micro_pck:.4f}")
    print("Done.")

    # --------------- FID ---------------
    print("Computing FID...")
    real_images_dir = "./real/realpose/"

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
        f.write(f"PCK@{PCK_THRESHOLD}: {mean_pck:.4f}\n")
        f.write(f"FID: {fid_score:.4f}\n")
        f.write(f"CLIP Score: {clip_score:.4f}\n")
        f.write("-" * 50 + "\n")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)