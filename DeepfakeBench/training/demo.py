import numpy as np
import cv2
import random
import yaml
import pickle
from tqdm import tqdm
from PIL import Image as pil_image
import dlib
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
from trainer.trainer import Trainer
from detectors import DETECTOR
from collections import defaultdict
from PIL import Image as pil_image
from imutils import face_utils
from skimage import transform as trans
import torchvision.transforms as T
import os
import sys
import json
import csv
from os.path import join
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

"""
Usage:
    python demo.py \
        --detector_config ./training/config/detector/effort.yaml \
        --weights ../../DeepfakeBenchv2/training/weights/easy_clipl14_cdf.pth \
        --image ./id9_id6_0009.jpg \
        --landmark_model ../../DeepfakeBenchv2/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat
"""

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def inference(model, data_dict):
    data, label = data_dict['image'], data_dict['label']
    # move data to GPU
    data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
    predictions = model(data_dict, inference=True)
    return predictions


# preprocess the input image --> cropped face, resize = 256, adding a dimension of batch (output shape: 1x3x256x256)
def get_keypts(image, face, predictor, face_detector):
    # detect the facial landmarks for the selected face
    shape = predictor(image, face)
    
    # select the key points for the eyes, nose, and mouth
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)

    return pts

def extract_aligned_face_dlib(face_detector, predictor, image, res=224, mask=None):
    def img_align_crop(img, landmark=None, outsize=None, scale=1.3, mask=None):
        """ 
        align and crop the face according to the given bbox and landmarks
        landmark: 5 key points
        """

        M = None
        target_size = [112, 112]
        dst = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        if target_size[1] == 112:
            dst[:, 0] += 8.0

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        target_size = outsize

        margin_rate = scale - 1
        x_margin = target_size[0] * margin_rate / 2.
        y_margin = target_size[1] * margin_rate / 2.

        # move
        dst[:, 0] += x_margin
        dst[:, 1] += y_margin

        # resize
        dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
        dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        # use skimage tranformation
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]

        # M: use opencv
        # M = cv2.getAffineTransform(src[[0,1,2],:],dst[[0,1,2],:])

        img = cv2.warpAffine(img, M, (target_size[1], target_size[0]))

        if outsize is not None:
            img = cv2.resize(img, (outsize[1], outsize[0]))
        
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (target_size[1], target_size[0]))
            mask = cv2.resize(mask, (outsize[1], outsize[0]))
            return img, mask
        else:
            return img

    # Image size
    height, width = image.shape[:2]

    # Convert to rgb
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect with dlib
    faces = face_detector(rgb, 1)
    if len(faces):
        # For now only take the biggest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Get the landmarks/parts for the face in box d only with the five key points
        landmarks = get_keypts(rgb, face, predictor, face_detector)

        # Align and crop the face
        cropped_face = img_align_crop(rgb, landmarks, outsize=(res, res), mask=mask)
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR)
        
        # Extract the all landmarks from the aligned face
        face_align = face_detector(cropped_face, 1)
        landmark = predictor(cropped_face, face_align[0])
        landmark = face_utils.shape_to_np(landmark)

        return cropped_face, landmark,face
    
    else:
        return None, None


def load_detector(detector_cfg: str, weights: str):
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)  # FIXME ⚠
    model.eval()
    print("[✓] Detector loaded.")
    return model


def preprocess_face(img_bgr: np.ndarray):
    """BGR → tensor"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711]),
    ])
    return transform(pil_image.fromarray(img_rgb)).unsqueeze(0)  # 1×3×H×W


@torch.inference_mode()
def infer_single_image(
    img_bgr: np.ndarray,
    face_detector,
    landmark_predictor,
    model,
) -> Tuple[int, float]:
    """Return (cls_out, prob)"""
    if face_detector is None or landmark_predictor is None:
        face_aligned = img_bgr
    else:
        face_aligned, _, _ = extract_aligned_face_dlib(
            face_detector, landmark_predictor, img_bgr, res=224
        )

    face_tensor = preprocess_face(face_aligned).to(device)
    data = {"image": face_tensor, "label": torch.tensor([0]).to(device)}
    preds = inference(model, data)
    
    # Handle different output formats
    cls_out = preds["cls"].squeeze().cpu().numpy()
    prob = preds["prob"].squeeze().cpu().numpy()
    
    # Convert to scalar - take first element if array, or convert if already scalar
    if isinstance(cls_out, np.ndarray):
        if cls_out.size == 1:
            cls_out = cls_out.item()
        else:
            # For multi-class output, take argmax
            cls_out = int(np.argmax(cls_out))
    else:
        cls_out = int(cls_out)
    
    if isinstance(prob, np.ndarray):
        if prob.size == 1:
            prob = prob.item()
        else:
            # For multi-class prob, take the probability of class 1 (fake)
            # If binary, prob might be [prob_real, prob_fake]
            if prob.size == 2:
                prob = float(prob[1])  # probability of fake class
            else:
                prob = float(np.max(prob))
    else:
        prob = float(prob)
    
    return cls_out, prob


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def collect_image_paths_with_labels(path_str: str) -> List[Tuple[Path, int]]:
    """
    Collect image paths with ground truth labels based on folder structure.
    Expected structure:
      path_str/
        real/
          image1.jpg
          image2.jpg
        fake/
          image3.jpg
          image4.jpg
    
    Returns:
        List of tuples: (image_path, label) where label is 0 for real, 1 for fake
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"[Error] Path does not exist: {path_str}")

    # Check if this is a single file
    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"[Error] Invalid image format: {p.name}")
        # No ground truth available for single file
        return [(p, None)]

    # Check for real/fake subdirectories
    real_dir = p / 'real'
    fake_dir = p / 'fake'
    
    img_label_list = []
    
    if real_dir.exists() and real_dir.is_dir():
        real_imgs = [fp for fp in real_dir.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS]
        img_label_list.extend([(fp, 0) for fp in real_imgs])
        print(f"[✓] Found {len(real_imgs)} images in 'real' folder")
    
    if fake_dir.exists() and fake_dir.is_dir():
        fake_imgs = [fp for fp in fake_dir.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS]
        img_label_list.extend([(fp, 1) for fp in fake_imgs])
        print(f"[✓] Found {len(fake_imgs)} images in 'fake' folder")
    
    # If no real/fake subdirectories, just collect all images without labels
    if not img_label_list:
        print("[Warning] No 'real' or 'fake' subdirectories found. Collecting all images without ground truth labels.")
        img_list = [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS]
        img_label_list = [(fp, None) for fp in img_list]
    
    if not img_label_list:
        raise RuntimeError(f"[Error] No valid image files found in directory: {path_str}")

    return sorted(img_label_list, key=lambda x: x[0])


def parse_args():
    p = argparse.ArgumentParser(
        description="Deepfake image inference (single image version)"
    )
    p.add_argument("--detector_config", default='training/config/detector/effort.yaml',
                   help="YAML 配置文件路径")
    p.add_argument("--weights", required=True,
                   help="Detector 预训练权重")
    p.add_argument("--image", required=True,
                   help="Path to image file or directory (with 'real' and 'fake' subdirectories for ground truth)")
    p.add_argument("--landmark_model", default=False,
                   help="dlib 81 landmarks dat 文件 / 如果不需要裁剪人脸就是False")
    p.add_argument("--results_dir", default=None,
                   help="Output directory for all results (CSV, JSON, metrics). Default: ./inference_results_<timestamp>")
    return p.parse_args()


def export_results(results: List[Dict], results_dir: Path):
    """Export inference results to both CSV and JSON formats in the results directory"""
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Export CSV
    csv_path = results_dir / "results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    print(f"[✓] Results CSV exported to: {csv_path}")
    
    # Export JSON
    json_path = results_dir / "results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_images": len(results),
            "results": results
        }, f, indent=2)
    print(f"[✓] Results JSON exported to: {json_path}")


def calculate_metrics(results: List[Dict]):
    """Calculate and display confusion matrix and metrics"""
    # Extract labels from results (ground_truth field)
    y_true = []
    y_pred = []
    y_prob = []
    
    matched = 0
    for result in results:
        if result['status'] != 'success':
            continue
        if result.get('ground_truth') is None:
            continue
        
        y_true.append(result['ground_truth'])
        y_pred.append(result['prediction'])
        y_prob.append(result['fake_probability'])
        matched += 1
    
    if matched == 0:
        print("\n[Warning] No ground truth labels available for metric calculation")
        return None
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = None
    
    metrics = {
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'auc': float(auc) if auc is not None else None,
        'matched_samples': matched
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """Pretty print metrics and confusion matrix"""
    if metrics is None:
        return
    
    cm = metrics['confusion_matrix']
    
    print(f"\n{'='*60}")
    print("CONFUSION MATRIX:")
    print(f"{'='*60}")
    print(f"                  Predicted")
    print(f"                Real    Fake")
    print(f"Actual  Real    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"        Fake    {cm[1,0]:4d}    {cm[1,1]:4d}")
    print(f"{'='*60}")
    
    print(f"\nMETRICS (based on {metrics['matched_samples']} matched samples):")
    print(f"{'='*60}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:     {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall/TPR:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  Specificity:   {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  F1-Score:      {metrics['f1_score']:.4f}")
    if metrics['auc'] is not None:
        print(f"  AUC:           {metrics['auc']:.4f}")
    print(f"{'='*60}")
    print(f"\nDetailed Counts:")
    print(f"  True Positives  (TP): {metrics['true_positives']}")
    print(f"  True Negatives  (TN): {metrics['true_negatives']}")
    print(f"  False Positives (FP): {metrics['false_positives']}")
    print(f"  False Negatives (FN): {metrics['false_negatives']}")


def main():
    args = parse_args()

    model = load_detector(args.detector_config, args.weights)
    if args.landmark_model:
        face_det = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(args.landmark_model)
    else:
        face_det, shape_predictor = None, None

    # Collect images with labels from folder structure
    img_label_pairs = collect_image_paths_with_labels(args.image)
    has_ground_truth = any(label is not None for _, label in img_label_pairs)
    
    # Prepare results directory
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"inference_results_{timestamp}")
    else:
        results_dir = Path(args.results_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[✓] Processing {len(img_label_pairs)} image(s)...")
    print(f"[✓] Results directory: {results_dir.absolute()}")
    if has_ground_truth:
        print(f"[✓] Ground truth labels detected from folder structure")
    
    # Store results
    results = []
    
    # ---------- infer with progress bar ----------
    for img_path, gt_label in tqdm(img_label_pairs, desc="Inferencing", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n[Warning] Failed to load image, skipping: {img_path}", file=sys.stderr)
            results.append({
                "filename": img_path.name,
                "path": str(img_path),
                "prediction": None,
                "fake_probability": None,
                "ground_truth": gt_label,
                "status": "error"
            })
            continue

        cls, prob = infer_single_image(img, face_det, shape_predictor, model)
        
        result = {
            "filename": img_path.name,
            "path": str(img_path),
            "prediction": cls,
            "label": "Fake" if cls == 1 else "Real",
            "fake_probability": prob,
            "status": "success"
        }
        
        # Add ground truth if available
        if gt_label is not None:
            result["ground_truth"] = gt_label
            result["ground_truth_label"] = "Fake" if gt_label == 1 else "Real"
            result["correct"] = cls == gt_label
        
        results.append(result)
    
    # Export results
    print(f"\n{'='*60}")
    print("EXPORTING RESULTS:")
    print(f"{'='*60}")
    export_results(results, results_dir)
    
    # Calculate and display metrics if ground truth is available
    if has_ground_truth:
        metrics = calculate_metrics(results)
        if metrics:
            print_metrics(metrics)
            
            # Export metrics to results directory
            metrics_path = results_dir / "results_metrics.json"
            # Convert numpy arrays to lists for JSON serialization
            metrics_export = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                            for k, v in metrics.items()}
            with open(metrics_path, 'w') as f:
                json.dump(metrics_export, f, indent=2)
            print(f"[✓] Metrics exported to: {metrics_path}")
    
    # Print summary
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    fake_count = sum(1 for r in results if r.get("prediction") == 1)
    real_count = sum(1 for r in results if r.get("prediction") == 0)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"  Total images:      {len(results)}")
    print(f"  Successful:        {successful}")
    print(f"  Failed:            {failed}")
    print(f"  Predicted as Real: {real_count}")
    print(f"  Predicted as Fake: {fake_count}")
    
    if has_ground_truth:
        correct_count = sum(1 for r in results if r.get("correct") == True)
        total_with_gt = sum(1 for r in results if r.get("ground_truth") is not None and r["status"] == "success")
        if total_with_gt > 0:
            accuracy = correct_count / total_with_gt
            print(f"  Correct:           {correct_count}/{total_with_gt} ({accuracy*100:.2f}%)")
    
    print(f"{'='*60}")
    print(f"\nAll results saved to: {results_dir.absolute()}")
    print(f"{'='*60}")




if __name__ == "__main__":
    main()
