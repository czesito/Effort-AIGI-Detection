import numpy as np
import cv2
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image as pil_image
from torchvision import transforms
from detectors import DETECTOR
from pathlib import Path
from datetime import datetime
import json
import csv
import sys
import argparse
from typing import Tuple, List, Dict, Optional
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

"""
Enhanced Inference Script for Large Poster Images with AI-Generated Content Detection

Key Features:
1. Patch-based processing for large images
2. Configurable aggregation strategies (max, mean)
3. Visualization of detection results with patch-level analysis
4. Handles composite poster images with AI-generated elements
5. Organized output: all results saved to a single directory

Usage:
python inference.py --detector_config ./config/detector/effort.yaml --weights ./weights/effort_clip_L14_trainOn_sdv14.pth --image ../../datasets/TISDC2025/ --patch_size 512 --size_threshold 1024 --aggregation max --patch_threshold 0.5 --batch_size 32 --visualize --results_dir ./my_results

Output Structure:
    results_dir/
        ├── results.csv              # Detection results in CSV format
        ├── results.json             # Detection results in JSON format
        ├── results_metrics.json     # Performance metrics (if ground truth available)
        └── visualizations/          # Visualization images (if --visualize enabled)
            ├── fake/               # Visualizations for images predicted as fake
            └── real/               # Visualizations for images predicted as real
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# GPU Optimization Settings
ENABLE_AMP = True  # Automatic Mixed Precision for faster inference
BATCH_SIZE = 32  # Process multiple patches in parallel
PREFETCH_FACTOR = 2  # Prefetch batches for better GPU utilization
NUM_WORKERS = 4  # Number of data loading workers
PIN_MEMORY = True  # Pin memory for faster data transfer to GPU


def load_detector(detector_cfg: str, weights: str):
    """Load the detector model from config and weights"""
    with open(detector_cfg, "r") as f:
        cfg = yaml.safe_load(f)

    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)

    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    
    # Optimize model for inference
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    
    # Enable inference mode optimizations
    for param in model.parameters():
        param.requires_grad = False
    
    print("[✓] Detector loaded.")
    return model


def preprocess_image(img_bgr: np.ndarray, target_size: int = 224) -> torch.Tensor:
    """
    Preprocess image for model inference
    Args:
        img_bgr: BGR image array
        target_size: Target size for resizing (default: 224)
    Returns:
        Preprocessed tensor (1×3×H×W)
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                           [0.26862954, 0.26130258, 0.27577711]),
    ])
    return transform(pil_image.fromarray(img_rgb)).unsqueeze(0)


@torch.inference_mode()
def inference_single(model, img_tensor: torch.Tensor) -> Tuple[int, float]:
    """
    Run inference on a single image tensor
    Returns: (predicted_class, probability)
    """
    img_tensor = img_tensor.to(device, non_blocking=True)
    data = {"image": img_tensor, "label": torch.tensor([0]).to(device)}
    
    # Use automatic mixed precision for faster inference
    if ENABLE_AMP:
        with torch.cuda.amp.autocast():
            preds = model(data, inference=True)
    else:
        preds = model(data, inference=True)
    
    cls_out = preds["cls"].squeeze().cpu().numpy()
    prob = preds["prob"].squeeze().cpu().numpy()
    
    # Convert to scalar
    if isinstance(cls_out, np.ndarray):
        if cls_out.size == 1:
            cls_out = cls_out.item()
        else:
            cls_out = int(np.argmax(cls_out))
    else:
        cls_out = int(cls_out)
    
    if isinstance(prob, np.ndarray):
        if prob.size == 1:
            prob = prob.item()
        else:
            if prob.size == 2:
                prob = float(prob[1])  # probability of fake class
            else:
                prob = float(np.max(prob))
    else:
        prob = float(prob)
    
    return cls_out, prob


@torch.inference_mode()
def inference_batch(model, img_tensors: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a batch of image tensors
    Args:
        img_tensors: Batch of image tensors (B×3×H×W)
    Returns:
        (predicted_classes, probabilities) as numpy arrays
    """
    img_tensors = img_tensors.to(device, non_blocking=True)
    batch_size = img_tensors.size(0)
    data = {"image": img_tensors, "label": torch.zeros(batch_size, dtype=torch.long).to(device)}
    
    # Use automatic mixed precision for faster inference
    if ENABLE_AMP:
        with torch.cuda.amp.autocast():
            preds = model(data, inference=True)
    else:
        preds = model(data, inference=True)
    
    cls_out = preds["cls"].cpu().numpy()
    prob = preds["prob"].cpu().numpy()
    
    # Handle different output formats
    if len(cls_out.shape) == 1:
        # Already class indices
        classes = cls_out.astype(int)
    else:
        # Need to argmax
        classes = np.argmax(cls_out, axis=1).astype(int)
    
    if len(prob.shape) == 1:
        # Single probability per sample
        probs = prob.astype(float)
    elif prob.shape[1] == 2:
        # Binary classification - use fake class probability
        probs = prob[:, 1].astype(float)
    else:
        # Use max probability
        probs = np.max(prob, axis=1).astype(float)
    
    return classes, probs


def extract_patches(img: np.ndarray, patch_size: int) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """
    Extract non-overlapping square patches from image
    Args:
        img: Input image (H×W×C)
        patch_size: Size of square patches
    Returns:
        patches: List of patch images
        positions: List of (row, col) positions for each patch
    """
    h, w = img.shape[:2]
    patches = []
    positions = []
    
    # Calculate number of patches needed
    n_rows = (h + patch_size - 1) // patch_size
    n_cols = (w + patch_size - 1) // patch_size
    
    for i in range(n_rows):
        for j in range(n_cols):
            # Calculate patch boundaries
            y1 = i * patch_size
            y2 = min(y1 + patch_size, h)
            x1 = j * patch_size
            x2 = min(x1 + patch_size, w)
            
            # Extract patch
            patch = img[y1:y2, x1:x2]
            
            # Pad if necessary to make it square
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                padded = np.zeros((patch_size, patch_size, 3), dtype=img.dtype)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            
            patches.append(patch)
            positions.append((i, j))
    
    return patches, positions


def aggregate_predictions(
    probs: List[float],
    method: str = "max",
    threshold: float = 0.5
) -> float:
    """
    Aggregate patch-level predictions
    Args:
        probs: List of probabilities from each patch
        method: Aggregation method ('max' or 'mean')
        threshold: Minimum probability to include in aggregation
    Returns:
        Aggregated probability
    """
    if not probs:
        return 0.0
    
    # Filter by threshold
    filtered_probs = [p for p in probs if p >= threshold]
    
    if not filtered_probs:
        # If no patches exceed threshold, return max of all patches
        return max(probs)
    
    if method == "max":
        return max(filtered_probs)
    elif method == "mean":
        return np.mean(filtered_probs)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def visualize_patches(
    original_img: np.ndarray,
    patches: List[np.ndarray],
    positions: List[Tuple[int, int]],
    probs: List[float],
    patch_size: int,
    final_prob: float,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Create visualization showing original image and annotated patch grid
    Args:
        original_img: Original input image
        patches: List of patches
        positions: List of (row, col) positions
        probs: List of probabilities for each patch
        patch_size: Size of patches
        final_prob: Final aggregated probability
        threshold: Threshold for highlighting patches
    Returns:
        Visualization image (original | annotated patches side by side)
    """
    h, w = original_img.shape[:2]
    n_rows = (h + patch_size - 1) // patch_size
    n_cols = (w + patch_size - 1) // patch_size
    
    # Create patch grid canvas
    grid_h = n_rows * patch_size
    grid_w = n_cols * patch_size
    patch_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    # Place patches and annotate
    for patch, (row, col), prob in zip(patches, positions, probs):
        y1 = row * patch_size
        x1 = col * patch_size
        
        # Copy patch to grid
        patch_rgb = patch.copy()
        
        # Add colored overlay based on probability
        overlay = patch_rgb.copy()
        if prob >= threshold:
            # Red for fake (high probability)
            color = np.array([255, 0, 0], dtype=np.uint8)
            alpha = min(0.3 + (prob - threshold) * 0.7, 0.7)
        else:
            # Green for real (low probability)
            color = np.array([0, 255, 0], dtype=np.uint8)
            alpha = 0.2
        
        overlay[:] = color
        patch_rgb = cv2.addWeighted(patch_rgb, 1 - alpha, overlay, alpha, 0)
        
        # Add probability text
        text = f"{prob:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (patch_size - text_size[0]) // 2
        text_y = (patch_size + text_size[1]) // 2
        
        # Add text background
        cv2.rectangle(patch_rgb, 
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        cv2.putText(patch_rgb, text, (text_x, text_y),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw border
        border_color = (0, 0, 255) if prob >= threshold else (0, 255, 0)
        cv2.rectangle(patch_rgb, (0, 0), (patch_size-1, patch_size-1), 
                     border_color, 3)
        
        patch_grid[y1:y1+patch_size, x1:x1+patch_size] = patch_rgb
    
    # Crop grid to original size
    patch_grid = patch_grid[:h, :w]
    
    # Resize original to match if needed
    original_resized = original_img.copy()
    
    # Add final prediction text to both images
    label = "FAKE" if final_prob >= 0.5 else "REAL"
    color = (0, 0, 255) if final_prob >= 0.5 else (0, 255, 0)
    
    for img in [original_resized, patch_grid]:
        text = f"{label}: {final_prob:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Add text at top center with background
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = 50
        cv2.rectangle(img,
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        cv2.putText(img, text, (text_x, text_y),
                   font, font_scale, color, thickness)
    
    # Concatenate horizontally
    vis = np.hstack([original_resized, patch_grid])
    
    return vis


def infer_image(
    img_path: Path,
    model,
    patch_size: int,
    size_threshold: int,
    aggregation: str,
    patch_threshold: float,
    visualize: bool = False,
    vis_output: Optional[Path] = None
) -> Dict:
    """
    Perform inference on a single image with patch-based processing
    Args:
        img_path: Path to input image
        model: Loaded detector model
        patch_size: Size of square patches
        size_threshold: Minimum dimension to trigger patching
        aggregation: Aggregation method ('max' or 'mean')
        patch_threshold: Threshold for filtering patches
        visualize: Whether to generate visualization
        vis_output: Output directory for visualizations
    Returns:
        Dictionary with inference results
    """
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            "filename": img_path.name,
            "path": str(img_path),
            "status": "error",
            "error": "Failed to load image"
        }
    
    h, w = img.shape[:2]
    use_patches = (h > size_threshold or w > size_threshold)
    
    result = {
        "filename": img_path.name,
        "path": str(img_path),
        "image_size": f"{w}x{h}",
        "used_patches": use_patches,
    }
    
    if use_patches:
        # Extract patches
        patches, positions = extract_patches(img, patch_size)
        result["num_patches"] = len(patches)
        
        # Preprocess all patches and batch them
        patch_tensors = []
        for patch in patches:
            tensor = preprocess_image(patch)
            patch_tensors.append(tensor)
        
        # Stack all patches into a single batch tensor
        patch_tensors = torch.cat(patch_tensors, dim=0)  # Shape: (num_patches, 3, 224, 224)
        
        # Process patches in batches for better GPU utilization
        patch_probs = []
        patch_classes = []
        num_patches = len(patches)
        
        for i in range(0, num_patches, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, num_patches)
            batch_tensors = patch_tensors[i:batch_end]
            
            # Batch inference
            batch_classes, batch_probs = inference_batch(model, batch_tensors)
            patch_classes.extend(batch_classes.tolist())
            patch_probs.extend(batch_probs.tolist())
        
        # Aggregate predictions
        final_prob = aggregate_predictions(patch_probs, aggregation, patch_threshold)
        final_cls = 1 if final_prob >= 0.5 else 0
        
        result.update({
            "prediction": final_cls,
            "label": "Fake" if final_cls == 1 else "Real",
            "fake_probability": final_prob,
            "aggregation_method": aggregation,
            "patch_threshold": patch_threshold,
            "patches_above_threshold": sum(1 for p in patch_probs if p >= patch_threshold),
            "max_patch_prob": max(patch_probs),
            "min_patch_prob": min(patch_probs),
            "mean_patch_prob": np.mean(patch_probs),
            "status": "success"
        })
        
        # Generate visualization if requested
        if visualize and vis_output:
            vis_img = visualize_patches(
                img, patches, positions, patch_probs,
                patch_size, final_prob, patch_threshold
            )
            
            # Create output directory structure
            pred_dir = vis_output / ("fake" if final_cls == 1 else "real")
            pred_dir.mkdir(parents=True, exist_ok=True)
            
            vis_path = pred_dir / f"{img_path.stem}_vis.jpg"
            cv2.imwrite(str(vis_path), vis_img)
            result["visualization_path"] = str(vis_path)
    
    else:
        # Process as single image
        tensor = preprocess_image(img)
        cls, prob = inference_single(model, tensor)
        
        result.update({
            "prediction": cls,
            "label": "Fake" if cls == 1 else "Real",
            "fake_probability": prob,
            "status": "success"
        })
    
    return result


def collect_image_paths_with_labels(path_str: str) -> List[Tuple[Path, Optional[int]]]:
    """
    Collect image paths with ground truth labels based on folder structure.
    Expected structure:
      path_str/
        real/
          image1.jpg
        fake/
          image2.jpg
    
    Returns:
        List of tuples: (image_path, label) where label is 0 for real, 1 for fake, None if unknown
    """
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"[Error] Path does not exist: {path_str}")

    # Check if this is a single file
    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"[Error] Invalid image format: {p.name}")
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
    
    # If no real/fake subdirectories, collect all images
    if not img_label_list:
        print("[Warning] No 'real' or 'fake' subdirectories found. Collecting all images without ground truth labels.")
        img_list = [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() in IMG_EXTS]
        img_label_list = [(fp, None) for fp in img_list]
    
    if not img_label_list:
        raise RuntimeError(f"[Error] No valid image files found in directory: {path_str}")

    return sorted(img_label_list, key=lambda x: x[0])


def calculate_metrics(results: List[Dict]) -> Optional[Dict]:
    """Calculate evaluation metrics from results"""
    y_true = []
    y_pred = []
    y_prob = []
    
    for result in results:
        if result['status'] != 'success':
            continue
        if result.get('ground_truth') is None:
            continue
        
        y_true.append(result['ground_truth'])
        y_pred.append(result['prediction'])
        y_prob.append(result['fake_probability'])
    
    if len(y_true) == 0:
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
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = None
    
    return {
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
        'matched_samples': len(y_true)
    }


def print_metrics(metrics: Dict):
    """Pretty print metrics"""
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


def export_results(results: List[Dict], results_dir: Path):
    """Export results to both CSV and JSON formats in the results directory"""
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Enhanced inference for AI-generated poster image detection"
    )
    p.add_argument("--detector_config", default='training/config/detector/effort.yaml',
                   help="Path to detector YAML config file")
    p.add_argument("--weights", required=True,
                   help="Path to detector weights")
    p.add_argument("--image", required=True,
                   help="Path to image file or directory (with optional 'real'/'fake' subdirectories)")
    
    # Patch processing parameters
    p.add_argument("--patch_size", type=int, default=512,
                   help="Size of square patches (default: 512)")
    p.add_argument("--size_threshold", type=int, default=1024,
                   help="Minimum image dimension to trigger patch processing (default: 1024)")
    p.add_argument("--aggregation", choices=["max", "mean"], default="max",
                   help="Aggregation method for patch predictions (default: max)")
    p.add_argument("--patch_threshold", type=float, default=0.5,
                   help="Minimum probability to include patch in aggregation (default: 0.5)")
    
    # Performance optimization parameters
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for processing patches in parallel (default: 32, increase for better GPU utilization)")
    p.add_argument("--disable_amp", action="store_true",
                   help="Disable automatic mixed precision (AMP) inference")
    
    # Visualization parameters
    p.add_argument("--visualize", action="store_true",
                   help="Generate visualization of patch-level predictions (saved in results_dir/visualizations)")
    
    # Output parameters
    p.add_argument("--results_dir", default=None,
                   help="Output directory for all results (CSV, JSON, metrics, visualizations). Default: ./inference_results_<timestamp>")
    
    return p.parse_args()


def main():
    global BATCH_SIZE, ENABLE_AMP
    
    args = parse_args()
    
    # Update global settings based on arguments
    BATCH_SIZE = args.batch_size
    ENABLE_AMP = not args.disable_amp
    
    # Load model
    print(f"[*] Loading detector from {args.detector_config}...")
    model = load_detector(args.detector_config, args.weights)
    
    # Enable optimizations
    if torch.cuda.is_available():
        print(f"[✓] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[✓] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
        print(f"[✓] cuDNN benchmark mode: Enabled")
        print(f"[✓] Automatic Mixed Precision (AMP): {'Enabled' if ENABLE_AMP else 'Disabled'}")
        print(f"[✓] Batch size for parallel patch processing: {BATCH_SIZE}")
    
    # Collect images
    img_label_pairs = collect_image_paths_with_labels(args.image)
    has_ground_truth = any(label is not None for _, label in img_label_pairs)
    
    # Prepare results directory
    if args.results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"inference_results_{timestamp}")
    else:
        results_dir = Path(args.results_dir)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set visualization output to be inside results directory
    vis_output = results_dir / "visualizations" if args.visualize else None
    if vis_output:
        vis_output.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print(f"\n{'='*60}")
    print("CONFIGURATION:")
    print(f"{'='*60}")
    print(f"  Images to process:    {len(img_label_pairs)}")
    print(f"  Patch size:           {args.patch_size}x{args.patch_size}")
    print(f"  Size threshold:       {args.size_threshold}")
    print(f"  Aggregation method:   {args.aggregation}")
    print(f"  Patch threshold:      {args.patch_threshold}")
    print(f"  Batch size:           {args.batch_size}")
    print(f"  Visualization:        {'Enabled' if args.visualize else 'Disabled'}")
    print(f"  Results directory:    {results_dir.absolute()}")
    if args.visualize:
        print(f"  Visualization output: {vis_output}")
    print(f"  Ground truth:         {'Available' if has_ground_truth else 'Not available'}")
    print(f"{'='*60}\n")
    
    # Process images
    results = []
    for idx, (img_path, gt_label) in enumerate(tqdm(img_label_pairs, desc="Processing images", unit="img")):
        result = infer_image(
            img_path, model, args.patch_size, args.size_threshold,
            args.aggregation, args.patch_threshold,
            args.visualize, vis_output
        )
        
        # Add ground truth if available
        if gt_label is not None:
            result["ground_truth"] = gt_label
            result["ground_truth_label"] = "Fake" if gt_label == 1 else "Real"
            if result["status"] == "success":
                result["correct"] = result["prediction"] == gt_label
        
        results.append(result)
        
        # Periodic memory cleanup (every 100 images)
        if (idx + 1) % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Export results
    print(f"\n{'='*60}")
    print("EXPORTING RESULTS:")
    print(f"{'='*60}")
    export_results(results, results_dir)
    
    # Calculate and display metrics
    if has_ground_truth:
        metrics = calculate_metrics(results)
        if metrics:
            print_metrics(metrics)
            
            # Export metrics to results directory
            metrics_path = results_dir / "results_metrics.json"
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
    patched_count = sum(1 for r in results if r.get("used_patches") == True)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"  Total images:         {len(results)}")
    print(f"  Successful:           {successful}")
    print(f"  Failed:               {failed}")
    print(f"  Used patch processing: {patched_count}")
    print(f"  Predicted as Real:    {real_count}")
    print(f"  Predicted as Fake:    {fake_count}")
    
    if has_ground_truth:
        correct_count = sum(1 for r in results if r.get("correct") == True)
        total_with_gt = sum(1 for r in results if r.get("ground_truth") is not None and r["status"] == "success")
        if total_with_gt > 0:
            accuracy = correct_count / total_with_gt
            print(f"  Correct predictions:  {correct_count}/{total_with_gt} ({accuracy*100:.2f}%)")
    
    print(f"{'='*60}")
    print(f"\nAll results saved to: {results_dir.absolute()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

