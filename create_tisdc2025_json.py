#!/usr/bin/env python3
"""
Script to create a JSON file for the TISDC2025 dataset.
Images are labeled based on filename containing "real" or "fake".
"""
import os
import json
from pathlib import Path
from collections import defaultdict

def create_tisdc2025_json():
    """Create JSON file for TISDC2025 dataset."""
    
    dataset_path = Path("./datasets/TISDC2025")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return
    
    # Initialize the dataset structure
    dataset_json = {
        "TISDC2025": {
            "0": {  # Real images
                "test": {}
            },
            "1": {  # Fake images
                "test": {}
            }
        }
    }
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    for file_path in dataset_path.iterdir():
        if file_path.is_file() and file_path.suffix in image_extensions:
            image_files.append(file_path)
    
    print(f"Found {len(image_files)} images in {dataset_path}")
    
    # Organize images by label
    real_count = 0
    fake_count = 0
    
    for img_path in image_files:
        filename = img_path.name.lower()
        
        # Determine label based on filename
        if 'real' in filename:
            label = "0"  # Real
            label_str = "TISDC2025_Real"
            real_count += 1
        elif 'fake' in filename:
            label = "1"  # Fake
            label_str = "TISDC2025_Fake"
            fake_count += 1
        else:
            print(f"Warning: Cannot determine label for {img_path.name}, skipping...")
            continue
        
        # Use the image filename (without extension) as the "video" name
        # This is because the dataset loader expects video-level organization
        video_name = img_path.stem
        
        # Get absolute path to the image
        abs_path = str(img_path.resolve())
        
        # Add to dataset structure
        dataset_json["TISDC2025"][label]["test"][video_name] = {
            "label": label_str,
            "frames": [abs_path]  # Single frame per "video"
        }
    
    print(f"\nDataset summary:")
    print(f"  Real images: {real_count}")
    print(f"  Fake images: {fake_count}")
    print(f"  Total: {real_count + fake_count}")
    
    # Save JSON file
    output_path = Path("./DeepfakeBench/preprocessing/dataset_json/TISDC2025.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nJSON file created: {output_path}")
    print("\nNext steps:")
    print("1. The JSON file has been created successfully")
    print("2. Run the test command:")
    print("\n   cd DeepfakeBench/")
    print("   python3 training/test.py \\")
    print("     --detector_path ./training/config/detector/effort.yaml \\")
    print("     --test_dataset TISDC2025 \\")
    print("     --weights_path ./training/weights/{CKPT}.pth")

if __name__ == "__main__":
    create_tisdc2025_json()
