#!/usr/bin/env python3
"""
Error Analysis Script

This script analyzes inference results and organizes images for error analysis.
It creates a folder structure separating correct and incorrect predictions,
with filenames prefixed by their ground truth labels.

Usage:
    python training/error_analysis.py --results results.csv --output error_analysis
    
    # With custom dataset path (if different from results CSV):
    python training/error_analysis.py --results results.csv --output error_analysis \
        --dataset-root /path/to/dataset
    
    # Include correct predictions too:
    python training/error_analysis.py --results results.csv --output error_analysis --include-correct
    
    # Dry run (preview without copying):
    python training/error_analysis.py --results results.csv --output error_analysis --dry-run
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform error analysis on inference results"
    )
    parser.add_argument(
        "--results", "-r",
        required=True,
        help="Path to results CSV or JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        default="error_analysis",
        help="Output directory for error analysis (default: error_analysis)"
    )
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Root directory of dataset (if paths in CSV are relative or need override)"
    )
    parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include correctly predicted images in the analysis (default: only errors)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        default=True,
        help="Copy files instead of creating symlinks (default: copy)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be done without actually copying files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information for each file"
    )
    return parser.parse_args()


def load_results(results_path: str) -> List[Dict]:
    """Load results from CSV or JSON file"""
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    results = []
    
    if results_path.suffix.lower() == '.json':
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both formats: direct list or wrapped in "results" key
            if isinstance(data, list):
                results = data
            elif isinstance(data, dict) and 'results' in data:
                results = data['results']
            else:
                raise ValueError("Invalid JSON format")
    
    elif results_path.suffix.lower() == '.csv':
        with open(results_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert string values to appropriate types
                if 'prediction' in row and row['prediction']:
                    row['prediction'] = int(row['prediction']) if row['prediction'] != 'None' else None
                if 'ground_truth' in row and row['ground_truth']:
                    row['ground_truth'] = int(row['ground_truth']) if row['ground_truth'] != 'None' else None
                if 'correct' in row and row['correct']:
                    row['correct'] = row['correct'].lower() in ('true', '1', 'yes')
                if 'fake_probability' in row and row['fake_probability']:
                    row['fake_probability'] = float(row['fake_probability']) if row['fake_probability'] != 'None' else None
                results.append(row)
    else:
        raise ValueError(f"Unsupported file format: {results_path.suffix}")
    
    print(f"[✓] Loaded {len(results)} results from {results_path}")
    return results


def categorize_results(results: List[Dict], include_correct: bool = False) -> Dict[str, List[Dict]]:
    """Categorize results into true/false predictions"""
    categories = {
        'true_positive': [],    # Correctly predicted as fake
        'true_negative': [],    # Correctly predicted as real
        'false_positive': [],   # Incorrectly predicted as fake (actually real)
        'false_negative': [],   # Incorrectly predicted as real (actually fake)
        'no_ground_truth': [],  # No ground truth available
        'failed': []            # Failed to process
    }
    
    for result in results:
        # Skip failed results
        if result.get('status') == 'error' or result.get('prediction') is None:
            categories['failed'].append(result)
            continue
        
        # Skip if no ground truth
        if result.get('ground_truth') is None:
            categories['no_ground_truth'].append(result)
            continue
        
        gt = result['ground_truth']
        pred = result['prediction']
        
        # Categorize based on ground truth and prediction
        if gt == 1 and pred == 1:
            categories['true_positive'].append(result)
        elif gt == 0 and pred == 0:
            categories['true_negative'].append(result)
        elif gt == 0 and pred == 1:
            categories['false_positive'].append(result)
        elif gt == 1 and pred == 0:
            categories['false_negative'].append(result)
    
    return categories


def print_analysis_summary(categories: Dict[str, List[Dict]]):
    """Print summary statistics"""
    tp = len(categories['true_positive'])
    tn = len(categories['true_negative'])
    fp = len(categories['false_positive'])
    fn = len(categories['false_negative'])
    failed = len(categories['failed'])
    no_gt = len(categories['no_ground_truth'])
    
    total = tp + tn + fp + fn
    correct = tp + tn
    incorrect = fp + fn
    
    print(f"\n{'='*70}")
    print("ERROR ANALYSIS SUMMARY")
    print(f"{'='*70}")
    
    if total > 0:
        print(f"\nConfusion Matrix:")
        print(f"                    Predicted")
        print(f"                Real      Fake")
        print(f"Actual  Real    {tn:4d}      {fp:4d}   (FP: False Positives)")
        print(f"        Fake    {fn:4d}      {tp:4d}   (FN: False Negatives)")
        
        print(f"\nOverall Statistics:")
        print(f"  Total samples with GT:  {total}")
        print(f"  Correct predictions:    {correct:4d} ({correct/total*100:.2f}%)")
        print(f"  Incorrect predictions:  {incorrect:4d} ({incorrect/total*100:.2f}%)")
        
        print(f"\nError Breakdown:")
        print(f"  False Positives (FP):   {fp:4d} - Real images predicted as Fake")
        print(f"  False Negatives (FN):   {fn:4d} - Fake images predicted as Real")
        
        if fp > 0 or fn > 0:
            print(f"\nError Rate:")
            print(f"  FP Rate (of real):      {fp/(tn+fp)*100:.2f}%" if (tn+fp) > 0 else "  FP Rate: N/A")
            print(f"  FN Rate (of fake):      {fn/(fn+tp)*100:.2f}%" if (fn+tp) > 0 else "  FN Rate: N/A")
    
    if no_gt > 0:
        print(f"\nNo ground truth:          {no_gt}")
    if failed > 0:
        print(f"Failed to process:        {failed}")
    
    print(f"{'='*70}\n")


def organize_error_analysis(
    categories: Dict[str, List[Dict]],
    output_dir: Path,
    dataset_root: Path = None,
    include_correct: bool = False,
    use_symlink: bool = False,
    dry_run: bool = False,
    verbose: bool = False
):
    """Organize images for error analysis"""
    
    # Create output directory structure
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define folder structure
    folders = {
        'false/false_positive': categories['false_positive'],
        'false/false_negative': categories['false_negative'],
    }
    
    if include_correct:
        folders['true/true_positive'] = categories['true_positive']
        folders['true/true_negative'] = categories['true_negative']
    
    # Create folders
    if not dry_run:
        for folder_path in folders.keys():
            (output_dir / folder_path).mkdir(parents=True, exist_ok=True)
    
    # Copy/link files
    total_files = sum(len(items) for items in folders.values())
    
    if dry_run:
        print(f"[DRY RUN] Would organize {total_files} files:")
        for folder_path, items in folders.items():
            if items:
                print(f"\n  {folder_path}/: {len(items)} files")
                for item in items[:5]:
                    gt_label = "fake" if item.get('ground_truth') == 1 else "real"
                    print(f"    - {gt_label}_{item['filename']}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
        return
    
    print(f"[✓] Organizing {total_files} files for error analysis...")
    
    stats = Counter()
    errors = []
    
    for folder_path, items in folders.items():
        if not items:
            continue
        
        dest_folder = output_dir / folder_path
        
        for item in tqdm(items, desc=f"Processing {folder_path}", unit="img"):
            try:
                # Determine source path
                if dataset_root:
                    src_path = dataset_root / item['filename']
                else:
                    src_path = Path(item['path'])
                
                if not src_path.exists():
                    errors.append(f"Source not found: {src_path}")
                    stats['not_found'] += 1
                    continue
                
                # Create destination filename with ground truth prefix
                gt_label = "fake" if item.get('ground_truth') == 1 else "real"
                
                # Format: gt_originalname.ext
                # e.g., real_image001.jpg or fake_image002.jpg
                dest_name = f"{gt_label}_{src_path.name}"
                dest_path = dest_folder / dest_name
                
                # Handle duplicates
                counter = 1
                while dest_path.exists():
                    stem = src_path.stem
                    ext = src_path.suffix
                    dest_name = f"{gt_label}_{stem}_{counter}{ext}"
                    dest_path = dest_folder / dest_name
                    counter += 1
                
                # Copy or symlink
                if use_symlink:
                    dest_path.symlink_to(src_path.absolute())
                    stats['symlinked'] += 1
                else:
                    shutil.copy2(src_path, dest_path)
                    stats['copied'] += 1
                
                if verbose:
                    print(f"  {src_path.name} -> {folder_path}/{dest_name}")
                
            except Exception as e:
                error_msg = f"Error processing {item.get('filename', 'unknown')}: {e}"
                errors.append(error_msg)
                stats['errors'] += 1
                if verbose:
                    print(f"  [Error] {error_msg}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("ORGANIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Files copied:     {stats['copied']}")
    print(f"  Files symlinked:  {stats['symlinked']}")
    if stats['not_found'] > 0:
        print(f"  Not found:        {stats['not_found']}")
    if stats['errors'] > 0:
        print(f"  Errors:           {stats['errors']}")
    print(f"{'='*70}")
    
    if errors:
        error_log = output_dir / "errors.log"
        with open(error_log, 'w') as f:
            f.write('\n'.join(errors))
        print(f"\n[!] Errors logged to: {error_log}")
    
    # Create README
    create_readme(output_dir, categories, include_correct)
    
    print(f"\n[✓] Error analysis complete!")
    print(f"[✓] Output directory: {output_dir.absolute()}")


def create_readme(output_dir: Path, categories: Dict, include_correct: bool):
    """Create README file explaining the organization"""
    readme_content = f"""# Error Analysis Results

## Directory Structure

```
{output_dir.name}/
├── false/                    # Incorrect predictions
│   ├── false_positive/       # Real images predicted as Fake ({len(categories['false_positive'])} files)
│   └── false_negative/       # Fake images predicted as Real ({len(categories['false_negative'])} files)
"""
    
    if include_correct:
        readme_content += f"""├── true/                     # Correct predictions
│   ├── true_positive/        # Fake images correctly predicted as Fake ({len(categories['true_positive'])} files)
│   └── true_negative/        # Real images correctly predicted as Real ({len(categories['true_negative'])} files)
"""
    
    readme_content += """└── README.md                 # This file

## Filename Format

Each file is prefixed with its ground truth label:
```
{ground_truth}_{original_filename}
```

**Examples:**
- `real_image001.jpg` - A real image
- `fake_image002.jpg` - A fake image

The folder path tells you what the prediction was, so:
- `false/false_positive/real_image001.jpg` - Real image incorrectly predicted as Fake
- `false/false_negative/fake_image002.jpg` - Fake image incorrectly predicted as Real

## Interpretation

### False Positives (FP)
- **What:** Real images that were incorrectly predicted as Fake
- **Impact:** May cause genuine content to be flagged
- **Analysis:** Look for common patterns in these images

### False Negatives (FN)
- **What:** Fake images that were incorrectly predicted as Real
- **Impact:** Fake content that slipped through detection
- **Analysis:** These are critical misses that need investigation

## Quick Analysis Tips

1. **Review False Positives:** What characteristics do real images have that confused the model?
2. **Review False Negatives:** What makes these fake images look real to the model?
3. **Sort by probability:** Files with mid-range probabilities (0.4-0.6) are harder cases
4. **Look for patterns:** Similar image types, sources, or artifacts

## Statistics

"""
    
    tp = len(categories['true_positive'])
    tn = len(categories['true_negative'])
    fp = len(categories['false_positive'])
    fn = len(categories['false_negative'])
    total = tp + tn + fp + fn
    
    if total > 0:
        readme_content += f"""- Total samples: {total}
- Correct predictions: {tp + tn} ({(tp + tn)/total*100:.2f}%)
- Incorrect predictions: {fp + fn} ({(fp + fn)/total*100:.2f}%)
- False Positive Rate: {fp/(tn+fp)*100:.2f}% ({fp} out of {tn+fp} real images)
- False Negative Rate: {fn/(fn+tp)*100:.2f}% ({fn} out of {fn+tp} fake images)
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"[✓] Created README: {readme_path}")


def main():
    args = parse_args()
    
    # Load results
    results = load_results(args.results)
    
    # Categorize results
    categories = categorize_results(results, args.include_correct)
    
    # Print summary
    print_analysis_summary(categories)
    
    # Determine if we should use symlinks
    use_symlink = args.symlink and not args.copy
    
    # Organize files
    output_dir = Path(args.output)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    
    organize_error_analysis(
        categories=categories,
        output_dir=output_dir,
        dataset_root=dataset_root,
        include_correct=args.include_correct,
        use_symlink=use_symlink,
        dry_run=args.dry_run,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
