# Testing Effort Model with TISDC2025 Dataset

## Setup Complete! ✅

Your TISDC2025 dataset has been successfully configured for testing with the Effort model.

### Dataset Summary
- **Location**: `datasets/TISDC2025/`
- **Total images**: 561
  - Real images: 536
  - Fake images: 25
- **Label method**: Based on filename (contains "real" or "fake")

### Files Created/Modified

1. **JSON configuration**: `DeepfakeBench/preprocessing/dataset_json/TISDC2025.json`
   - Contains the dataset structure required by the Effort framework
   - Maps each image to its label based on filename

2. **Test configuration**: `DeepfakeBench/training/config/test_config.yaml`
   - Added TISDC2025 label mappings:
     - `TISDC2025_Real: 0`
     - `TISDC2025_Fake: 1`

## How to Test

### Option 1: Test with the Pre-trained Checkpoint

Run the following command to test with the existing checkpoint trained on SDv1.4:

```bash
cd DeepfakeBench/

python3 training/test.py \
  --detector_path ./training/config/detector/effort.yaml \
  --test_dataset TISDC2025 \
  --weights_path ./training/weights/effort_clip_L14_trainOn_sdv14.pth
```

### Option 2: Test with Your Own Checkpoint

If you have downloaded or trained a different checkpoint:

```bash
cd DeepfakeBench/

python3 training/test.py \
  --detector_path ./training/config/detector/effort.yaml \
  --test_dataset TISDC2025 \
  --weights_path /path/to/your/checkpoint.pth
```

### Option 3: Test Multiple Datasets at Once

You can test TISDC2025 along with other datasets:

```bash
cd DeepfakeBench/

python3 training/test.py \
  --detector_path ./training/config/detector/effort.yaml \
  --test_dataset TISDC2025 Celeb-DF-v2 DFDC \
  --weights_path ./training/weights/effort_clip_L14_trainOn_sdv14.pth
```

## Expected Output

The test script will output:
- **Per-dataset metrics**:
  - AUC (Area Under Curve)
  - ACC (Accuracy)
  - EER (Equal Error Rate)
  - AP (Average Precision)
- **Prediction probabilities** for each image
- **Feature vectors** extracted by the model

Example output:
```
dataset: TISDC2025
auc: 0.9523
acc: 0.9125
eer: 0.0876
ap: 0.9634
```

## Notes

### Image Resolution
- The model will automatically resize images to the configured resolution (default: 224x224 or 256x256)
- Original aspect ratios are not preserved during resizing

### Batch Size
- Default test batch size is defined in the config files
- If you encounter memory issues, you can modify the batch size in `training/config/test_config.yaml`

### Face Detection
- The checkpoint `effort_clip_L14_trainOn_sdv14.pth` was trained on general AI-generated images (not faces)
- If your TISDC2025 images are natural scene images (not faces), this checkpoint should work well
- If they are face images and you want face-specific processing, add the landmark model parameter:
  ```bash
  --landmark_model ./preprocessing/shape_predictor_81_face_landmarks.dat
  ```

## Troubleshooting

### If you get "dataset not found" error:
- Check that `TISDC2025.json` exists in `DeepfakeBench/preprocessing/dataset_json/`
- Verify the JSON file contains the correct absolute paths to your images

### If you get "label not found" error:
- Check that `test_config.yaml` contains the TISDC2025 label definitions
- Ensure image filenames contain either "real" or "fake"

### If some images are skipped:
- Check the console output for warnings about unlabeled images
- Image filenames must contain "real" or "fake" (case-insensitive)

## Customization

### To modify the labeling logic:
Edit `create_tisdc2025_json.py` and rerun it to regenerate the JSON file.

### To test only real or fake images:
Manually edit `TISDC2025.json` to include only the desired subset.

### To change the test/train split:
Currently all images are in the "test" split. You can manually reorganize the JSON structure to create a "train" split if needed.

## Additional Resources

- **Main README**: See `README.md` for more information about the Effort model
- **DeepfakeBench**: https://github.com/SCLBD/DeepfakeBench
- **Paper**: https://arxiv.org/abs/2411.15633
