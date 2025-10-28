#!/bin/bash

# Simple Grid Search
# Edit the arrays below to test different parameters

# Parameters to test
PATCH_SIZES=(512 1024 2048)
SIZE_THRESHOLDS=(1024 2048)
AGGREGATIONS=(max mean)
PATCH_THRESHOLDS=(0.4 0.5 0.6)

# Fixed settings
CONFIG="./config/detector/effort.yaml"
WEIGHTS="./weights/effort_clip_L14_trainOn_sdv14.pth"
IMAGE="../../datasets/TISDC2025/"
RESULTS="./my_results/grid"

mkdir -p "$RESULTS"
echo "run,patch_size,size_threshold,aggregation,patch_threshold,accuracy,f1_score" > "$RESULTS/summary.csv"

run=0
for ps in "${PATCH_SIZES[@]}"; do
  for st in "${SIZE_THRESHOLDS[@]}"; do
    for agg in "${AGGREGATIONS[@]}"; do
      for pt in "${PATCH_THRESHOLDS[@]}"; do
        run=$((run+1))
        dir="$RESULTS/run_${run}"
        mkdir -p "$dir"
        
        echo "[$run] ps=$ps st=$st agg=$agg pt=$pt"
        
        python inference.py \
          --detector_config "$CONFIG" \
          --weights "$WEIGHTS" \
          --image "$IMAGE" \
          --patch_size $ps \
          --size_threshold $st \
          --aggregation $agg \
          --patch_threshold $pt \
          --batch_size 32 \
          --results_dir "$dir" \
          > "$dir/log.txt" 2>&1
        
        # Extract metrics
        metrics=$(find "$dir" -name "*_metrics.json" | head -n 1)
        if [ -f "$metrics" ]; then
          acc=$(python3 -c "import json; print(json.load(open('$metrics'))['accuracy'])")
          f1=$(python3 -c "import json; print(json.load(open('$metrics'))['f1_score'])")
          echo "$run,$ps,$st,$agg,$pt,$acc,$f1" >> "$RESULTS/summary.csv"
          echo "  acc=$acc f1=$f1"
        fi
      done
    done
  done
done

echo ""
echo "Done! Results in: $RESULTS/summary.csv"
echo ""
echo "Best result:"
python3 -c "
import pandas as pd
df = pd.read_csv('$RESULTS/summary.csv')
best = df.loc[df['f1_score'].idxmax()]
print(f\"  Run {int(best['run'])}: F1={best['f1_score']:.4f}\")
print(f\"  --patch_size {int(best['patch_size'])} --size_threshold {int(best['size_threshold'])} --aggregation {best['aggregation']} --patch_threshold {best['patch_threshold']}\")
"

