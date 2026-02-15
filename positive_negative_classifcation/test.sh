#!/bin/bash
set -euo pipefail

# update according to your python path
PYTHON="python3"

declare -A models=(
  [distribution]="best_model__2026_01_19__13_53_21_distribution"
  [mean]="best_model__2026_01_19__14_40_37_mean"
  [attention]="best_model__2026_01_19__16_56_38_attention"
  [max]="best_model__2026_01_19__18_23_53_max"
  [gmm]="best_model__2026_01_30__10_18_20_gmm"
)

# Filters to evaluate (order matters only if you keep num_features_arr aligned by index)
mil_filter_array=( "distribution" "mean" "attention" "max" "gmm" )

# One num_features per filter (aligned by index with mil_filter_array)
num_features_arr=( 32 32 32 32 32 )

# Sanity check: arrays must match length
if [[ ${#mil_filter_array[@]} -ne ${#num_features_arr[@]} ]]; then
  echo "Error: mil_filter_array and num_features_arr must have the same length" >&2
  exit 1
fi

for i in "${!mil_filter_array[@]}"; do
  MIL_FILTER="${mil_filter_array[$i]}"
  NUM_FEATURES="${num_features_arr[$i]}"

  # Lookup model name by filter key
  if [[ -z "${models[$MIL_FILTER]:-}" ]]; then
    echo "Error: no model configured for mil_filter '$MIL_FILTER'" >&2
    exit 1
  fi
  model_name="${models[$MIL_FILTER]}"

  echo "MIL_FILTER=$MIL_FILTER"
  MODEL_FILE="saved_models/${model_name}.pth"
  echo "MODEL_FILE=$MODEL_FILE"
  echo "NUM_FEATURES=$NUM_FEATURES"

  "$PYTHON" test.py \
    --mil_pooling_filter "$MIL_FILTER" \
    --init_model_file "$MODEL_FILE" \
    --num_features "$NUM_FEATURES"

  DATA_FOLDER_PATH="test_metrics/${model_name}/test"
  echo "DATA_FOLDER_PATH=$DATA_FOLDER_PATH"

  "$PYTHON" collect_statistics_over_bag_predictions.py \
    --data_folder_path "$DATA_FOLDER_PATH"

  echo "----"
done
