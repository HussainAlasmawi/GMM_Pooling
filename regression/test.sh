#!/bin/bash
set -euo pipefail

PYTHON="python3"
MODELS_DIR="saved_models"

declare -A NUM_FEATURES_BY_FILTER=(
  [distribution]=32
  [mean]=32
  [attention]=32
  [max]=32
  [gmm]=32
)

ALLOWED_FILTERS_REGEX='^(distribution|mean|attention|max|gmm)$'

shopt -s nullglob
model_files=( "${MODELS_DIR}"/best_model__*.pth )

if (( ${#model_files[@]} == 0 )); then
  echo "No models found matching ${MODELS_DIR}/best_model__*.pth" >&2
  exit 1
fi

for MODEL_FILE in "${model_files[@]}"; do
  filename="$(basename "$MODEL_FILE")"
  model_name="${filename%.pth}"

  # Parse filter from:
  # best_model__YYYY_MM_DD__HH_MM_SS_<filter>_seedXYZ.pth
  base="${filename%.pth}"
  pre_seed="${base%_seed*}"      # -> best_model__...__HH_MM_SS_filter
  MIL_FILTER="${pre_seed##*_}"   # -> filter

  if [[ -z "$MIL_FILTER" || ! "$MIL_FILTER" =~ $ALLOWED_FILTERS_REGEX ]]; then
    echo "Skipping (could not parse/unsupported MIL_FILTER='$MIL_FILTER'): $filename" >&2
    echo "----"
    continue
  fi

  NUM_FEATURES="${NUM_FEATURES_BY_FILTER[$MIL_FILTER]:-}"
  if [[ -z "$NUM_FEATURES" ]]; then
    echo "Skipping (no NUM_FEATURES configured for '$MIL_FILTER'): $filename" >&2
    echo "----"
    continue
  fi

  echo "MIL_FILTER=$MIL_FILTER"
  echo "MODEL_FILE=$MODEL_FILE"
  echo "NUM_FEATURES=$NUM_FEATURES"
  DATA_FOLDER_PATH="test_metrics/${model_name}/test"

  if [[ -d "$DATA_FOLDER_PATH" ]]; then 
    echo "Skipping (results already exist): $DATA_FOLDER_PATH"
    echo "----"
    continue
  fi
  echo "DATA_FOLDER_PATH=$DATA_FOLDER_PATH"

  "$PYTHON" test.py \
    --mil_pooling_filter "$MIL_FILTER" \
    --init_model_file "$MODEL_FILE" \
    --num_features "$NUM_FEATURES" \

  "$PYTHON" collect_statistics_over_bag_predictions.py \
    --data_folder_path "$DATA_FOLDER_PATH"


  echo "----"
done
