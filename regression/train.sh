#!/bin/bash

# update according to your python path
PYTHON="python3"

model_file_arr=( "None" "None" "None" "None" "None" )

num_features_arr=( 32 32 32 32 32 )

num_epochs_arr=( 5000 5000 5000 5000 5000 )

mil_filter_array=( "distribution" "mean" "attention" "max" "gmm" )

count=0
seeds=( 10 42 1337)

M=10
K=5
T=40

for MIL_FILTER in "${mil_filter_array[@]}"
do
        for seed in "${seeds[@]}"
        do
                echo "MIL_FILTER="$MIL_FILTER

                MODEL_FILE=${model_file_arr[$count]}

                NUM_FEATURES=${num_features_arr[$count]}

                NUM_EPOCHS=${num_epochs_arr[$count]}

                $PYTHON train.py --mil_pooling_filter $MIL_FILTER --init_model_file $MODEL_FILE --num_features $NUM_FEATURES --num_epochs $NUM_EPOCHS --seed $seed --M $M --K $K --T $T --no_sigmoid  


        done
        count=$((count + 1))
done