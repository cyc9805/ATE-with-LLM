#!/bin/bash
cd ../
model=(facebook/bart-large)
dataset=(ACTER ACL-RD BCGM)
for j in "${model[@]}"; do
    for k in "${dataset[@]}"; do
        CUDA_VISIBLE_DEVICES=4,7 python main.py \
        --config_path configs/train.json \
        --model_name $j \
        --dataset_name $k
    done
done