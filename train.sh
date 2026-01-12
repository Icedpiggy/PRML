#!/bin/bash

# Simple training script for Transformer policy network

python train.py \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --lr 1e-4 \
    --d-model 128 \
    --nhead 8 \
    --num-layers 4 \
    --dim-feedforward 512 \
    --dropout 0.1 \
    --max-seq-len 5000 \
    --seed 42 \
    --save-dir checkpoints \
    --device cuda \
    --early-stopping-patience 100 \
    --early-stopping-delta 1e-6 \
    --obs-embed-hidden 256 \
    --obs-embed-layers 3
