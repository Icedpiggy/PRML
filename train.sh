#!/bin/bash

# Training script for Transformer policy network with class imbalance solutions

python train.py \
    --data-dir data \
    --epochs 1000 \
    --batch-size 8 \
    --lr 1e-4 \
    --d-model 64 \
    --nhead 8 \
    --num-layers 4 \
    --dim-feedforward 256 \
    --dropout 0.1 \
    --max-seq-len 5000 \
    --seed 42 \
    --save-dir checkpoints \
    --device cuda \
    --early-stopping-patience 1000 \
    --early-stopping-delta 1e-6 \
    --pos-speed 0.5 \
    --rot-speed 0.5 \
    --obs-embed-hidden 128 \
    --obs-embed-layers 3 \
    --use-class-weights \
    --entropy-weight 0.01 \
    --easy
