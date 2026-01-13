#!/bin/bash

# Training script for Transformer policy network with aggressive regularization to prevent overfitting

python train.py \
    --data-dir data_r \
    --epochs 1000 \
    --batch-size 8 \
    --lr 5e-4 \
    --d-model 64 \
    --nhead 8 \
    --num-layers 4 \
    --dim-feedforward 256 \
    --dropout 0 \
    --max-seq-len 2000 \
    --seed 42 \
    --save-dir checkpoints_r \
    --device cuda \
    --early-stopping-patience 1000 \
    --early-stopping-delta 1e-6 \
    --obs-embed-hidden 128 \
    --obs-embed-layers 3 \
    --use-class-weights \
    --entropy-weight 0.03

# no dropout
# no early-stop