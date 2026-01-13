#!/bin/bash

# Simple testing script for trained Transformer policy network

python test.py \
    --model-path checkpoints_r/checkpoint_epoch_800.pth \
    --episodes 100 \
    --max-steps 2000 \
    --device cuda \
    --seed 42 \
    --view front \
    --randomize \
    --no-render
