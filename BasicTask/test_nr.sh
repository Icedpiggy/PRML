#!/bin/bash

# Simple testing script for trained Transformer policy network

python test.py \
    --model-path checkpoints_nr/final_model.pth \
    --episodes 100 \
    --max-steps 2000 \
    --device cuda \
    --seed 42 \
    --view front \
    --no-render
