#!/bin/bash

# Simple testing script for trained Transformer policy network

python test.py \
    --model-path checkpoints/final_model.pth \
    --episodes 1 \
    --max-steps 2000 \
    --device cuda \
    --view front \
    --randomize \
    --debug \
    --speed 1.0
