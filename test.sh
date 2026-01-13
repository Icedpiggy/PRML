#!/bin/bash

# Simple testing script for trained Transformer policy network

python test.py \
    --model-path checkpoints/best_model.pth \
    --episodes 10 \
    --max-steps 2000 \
    --device cuda \
    --view front \
    --randomize \
    --debug \
    --speed 1.0 \
    --save-dir test_env_results
