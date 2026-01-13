#!/bin/bash

# Simple testing script for trained Transformer policy network

python test.py \
    --model-path checkpoints_ez/final_model.pth \
    --episodes 10 \
    --max-steps 5000 \
    --device cuda \
    --view front \
    --randomize \
    --debug \
    --no-render \
    --seed 42 \
    --save-dir test_env_results \
    --easy
