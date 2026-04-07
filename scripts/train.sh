#!/bin/bash
# LAM Training Script
# Usage: bash scripts/train.sh [config_name]
# Example: bash scripts/train.sh lam_small
#          bash scripts/train.sh lam_medium

CONFIG_NAME=${1:-lam_small}
CONFIG_PATH="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

echo "Training LAM with config: $CONFIG_PATH"

cd "$(dirname "$0")/.."

python train.py fit --config "$CONFIG_PATH"
