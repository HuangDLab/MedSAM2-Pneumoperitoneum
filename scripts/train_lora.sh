#!/bin/bash
# =============================================================================
# Launch LoRA fine-tuning.
#
#   bash scripts/train_lora.sh [config] [output_dir]
#
# Defaults to the stage-3 configuration used for the reported model. Earlier
# stages used the same configuration with a different initial checkpoint and a
# different training NPZ folder; see the README.
#
# The base SAM 2.1 checkpoint must be at checkpoints/sam2.1_hiera_tiny.pt.
# =============================================================================

set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH

CONFIG="${1:-configs/lora_encode_decode_sam2.1_hiera_tiny512_FLARE_RECIST.yaml}"
OUTPUT_PATH="${2:-./exp_log/lora_lung_window}"

# Single-GPU training. The launcher is told --num-gpus 1, so exactly one device
# must be visible; exposing more would be ignored and is misleading.
export CUDA_VISIBLE_DEVICES=0

echo "config : $CONFIG"
echo "output : $OUTPUT_PATH"

python training/train.py \
    -c "$CONFIG" \
    --output-path "$OUTPUT_PATH" \
    --use-cluster 0 \
    --num-gpus 1 \
    --num-nodes 1

echo "training done"
# The settings actually used are written to $OUTPUT_PATH/config_resolved.yaml
# Checkpoints are saved every 10 epochs to $OUTPUT_PATH/checkpoints/
# Merge the LoRA weights afterwards with scripts/merge_lora_weights.py
