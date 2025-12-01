#!/bin/bash

# Example script for running SONIC video inpainting
# Spectral Optimization of Noise for Video Inpainting with Consistency

# Navigate to the script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default to checkpoints/Wan2.1-T2V-1.3B relative to the repo root but allow overrides.
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WAN21_T2V_CKPT_DIR="${WAN21_T2V_CKPT_DIR:-$REPO_ROOT/checkpoints/Wan2.1-T2V-1.3B}"

# Example 1: Basic usage with auto center mask (default)
python sonic_video_inpaint.py \
  --target_video video_samples/masked_videos/flower_832_480_pixel_inpainted.mp4 \
  --ckpt_dir "$WAN21_T2V_CKPT_DIR" \
  --task t2v-1.3B \
  --size 832*480 \
  --prompt "The video shows a cluster of daffodils in a bright, realistic photographic style, their white petals and yellow centers glowing in clean sunlight. Foward and backward movement of the daffodills. Slender green leaves surround the blossoms, with the foreground flowers kept in crisp focus while the background gently softens. The scene feels natural, with subtle highlights and shadows enhancing the flowers’ delicate textures." \
  --negative_prompt "" \
  --num_iterations 50 \
  --step_nums 20 \
  --sample_shift 8.0 \
  --CFG_scale 2.0 \
  --learning_rate 25.0 \
  --device_id 0 \
  --seed 200 \
  --out_dir video_results \
  --save_every 5 \
  --center_mask_fraction 0.5

# Example 2: Using a custom mask video
# python sonic_video_inpaint.py \
#   --target_video video_samples/masked_videos/hike.mp4 \
#   --mask video_samples/masks/hike_mask.mp4 \
#   --ckpt_dir "$WAN21_T2V_CKPT_DIR"  \
#   --task t2v-1.3B \
#   --size 832*480 \
#   --prompt "A side view of a man hiking in a rocky mountain landscape, wearing a checkered shirt, gray cargo pants, and a large black backpack. He is walking on a gravel trail surrounded by boulders and cliffs, with steep rocky slopes and dramatic mountains in the background under daylight." \
#   --num_iterations 75 \
#   --step_nums 20 \
#   --sample_shift 8.0 \
#   --CFG_scale 2.0 \
#   --learning_rate 9.0 \
#   --device_id 0 \
#   --seed 222 \
#   --out_dir video_results \
#   --save_every 1
