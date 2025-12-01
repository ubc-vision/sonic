# SONIC Video Inpainting


## Video Inpainting Example

<table>
  <tr>
    <td align="center">Masked Ground Truth</td>
    <td align="center">Inpainted Output</td>
  </tr>
  <tr>
    <td><img src="video_samples/masked_videos/flower_masked.gif" width="320"></td>
    <td><img src="video_inpaint_output_samples/flower_output.gif" width="320"></td>
  </tr>
</table>

```
The video shows a cluster of daffodils in a bright, realistic photographic style, their white petals and yellow centers glowing in clean sunlight. Forward and backward movement of the daffodils. Slender green leaves surround the blossoms, with the foreground flowers kept in crisp focus while the background gently softens. The scene feels natural, with subtle highlights and shadows enhancing the flowers’ delicate textures.
```

Run the video inpainting example with:

```bash
bash run_sonic_video_inpainting.sh
```

## Environment Setup

### Installation

1. **Initialize the Wan2.1 submodule**
```bash
git submodule update --init --recursive
```

2. **Install required packages**
```bash
pip install -r video_requirements.txt
```

## Usage

### Model checkpoints

The scripts assume `Wan2.1-T2V-1.3B` lives under `checkpoints/` at the repository root so that the repo can be shared without shipping large weights. Download or symlink the released weights into that directory and (optionally) set `WAN21_T2V_CKPT_DIR` if you prefer to keep the weights elsewhere:

```bash
mkdir -p checkpoints
ln -s /path/to/Wan2.1-T2V-1.3B checkpoints/Wan2.1-T2V-1.3B
export WAN21_T2V_CKPT_DIR=/path/to/Wan2.1-T2V-1.3B
```

`run_sonic_video_inpainting.sh` already honors `WAN21_T2V_CKPT_DIR`, but if you call `sonic_video_inpaint.py` directly you can still pass `--ckpt_dir` to override the default path.

### Basic usage with auto center mask

```bash
python sonic_video_inpaint.py \
  --target_video video_samples/masked_videos/flower_832_480_pixel_inpainted.mp4 \
  --ckpt_dir checkpoints/Wan2.1-T2V-1.3B \
  --task t2v-1.3B \
  --size 832*480 \
  --prompt "The video shows a cluster of daffodils in a bright, realistic photographic style, their white petals and yellow centers glowing in clean sunlight. Foward and backward movement of the daffodills. Slender green leaves surround the blossoms, with the foreground flowers kept in crisp focus while the background gently softens. The scene feels natural, with subtle highlights and shadows enhancing the flowers’ delicate textures." \
  --num_iterations 50 \
  --step_nums 20 \
  --learning_rate 25.0 \
  --center_mask_fraction 0.5
```

### Using a custom mask video

```bash
python sonic_video_inpaint.py \
  --target_video video_samples/masked_videos/hike.mp4 \
  --mask video_samples/masks/hike_mask.mp4 \
  --ckpt_dir checkpoints/Wan2.1-T2V-1.3B \
  --task t2v-1.3B \
  --size 832*480 \
  --prompt "A side view of a man hiking in a rocky mountain landscape, wearing a checkered shirt, gray cargo pants, and a large black backpack. He is walking on a gravel trail surrounded by boulders and cliffs, with steep rocky slopes and dramatic mountains in the background under daylight." \
  --num_iterations 75 \
  --step_nums 20 \
  --learning_rate 9.0
```

See [run_sonic_video_inpainting.sh](run_sonic_video_inpainting.sh) for more examples.

### Output

Results are saved to `video_results/{timestamp}_{video_name}_steps{step_nums}_iter{num_iterations}/`:
- `{video_name}_masked.mp4` - Masked target video
- `epsilon/` - Optimized noise at each iteration
- `x_0_hat/` - Predicted clean videos during optimization
- `mask_latent.pt` - Latent space mask


