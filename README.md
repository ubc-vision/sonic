# SONIC: Spectral Optimization of Noise for Inpainting with Consistency

[![button](https://img.shields.io/badge/Project%20Website-orange?style=for-the-badge)](https://ubc-vision.github.io/sonic/)
[![button](https://img.shields.io/badge/Paper-blue?style=for-the-badge)](https://arxiv.org/abs/2511.19985)



<span class="author-block">
  <a href="">Seungyeon Baek</a>,
</span>
<span class="author-block">
  <a href="">Erqun Dong</a>,
</span>
<span class="author-block">
  <a href="">Shadan Namazifard</a>,
</span>
<span class="author-block">
  <a href="">Mark J. Matthews</a>,
</span>
<span class="author-block">
  <a href="https://www.cs.ubc.ca/~kmyi/">Kwang Moo Yi</a>
</span>

<hr>

## Inpainting Example
Given a nearest-pixel inpainted image (left) and its corresponding prompt, we can inpaint the image via noise optimization.

<table>
  <tr>
    <td align="center">Input</td>
    <td align="center">Inpainted Output</td>
  </tr>
  <tr>
    <td><img src="samples/FFHQ/00064.png" width="300"/></td>
    <td><img src="inpaint_output_samples/FFHQ/inpainted_00064.png" width="300"/></td>
  </tr>
</table>

```
"A young man with short black hair styled upward, dark brown eyes, and fair skin with light stubble. He has well-defined eyebrows and is wearing a black collar or shirt. The background is a clean white."
```

<hr>

## Environment Setup

### Installation

1. **Clone the repository**
```bash
git clone git@github.com:ubc-vision/sonic.git
cd sonic
```

2. **Install PyTorch**

Install PyTorch with CUDA support. Visit [https://pytorch.org](https://pytorch.org) for installation instructions.

3. **Install required packages**
```bash
pip install diffusers transformers accelerate pillow numpy
```

## Usage

Run the inpainting script with the following command:

```bash
python sonic_inpaint.py \
    --dataset_name FFHQ \
    --image_index 00064 \
    --num_iterations 20 \
    --step_nums 20 \
    --CFG_scale 2.0 \
    --learning_rate 3.0
```

### Arguments

- `--dataset_name`: Dataset name (`BrushBench`, `DIV2K`, or `FFHQ`) [required]
- `--image_index`: Image index/name (e.g., `000000069` for BrushBench, `00088` for DIV2K/FFHQ) [required]
- `--image_path`: Path to input image (optional, auto-constructed if not provided)
- `--mask_path`: Path to inpainting mask (optional, auto-constructed if not provided)
- `--prompt`: Text prompt for inpainting (optional, read from txt file if not provided)
- `--num_iterations`: Number of optimization iterations (default: 20)
- `--step_nums`: Number of steps for ODE solver (default: 20)
- `--CFG_scale`: CFG scale for velocity prediction (default: 2.0)
- `--learning_rate`: Learning rate for optimization (default: 3.0)
- `--seed`: Random seed for reproducibility (default: 200)

### Output

Results are saved to `inpaint_results/{dataset_name}_{image_name}_steps{step_nums}_iter{num_iterations}/`:
- `target_image.png` - Masked target image
- `epsilon/` - Optimized noise at each iteration
- `x_0_hat/` - Predicted clean images during optimization
- `inpainted_output/inpainted.png` - Final inpainted result

## Code Release
- ✅ Inpainting code with sample images and prompts
- ✅ Environment setup guide
- ⬜ Metrics code from FLAIR and BrushNet

