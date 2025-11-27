
## Metrics Evaluation

### Reconstruction Metrics (PSNR, LPIPS, SSIM, FID)

Evaluate inpainting results using reconstruction metrics:

```bash
python metrics/reconstruction_metrics.py \
    --folder_path /path/to/inpainted/images \
    --gt_folder_path /path/to/ground/truth/images \
    --resolution 1024
```

**Output:**
- `metrics_results/reconstruction_metrics_{timestamp}.txt` - Detailed metrics with standard deviations
- `metrics_results/reconstruction_metrics_log.csv` - Global CSV log of all runs

### Perceptual Quality Metrics (Image Reward, HPS v2.1, Aesthetic Score, CLIP Similarity)

Evaluate inpainting results using prompt-based quality metrics:

```bash
python metrics/perceptual_metrics.py \
    --folder_path /path/to/inpainted/images \
    --prompts_json /path/to/prompts.json \
    --resolution 1024
```

**Output:**
- `metrics_results/perceptual_metrics_summary_{timestamp}.csv` - Summary of averaged metrics
- `metrics_results/perceptual_metrics_detailed_{timestamp}.csv` - Per-image detailed results
- `metrics_results/perceptual_metrics_log.csv` - Global CSV log of all runs